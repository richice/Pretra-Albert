from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    AlbertForMaskedLM,
    DataCollatorForLanguageModeling,
    AdamW,
    get_cosine_schedule_with_warmup,
    TrainingArguments,
    set_seed,
)
from transformers.adapters import (AdapterType,
                                   AdapterConfig,
                                   AdapterCompositionBlock,
                                   AdapterTrainer)

# 部分1
# 载入模型和分词器，并冻结模型参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("voidful/albert_chinese_base")
new_vocab_file = "/content/drive/MyDrive/BERT-NER-dev/data/new_vocab.txt"
tokenizer.add_tokens(new_vocab_file, special_tokens=True)
model = AlbertForMaskedLM.from_pretrained("voidful/albert_chinese_base")
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# 冻结模型参数
for name, param in model.named_parameters():
    if 'adapter' not in name:
        param.requires_grad = False

# 添加适配器层
adapter_name = "mlm"
config = AdapterConfig.load(adapter_name)
config.non_linearity = "swish"
config.adapter_dim = 256
config.adapter_type = AdapterType.text_lang
model.add_adapter(adapter_name, config=config)
model.train_adapter([adapter_name])
model.set_active_adapters(adapter_name)
model.config.gradient_checkpointing = True

adapter_layer = model.albert.encoder.albert_layer_groups[0].albert_layers[2].output
if not adapter_layer.adapters:
    adapter_layer.adapters = AdapterCompositionBlock()
    adapter_layer.adapters.add(adapter_name)

# Load dataset
class TextDataset(Dataset):
    """
    数据集类，用于读取和处理文本数据集。
    """

    def __init__(self, tokenizer, file_path, block_size):
        """
        初始化函数，读取文本数据集并进行分块处理。

        参数：
        tokenizer: 用于将文本转化为模型输入的分词器。
        file_path: 数据集的路径。
        block_size: 分块大小，即每个输入序列的长度。

        返回：
        无。
        """
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
            )

    def __len__(self):
        """
        返回数据集大小。
        """
        return len(self.examples)

    def __getitem__(self, i):
        """
        获取数据集中的一条样本。
        参数：        i: 样本的索引。
        返回：        一个PyTorch张量，代表输入序列的编码结果。
        """
        return torch.tensor(self.examples[i])

# 部分2
block_size = 128
num_train_epochs = 8
per_device_train_batch_size = 16 # should be 16 to match gradient_accumulation_steps=2
warmup_steps = 1000
mlm_probability = 0.1
dropout_rate = 0.3
weight_decay = 0.01
gradient_accumulation_steps = 4
learning_rate = 2.5e-6 # should be the original value divided by gradient_accumulation_steps
learning_rate_scheduler = 'cosine'
adapter_learning_rate = 1e-4

# Set seed for reproducibility
set_seed(42)

dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="/content/drive/MyDrive/BERT-NER-dev/data/combined.txt",
    block_size=block_size,
)

# Create data collator for Masked Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
)

# 修改优化器
optimizer = AdamW(
    [        {"params": model.albert.encoder.albert_layer_groups[0].albert_layers[2].output.adapters[0].parameters(), "lr": adapter_learning_rate}
    ],
    lr=adapter_learning_rate,
    correct_bias=False,
    weight_decay=weight_decay
)


scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=len(dataset) * num_train_epochs,
    num_cycles=0.5
)


# 部分3
# Pretraining training arguments
training_args = TrainingArguments(
    output_dir="./albert-matching-model",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    save_steps=5000,
    save_total_limit=2,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    lr_scheduler_type=learning_rate_scheduler,
    warmup_steps=warmup_steps,
    logging_dir='./logs',
    logging_steps=500,
    report_to="none",
)

# Start training
adapter_trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    adapters=[adapter_name],
    adapter_names=[adapter_name],
    optimizers=[optimizer],
    scheduler=scheduler,
    disable_tqdm=False
)
adapter_trainer.train()


adapter_optimizer, adapter_scheduler = AdapterTrainer.create_optimizer_and_scheduler(
    model, adapter_names=["adapter"],
    learning_rate=learning_rate, adapter_learning_rate=adapter_learning_rate,
    adapter_warmup_steps=warmup_steps, adapter_training_steps=len(dataset) * num_train_epochs,
    adapter_learning_rates=[adapter_learning_rate]
)
# # set up logger
# logger = SimpleLogger()

## train loop
per_device_train_batch_size = 16  # should be 16 to match gradient_accumulation_steps=2
gradient_accumulation_steps = 4
batch_size = per_device_train_batch_size * gradient_accumulation_steps
num_train_epochs = 8
total_steps = len(dataset) * num_train_epochs // batch_size
num_warmup_steps = 1000

# Create dataloader
train_dataloader = DataLoader(
    dataset, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=data_collator
)

model.train()
model.zero_grad()
optimizer.zero_grad()

# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
adapter_optimizer = torch.optim.Adam(model.get_adapter(adapter_name).parameters(), lr=adapter_learning_rate)
total_steps = len(dataset) * num_train_epochs // gradient_accumulation_steps
progress_bar = tqdm(range(total_steps), desc="Training")

for epoch in range(num_train_epochs):
    epoch_loss = 0.0
    epoch_steps = 0

    for step, inputs in enumerate(DataLoader(dataset, batch_size=per_device_train_batch_size)):

        inputs = inputs.to(device)
        inputs_masked, labels = mask_tokens(inputs, tokenizer, mlm_probability)
        inputs_masked = inputs_masked.to(device)
        labels = labels.to(device)

        outputs = model(inputs_masked, attention_mask=(inputs_masked != tokenizer.pad_token_id))
        logits = outputs.logits

        # 只计算适配器层的损失
        adapter_outputs = model.get_adapter(adapter_name)(outputs.last_hidden_state)
        adapter_logits = adapter_outputs.mean(dim=1)
        adapter_loss = loss_fn(adapter_logits.view(-1, model.config.vocab_size), labels.view(-1))

        # 反向传播并更新适配器层参数
        adapter_loss.backward()
        adapter_optimizer.step()
        adapter_optimizer.zero_grad()

        # 每gradient_accumulation_steps个batch更新一次主模型参数
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += adapter_loss.item()
        epoch_steps += 1
        progress_bar.update(1)
        progress_bar.set_postfix(epoch=epoch + 1, loss=epoch_loss / epoch_steps)

    # 输出每个epoch的平均损失
    print(f"Epoch {epoch}/{num_train_epochs}, Average loss: {epoch_loss / epoch_steps:.4f}")

## save model to disk once training is done
model.remove_adapter(adapter_name)
model.save_pretrained("./aec_albert-adapter")
tokenizer.save_pretrained("./aec_albert-adapter")
