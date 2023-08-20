# Pretrain-Albert
Fine-tunes albert-based-chinese model using MLM (masked language modeling) on custom dataset. 
| | Masked LM Fine-tuning | Regular Fine-tuning |
|-|------------------|--------------------|
| Objective | Masked language modeling | Downstream task objective (classification, QA etc) |   
| Input | Original text with random masked tokens | Original raw text as input |
| Optimization | Continue pretraining with MLM | Directly optimize for downstream task |
| Data | Unlabeled data can be used | Requires labeled data for downstream task |
| Pros | Utilize unlabeled data, improve language understanding | Faster convergence with supervised data |
| Cons | Require constructing auxiliary MLM task | Need annotated data for every downstream task |

## Usage

- Update block_size, num_train_epochs, per_device_train_batch_size etc.
- Run training loop
- Model saved to `aec_albert` directory

# Update Log

## 2023-02-27

- Added adapter layer to pretrained Albert model
- Froze pretrained model weights 
- Only update adapter parameters during training
- Used separate optimizer and learning rate for adapter

## 2023-03-05  

- Switched to AdapterTrainer for training loop
- Implemented gradient accumulation 
- Added learning rate scheduler 
- Updated to dynamic batch size

## 2023-03-12

- Refactored into dataloader, train loop
- Calculated adapter loss separately 
- Optimization steps for adapter vs main model
- Added progressbar for tracking
