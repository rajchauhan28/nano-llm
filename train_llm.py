# train_llm.py
import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 1. Load Tokenizer and Model
tokenizer = GPT2TokenizerFast.from_pretrained("./nanollm-tokenizer")
tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_embd=512,
    n_layer=6,
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
print(f"Model size: {model.num_parameters()/1e6:.2f}M parameters")


# 2. Load and Process Dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")
# Split dataset into train and validation sets
dataset = dataset.train_test_split(test_size=0.05)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=config.n_positions)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # We are training a Causal LM, not a Masked LM
)


# 3. Set Up Trainer
training_args = TrainingArguments(
    output_dir="./nanollm-results",
    overwrite_output_dir=True,
    num_train_epochs=1, # Start with 1 epoch, increase later
    per_device_train_batch_size=8,  # Reduce if you get "CUDA out of memory"
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=1000,
    #evaluation_strategy="steps",
    eval_steps=10_000,
    fp16=torch.cuda.is_available(), # Use mixed precision for speed if possible
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)


# 4. Start Training! 🚀
print("Starting training...")
trainer.train()

# 5. Save the final model
trainer.save_model("./nanollm-final-model")
tokenizer.save_pretrained("./nanollm-final-model")
print("Training complete and model saved.")
