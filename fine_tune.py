from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

file_path = "spiritual_guidance.txt"  # Dataset file
dataset = load_dataset(file_path, tokenizer)

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # False for causal language modeling
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-spiritual",  # Directory to save the model
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust epochs based on dataset size
    per_device_train_batch_size=1,  # Use small batch size due to memory constraints
    save_steps=200,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
)

# Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gpt2-finetuned-spiritual")
tokenizer.save_pretrained("./gpt2-finetuned-spiritual")
