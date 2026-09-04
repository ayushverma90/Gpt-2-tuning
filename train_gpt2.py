# GPT-2 Fine-Tuning for Coherent Text Generation
# Install: pip install -r requirements.txt
# Run: python train_gpt2.py
#
# The script fine-tunes GPT-2 on a small custom dataset and generates text.
# For a stronger model, replace data/train.txt with a larger, domain-specific corpus.

from pathlib import Path
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
import torch

MODEL_NAME = "gpt2"
DATA_FILE = "data/train.txt"
OUTPUT_DIR = "./gpt2-finetuned"

def main():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("text", data_files={"train": DATA_FILE})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_fp16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    generator = pipeline(
        "text-generation",
        model=OUTPUT_DIR,
        tokenizer=OUTPUT_DIR,
    )

    prompt = "Artificial intelligence is"
    result = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    print("\nPROMPT:\n", prompt)
    print("\nGENERATED TEXT:\n", result)

if __name__ == "__main__":
    main()
