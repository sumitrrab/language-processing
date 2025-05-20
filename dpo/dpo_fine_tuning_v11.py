import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig
import torch
import os

# ====================== Configuration ======================
model_path = "/home/stud447/phi3p5_lora_styleA"
data_path = "/home/stud447/labeled_titles.csv"
output_dir = "./dpo_fine_tuned_model"

# ====================== Dataset Loading ======================
def load_and_format_data(file_path):
    """Load and format dataset with explicit prompts"""
    df = pd.read_csv(file_path)
    
    processed_data = []
    for _, row in df.iterrows():
        if row["Candidate 2"].strip().lower() == "both are similar":
            continue
        
        prompt = (
            "Hey there! Check out this awesome place:\n"
            + row["Description"]
            + "\n\nPlease provide ONLY a short, fun title. End your response with <END>."
        )
        chosen = row["Chosen Title"]
        rejected = row["Candidate 2"] if chosen == row["Candidate 1"] else row["Candidate 1"]
        
        processed_data.append({
            "prompt": prompt,
            "chosen": chosen + " <END>",  # Add explicit end token
            "rejected": rejected + " <END>"
        })
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))

# ====================== Model Loading ======================
def load_model_and_tokenizer(model_path):
    """Load base model with PEFT adapters and tokenizer"""
    print(f" Loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
        )
    print(f" Loading base model from {model_path}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,  # 4-bit quantization
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(" Wrapping model with PEFT adapters", flush=True)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.enable_adapters()
    
    return model, tokenizer

# ====================== Training Setup ======================
def get_peft_config():
    """LoRA configuration optimized for Phi-3"""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "o_proj"],  # Critical for Phi-3
        task_type="CAUSAL_LM"
    )

def get_dpo_config():
    """DPO-specific training configuration"""
    return DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        beta=0.1,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True  # Enable gradient checkpointing
    )

# ====================== Main Execution ======================
if __name__ == "__main__":
    # Load and prepare data
    print("\n=== Loading dataset ===", flush=True)
    dataset = load_and_format_data(data_path)

    # Load model components
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Tokenization
    print("\n=== Tokenizing dataset ===", flush=True)
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["prompt"],
            text_pair=examples["chosen"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize rejected separately
        rejected_tokens = tokenizer(
            examples["prompt"],
            text_pair=examples["rejected"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"]
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Initialize DPO Trainer
    print("\n=== Initializing DPO Trainer ===", flush=True)
    dpo_trainer = DPOTrainer(
        model=model,
        args=get_dpo_config(),
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config()
    )

    # Start training
    print("\n Starting training...", flush=True)
    dpo_trainer.train()

    # Save final model
    print("\n Saving trained model...", flush=True)
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n Training complete! Model saved to {output_dir}", flush=True)
