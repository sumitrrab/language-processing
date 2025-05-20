#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel

# Define paths
base_model_id = "microsoft/phi-3.5-mini-instruct"
adapter_dir = "/home/stud447/phi3p5_lora_styleA"
input_csv_path = "/home/stud447/airbnb_tabular.csv"
output_csv_path = "/home/stud447/generated_titles.csv"

# Ensure adapter directory exists
if not os.path.exists(adapter_dir):
    raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

print(" Script started", flush=True)

# Load model with progress tracking
print(" Loading base model...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("⏳ Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(base_model, adapter_dir)

# GPU status
gpu_available = torch.cuda.is_available()
print(f" Model loaded. GPU available? {gpu_available}", flush=True)
if gpu_available:
    print(f" GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f" GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

# Load and filter data
print(" Loading dataset...", flush=True)
df = pd.read_csv(input_csv_path, engine="python", on_bad_lines="skip")
df = df[df["in_top_third"] == 1].head(50)  # Increased to 50 listings

if "description" not in df.columns:
    raise KeyError("CSV must contain 'description' column")

# Enhanced title cleaner
def clean_title(title):
    """Remove unwanted characters and truncate"""
    title = (
        title.strip()
        .replace("Amen", "")
        .replace("**", "")
        .replace("#", "")
        .strip("|().\"'")
    )
    # Remove emojis and special characters
    title = title.encode("ascii", "ignore").decode()
    return title[:60]  # Truncate to 60 characters

# Optimized prompt template
PROMPT_TEMPLATE = """Craft a catchy Airbnb title under 60 characters for this property. 
Focus on:
- Key amenities/features
- Location benefits
- Emotional appeal
Avoid markdown and special characters.

Example good title: "Sunny Downtown Studio Steps from Metro & Cafes!"

Property description:
{description}

Title:"""

def generate_titles(description, num_candidates=2):
    """Generate marketing-optimized titles"""
    candidates = []
    for _ in range(num_candidates):
        prompt = PROMPT_TEMPLATE.format(description=description)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,  # Reduce repetition
            pad_token_id=tokenizer.eos_token_id
        )
        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract text after last "Title:" 
        title = raw_text.split("Title:")[-1].strip()
        candidates.append(clean_title(title))
    
    return [t for t in candidates if t]  # Filter empty titles

# Processing loop with progress tracking
print(f" Starting generation for {len(df)} listings...", flush=True)
generated_data = []
for idx, row in df.iterrows():
    try:
        description = row["description"]
        if pd.isna(description) or not isinstance(description, str):
            continue
            
        print(f"\n Processing listing {idx+1}/{len(df)}", flush=True)
        print(f" Description: {description[:100]}...", flush=True)
        
        candidate_titles = generate_titles(description)
        print(f" Generated titles: {candidate_titles}", flush=True)
        
        generated_data.append([description] + candidate_titles)
        
    except Exception as e:
        print(f" Error processing listing {idx+1}: {str(e)}", flush=True)
        continue

# Save results
print("\n Saving results...", flush=True)
output_df = pd.DataFrame(generated_data, columns=["Description", "Candidate 1", "Candidate 2"])
output_df.to_csv(output_csv_path, index=False, encoding="utf-8")

print(f" Success! Generated {len(output_df)} titles saved to {output_csv_path}", flush=True)
