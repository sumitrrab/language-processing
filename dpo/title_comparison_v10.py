import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import os

# ====================== Configuration ======================
original_lora_path = "/home/stud447/phi3p5_lora_styleA"  # Path to the original LoRA model
dpo_lora_path = "./dpo_fine_tuned_model"  # Path to the DPO-fine-tuned LoRA model
unseen_data_path = "/home/stud447/unseen_listings.csv"  # Path to unseen listings
output_csv_path = "./generated_titles.csv"  # Output file for generated titles

# Check if paths exist
assert os.path.exists(original_lora_path), "Original LoRA model path missing"
assert os.path.exists(dpo_lora_path), "DPO LoRA model path missing"
assert os.path.exists(unseen_data_path), "Unseen data missing"

# ====================== Load Models and Tokenizer ======================
def load_model_and_tokenizer(model_path):
    """Load model with PEFT adapter and quantization"""
    print(f"Loading model and tokenizer from {model_path}...")
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter if present
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model
    
    return model, tokenizer

# Load original LoRA model
print("Loading original LoRA model...")
original_model, original_tokenizer = load_model_and_tokenizer(original_lora_path)
original_model.eval()

# Load DPO LoRA model
print("Loading DPO LoRA model...")
dpo_model, dpo_tokenizer = load_model_and_tokenizer(dpo_lora_path)
dpo_model.eval()

# ====================== Title Generation Functions ======================
def generate_title_original(model, tokenizer, desc):
    """
    Uses the original prompt style:
    'Hey there! Check out this awesome place:\n{desc}\n\n
     Can you come up with a short, fun title that makes people want to click?'
    """
    prompt = (
        f"Hey there! Check out this awesome place:\n{desc}\n\n"
        "Can you come up with a short, fun title that makes people want to click?"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("click?")[-1].strip()

def generate_title_dpo(model, tokenizer, desc):
    """
    Uses the new DPO-style prompt with <END>:
    'Hey there! Check out this awesome place:\n{desc}\n\n
     Please provide ONLY a short, fun title. End your response with <END>.'
    """
    prompt = (
        f"Hey there! Check out this awesome place:\n{desc}\n\n"
        "Please provide ONLY a short, fun title. End your response with <END>."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = raw_output.split("<END>")
    
    if len(parts) >= 2:
        title = parts[1].strip()
        # Remove a leading period if present
        if title.startswith("."):
            title = title[1:].strip()
        return title
    else:
        # Fallback if there's only one <END> or none
        print("No second <END> found, raw output is:", raw_output)
        return raw_output.strip()

# ====================== Main Execution ======================
if __name__ == "__main__":
    # Load unseen listings
    print("Loading unseen listings...")
    df = pd.read_csv(unseen_data_path)
    unseen_listings = df["description"].tolist()[:20]  # Use the first 20 listings for evaluation
    
    # Generate titles
    results = []
    for i, description in enumerate(unseen_listings):
        print(f"\n=== ITEM #{i} ===")
        print("Description:", description)
        
        # Generate titles
        title_orig = generate_title_original(original_model, original_tokenizer, description)
        title_dpo = generate_title_dpo(dpo_model, dpo_tokenizer, description)
        
        # Save results
        results.append({
            "listing_description": description,
            "original_title": title_orig,
            "dpo_title": title_dpo
        })
        
        print("\nOriginal Model Title:")
        print(title_orig)
        print("\nDPO Model Title:")
        print(title_dpo)
        print("--------------------------------------------------")
    
    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nGenerated titles saved to {output_csv_path}")
