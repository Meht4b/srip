# smoke_test_lora.py
# Tests: BioGPT loads → LoRA wraps it → forward pass works → adapter saves/loads
# Run on CPU, no GPU needed. Should complete in ~2-3 minutes.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os

print("=" * 50)
print("SMOKE TEST: BioGPT + LoRA")
print("=" * 50)

MODEL_NAME = "microsoft/biogpt"
SAVE_PATH = "./test_lora_adapter"

# Step 1: Load BioGPT
print("\n[1/5] Loading BioGPT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
print(f"    Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 2: Wrap with LoRA
print("\n[2/5] Applying LoRA (r=8, targeting q_proj and v_proj)...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 3: Forward pass
print("\n[3/5] Running a forward pass...")
inputs = tokenizer(
    "The patient was diagnosed with type 2 diabetes and prescribed",
    return_tensors="pt"
)
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
print(f"    Loss: {outputs.loss.item():.4f}  ✓")

# Step 4: Save adapter
print(f"\n[4/5] Saving LoRA adapter to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
saved_files = os.listdir(SAVE_PATH)
print(f"    Saved files: {saved_files}  ✓")

# Step 5: Reload and verify
print("\n[5/5] Reloading adapter from disk...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
reloaded = PeftModel.from_pretrained(base_model, SAVE_PATH)
print("    Adapter reloaded successfully  ✓")

print("\n" + "=" * 50)
print("SMOKE TEST PASSED — LoRA pipeline is working!")
print("=" * 50)