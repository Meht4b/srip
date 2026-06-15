# test_biomistral.py
# Loads BioMistral-7B in 4-bit QLoRA mode and runs a forward pass
# WARNING: Will download ~4GB, be patient

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

print("=" * 50)
print("TEST: BioMistral-7B in 4-bit QLoRA")
print("=" * 50)

MODEL_NAME = "BioMistral/BioMistral-7B"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("    Tokenizer loaded ✓")

print("\n[2/4] Loading BioMistral-7B in 4-bit...")
# Change this:
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
)
mem = torch.cuda.memory_allocated() / 1024**3
print(f"    Model loaded ✓  |  GPU memory used: {mem:.2f} GB")

print("\n[3/4] Applying LoRA...")
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

print("\n[4/4] Running forward pass...")
inputs = tokenizer(
    "The patient presented with chest pain and was diagnosed with",
    return_tensors="pt"
).to("cuda")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
print(f"    Loss: {outputs.loss.item():.4f} ✓")

print("\n" + "=" * 50)
print("BioMistral-7B QLoRA TEST PASSED!")
print("=" * 50)