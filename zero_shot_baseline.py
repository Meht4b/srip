# zero_shot_baseline.py
# Runs BioMistral-7B zero-shot on BC5CDR test set
# Measures entity hallucination rate before any fine-tuning
# Usage: python zero_shot_baseline.py --data_dir ./BioCreative-V-CDR-Corpus

import os
import re
import argparse
import torch
import wandb
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to BioCreative-V-CDR-Corpus folder")
parser.add_argument("--max_samples", type=int, default=50,
                    help="How many test samples to run (50 is enough for baseline)")
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "BioMistral/BioMistral-7B"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = args.max_samples

# ── wandb ────────────────────────────────────────────────────────────────────
wandb.init(
    project="kgpeft",
    name="zero-shot-baseline-bc5cdr",
    config={"model": MODEL_NAME, "task": "NER", "dataset": "BC5CDR",
            "mode": "zero-shot", "max_samples": MAX_SAMPLES}
)

# ── Load scispaCy for hallucination metric ───────────────────────────────────
print("[1/4] Loading scispaCy...")
nlp = spacy.load("en_core_sci_lg")

# ── Parse BC5CDR PubTator format ─────────────────────────────────────────────
def parse_pubtator(filepath):
    """Returns list of (pmid, title+abstract, set_of_gold_entities)"""
    samples = []
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        text = ""
        gold_entities = set()
        pmid = None
        for line in lines:
            if "|t|" in line:
                pmid = line.split("|")[0]
                text += line.split("|t|")[1] + " "
            elif "|a|" in line:
                text += line.split("|a|")[1] + " "
            elif "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 5:
                    gold_entities.add(parts[3].lower().strip())
        if pmid and text:
            samples.append((pmid, text.strip(), gold_entities))
    return samples

# ── Load model ───────────────────────────────────────────────────────────────
print("[2/4] Loading BioMistral-7B in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.eval()
print("    Model loaded ✓")

# ── Load BC5CDR test set ──────────────────────────────────────────────────────
print("[3/4] Loading BC5CDR test set...")
test_file = os.path.join(args.data_dir, "CDR_Data", "CDR.Corpus.v010516",
                         "CDR_TestSet.PubTator.txt")
samples = parse_pubtator(test_file)[:MAX_SAMPLES]
print(f"    Loaded {len(samples)} test samples ✓")

# ── Zero-shot inference + hallucination metric ───────────────────────────────
print("[4/4] Running zero-shot inference...")

def get_model_entities(text):
    """Ask BioMistral to extract entities, return set of extracted entities"""
    prompt = (
        f"Extract all chemicals and diseases mentioned in the following text.\n"
        f"Text: {text[:300]}\n"
        f"Entities:"
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the part after "Entities:"
    if "Entities:" in decoded:
        entity_text = decoded.split("Entities:")[-1].strip()
    else:
        entity_text = decoded
    # Use scispaCy to extract entities from model output
    doc = nlp(entity_text)
    return set(e.text.lower().strip() for e in doc.ents)

def entity_hallucination_rate(predicted_entities, gold_entities):
    """Proportion of predicted entities NOT in the source gold set"""
    if not predicted_entities:
        return 0.0
    hallucinated = predicted_entities - gold_entities
    return len(hallucinated) / len(predicted_entities)

total_ehr = []
for pmid, text, gold in tqdm(samples, desc="Evaluating"):
    predicted = get_model_entities(text)
    ehr = entity_hallucination_rate(predicted, gold)
    total_ehr.append(ehr)
    wandb.log({"hallucination_rate": ehr, "pmid": pmid})

mean_ehr = sum(total_ehr) / len(total_ehr)
print(f"\n{'='*50}")
print(f"ZERO-SHOT BASELINE RESULTS on BC5CDR ({MAX_SAMPLES} samples)")
print(f"Mean Entity Hallucination Rate: {mean_ehr:.4f} ({mean_ehr*100:.2f}%)")
print(f"{'='*50}")

wandb.log({"mean_hallucination_rate": mean_ehr})
wandb.finish()