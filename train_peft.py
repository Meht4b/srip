# train_peft.py  — KG-PEFT Project, Phase 2
# Generative NER using AutoModelForCausalLM (BioMistral-7B)
# All five PEFT methods share one training loop.
#
# Usage:
#   python train_peft.py --method lora   --n_samples 10  --dataset bc5cdr
#   python train_peft.py --method qlora  --n_samples 50  --dataset bc5cdr
#   python train_peft.py --method adalora --n_samples 100 --dataset bc5cdr
#   python train_peft.py --method prefix --n_samples 500 --dataset bc5cdr
#   python train_peft.py --method ia3    --n_samples full --dataset bc5cdr

import os
import re
import json
import random
import argparse
import numpy as np
import torch
import wandb
import spacy
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    AdaLoraConfig,
    PrefixTuningConfig,
    IA3Config,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--method",     required=True,
                    choices=["lora","qlora","adalora","prefix","ia3"])
parser.add_argument("--n_samples",  required=True,
                    help="10 | 50 | 100 | 500 | full")
parser.add_argument("--dataset",    required=True,
                    choices=["bc5cdr","n2c2"])
parser.add_argument("--data_dir",   default=r"C:\Users\sayan\mehthab\kg-peft\~kg-peft\data")
parser.add_argument("--output_dir", default="./checkpoints")
parser.add_argument("--lora_r",     type=int, default=8)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--epochs",     type=int, default=3)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--max_length", type=int, default=512)
args = parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(args.seed)

MODEL_NAME = "BioMistral/BioMistral-7B"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES  = None if args.n_samples == "full" else int(args.n_samples)
RUN_NAME   = f"{args.method}_r{args.lora_r}_{args.dataset}_n{args.n_samples}_seed{args.seed}"
SAVE_PATH  = os.path.join(args.output_dir, RUN_NAME)
os.makedirs(SAVE_PATH, exist_ok=True)

print("=" * 60)
print(f"KG-PEFT | {RUN_NAME}")
print(f"Model: BioMistral-7B | Device: {DEVICE}")
print("=" * 60)

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="kgpeft",
    name=RUN_NAME,
    config=vars(args),
)

# ── scispaCy for hallucination metric ─────────────────────────────────────────
print("[Setup] Loading scispaCy...")
nlp = spacy.load("en_core_sci_lg")

def extract_entities_scispacy(text):
    doc = nlp(text[:1000])
    return set(e.text.lower().strip() for e in doc.ents)

def entity_hallucination_rate(generated_text, source_text):
    gen_ents = extract_entities_scispacy(generated_text)
    src_ents = extract_entities_scispacy(source_text)
    if not gen_ents:
        return 0.0
    hallucinated = gen_ents - src_ents
    return len(hallucinated) / len(gen_ents)

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(source_text, entities=None):
    prompt = (
        "You are a clinical NER system. "
        "Extract all chemicals and diseases from the text below.\n"
        "List each entity on a new line. "
        "If none found, write NONE.\n\n"
        f"Text: {source_text.strip()}\n\n"
        "Entities:\n"
    )
    if entities is not None:
        answer = "\n".join(sorted(entities)) if entities else "NONE"
        prompt += answer + "\n"
    return prompt

# ── BC5CDR parser ─────────────────────────────────────────────────────────────
def parse_pubtator(filepath):
    samples = []
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        text, entities, pmid = "", set(), None
        for line in lines:
            if "|t|" in line:
                pmid = line.split("|")[0]
                text += line.split("|t|")[1] + " "
            elif "|a|" in line:
                text += line.split("|a|")[1]
            elif "\t" in line and pmid:
                parts = line.split("\t")
                if len(parts) >= 4:
                    entities.add(parts[3].lower().strip())
        if pmid and text.strip():
            samples.append((text.strip(), entities))
    return samples

# ── Stratified subsampling ────────────────────────────────────────────────────
def stratified_subsample(samples, n, seed=42):
    if n is None or n >= len(samples):
        return samples
    rng = random.Random(seed)
    result = rng.sample(samples, n)
    return result

# ── Dataset ───────────────────────────────────────────────────────────────────
class GenerativeNERDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source_text, entities = self.samples[idx]
        prompt = build_prompt(source_text, entities)

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Loss only on the entity answer, not the prompt
        prefix     = build_prompt(source_text, entities=None)
        prefix_len = len(self.tokenizer(prefix, truncation=True,
                                        max_length=self.max_length)["input_ids"])

        labels = input_ids.clone()
        labels[:prefix_len]          = -100
        labels[attention_mask == 0]  = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

# ── PEFT configs ──────────────────────────────────────────────────────────────
def get_peft_config():
    if args.method in ("lora", "qlora"):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
    elif args.method == "adalora":
        return AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            init_r=12,
            target_r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
    elif args.method == "prefix":
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
        )
    elif args.method == "ia3":
        return IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    print(f"\n[Model] Loading BioMistral-7B in 4-bit for {args.method.upper()}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, get_peft_config())
    model.print_trainable_parameters()
    return model

# ── Generate entities at inference ───────────────────────────────────────────
def generate_entities(model, tokenizer, source_text):
    prompt = build_prompt(source_text, entities=None)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "Entities:" in decoded:
        entity_block = decoded.split("Entities:")[-1].strip()
    else:
        entity_block = decoded
    entities = set()
    for line in entity_block.split("\n"):
        line = line.strip().lower()
        if line and line != "none" and len(line) < 100:
            entities.add(line)
    return entities

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, tokenizer, samples, split_name="val"):
    model.eval()
    all_ehr, all_f1, all_p, all_r = [], [], [], []

    for source_text, gold_entities in tqdm(samples, desc=f"  [{split_name}]"):
        pred_entities = generate_entities(model, tokenizer, source_text)

        ehr = entity_hallucination_rate(" ".join(pred_entities), source_text)
        all_ehr.append(ehr)

        if gold_entities:
            tp = len(pred_entities & gold_entities)
            fp = len(pred_entities - gold_entities)
            fn = len(gold_entities - pred_entities)
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
        else:
            p = r = f1 = 0.0

        all_p.append(p); all_r.append(r); all_f1.append(f1)

    return {
        f"{split_name}_ehr":  float(np.mean(all_ehr)),
        f"{split_name}_f1":   float(np.mean(all_f1)),
        f"{split_name}_prec": float(np.mean(all_p)),
        f"{split_name}_rec":  float(np.mean(all_r)),
    }

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, train_loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"  Epoch {epoch} [Train]"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[Setup] Loading dataset...")
    if args.dataset == "bc5cdr":
        base = os.path.join(args.data_dir, "bc5cdr", "CDR_Data", "CDR.Corpus.v010516")
        train_samples = parse_pubtator(os.path.join(base, "CDR_TrainingSet.PubTator.txt"))
        val_samples   = parse_pubtator(os.path.join(base, "CDR_DevelopmentSet.PubTator.txt"))
        test_samples  = parse_pubtator(os.path.join(base, "CDR_TestSet.PubTator.txt"))
    else:
        raise NotImplementedError("n2c2 loader will be added once data is available")

    train_samples = stratified_subsample(train_samples, N_SAMPLES, seed=args.seed)
    print(f"  Train: {len(train_samples)} | Val (eval 50): 50 | Test (eval 50): 50")

    val_eval   = val_samples[:50]
    test_eval  = test_samples[:50]

    train_dataset = GenerativeNERDataset(train_samples, tokenizer, args.max_length)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = load_model()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss   = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_metrics = evaluate(model, tokenizer, val_eval, "val")

        print(f"  Epoch {epoch}: Loss={avg_loss:.4f} | "
              f"F1={val_metrics['val_f1']:.4f} | EHR={val_metrics['val_ehr']:.4f}")

        wandb.log({"epoch": epoch, "train_loss": avg_loss, **val_metrics})

        if val_metrics["val_f1"] > best_f1:
            best_f1 = val_metrics["val_f1"]
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"  ✓ Checkpoint saved → {SAVE_PATH}")

    print("\n[Final] Test evaluation...")
    test_metrics = evaluate(model, tokenizer, test_eval, "test")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {RUN_NAME}")
    print(f"F1:   {test_metrics['test_f1']:.4f}")
    print(f"EHR:  {test_metrics['test_ehr']:.4f}")
    print(f"Prec: {test_metrics['test_prec']:.4f}")
    print(f"Rec:  {test_metrics['test_rec']:.4f}")
    print(f"Checkpoint: {SAVE_PATH}")
    print(f"{'='*60}")

    wandb.log(test_metrics)

    results = {
        "run_name":   RUN_NAME,
        "method":     args.method,
        "lora_r":     args.lora_r,
        "dataset":    args.dataset,
        "n_samples":  args.n_samples,
        "seed":       args.seed,
        "checkpoint": SAVE_PATH,
        **test_metrics,
    }
    with open(os.path.join(SAVE_PATH, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results JSON → {SAVE_PATH}/results.json")
    print(f"Wandb       → {wandb.run.get_url()}")
    wandb.finish()

if __name__ == "__main__":
    main()