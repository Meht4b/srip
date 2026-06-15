# scripts/stratified_sampling.py
"""
Creates stratified N subsets for all datasets.
N values: 10, 50, 100, 500, full
- NER datasets: stratify by entity type distribution
- MIMIC-III: stratify by ICD code frequency distribution
"""
import json
import random
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = [42, 123, 456]  # Three fixed seeds for all experiments
N_VALUES = [10, 50, 100, 500]  # 'full' handled separately


def stratified_sample_ner(documents, n, seed):
    """
    Sample n documents from NER dataset preserving entity type distribution.
    If n >= len(documents), returns all documents.
    """
    if n >= len(documents):
        return documents

    random.seed(seed)
    np.random.seed(seed)

    # Count entity type distribution in full dataset
    type_counts = Counter()
    for doc in documents:
        for ent in doc['entities']:
            type_counts[ent['type']] += 1
    total_entities = sum(type_counts.values())

    # Group documents by their dominant entity type
    type_to_docs = defaultdict(list)
    for doc in documents:
        if not doc['entities']:
            type_to_docs['NONE'].append(doc)
            continue
        doc_types = Counter(e['type'] for e in doc['entities'])
        dominant_type = doc_types.most_common(1)[0][0]
        type_to_docs[dominant_type].append(doc)

    # Calculate how many docs to sample per type (proportional)
    sampled = []
    remaining_n = n
    type_list = [t for t in type_to_docs if t != 'NONE']

    for i, etype in enumerate(type_list):
        if i == len(type_list) - 1:
            # Last type gets whatever is left to avoid rounding errors
            n_from_type = remaining_n
        else:
            proportion = type_counts[etype] / total_entities
            n_from_type = max(1, round(proportion * n))
            n_from_type = min(n_from_type, len(type_to_docs[etype]), remaining_n)

        docs_for_type = type_to_docs[etype].copy()
        random.shuffle(docs_for_type)
        sampled.extend(docs_for_type[:n_from_type])
        remaining_n -= n_from_type

        if remaining_n <= 0:
            break

    # If we still need more docs, fill from NONE or random
    if remaining_n > 0 and type_to_docs['NONE']:
        random.shuffle(type_to_docs['NONE'])
        sampled.extend(type_to_docs['NONE'][:remaining_n])

    # Final shuffle
    random.shuffle(sampled)
    return sampled[:n]


def stratified_sample_icd(documents, n, seed):
    """
    Sample n documents from MIMIC-III preserving ICD code frequency distribution.
    """
    if n >= len(documents):
        return documents

    random.seed(seed)
    np.random.seed(seed)

    # Count how often each ICD code appears across all documents
    code_counts = Counter()
    for doc in documents:
        for code in doc['icd_codes']:
            code_counts[code] += 1
    total_code_occurrences = sum(code_counts.values())

    # Group documents by their most frequent ICD code
    code_to_docs = defaultdict(list)
    for doc in documents:
        if not doc['icd_codes']:
            code_to_docs['NONE'].append(doc)
            continue
        doc_code_counts = Counter(doc['icd_codes'])
        dominant_code = doc_code_counts.most_common(1)[0][0]
        code_to_docs[dominant_code].append(doc)

    # Sample proportionally
    sampled = []
    remaining_n = n
    code_list = [c for c in code_to_docs if c != 'NONE']

    for i, code in enumerate(code_list):
        if i == len(code_list) - 1:
            n_from_code = remaining_n
        else:
            proportion = code_counts[code] / total_code_occurrences
            n_from_code = max(1, round(proportion * n))
            n_from_code = min(n_from_code, len(code_to_docs[code]), remaining_n)

        docs_for_code = code_to_docs[code].copy()
        random.shuffle(docs_for_code)
        sampled.extend(docs_for_code[:n_from_code])
        remaining_n -= n_from_code

        if remaining_n <= 0:
            break

    random.shuffle(sampled)
    return sampled[:n]


def create_all_subsets(dataset_name, documents, task_type, output_dir):
    """
    Creates N subsets for all N values and all 3 seeds.
    Saves as JSON files named: {dataset}_{N}_{seed}.json
    Also saves the full dataset as: {dataset}_full.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_fn = stratified_sample_ner if task_type == 'ner' else stratified_sample_icd

    # Save full dataset
    full_path = output_dir / f"{dataset_name}_full.json"
    with open(full_path, 'w') as f:
        json.dump(documents, f)
    print(f"Saved {dataset_name}_full: {len(documents)} docs")

    # Save each N x seed combination
    for n in N_VALUES:
        if n > len(documents):
            print(f"  Skipping N={n} for {dataset_name}: only {len(documents)} docs available")
            continue
        for seed in SEEDS:
            subset = sample_fn(documents, n, seed)
            out_path = output_dir / f"{dataset_name}_N{n}_seed{seed}.json"
            with open(out_path, 'w') as f:
                json.dump(subset, f)
        print(f"  Saved {dataset_name}_N{n} (3 seeds)")

    print(f"Done: {dataset_name}\n")


def verify_stratification(documents, subset, task_type):
    """Quick check that entity/code distribution is roughly preserved."""
    if task_type == 'ner':
        full_types = Counter(e['type'] for doc in documents for e in doc['entities'])
        sub_types = Counter(e['type'] for doc in subset for e in doc['entities'])
    else:
        full_types = Counter(c for doc in documents for c in doc['icd_codes'])
        sub_types = Counter(c for doc in subset for c in doc['icd_codes'])

    full_total = sum(full_types.values())
    sub_total = sum(sub_types.values())

    print("  Stratification check (full% vs subset%):")
    for label in list(full_types.keys())[:5]:
        full_pct = 100 * full_types[label] / full_total if full_total else 0
        sub_pct = 100 * sub_types.get(label, 0) / sub_total if sub_total else 0
        print(f"    {label}: full={full_pct:.1f}%  subset={sub_pct:.1f}%")


if __name__ == "__main__":
    processed_dir = Path("../data/processed")
    subsets_dir = Path("../data/subsets")

    # BC5CDR
    bc5cdr_path = processed_dir / "bc5cdr_train.json"
    if bc5cdr_path.exists():
        with open(bc5cdr_path) as f:
            bc5cdr_docs = json.load(f)
        create_all_subsets("bc5cdr", bc5cdr_docs, "ner", subsets_dir)
        # Verify stratification on N=100 seed=42
        subset_check = stratified_sample_ner(bc5cdr_docs, 100, 42)
        verify_stratification(bc5cdr_docs, subset_check, "ner")
    else:
        print("BC5CDR not found yet, run load_bc5cdr.py first")

    # n2c2 (run after you get access)
    n2c2_path = processed_dir / "n2c2_train.json"
    if n2c2_path.exists():
        with open(n2c2_path) as f:
            n2c2_docs = json.load(f)
        create_all_subsets("n2c2", n2c2_docs, "ner", subsets_dir)
    else:
        print("n2c2 not available yet - run after access is granted")

    # MIMIC-III (run after you get access)
    mimic_path = processed_dir / "mimic3_50_full.json"
    if mimic_path.exists():
        with open(mimic_path) as f:
            mimic_docs = json.load(f)
        create_all_subsets("mimic3_50", mimic_docs, "icd", subsets_dir)
    else:
        print("MIMIC-III not available yet - run after access is granted")