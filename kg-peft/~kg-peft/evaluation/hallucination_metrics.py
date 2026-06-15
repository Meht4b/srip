# evaluation/hallucination_metrics.py
import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path("C:/Users/tenac/OneDrive/Desktop/kg-peft/~kg-peft")

# ── Lazy globals ──────────────────────────────────────────────────────────────
NLP = None


def get_nlp():
    global NLP
    if NLP is None:
        import spacy
        print("Loading scispaCy en_core_sci_lg...")
        NLP = spacy.load("en_core_sci_lg")
        print("scispaCy loaded.")
    return NLP


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 1: Entity Hallucination Rate (EHR)
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities(text):
    """Extract medical entity strings from text using scispaCy."""
    nlp = get_nlp()
    doc = nlp(text)
    return set(ent.text.lower().strip() for ent in doc.ents)


def entity_hallucination_rate(generated, source):
    """
    EHR = proportion of entities in generated text
    that do NOT appear in the source text.
    Range: 0.0 (no hallucination) to 1.0 (all hallucinated).
    """
    gen_ents = extract_entities(generated)
    src_ents = extract_entities(source)

    if not gen_ents:
        return 0.0  # nothing generated = nothing hallucinated

    hallucinated = gen_ents - src_ents
    return len(hallucinated) / len(gen_ents)


def batch_ehr(generated_texts, source_texts):
    """Compute EHR for a list of (generated, source) pairs."""
    assert len(generated_texts) == len(source_texts), \
        "generated_texts and source_texts must be same length"

    scores = []
    for i, (gen, src) in enumerate(zip(generated_texts, source_texts)):
        score = entity_hallucination_rate(gen, src)
        scores.append(score)
        print(f"  Sample {i+1}: EHR = {score:.4f}")

    return {
        "per_sample": scores,
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores))
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 2: Med-HALT Score
# ─────────────────────────────────────────────────────────────────────────────

def score_medhalt(model_answers, correct_answers):
    """
    Med-HALT accuracy: proportion of correct answers.
    Hallucination rate = 1 - accuracy.

    Args:
        model_answers:   list of strings output by the model
        correct_answers: list of ground truth answer strings
    """
    assert len(model_answers) == len(correct_answers), \
        "model_answers and correct_answers must be same length"

    results = []
    for pred, gold in zip(model_answers, correct_answers):
        # Check if gold answer appears anywhere in model's answer
        is_correct = gold.strip().lower() in pred.strip().lower()
        results.append({
            "model_answer":   pred,
            "correct_answer": gold,
            "correct":        is_correct,
            "score":          int(is_correct)
        })

    accuracy = float(np.mean([r["score"] for r in results]))
    return {
        "accuracy":          accuracy,
        "hallucination_rate": round(1.0 - accuracy, 4),
        "per_sample":        results
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRIC 3: BERTScore-Clinical
# ─────────────────────────────────────────────────────────────────────────────

def bertscore_clinical(generated_texts, reference_texts, batch_size=8):
    """
    BERTScore using ClinicalBERT embeddings.
    Higher F1 = better factual consistency = less hallucination.

    First run will download ClinicalBERT (~400MB). Be patient.

    Args:
        generated_texts: list of model output strings
        reference_texts: list of reference/ground truth strings
        batch_size:      reduce to 4 if you get memory errors

    Returns dict with mean precision, recall, F1 and per-sample F1.
    """
    import torch
    from bert_score import score as bs_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Running BERTScore on device: {device}")
    print(f"  Downloading ClinicalBERT if first run (may take a few minutes)...")

    P, R, F1 = bs_score(
        cands=generated_texts,
        refs=reference_texts,
        model_type="emilyalsentzer/Bio_ClinicalBERT",
        num_layers=12,
        batch_size=batch_size,
        device=device,
        verbose=True
    )

    return {
        "mean_precision":  float(P.mean()),
        "mean_recall":     float(R.mean()),
        "mean_f1":         float(F1.mean()),
        "std_f1":          float(F1.std()),
        "per_sample_f1":   F1.numpy().tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    generated_texts,
    source_texts,
    reference_texts=None,
    medhalt_answers=None,
    medhalt_correct=None,
    run_bertscore=True,
    save_name="test_run"
):
    """
    Run all three hallucination metrics and save results to JSON.

    Args:
        generated_texts: model outputs (list of strings)
        source_texts:    original input notes (for EHR)
        reference_texts: ground truth outputs (for BERTScore, optional)
        medhalt_answers: model answers to Med-HALT questions (optional)
        medhalt_correct: correct Med-HALT answers (optional)
        run_bertscore:   set False to skip BERTScore (slow on CPU)
        save_name:       filename prefix for saved results JSON
    """
    results = {}

    # ── Metric 1 ──
    print("\n" + "="*50)
    print("METRIC 1: Entity Hallucination Rate (EHR)")
    print("="*50)
    results["ehr"] = batch_ehr(generated_texts, source_texts)
    print(f"\n  >> Mean EHR : {results['ehr']['mean']:.4f}")
    print(f"  >> Std  EHR : {results['ehr']['std']:.4f}")

    # ── Metric 2 ──
    if medhalt_answers is not None and medhalt_correct is not None:
        print("\n" + "="*50)
        print("METRIC 2: Med-HALT Score")
        print("="*50)
        results["medhalt"] = score_medhalt(medhalt_answers, medhalt_correct)
        print(f"\n  >> Accuracy           : {results['medhalt']['accuracy']:.4f}")
        print(f"  >> Hallucination Rate : {results['medhalt']['hallucination_rate']:.4f}")

    # ── Metric 3 ──
    if run_bertscore and reference_texts is not None:
        print("\n" + "="*50)
        print("METRIC 3: BERTScore-Clinical")
        print("="*50)
        results["bertscore"] = bertscore_clinical(generated_texts, reference_texts)
        print(f"\n  >> Mean F1  : {results['bertscore']['mean_f1']:.4f}")
        print(f"  >> Std  F1  : {results['bertscore']['std_f1']:.4f}")

    # ── Save ──
    out_dir = BASE_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{save_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*50}")
    print(f"Results saved -> {out_path}")
    print("="*50)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  (runs when you execute this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n>>> Running hallucination metrics test with dummy data\n")

    # ── Test data ──────────────────────────────────────────────────────────
    # Two generated texts vs their source notes
    # Sample 1: model added 'lisinopril' and 'hypertension' which are NOT in source
    #           -> should show hallucination
    # Sample 2: model output closely matches source
    #           -> should show low hallucination
    generated_texts = [
        "Patient was treated with metformin and lisinopril for hypertension and diabetes.",
        "The patient presented with chest pain and was diagnosed with myocardial infarction."
    ]
    source_texts = [
        "Patient was treated with metformin for diabetes.",
        "The patient presented with chest pain and was diagnosed with myocardial infarction."
    ]

    # Reference texts for BERTScore
    reference_texts = [
        "Patient diagnosed with type 2 diabetes mellitus, started on metformin 500mg.",
        "Patient came in with chest pain, confirmed myocardial infarction on ECG."
    ]

    # Med-HALT dummy QA pairs
    # 2 out of 3 correct -> accuracy=0.667, hallucination_rate=0.333
    medhalt_answers = [
        "The answer is aspirin",
        "I think the drug is paracetamol",
        "metformin is used for diabetes"
    ]
    medhalt_correct = [
        "aspirin",
        "ibuprofen",
        "metformin"
    ]

    # ── Run ───────────────────────────────────────────────────────────────
    results = run_full_evaluation(
        generated_texts=generated_texts,
        source_texts=source_texts,
        reference_texts=reference_texts,
        medhalt_answers=medhalt_answers,
        medhalt_correct=medhalt_correct,
        run_bertscore=True,
        save_name="dummy_test"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n>>> FINAL SUMMARY")
    print(f"    EHR mean              : {results['ehr']['mean']:.4f}  (lower is better)")
    if "medhalt" in results:
        print(f"    Med-HALT halluc rate  : {results['medhalt']['hallucination_rate']:.4f}  (lower is better)")
    if "bertscore" in results:
        print(f"    BERTScore-Clinical F1 : {results['bertscore']['mean_f1']:.4f}  (higher is better)")