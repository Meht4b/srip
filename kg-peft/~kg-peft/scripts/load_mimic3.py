# scripts/load_mimic3.py
"""
MIMIC-III ICD coding loader for MIMIC-III-50 subset.
Follows Mullenbach et al. 2018 split.
Run only after PhysioNet access is granted.
"""
import pandas as pd
import json
from pathlib import Path


# Top 50 ICD codes from Mullenbach et al. 2018
MIMIC3_50_CODES = [
    '401.9','38.93','428.0','427.31','414.01','96.04','96.6','584.9','250.00',
    '96.71','272.4','518.81','99.04','39.61','599.0','530.81','96.72','272.0',
    '99.15','285.9','88.56','V58.61','80.51','244.9','486','38.91','285.1',
    '36.15','276.2','496','99.60','276.1','414.00','V45.81','425.4','570',
    '45.13','38.93','311','305.1','37.22','412','33.24','39.95','287.5',
    '410.71','276.9','V15.82','37.10','403.90'
]


def load_mimic3_50(mimic_dir):
    """
    mimic_dir should contain: NOTEEVENTS.csv, DIAGNOSES_ICD.csv, ADMISSIONS.csv
    """
    mimic_dir = Path(mimic_dir)

    print("Loading NOTEEVENTS (discharge summaries)...")
    notes = pd.read_csv(mimic_dir / 'NOTEEVENTS.csv', low_memory=False)
    notes = notes[notes['CATEGORY'] == 'Discharge summary']
    # Keep latest per admission
    notes = notes.sort_values('CHARTDATE').groupby('HADM_ID').last().reset_index()
    print(f"  Discharge summaries: {len(notes)}")

    print("Loading ICD diagnoses...")
    diagnoses = pd.read_csv(mimic_dir / 'DIAGNOSES_ICD.csv', low_memory=False)
    # Keep only top-50 codes
    diagnoses = diagnoses[diagnoses['ICD9_CODE'].isin(MIMIC3_50_CODES)]
    # Group codes per admission
    icd_per_admission = (
        diagnoses.groupby('HADM_ID')['ICD9_CODE']
        .apply(list)
        .reset_index()
        .rename(columns={'ICD9_CODE': 'icd_codes'})
    )
    print(f"  Admissions with top-50 ICD codes: {len(icd_per_admission)}")

    # Merge notes with ICD codes
    merged = notes.merge(icd_per_admission, on='HADM_ID', how='inner')
    print(f"  After merge: {len(merged)} records")

    documents = []
    for _, row in merged.iterrows():
        documents.append({
            'hadm_id': str(row['HADM_ID']),
            'text': str(row['TEXT']),
            'icd_codes': row['icd_codes']
        })

    return documents


if __name__ == "__main__":
    docs = load_mimic3_50("../data/mimic3/raw")
    print(f"\nTotal documents: {len(docs)}")
    print(f"Sample hadm_id: {docs[0]['hadm_id']}")
    print(f"Sample ICD codes: {docs[0]['icd_codes']}")
    print(f"Text snippet: {docs[0]['text'][:300]}")

    with open("../data/processed/mimic3_50_full.json", 'w') as f:
        json.dump(docs, f, indent=2)
    print("Saved mimic3_50_full.json")