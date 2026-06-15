import xml.etree.ElementTree as ET
import json
from pathlib import Path

BASE_DIR = Path("C:/Users/tenac/OneDrive/Desktop/kg-peft/~kg-peft")

def parse_bc5cdr_bioc(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    documents = []

    for doc in root.findall('document'):
        doc_id = doc.find('id').text
        full_text = ""
        entities = []

        for passage in doc.findall('passage'):
            text_el = passage.find('text')
            text = text_el.text if text_el is not None else ""
            full_text += text + " "

            for ann in passage.findall('annotation'):
                entity_text_el = ann.find('text')
                entity_text = entity_text_el.text if entity_text_el is not None else ""

                infon_type = None
                for infon in ann.findall('infon'):
                    if infon.get('key') == 'type':
                        infon_type = infon.text

                location = ann.find('location')
                if location is not None:
                    start = int(location.get('offset'))
                    length = int(location.get('length'))
                    entities.append({
                        'text': entity_text,
                        'type': infon_type,
                        'start': start,
                        'end': start + length
                    })

        documents.append({
            'doc_id': doc_id,
            'text': full_text.strip(),
            'entities': entities
        })

    return documents


def load_bc5cdr_all():
    corpus_path = BASE_DIR / 'data' / 'bc5cdr' / 'CDR_Data' / 'CDR.Corpus.v010516'
    out_dir = BASE_DIR / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split, filename in [
        ('train', 'CDR_TrainingSet.BioC.xml'),
        ('dev',   'CDR_DevelopmentSet.BioC.xml'),
        ('test',  'CDR_TestSet.BioC.xml')
    ]:
        filepath = corpus_path / filename
        print(f"Loading {filepath}")
        splits[split] = parse_bc5cdr_bioc(filepath)
        print(f"  BC5CDR {split}: {len(splits[split])} documents")

        out_path = out_dir / f"bc5cdr_{split}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(splits[split], f, indent=2)
        print(f"  Saved -> {out_path}")

    # Quick sanity check
    sample = splits['train'][0]
    print(f"\nSample doc_id : {sample['doc_id']}")
    print(f"Text snippet  : {sample['text'][:200]}")
    print(f"First 3 entities: {sample['entities'][:3]}")
    return splits


if __name__ == "__main__":
    load_bc5cdr_all()