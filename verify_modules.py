# save as verify_env.py and run: python verify_env.py

import torch
import transformers
import peft
import bitsandbytes
import seqeval
import scispacy
import spacy

print("=" * 40)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("CUDA available:", torch.cuda.is_available())
print("=" * 40)

# Test scispaCy model loads
nlp = spacy.load("en_core_sci_lg")
doc = nlp("The patient was treated with metformin for type 2 diabetes.")
print("scispaCy entities found:")
for ent in doc.ents:
    print(f"  {ent.text} -> {ent.label_}")
print("=" * 40)
print("ALL CHECKS PASSED")
