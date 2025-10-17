#!/usr/bin/env python3
"""
Quick check to verify ROUGE-L vs ROUGE-Lsum calculation
"""

from evaluate import load
import re

def to_lsum_text(s):
    """Convert text to sentence-level format for ROUGE-Lsum"""
    s = re.sub(r"\s+", " ", s).strip()
    sents = re.split(r"(?<=[.?!])\s+", s)  # simple splitter
    return "\n".join(sents)

# Load some sample data from your results
preds = [
    "The chest shows significant air trapping. There are long-term changes at the top of both lungs. There is a curvature of the spine in the upper back. No signs of air leakage in the chest.",
    "A central venous catheter is inserted through the left jugular vein with its tip in the superior vena cava. Everything else looks the same.",
    "Long-term changes in the lungs are present.",
    "The x-ray shows signs of air trapping, flattened diaphragm, and increased space behind the breastbone. There are calcium deposits in the pleura, which are the membranes around the lungs. The left lung has less volume, and there are linear opacities in the lower part of the lung. These findings are related to long-term inflammation caused by asbestos exposure. The previous CT scan shows no significant changes compared to the scan from March 3, 2009.",
    "A calcified granuloma is present in the right lung's edge."
]

refs = [
    "The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs.",
    "A central venous catheter is going through the left jugular vein and its tip is in the superior vena cava. Everything else is the same as before.",
    "Long-term changes in the lungs are seen.",
    "The X-ray shows signs of trapped air, a flattened muscle under the lungs, and more space behind the breastbone. There are also hardened areas on the lung lining on the left side. The left lung has lost some volume and has some linear shadows near the outer lining. These findings are related to long-term inflammation caused by exposure to asbestos. Looking at the previous CT scan, there are no significant changes compared to the scanogram dated 3/4/2009.",
    "There is a calcified granuloma located at the top of the right lung."
]

rouge = load("rouge")

print("=== ROUGE-L vs ROUGE-Lsum Check ===")
print(f"Sample predictions: {len(preds)}")
print(f"Sample references: {len(refs)}")
print()

# Plain strings (current method)
print("1. Current method (plain strings):")
rL = rouge.compute(predictions=preds, references=refs,
                   rouge_types=["rougeL"], use_stemmer=True)
print(f"   ROUGE-L: {rL['rougeL']:.6f}")

# Newline-separated sentences for Lsum
print("2. Sentence-level method (newline-separated):")
preds_lsum = [to_lsum_text(p) for p in preds]
refs_lsum = [to_lsum_text(r) for r in refs]
rLsum = rouge.compute(predictions=preds_lsum, references=refs_lsum,
                      rouge_types=["rougeLsum"], use_stemmer=True)
print(f"   ROUGE-Lsum: {rLsum['rougeLsum']:.6f}")

print()
print("=== Sample sentence splitting ===")
print("Original prediction:")
print(f"  '{preds[0]}'")
print("Sentence-split for Lsum:")
print(f"  '{preds_lsum[0]}'")
print()
print("Difference:", abs(rL['rougeL'] - rLsum['rougeLsum']))
print("Match within tolerance (0.001):", abs(rL['rougeL'] - rLsum['rougeLsum']) < 0.001)
