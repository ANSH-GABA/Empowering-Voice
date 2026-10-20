#!/usr/bin/env python3
"""Evaluate the Hugging Face model on YOUR OWN labelled photos.

Folder layout (one folder per sign, photos from people NOT used elsewhere):
    eval_images/A/p05_001.jpg
    eval_images/B/p05_002.jpg ...

    python backend/scripts/evaluate_hf.py eval_images
Writes docs/experiments/hf_eval.csv (per-sign accuracy) and prints the most confused pairs.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.hf_classifier import MODEL_ID, classify  # noqa: E402


def main(folder: str) -> None:
    root = Path(folder)
    per_sign, confusion = defaultdict(lambda: [0, 0]), Counter()
    for sign_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(sign_dir.glob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            pred = classify(img.read_bytes(), top_k=1)[0]["label"]
            per_sign[sign_dir.name][1] += 1
            if pred == sign_dir.name:
                per_sign[sign_dir.name][0] += 1
            else:
                confusion[(sign_dir.name, pred)] += 1
    if not per_sign:
        sys.exit(f"No images found in {folder}/<SIGN>/")
    correct = sum(c for c, _ in per_sign.values())
    total = sum(n for _, n in per_sign.values())
    out = ROOT / "docs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hf_eval.csv", "w") as f:
        f.write("sign,correct,total,accuracy\n")
        for s, (c, n) in sorted(per_sign.items()):
            f.write(f"{s},{c},{n},{c / n:.3f}\n")
    print(f"Model: {MODEL_ID}\nOverall accuracy: {correct}/{total} = {correct / total:.1%}")
    worst = sorted(per_sign.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
    print("Weakest signs:", ", ".join(f"{s} ({c}/{n})" for s, (c, n) in worst))
    print("Most confused (true -> predicted):", ", ".join(f"{a}->{b} ({n})" for (a, b), n in confusion.most_common(5)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_images")
