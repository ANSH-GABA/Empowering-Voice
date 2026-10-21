#!/usr/bin/env python3
"""Phase 4 + 5: compare baselines, then export the MLP for the browser.

    python Model/scripts/train_static.py

What it does
  1. Loads every CSV in Model/data/processed/static/
  2. Splits BY PARTICIPANT (train / validation / test)
  3. Trains KNN, SVM, Random Forest and MLP and prints accuracy,
     precision, recall, F1 and single-sample latency
  4. Saves a confusion matrix of the MLP on the test set
  5. Refits the MLP on train+validation and writes
     frontend/public/models/static_mlp.json (used by the Translate page)
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

FEATURE_DIM = 129  # must match frontend/src/ml/features/featureExtractor.ts
SEED = 42
ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]


def make_mlp():
    # Dense 128 -> ReLU -> Dense 64 -> ReLU -> softmax (the README architecture).
    # Dropout is not available in scikit-learn; alpha (L2) + early stopping regularise instead.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=300,
            random_state=SEED,
        ),
    )


def candidates():
    return {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "MLP": make_mlp(),
    }


def load(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {data_dir}. Record data on the Collect page first.")
    df = pd.concat(
        [pd.read_csv(f, dtype={"participant_id": str, "label": str}) for f in files],
        ignore_index=True,
    )
    feat = [f"f{i}" for i in range(FEATURE_DIM)]
    if any(c not in df.columns for c in feat):
        sys.exit(f"CSV must contain columns f0..f{FEATURE_DIM - 1}.")
    df = df.dropna(subset=feat)
    df = df[df[feat].abs().sum(axis=1) > 0]  # drop all-zero rows
    return df.drop_duplicates(subset=["participant_id", "label"] + feat).reset_index(drop=True)


def split(df: pd.DataFrame, val: list[str], test: list[str]):
    people = sorted(df["participant_id"].unique())
    if val or test:
        val, test = list(val), list(test)
    elif len(people) >= 3:
        val, test = [people[-2]], [people[-1]]
    else:
        print("\n!! WARNING: fewer than 3 participants -> using a RANDOM split.")
        print("!! Scores will be optimistic and are NOT valid for your report.\n")
        tr, rest = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
        va, te = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
        return tr, va, te, "random (NOT participant-independent)", [], []
    is_val, is_test = df["participant_id"].isin(val), df["participant_id"].isin(test)
    return df[~is_val & ~is_test], df[is_val], df[is_test], "by participant", val, test


def score(model, X, y):
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f}


def latency_ms(model, X, n=100):
    X = X[: min(n, len(X))]
    t = time.perf_counter()
    for row in X:
        model.predict(row.reshape(1, -1))
    return (time.perf_counter() - t) / len(X) * 1000


def export_mlp(pipe, path: Path, class_names):
    scaler, mlp = pipe.named_steps["standardscaler"], pipe.named_steps["mlpclassifier"]
    payload = {
        "version": 1,
        "featureDim": FEATURE_DIM,
        "classes": [str(c) for c in class_names],
        "mean": np.round(scaler.mean_, 6).tolist(),
        "scale": np.round(scaler.scale_, 6).tolist(),
        "layers": [
            {"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist()}
            for W, b in zip(mlp.coefs_, mlp.intercepts_)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


def forward_check(payload, pipe, X):
    """Re-implements the browser forward pass in numpy and compares with scikit-learn."""
    x = (X - np.array(payload["mean"])) / np.array(payload["scale"])
    layers = payload["layers"]
    for i, layer in enumerate(layers):
        x = x @ np.array(layer["W"]) + np.array(layer["b"])
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    probs = x / x.sum(axis=1, keepdims=True)
    ref = pipe.predict_proba(X)
    assert np.allclose(probs, ref, atol=1e-3), "Exported JSON does not match scikit-learn!"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(MODEL_ROOT / "data" / "processed" / "static"))
    ap.add_argument("--out", default=str(ROOT / "frontend" / "public" / "models" / "static_mlp.json"))
    ap.add_argument("--report", default=str(ROOT / "docs" / "experiments"))
    ap.add_argument("--val", nargs="*", default=[], help="validation participant IDs, e.g. --val P05")
    ap.add_argument("--test", nargs="*", default=[], help="test participant IDs, e.g. --test P06")
    args = ap.parse_args()

    df = load(Path(args.data))
    feat = [f"f{i}" for i in range(FEATURE_DIM)]
    tr, va, te, how, val_ids, test_ids = split(df, args.val, args.test)
    print(f"Samples: {len(df)} | signs: {df['label'].nunique()} | participants: {df['participant_id'].nunique()}")
    print(f"Split ({how}): train {len(tr)} | validation {len(va)} | test {len(te)}")
    if how == "by participant":
        print(f"  validation: {sorted(va['participant_id'].unique())}  test: {sorted(te['participant_id'].unique())}")

    # Integer labels: newer scikit-learn MLPs break on string labels with early stopping.
    le = LabelEncoder().fit(df["label"])
    Xtr, ytr = tr[feat].to_numpy(np.float32), le.transform(tr["label"])
    Xva, yva = va[feat].to_numpy(np.float32), le.transform(va["label"])
    Xte, yte = te[feat].to_numpy(np.float32), le.transform(te["label"])

    rows, fitted = [], {}
    for name, model in candidates().items():
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        v, t = score(model, Xva, yva), score(model, Xte, yte)
        rows.append(
            {
                "model": name,
                "val_accuracy": v["accuracy"], "val_f1": v["f1"],
                "test_accuracy": t["accuracy"], "test_precision": t["precision"],
                "test_recall": t["recall"], "test_f1": t["f1"],
                "latency_ms": latency_ms(model, Xte), "train_seconds": fit_s,
            }
        )
        fitted[name] = model
        print(f"  {name:<13} val F1 {v['f1']:.3f} | test acc {t['accuracy']:.3f}  F1 {t['f1']:.3f} | {rows[-1]['latency_ms']:.2f} ms/sample")

    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).round(4)
    table.to_csv(report_dir / "static_baselines.csv", index=False)
    best = table.sort_values("val_f1", ascending=False).iloc[0]["model"]
    print(f"\nBest on validation: {best}. The browser uses the MLP, so the MLP is exported either way.")

    # Confusion matrix (MLP, test set)
    mlp = fitted["MLP"]
    labels = [str(c) for c in le.classes_]
    cm = confusion_matrix(yte, mlp.predict(Xte), labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.4)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation="vertical")
    ax.set_title(f"MLP confusion matrix (test, {how})")
    fig.tight_layout()
    fig.savefig(report_dir / "static_confusion_matrix.png", dpi=150)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if off[i, j]), reverse=True)[:5]
    if pairs:
        print("Most confused pairs (true -> predicted): " + ", ".join(f"{a}->{b} ({n})" for n, a, b in pairs))

    # Final model: train + validation, then export for the browser
    final = make_mlp().fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    payload = export_mlp(final, Path(args.out), le.classes_)
    forward_check(payload, final, Xte[:200])
    size_kb = Path(args.out).stat().st_size / 1024
    (report_dir / "static_final.json").write_text(
        json.dumps({"split": how, "validation_participants": val_ids or sorted(va["participant_id"].unique().tolist()),
                    "test_participants": test_ids or sorted(te["participant_id"].unique().tolist()),
                    "classes": payload["classes"]}, indent=2)
    )
    print(f"\nExported {args.out} ({size_kb:.0f} KB, {len(payload['classes'])} classes) - export check passed.")
    print("Next: cd frontend && npm run dev, open the Translate page.")


if __name__ == "__main__":
    main()
