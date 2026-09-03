#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Average/ensemble text prediction probabilities from FIVE causal [PREV]+[CURR]
text models trained on the SAME shared split.

Run after you have:
  1) seed 123 result:
     outputs/text/seed_123/
  2) seed 42 result:
     outputs/text/seed_42/
  3) seed 777 result:
     outputs/text/seed_777/

This creates:
  outputs/text/ensemble_5seed/
    val_predictions.csv
    test_predictions.csv
    val_metrics.json
    test_metrics.json
    test_classification_report.txt
    test_confusion_matrix.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredum"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}

VAL_FILES = [
    "outputs/text/seed_123/val_predictions.csv",
    "outputs/text/seed_42/val_predictions.csv",
    "outputs/text/seed_777/val_predictions.csv",
    "outputs/text/seed_2024/val_predictions.csv",
    "outputs/text/seed_2025/val_predictions.csv",
]

TEST_FILES = [
    "outputs/text/seed_123/test_predictions.csv",
    "outputs/text/seed_42/test_predictions.csv",
    "outputs/text/seed_777/test_predictions.csv",
    "outputs/text/seed_2024/test_predictions.csv",
    "outputs/text/seed_2025/test_predictions.csv",
]

OUT_DIR = Path("outputs/text/ensemble_5seed")
EPS = 1e-12


def norm_label(x):
    s = "" if pd.isna(x) else str(x).strip()
    mp = {
        "neutral": "Neutral", "happy": "Happy", "angry": "Angry",
        "sad": "Sad", "fear": "Fear", "disgust": "Disgust",
        "boredom": "Boredum", "boredum": "Boredum",
    }
    return mp.get(s.lower(), s)


def norm_id(x):
    s = "" if pd.isna(x) else str(x).strip().replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.(wav|mp3|flac|m4a|aac|ogg)$", "", s, flags=re.I)
    return s


def prob_col(df, lab):
    candidates = [f"prob_{lab}", f"p_{lab}", lab]
    if lab == "Boredum":
        candidates += ["prob_Boredom", "p_Boredom", "Boredom"]
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(f"Missing probability column for {lab}. Columns: {list(df.columns)}")


def load_one(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {p}\n"
            "Run all text seed scripts first."
        )
    df = pd.read_csv(p)

    if "file_id" not in df.columns:
        if "path" in df.columns:
            df["file_id"] = df["path"].map(norm_id)
        else:
            raise ValueError(f"{p} missing file_id/path column")

    if "true_label" in df.columns:
        label_col = "true_label"
    elif "target_label" in df.columns:
        label_col = "target_label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(f"{p} missing label column")

    out = pd.DataFrame()
    out["file_id"] = df["file_id"].map(norm_id)
    out["true_label"] = df[label_col].map(norm_label)
    out["true_id"] = out["true_label"].map(LABEL2ID).astype(int)

    if "path" in df.columns:
        out["path"] = df["path"].astype(str)
    else:
        out["path"] = out["file_id"]

    probs = np.stack([df[prob_col(df, lab)].astype(float).to_numpy() for lab in LABELS], axis=1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), EPS)

    return out, probs


def metric_dict(y, pred):
    per = f1_score(y, pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y, pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y, pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "min_class_f1": float(per.min()),
    }
    for i, lab in enumerate(LABELS):
        out[f"f1_{lab}"] = float(per[i])
    return out


def ensemble(files: List[str], split_name: str):
    base_df = None
    probs_list = []

    for path in files:
        df, probs = load_one(path)
        if base_df is None:
            base_df = df
        else:
            if list(base_df["file_id"]) != list(df["file_id"]):
                raise RuntimeError(
                    f"{split_name}: file order mismatch in {path}. "
                    "All seed prediction files must use the same shared split and same row order."
                )
            if list(base_df["true_id"]) != list(df["true_id"]):
                raise RuntimeError(f"{split_name}: true label mismatch in {path}.")

        probs_list.append(probs)

    ens = np.mean(np.stack(probs_list, axis=0), axis=0)
    pred = ens.argmax(axis=1)
    y = base_df["true_id"].to_numpy(dtype=int)

    out = base_df.copy()
    out["pred_id"] = pred
    out["pred_label"] = [ID2LABEL[int(i)] for i in pred]
    out["correct"] = y == pred
    out["confidence"] = ens.max(axis=1)

    for i, lab in enumerate(LABELS):
        out[f"prob_{lab}"] = ens[:, i]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
    out[~out["correct"]].to_csv(OUT_DIR / f"{split_name}_misclassified_samples.csv", index=False, encoding="utf-8-sig")

    m = metric_dict(y, pred)
    (OUT_DIR / f"{split_name}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    rep = classification_report(y, pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0)
    (OUT_DIR / f"{split_name}_classification_report.txt").write_text(rep, encoding="utf-8")
    cm = confusion_matrix(y, pred, labels=list(range(len(LABELS))))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT_DIR / f"{split_name}_confusion_matrix.csv", encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(f"TEXT 5-SEED ENSEMBLE {split_name.upper()} RESULTS")
    print("=" * 80)
    print(json.dumps(m, indent=2))
    print(rep)
    print(cm)

    return m


def main():
    print("Text ensemble output:", OUT_DIR)
    val_m = ensemble(VAL_FILES, "val")
    test_m = ensemble(TEST_FILES, "test")
    summary = {"val_metrics": val_m, "test_metrics": test_m, "val_files": VAL_FILES, "test_files": TEST_FILES}
    (OUT_DIR / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved text ensemble in:", OUT_DIR)
    print("Use for fusion:")
    print("  TEXT_VAL_PRED_CSV :", OUT_DIR / "val_predictions.csv")
    print("  TEXT_TEST_PRED_CSV:", OUT_DIR / "test_predictions.csv")


if __name__ == "__main__":
    main()
