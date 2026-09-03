#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V24 VALIDATION-PROTECTED MULTI-BASE ACOUSTIC ENSEMBLE
=============================================

Purpose
-------
Introduce a genuinely new signal by extracting handcrafted acoustic/prosodic
features directly from WAV files, then fuse them protectively with V21.

Models
------
1. RBF SVM on standardized acoustic features
2. ExtraTrees classifier
3. Logistic regression
4. Validation-selected acoustic ensemble
5. Validation-only protected blend with V21
6. Optional selective acoustic correction gate

Features
--------
- duration
- RMS energy statistics
- zero-crossing rate
- spectral centroid/bandwidth/rolloff/flatness
- MFCC means/std/min/max
- delta MFCC means/std
- chroma means/std
- pitch (F0) statistics
- voiced ratio
- onset strength statistics
- tempo estimate

No pretrained checkpoint is required.

Run
---
python V24_multibase_acoustic_gate.py 2>&1 | tee v24_result.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError as exc:
    raise SystemExit(
        "librosa is required. Install it with:\n"
        "  pip install librosa soundfile\n"
    ) from exc

from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredum"]
L2I = {x: i for i, x in enumerate(LABELS)}
I2L = {i: x for x, i in L2I.items()}
K = 7
EPS = 1e-12

ROOT = Path(".")
DEFAULT_MANIFEST = ROOT / "data/urdu_ser_manifest_v11_portable.csv"
DEFAULT_OUT = ROOT / "outputs/fusion/v24"
DEFAULT_BASE_VAL = ROOT / "outputs/fusion/v21/val_predictions.csv"
DEFAULT_BASE_TEST = ROOT / "outputs/fusion/v21/test_predictions.csv"
DEFAULT_V13_VAL = ROOT / "outputs/fusion/v13/val_predictions.csv"
DEFAULT_V13_TEST = ROOT / "outputs/fusion/v13/test_predictions.csv"


def norm_label(x):
    mapping = {
        "neutral": "Neutral",
        "happy": "Happy",
        "angry": "Angry",
        "sad": "Sad",
        "fear": "Fear",
        "disgust": "Disgust",
        "boredom": "Boredum",
        "boredum": "Boredum",
    }
    key = str(x).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown label: {x!r}")
    return mapping[key]


def normp(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def metric(y, p):
    p = normp(p)
    pred = p.argmax(1)
    per = f1_score(y, pred, labels=list(range(K)), average=None, zero_division=0)
    acc = accuracy_score(y, pred)
    macro = f1_score(y, pred, average="macro", zero_division=0)
    weighted = f1_score(y, pred, average="weighted", zero_division=0)

    out = {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "min_class_f1": float(per.min()),
        "selection_score": float(.45 * acc + .45 * macro + .10 * per.min()),
    }
    for i, label in enumerate(LABELS):
        out[f"f1_{label}"] = float(per[i])
    return out


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def find_col(df, names):
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"Missing one of columns: {names}")


def prob_col(df, label):
    names = [f"prob_{label}", f"p_{label}", label]
    if label == "Boredum":
        names += ["prob_Boredom", "p_Boredom", "Boredom"]
    return find_col(df, names)


def load_predictions(path):
    df = pd.read_csv(path)
    id_col = find_col(df, ["file_id", "id", "utterance_id", "sample_id", "filename"])
    label_col = find_col(df, ["true_label", "target_label", "label", "emotion"])

    ids = df[id_col].astype(str).tolist()
    y = df[label_col].map(norm_label).map(L2I).to_numpy(dtype=int)
    p = np.stack(
        [df[prob_col(df, label)].astype(float).to_numpy() for label in LABELS],
        axis=1,
    )
    return ids, y, normp(p)


def align(reference_ids, ids, y, p):
    index = {x: i for i, x in enumerate(ids)}
    missing = [x for x in reference_ids if x not in index]
    if missing:
        raise ValueError(f"Missing IDs: {missing[:5]}")
    order = [index[x] for x in reference_ids]
    return y[order], p[order]


def load_manifest(path):
    df = pd.read_csv(path).reset_index(drop=True)
    required = {"file_id", "split", "label", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    df["file_id"] = df["file_id"].astype(str)
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].map(norm_label)
    df["label_id"] = df["label"].map(L2I).astype(int)
    df["audio_path"] = df["audio_path"].astype(str)

    return df


def safe_stats(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.median(x)),
    ]


def feature_names(n_mfcc=20):
    names = ["duration"]
    for base in [
        "rms",
        "zcr",
        "centroid",
        "bandwidth",
        "rolloff",
        "flatness",
        "onset",
    ]:
        names += [f"{base}_{s}" for s in ["mean", "std", "min", "max", "median"]]

    for i in range(n_mfcc):
        names += [f"mfcc{i+1}_{s}" for s in ["mean", "std", "min", "max"]]

    for i in range(n_mfcc):
        names += [f"delta_mfcc{i+1}_{s}" for s in ["mean", "std"]]

    for i in range(12):
        names += [f"chroma{i+1}_{s}" for s in ["mean", "std"]]

    names += [
        "f0_mean",
        "f0_std",
        "f0_min",
        "f0_max",
        "f0_median",
        "voiced_ratio",
        "tempo",
    ]
    return names


def extract_one(path, sr, n_mfcc):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception:
        return np.zeros(len(feature_names(n_mfcc)), dtype=np.float32)

    if y is None or len(y) < 16:
        return np.zeros(len(feature_names(n_mfcc)), dtype=np.float32)

    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)
    y, _ = librosa.effects.trim(y, top_db=35)

    if len(y) < 16:
        return np.zeros(len(feature_names(n_mfcc)), dtype=np.float32)

    duration = len(y) / float(sr)
    n_fft = min(1024, 2 ** int(np.floor(np.log2(max(32, len(y))))))
    hop = max(128, n_fft // 4)

    feats = [float(duration)]

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    for arr in [rms, zcr, centroid, bandwidth, rolloff, flatness, onset]:
        feats.extend(safe_stats(arr))

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop,
    )
    for row in mfcc:
        s = safe_stats(row)
        feats.extend(s[:4])

    width = 9
    if mfcc.shape[1] < width:
        width = max(3, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)

    if width >= 3:
        delta = librosa.feature.delta(mfcc, width=width, mode="nearest")
    else:
        delta = np.zeros_like(mfcc)

    for row in delta:
        s = safe_stats(row)
        feats.extend(s[:2])

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
    )
    for row in chroma:
        s = safe_stats(row)
        feats.extend(s[:2])

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=60,
            fmax=500,
            sr=sr,
            frame_length=n_fft,
            hop_length=hop,
        )
        valid_f0 = f0[np.isfinite(f0)]
        feats.extend(safe_stats(valid_f0))
        voiced_ratio = float(np.mean(np.isfinite(f0))) if f0 is not None else 0.0
    except Exception:
        feats.extend([0.0] * 5)
        voiced_ratio = 0.0

    feats.append(voiced_ratio)

    try:
        tempo = librosa.feature.tempo(
            onset_envelope=onset,
            sr=sr,
            hop_length=hop,
        )
        feats.append(float(np.asarray(tempo).reshape(-1)[0]))
    except Exception:
        feats.append(0.0)

    feats = np.asarray(feats, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    expected = len(feature_names(n_mfcc))
    if len(feats) != expected:
        if len(feats) < expected:
            feats = np.pad(feats, (0, expected - len(feats)))
        else:
            feats = feats[:expected]

    return feats.astype(np.float32)


def extract_all(df, cache_path, sr, n_mfcc):
    cache_path = Path(cache_path)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_ids = data["file_ids"].astype(str).tolist()
        current_ids = df["file_id"].astype(str).tolist()

        if cached_ids == current_ids:
            print("Loading cached acoustic features:", cache_path)
            return data["features"].astype(np.float32)

        print("Feature cache IDs differ; rebuilding.")

    features = []
    total = len(df)

    for i, row in df.iterrows():
        path = row["audio_path"]
        features.append(extract_one(path, sr, n_mfcc))

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"Extracted {i + 1}/{total}")

    x = np.stack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        file_ids=df["file_id"].astype(str).to_numpy(),
        features=x,
        feature_names=np.asarray(feature_names(n_mfcc), dtype=object),
    )

    return x


def aligned_probs(model, x):
    raw = model.predict_proba(x)
    out = np.full((len(x), K), EPS, dtype=np.float64)

    for source_idx, class_id in enumerate(model.classes_.astype(int)):
        out[:, class_id] = raw[:, source_idx]

    return normp(out)


def model_weight_search(y, prob_list, names, step):
    if len(prob_list) == 1:
        return [1.0], metric(y, prob_list[0])

    best = None
    grid = np.arange(0, 1 + step / 2, step)

    if len(prob_list) == 3:
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                weights = [float(w1), float(w2), float(max(0.0, w3))]
                p = normp(sum(w * q for w, q in zip(weights, prob_list)))
                m = metric(y, p)
                key = (
                    m["selection_score"],
                    m["macro_f1"],
                    m["accuracy"],
                    m["min_class_f1"],
                )
                if best is None or key > best[0]:
                    best = (key, weights, m)
    else:
        weights = [1.0 / len(prob_list)] * len(prob_list)
        p = normp(sum(w * q for w, q in zip(weights, prob_list)))
        best = (None, weights, metric(y, p))

    return best[1], best[2]


def protected_blend_search(y, base, acoustic, max_weight, step):
    rows = []

    for w in np.arange(0, max_weight + step / 2, step):
        p = normp((1.0 - w) * base + w * acoustic)
        rows.append(
            {
                "acoustic_weight": float(w),
                "base_weight": float(1.0 - w),
                **metric(y, p),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            r["selection_score"],
            r["macro_f1"],
            r["accuracy"],
            r["min_class_f1"],
            -r["acoustic_weight"],
        ),
        reverse=True,
    )
    return rows[0], rows


def confidence_margin(p):
    order = np.argsort(p, axis=1)
    top = p[np.arange(len(p)), order[:, -1]]
    second = p[np.arange(len(p)), order[:, -2]]
    return top, top - second


def selective_gate_search(y, base, acoustic, args):
    base_pred = base.argmax(1)
    acoustic_pred = acoustic.argmax(1)
    base_conf, base_margin = confidence_margin(base)
    acoustic_conf, acoustic_margin = confidence_margin(acoustic)

    candidates = []

    for source in range(K):
        for target in range(K):
            if source == target:
                continue

            transition = (base_pred == source) & (acoustic_pred == target)
            if transition.sum() < args.min_support:
                continue

            for bc in [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
                for bm in [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]:
                    for ac in [0.30, 0.40, 0.50, 0.60, 0.70]:
                        for am in [0.00, 0.05, 0.10, 0.15, 0.20]:
                            mask = (
                                transition
                                & (base_conf <= bc)
                                & (base_margin <= bm)
                                & (acoustic_conf >= ac)
                                & (acoustic_margin >= am)
                            )

                            support = int(mask.sum())
                            if support < args.min_support:
                                continue

                            before = base_pred
                            after = before.copy()
                            after[mask] = target

                            corrected = int(((before != y) & (after == y) & mask).sum())
                            damaged = int(((before == y) & (after != y) & mask).sum())
                            net = corrected - damaged

                            if net < args.min_net:
                                continue

                            p = base.copy()
                            pair_mass = p[mask, source] + p[mask, target]
                            p[mask, source] = pair_mass * 0.35
                            p[mask, target] = pair_mass * 0.65
                            p = normp(p)

                            old_m = metric(y, base)
                            new_m = metric(y, p)

                            if (
                                new_m["accuracy"] < old_m["accuracy"] - 1e-12
                                or new_m["macro_f1"] < old_m["macro_f1"] - 1e-12
                            ):
                                continue

                            score = (
                                4 * net
                                + 200 * (new_m["macro_f1"] - old_m["macro_f1"])
                                + 100 * (new_m["min_class_f1"] - old_m["min_class_f1"])
                                - 0.03 * support
                            )

                            candidates.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "base_conf_max": bc,
                                    "base_margin_max": bm,
                                    "acoustic_conf_min": ac,
                                    "acoustic_margin_min": am,
                                    "support": support,
                                    "corrected": corrected,
                                    "damaged": damaged,
                                    "net": net,
                                    "score": score,
                                }
                            )

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda x: (
            x["score"],
            x["net"],
            -x["support"],
        ),
        reverse=True,
    )[0]


def apply_selective_rule(base, acoustic, rule):
    if rule is None:
        return base.copy(), np.zeros(len(base), dtype=bool)

    base_pred = base.argmax(1)
    acoustic_pred = acoustic.argmax(1)
    base_conf, base_margin = confidence_margin(base)
    acoustic_conf, acoustic_margin = confidence_margin(acoustic)

    mask = (
        (base_pred == rule["source"])
        & (acoustic_pred == rule["target"])
        & (base_conf <= rule["base_conf_max"])
        & (base_margin <= rule["base_margin_max"])
        & (acoustic_conf >= rule["acoustic_conf_min"])
        & (acoustic_margin >= rule["acoustic_margin_min"])
    )

    out = base.copy()
    source = rule["source"]
    target = rule["target"]

    pair_mass = out[mask, source] + out[mask, target]
    out[mask, source] = pair_mass * 0.35
    out[mask, target] = pair_mass * 0.65

    return normp(out), mask


def base_ensemble_search(y, named_probs, step=0.01):
    """Validation-only convex search over two protected baselines."""
    if len(named_probs) == 1:
        name, p = named_probs[0]
        return name, p, {name: 1.0}, [{"method": name, "weight": 1.0, **metric(y, p)}]
    if len(named_probs) != 2:
        raise ValueError("V24 currently supports one or two protected baselines.")

    (n1, p1), (n2, p2) = named_probs
    rows = []
    for w in np.arange(0.0, 1.0 + step / 2, step):
        q = normp(w * p1 + (1.0 - w) * p2)
        rows.append({
            "method": f"{n1}_{w:.2f}_{n2}_{1.0-w:.2f}",
            "weight_first": float(w),
            "weight_second": float(1.0 - w),
            **metric(y, q),
        })
    rows.sort(key=lambda r: (
        r["selection_score"], r["macro_f1"], r["accuracy"],
        r["min_class_f1"], -abs(r["weight_first"] - 0.5)
    ), reverse=True)
    b = rows[0]
    q = normp(b["weight_first"] * p1 + b["weight_second"] * p2)
    return b["method"], q, {n1: b["weight_first"], n2: b["weight_second"]}, rows


def apply_rule_sequence(base, acoustic, rules):
    out = base.copy()
    combined = np.zeros(len(base), dtype=bool)
    masks = []
    for rule in rules:
        out, mask = apply_selective_rule(out, acoustic, rule)
        combined |= mask
        masks.append(mask)
    return out, combined, masks


def sequential_rule_search(y, base, acoustic, args):
    """Select up to max_rules sequentially on validation only.

    Each accepted rule must improve the validation selection score and may not
    reduce either accuracy or macro-F1. The resulting frozen sequence is then
    applied unchanged to the test probabilities.
    """
    current = base.copy()
    rules = []
    history = []
    for _ in range(max(0, int(args.max_rules))):
        before = metric(y, current)
        rule = selective_gate_search(y, current, acoustic, args)
        if rule is None:
            break
        candidate, mask = apply_selective_rule(current, acoustic, rule)
        after = metric(y, candidate)
        if (after["selection_score"] <= before["selection_score"] + 1e-12 or
                after["accuracy"] < before["accuracy"] - 1e-12 or
                after["macro_f1"] < before["macro_f1"] - 1e-12):
            break
        rule = dict(rule)
        rule["validation_metrics_before"] = before
        rule["validation_metrics_after"] = after
        rule["validation_trigger_count"] = int(mask.sum())
        rules.append(rule)
        history.append({"rule": rule, "before": before, "after": after})
        current = candidate
    return current, rules, history


def save_outputs(out, split, ids, y, p, method):
    out.mkdir(parents=True, exist_ok=True)
    p = normp(p)
    pred = p.argmax(1)
    m = metric(y, p)

    df = pd.DataFrame(
        {
            "file_id": ids,
            "true_id": y,
            "true_label": [I2L[int(x)] for x in y],
            "pred_id": pred,
            "pred_label": [I2L[int(x)] for x in pred],
            "correct": pred == y,
            "confidence": p.max(1),
            "method": method,
        }
    )

    for i, label in enumerate(LABELS):
        df[f"prob_{label}"] = p[:, i]

    df.to_csv(out / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
    df.loc[~df.correct].to_csv(
        out / f"{split}_misclassified_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(out / f"{split}_metrics.json", m)

    report = classification_report(
        y,
        pred,
        labels=list(range(K)),
        target_names=LABELS,
        zero_division=0,
    )
    (out / f"{split}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y, pred, labels=list(range(K)))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        out / f"{split}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    print("\n" + "-" * 110)
    print(f"{method.upper()} | {split.upper()}")
    print("-" * 110)
    print(json.dumps(m, indent=2))
    print(report)
    print(cm)
    return m


def main():
    ap = argparse.ArgumentParser(description="V24 validation-protected multi-base acoustic ensemble")

    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--base-val", default=str(DEFAULT_BASE_VAL))
    ap.add_argument("--base-test", default=str(DEFAULT_BASE_TEST))
    ap.add_argument("--fallback-v13-val", default=str(DEFAULT_V13_VAL))
    ap.add_argument("--fallback-v13-test", default=str(DEFAULT_V13_TEST))

    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mfcc", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--svm-c", type=float, default=4.0)
    ap.add_argument("--svm-gamma", default="scale")
    ap.add_argument("--extra-trees", type=int, default=800)
    ap.add_argument("--logreg-c", type=float, default=2.0)

    ap.add_argument("--ensemble-step", type=float, default=0.05)
    ap.add_argument("--max-acoustic-weight", type=float, default=0.35)
    ap.add_argument("--blend-step", type=float, default=0.01)
    ap.add_argument("--base-ensemble-step", type=float, default=0.01)
    ap.add_argument("--max-rules", type=int, default=2)

    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--min-net", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=-1)

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest)

    cache_path = out / f"acoustic_features_sr{args.sample_rate}_mfcc{args.n_mfcc}.npz"
    x = extract_all(
        df,
        cache_path,
        args.sample_rate,
        args.n_mfcc,
    )

    split = df["split"].to_numpy()
    y = df["label_id"].to_numpy(dtype=int)

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    val_ids = df.loc[val_mask, "file_id"].astype(str).tolist()
    test_ids = df.loc[test_mask, "file_id"].astype(str).tolist()

    print(
        f"Dataset: train={len(x_train)}, "
        f"val={len(x_val)}, test={len(x_test)}, "
        f"features={x.shape[1]}"
    )

    models = {
        "svm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=args.svm_c,
                        gamma=args.svm_gamma,
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=args.extra_trees,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        ),
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=args.logreg_c,
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                        multi_class="auto",
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
    }

    val_probs = []
    test_probs = []
    model_names = []

    for name, model in models.items():
        print(f"\nTraining acoustic model: {name}")
        model.fit(x_train, y_train)

        vp = aligned_probs(model, x_val)
        tp = aligned_probs(model, x_test)

        val_probs.append(vp)
        test_probs.append(tp)
        model_names.append(name)

        save_outputs(
            out / name,
            "val",
            val_ids,
            y_val,
            vp,
            f"v24_{name}",
        )
        save_outputs(
            out / name,
            "test",
            test_ids,
            y_test,
            tp,
            f"v24_{name}",
        )

        dump(model, out / f"{name}.joblib")

    weights, ensemble_val_metrics = model_weight_search(
        y_val,
        val_probs,
        model_names,
        args.ensemble_step,
    )

    print("\nSelected acoustic ensemble weights:")
    for name, weight in zip(model_names, weights):
        print(f"  {name}: {weight:.2f}")

    acoustic_val = normp(
        sum(w * p for w, p in zip(weights, val_probs))
    )
    acoustic_test = normp(
        sum(w * p for w, p in zip(weights, test_probs))
    )

    acoustic_dir = out / "acoustic_ensemble"
    acoustic_val_m = save_outputs(
        acoustic_dir,
        "val",
        val_ids,
        y_val,
        acoustic_val,
        "v24_acoustic_ensemble",
    )
    acoustic_test_m = save_outputs(
        acoustic_dir,
        "test",
        test_ids,
        y_test,
        acoustic_test,
        "v24_acoustic_ensemble",
    )

    # ------------------------------------------------------------------
    # V24 protected multi-base ensemble (validation only)
    # ------------------------------------------------------------------
    protected = []
    protected_test = {}

    for name, vp_path, tp_path in [
        ("v21", Path(args.base_val), Path(args.base_test)),
        ("v13", Path(args.fallback_v13_val), Path(args.fallback_v13_test)),
    ]:
        if not (vp_path.exists() and tp_path.exists()):
            continue
        ids_v, y_v, p_v = load_predictions(vp_path)
        ids_t, y_t, p_t = load_predictions(tp_path)
        y_v, p_v = align(val_ids, ids_v, y_v, p_v)
        y_t, p_t = align(test_ids, ids_t, y_t, p_t)
        if not np.array_equal(y_v, y_val) or not np.array_equal(y_t, y_test):
            raise ValueError(f"{name} labels do not align with manifest.")
        protected.append((name, p_v))
        protected_test[name] = p_t

    if not protected:
        raise FileNotFoundError("No protected V21/V13 prediction files were found.")

    baseline_name, base_val, base_weights, base_ranking = base_ensemble_search(
        y_val, protected, args.base_ensemble_step
    )
    pd.DataFrame(base_ranking).to_csv(
        out / "validation_base_ensemble_ranking.csv", index=False, encoding="utf-8-sig"
    )
    base_test = normp(sum(base_weights[n] * protected_test[n] for n in base_weights))

    best_blend, ranking = protected_blend_search(
        y_val, base_val, acoustic_val, args.max_acoustic_weight, args.blend_step
    )
    pd.DataFrame(ranking).to_csv(
        out / "validation_acoustic_blend_ranking.csv", index=False, encoding="utf-8-sig"
    )

    blend_weight = best_blend["acoustic_weight"]
    blend_val = normp((1.0 - blend_weight) * base_val + blend_weight * acoustic_val)
    blend_test = normp((1.0 - blend_weight) * base_test + blend_weight * acoustic_test)

    selective_val, selective_rules, rule_history = sequential_rule_search(
        y_val, base_val, acoustic_val, args
    )
    selective_test, test_rule_mask, test_rule_masks = apply_rule_sequence(
        base_test, acoustic_test, selective_rules
    )

    candidates = [
        ("protected_" + baseline_name, base_val, base_test),
        (f"{baseline_name}_plus_v24_audio_{blend_weight:.2f}", blend_val, blend_test),
    ]
    if selective_rules:
        candidates.append(("v24_sequential_selective_acoustic_gate", selective_val, selective_test))

    ranked_candidates = []
    for method, vp, tp in candidates:
        vm = metric(y_val, vp)
        ranked_candidates.append((
            vm["selection_score"], vm["macro_f1"], vm["accuracy"],
            vm["min_class_f1"], method, vp, tp
        ))
    ranked_candidates.sort(reverse=True)
    _, _, _, _, method, final_val, final_test = ranked_candidates[0]

    fusion_dir = out / "fusion"
    final_val_m = save_outputs(
        fusion_dir,
        "val",
        val_ids,
        y_val,
        final_val,
        method,
    )
    final_test_m = save_outputs(
        fusion_dir,
        "test",
        test_ids,
        y_test,
        final_test,
        method,
    )

    base_test_m = metric(y_test, base_test)
    before = base_test.argmax(1)
    after = final_test.argmax(1)

    change = pd.DataFrame(
        {
            "file_id": test_ids,
            "true_label": [I2L[int(x)] for x in y_test],
            "base_pred": [I2L[int(x)] for x in before],
            "v24_pred": [I2L[int(x)] for x in after],
            "changed": before != after,
            "corrected": (before != y_test) & (after == y_test),
            "damaged": (before == y_test) & (after != y_test),
            "selective_rule_triggered": test_rule_mask,
        }
    )
    change.to_csv(
        out / "test_change_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta = {
        key: float(final_test_m[key] - base_test_m[key])
        for key in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "min_class_f1",
        ]
    }

    summary = {
        "selected_method": method,
        "protected_baseline": baseline_name,
        "protected_base_weights": {k: float(v) for k, v in base_weights.items()},
        "acoustic_models": model_names,
        "acoustic_ensemble_weights": {
            name: float(weight)
            for name, weight in zip(model_names, weights)
        },
        "acoustic_validation_metrics": acoustic_val_m,
        "acoustic_test_metrics": acoustic_test_m,
        "global_blend_weight": float(blend_weight),
        "selective_rules": selective_rules,
        "rule_history": rule_history,
        "final_validation_metrics": final_val_m,
        "final_test_metrics": final_test_m,
        "baseline_test_metrics": base_test_m,
        "delta_over_baseline": delta,
        "changed_predictions": int((before != after).sum()),
        "errors_corrected": int(
            ((before != y_test) & (after == y_test)).sum()
        ),
        "correct_predictions_damaged": int(
            ((before == y_test) & (after != y_test)).sum()
        ),
        "correct_test_predictions": int((after == y_test).sum()),
        "test_errors": int((after != y_test).sum()),
    }

    save_json(out / "final_v24_summary.json", summary)

    print("\n" + "=" * 118)
    print("FINAL V24 VALIDATION-PROTECTED MULTI-BASE ACOUSTIC SUMMARY")
    print("=" * 118)
    print("BASELINE   |", baseline_name)
    print("SELECTED   |", method)
    print(
        "AUDIO WTS  |",
        ", ".join(
            f"{name}={weight:.2f}"
            for name, weight in zip(model_names, weights)
        ),
    )
    print(
        f"ACOUSTIC   | val_acc={acoustic_val_m['accuracy']:.4f}, "
        f"test_acc={acoustic_test_m['accuracy']:.4f}, "
        f"test_macroF1={acoustic_test_m['macro_f1']:.4f}"
    )

    print("BASE WTS   |", ", ".join(f"{k}={v:.2f}" for k, v in base_weights.items()))
    if selective_rules:
        for idx, rule in enumerate(selective_rules, 1):
            print(
                f"RULE {idx:<4} | {I2L[rule['source']]} -> {I2L[rule['target']]} | "
                f"support={rule['support']} | val_net={rule['net']}"
            )
    else:
        print("RULE       | none")

    print(
        f"FINAL VAL  | acc={final_val_m['accuracy']:.4f}, "
        f"macroF1={final_val_m['macro_f1']:.4f}, "
        f"minF1={final_val_m['min_class_f1']:.4f}"
    )
    print(
        f"TEST       | acc={final_test_m['accuracy']:.4f}, "
        f"macroF1={final_test_m['macro_f1']:.4f}, "
        f"minF1={final_test_m['min_class_f1']:.4f}"
    )
    print(
        f"DELTA      | accuracy={delta['accuracy']:+.4f}, "
        f"macroF1={delta['macro_f1']:+.4f}, "
        f"minF1={delta['min_class_f1']:+.4f}"
    )
    print(
        f"CHANGES    | changed={(before != after).sum()}, "
        f"corrected={((before != y_test) & (after == y_test)).sum()}, "
        f"damaged={((before == y_test) & (after != y_test)).sum()}"
    )
    print("OUTPUTS    |", out)


if __name__ == "__main__":
    main()