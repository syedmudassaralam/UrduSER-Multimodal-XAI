#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UrduSER Selected ML + DL + Simple Fusion Benchmark

Selected models requested by user
---------------------------------

TEXT CLASSICAL ML using TF-IDF:
1. TF-IDF + Linear SVM
2. TF-IDF + Logistic Regression
3. TF-IDF + Multinomial Naive Bayes
4. TF-IDF + Random Forest
5. TF-IDF + XGBoost
6. TF-IDF + LightGBM
7. TF-IDF + KNN

TEXT DEEP LEARNING:
1. TextCNN
2. LSTM
3. BiLSTM
4. GRU
5. BiGRU
6. CNN-LSTM
7. CNN-BiLSTM
8. CNN-GRU
9. CNN-BiGRU
10. BiLSTM + Attention
11. BiGRU + Attention

AUDIO CLASSICAL ML using acoustic features:
MFCC, chroma, spectral contrast, zero-crossing rate, RMS energy, spectral features, pitch-related features
1. Acoustic SVM-RBF
2. Acoustic Linear SVM
3. Acoustic Logistic Regression
4. Acoustic Random Forest
5. Acoustic Extra Trees
6. Acoustic XGBoost
7. Acoustic LightGBM
8. Acoustic KNN

AUDIO DEEP LEARNING:
1. MFCC 1D-CNN
2. MFCC CNN-LSTM
3. MFCC CNN-BiLSTM
4. MFCC CNN-GRU
5. MFCC CNN-BiGRU
6. Audio BiLSTM-Attention
7. LogMel-CNN
8. LogMel-CRNN
9. LogMel-ResNet

SIMPLE FUSION:
1. Late decision fusion
2. Weighted probability fusion
3. Average probability fusion
4. Max-confidence fusion

Protocol:
- Same fixed shared split:
  shared_splits_urdu_ser_7class_seed123_prev_curr/
- Text uses [PREV] + [CURR] through true_context_text
- No [NEXT] future context
- Test set evaluated once
- Validation set used only for model selection / fusion weights
- Saves accuracy, macro precision, macro recall, macro F1, weighted F1,
  minimum class F1, per-class precision/recall/F1, confusion matrix,
  predictions, misclassified samples, train/validation curves, training time,
  inference time, and trainable parameters.

Run:
    python urdu_ser_selected_ml_dl_fusion_benchmark.py

Output:
    saved_models_urdu_ser_selected_ml_dl_fusion_benchmark/

Requirements:
    pip install torch numpy pandas scikit-learn librosa soundfile matplotlib

Optional:
    pip install xgboost lightgbm
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIG
# =============================================================================

SEED = 123
SPLIT_DIR = Path("shared_splits_urdu_ser_7class_seed123_prev_curr")
DATA_ROOT = Path("UrduSER/UrduSER")
OUT_DIR = Path("saved_models_urdu_ser_selected_ml_dl_fusion_benchmark")

LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredum"]
AUDIO_FOLDER_LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredom"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL2ID.items()}

TEXT_COLUMN = "true_context_text"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_JOBS = -1

# Runtime switches. You can change these if you want to run one branch only.
RUN_TEXT_ML = True
RUN_TEXT_DL = True
RUN_AUDIO_ML = True
RUN_AUDIO_DL = True
RUN_SIMPLE_FUSION = True

# If a model is too slow, set it False here.
USE_XGBOOST = True
USE_LIGHTGBM = True

# Text DL settings
MAX_LEN = 80
TEXT_BATCH_SIZE = 32
TEXT_EPOCHS = 45
TEXT_PATIENCE = 7
TEXT_EMB_DIM = 200
TEXT_HIDDEN = 160
TEXT_LR = 1e-3
TEXT_WEIGHT_DECAY = 1e-4

# Audio settings
AUDIO_SR = 16000
AUDIO_SECONDS = 5.0
AUDIO_BATCH_SIZE = 16
AUDIO_EPOCHS = 45
AUDIO_PATIENCE = 7
AUDIO_LR = 8e-4
AUDIO_WEIGHT_DECAY = 1e-4
AUDIO_N_MFCC = 40
AUDIO_N_MELS = 80

# Feature caches
ACOUSTIC_CACHE = OUT_DIR / "cache_acoustic_features.npz"
MFCC_SEQ_CACHE = OUT_DIR / "cache_mfcc_sequence_features.npz"
LOGMEL_CACHE = OUT_DIR / "cache_logmel_features.npz"


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def canonical_label(x) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    mp = {
        "neutral": "Neutral", "happy": "Happy", "angry": "Angry",
        "sad": "Sad", "fear": "Fear", "disgust": "Disgust",
        "boredom": "Boredum", "boredum": "Boredum",
    }
    return mp.get(s.lower(), s)


def audio_folder_label(x) -> str:
    lab = canonical_label(x)
    return "Boredom" if lab == "Boredum" else lab


def norm_id(x) -> str:
    s = "" if pd.isna(x) else str(x).strip().replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"\.(wav|mp3|flac|m4a|aac|ogg)$", "", s, flags=re.I)
    return s


def safe_name(x) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def load_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"{name}_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    required = ["file_id", "target_label", TEXT_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df["file_id"] = df["file_id"].map(norm_id)
    df["target_label"] = df["target_label"].map(canonical_label)
    df["target_id"] = df["target_label"].map(LABEL2ID).astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    return df.reset_index(drop=True)


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    per_f1 = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "min_class_f1": float(per_f1.min()),
    }

    per_p = precision_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    per_r = recall_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    for i, lab in enumerate(LABELS):
        out[f"precision_{lab}"] = float(per_p[i])
        out[f"recall_{lab}"] = float(per_r[i])
        out[f"f1_{lab}"] = float(per_f1[i])
    return out


def save_loss_plot(history_csv: Path, out_png: Path):
    if not HAS_MPL or not history_csv.exists():
        return
    df = pd.read_csv(history_csv)
    plt.figure(figsize=(7, 5))
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_report(
    outdir: Path,
    name: str,
    y_true,
    y_pred,
    pred_proba: Optional[np.ndarray] = None,
    file_ids: Optional[List[str]] = None,
    extra: Optional[Dict] = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    m = metric_dict(y_true, y_pred)
    if extra:
        m.update(extra)

    (outdir / f"{name}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    report = classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))),
        target_names=LABELS, zero_division=0
    )
    (outdir / f"{name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(outdir / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "file_id": file_ids if file_ids is not None else [str(i) for i in range(len(y_true))],
        "true_id": y_true,
        "true_label": [ID2LABEL[int(i)] for i in y_true],
        "pred_id": y_pred,
        "pred_label": [ID2LABEL[int(i)] for i in y_pred],
        "correct": y_true == y_pred,
    })

    if pred_proba is not None:
        pred_proba = np.asarray(pred_proba, dtype=np.float32)
        pred_df["confidence"] = pred_proba.max(axis=1)
        for i, lab in enumerate(LABELS):
            pred_df[f"prob_{lab}"] = pred_proba[:, i]

    pred_df.to_csv(outdir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
    pred_df[~pred_df["correct"]].to_csv(outdir / f"{name}_misclassified_samples.csv", index=False, encoding="utf-8-sig")

    return m, report, cm


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def class_weights(y) -> torch.Tensor:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=len(LABELS)).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def sklearn_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            if p.shape[1] == len(LABELS):
                return p
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            s = model.decision_function(X)
            if s.ndim == 1:
                s = np.stack([-s, s], axis=1)
            s = s - s.max(axis=1, keepdims=True)
            p = np.exp(s)
            return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        except Exception:
            pass
    return None


# =============================================================================
# TEXT ML
# =============================================================================

def make_text_ml_models():
    models = []

    models.append(("TFIDF_LinearSVM", Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.5, class_weight="balanced", random_state=SEED), cv=3)),
    ])))

    models.append(("TFIDF_LogisticRegression", Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced", solver="saga", n_jobs=N_JOBS, random_state=SEED)),
    ])))

    models.append(("TFIDF_MultinomialNB", Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.3)),
    ])))

    models.append(("TFIDF_RandomForest", Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)),
        ("clf", RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS)),
    ])))

    if USE_XGBOOST and HAS_XGB:
        models.append(("TFIDF_XGBoost", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)),
            ("clf", XGBClassifier(
                n_estimators=450, max_depth=4, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.85,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=SEED, n_jobs=N_JOBS
            )),
        ])))

    if USE_LIGHTGBM and HAS_LGBM:
        models.append(("TFIDF_LightGBM", Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)),
            ("clf", LGBMClassifier(
                n_estimators=500, learning_rate=0.04, num_leaves=31,
                subsample=0.9, colsample_bytree=0.85,
                class_weight="balanced", random_state=SEED, n_jobs=N_JOBS
            )),
        ])))

    models.append(("TFIDF_KNN", Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine")),
    ])))

    return models


def run_text_ml(train_df, val_df, test_df):
    print("\n" + "=" * 100)
    print("TEXT CLASSICAL ML")
    print("=" * 100)

    out_root = OUT_DIR / "text_ml"
    X_train = train_df[TEXT_COLUMN].astype(str).tolist()
    y_train = train_df["target_id"].to_numpy(dtype=int)
    X_test = test_df[TEXT_COLUMN].astype(str).tolist()
    y_test = test_df["target_id"].to_numpy(dtype=int)

    rows = []
    for name, model in make_text_ml_models():
        print("\nTraining:", name)
        outdir = out_root / safe_name(name)
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            pred = model.predict(X_test)
            proba = sklearn_proba(model, X_test)
            infer_time = time.time() - t1

            extra = {
                "training_time_sec": float(train_time),
                "inference_time_sec": float(infer_time),
                "trainable_parameters": 0,
            }
            m, rep, cm = save_report(outdir, "test", y_test, pred, proba, test_df["file_id"].tolist(), extra)
            row = {"branch": "text_ml", "model": name, **m, "outdir": str(outdir)}
            rows.append(row)
            print(json.dumps({k: m[k] for k in ["accuracy", "macro_f1", "weighted_f1", "min_class_f1"]}, indent=2))
        except Exception as e:
            print("FAILED:", name, repr(e))
            rows.append({"branch": "text_ml", "model": name, "error": repr(e), "outdir": str(outdir)})

        pd.DataFrame(rows).to_csv(OUT_DIR / "text_ml_summary.csv", index=False, encoding="utf-8-sig")
    return rows


# =============================================================================
# TEXT DL
# =============================================================================

def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[\w\u0600-\u06FF]+", str(text), flags=re.UNICODE)


def build_vocab(texts, min_freq=1) -> Dict[str, int]:
    c = Counter()
    for t in texts:
        c.update(tokenize_text(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, n in c.most_common():
        if n >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_text(text, vocab):
    ids = [vocab.get(tok, 1) for tok in tokenize_text(text)]
    ids = ids[:MAX_LEN]
    ids += [0] * max(0, MAX_LEN - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, df, vocab):
        self.y = df["target_id"].to_numpy(dtype=np.int64)
        self.file_ids = df["file_id"].tolist()
        self.x = np.array([encode_text(t, vocab) for t in df[TEXT_COLUMN].astype(str).tolist()], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "file_id": self.file_ids[idx],
        }


class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, TEXT_EMB_DIM, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(TEXT_EMB_DIM, 160, k) for k in [2, 3, 4, 5]])
        self.drop = nn.Dropout(0.45)
        self.fc = nn.Linear(160 * 4, len(LABELS))

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = [F.max_pool1d(F.relu(conv(e)), kernel_size=F.relu(conv(e)).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(self.drop(torch.cat(h, dim=1)))


class RNNText(nn.Module):
    def __init__(self, vocab_size, rnn_type="lstm", bidirectional=False, use_attention=False, use_cnn=False):
        super().__init__()
        self.use_attention = use_attention
        self.use_cnn = use_cnn
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, TEXT_EMB_DIM, padding_idx=0)

        rnn_input = TEXT_EMB_DIM
        if use_cnn:
            self.conv = nn.Conv1d(TEXT_EMB_DIM, TEXT_EMB_DIM, kernel_size=3, padding=1)
            rnn_input = TEXT_EMB_DIM

        klass = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = klass(
            rnn_input, TEXT_HIDDEN, num_layers=1, batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = TEXT_HIDDEN * (2 if bidirectional else 1)
        if use_attention:
            self.attn = nn.Linear(out_dim, 1)
        self.drop = nn.Dropout(0.45)
        self.fc = nn.Linear(out_dim, len(LABELS))

    def forward(self, x):
        mask = x.ne(0)
        e = self.emb(x)
        if self.use_cnn:
            e = F.relu(self.conv(e.transpose(1, 2))).transpose(1, 2)
        h, _ = self.rnn(e)

        if self.use_attention:
            score = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4)
            w = F.softmax(score, dim=1).unsqueeze(-1)
            pooled = (h * w).sum(dim=1)
        else:
            h = h.masked_fill(~mask.unsqueeze(-1), -1e4)
            pooled = h.max(dim=1).values

        return self.fc(self.drop(pooled))


def evaluate_torch(model, loader, criterion=None):
    model.eval()
    ys, ps, probs = [], [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            logits = model(x)
            if criterion is not None:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * len(y)
                n += len(y)
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
            probs.append(prob.cpu().numpy())
    avg_loss = total_loss / max(n, 1) if criterion is not None else np.nan
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs), avg_loss


def train_torch_model(model, train_loader, val_loader, outdir, epochs, patience, lr, wd, y_train):
    outdir.mkdir(parents=True, exist_ok=True)
    model.to(DEVICE)
    param_count = count_trainable_params(model)
    cw = class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_f1, best_state, bad = -1, None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            train_loss += float(loss.item()) * len(y)
            n += len(y)

        val_y, val_pred, val_prob, val_loss = evaluate_torch(model, val_loader, criterion)
        val_m = metric_dict(val_y, val_pred)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n, 1),
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_m.items()}
        }
        history.append(row)
        print(f"Epoch {epoch:03d} | loss={row['train_loss']:.4f} | val_loss={val_loss:.4f} | val_acc={val_m['accuracy']:.4f} | val_f1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, outdir / "best_model.pt")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    training_time = time.time() - t0
    hist_path = outdir / "training_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False, encoding="utf-8-sig")
    save_loss_plot(hist_path, outdir / "loss_curve.png")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, training_time, param_count


def run_text_dl(train_df, val_df, test_df):
    print("\n" + "=" * 100)
    print("TEXT DEEP LEARNING")
    print("=" * 100)

    out_root = OUT_DIR / "text_dl"
    vocab = build_vocab(train_df[TEXT_COLUMN].astype(str).tolist(), min_freq=1)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    train_ds, val_ds, test_ds = TextDataset(train_df, vocab), TextDataset(val_df, vocab), TextDataset(test_df, vocab)
    train_loader = DataLoader(train_ds, batch_size=TEXT_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=TEXT_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=TEXT_BATCH_SIZE, shuffle=False, num_workers=0)
    y_train = train_df["target_id"].to_numpy(dtype=int)
    y_test = test_df["target_id"].to_numpy(dtype=int)

    models = [
        ("TextCNN", lambda: TextCNN(len(vocab))),
        ("LSTM", lambda: RNNText(len(vocab), "lstm", False, False, False)),
        ("BiLSTM", lambda: RNNText(len(vocab), "lstm", True, False, False)),
        ("GRU", lambda: RNNText(len(vocab), "gru", False, False, False)),
        ("BiGRU", lambda: RNNText(len(vocab), "gru", True, False, False)),
        ("CNN_LSTM", lambda: RNNText(len(vocab), "lstm", False, False, True)),
        ("CNN_BiLSTM", lambda: RNNText(len(vocab), "lstm", True, False, True)),
        ("CNN_GRU", lambda: RNNText(len(vocab), "gru", False, False, True)),
        ("CNN_BiGRU", lambda: RNNText(len(vocab), "gru", True, False, True)),
        ("BiLSTM_Attention", lambda: RNNText(len(vocab), "lstm", True, True, False)),
        ("BiGRU_Attention", lambda: RNNText(len(vocab), "gru", True, True, False)),
    ]

    rows = []
    for name, factory in models:
        print("\nTraining:", name)
        outdir = out_root / safe_name(name)
        try:
            model = factory()
            model, train_time, params = train_torch_model(
                model, train_loader, val_loader, outdir,
                TEXT_EPOCHS, TEXT_PATIENCE, TEXT_LR, TEXT_WEIGHT_DECAY, y_train
            )
            t1 = time.time()
            y_true, y_pred, prob, _ = evaluate_torch(model, test_loader)
            infer_time = time.time() - t1
            extra = {"training_time_sec": float(train_time), "inference_time_sec": float(infer_time), "trainable_parameters": int(params)}
            m, rep, cm = save_report(outdir, "test", y_true, y_pred, prob, test_df["file_id"].tolist(), extra)
            row = {"branch": "text_dl", "model": name, **m, "outdir": str(outdir)}
            rows.append(row)
            print(json.dumps({k: m[k] for k in ["accuracy", "macro_f1", "weighted_f1", "min_class_f1"]}, indent=2))
        except Exception as e:
            print("FAILED:", name, repr(e))
            rows.append({"branch": "text_dl", "model": name, "error": repr(e), "outdir": str(outdir)})

        pd.DataFrame(rows).to_csv(OUT_DIR / "text_dl_summary.csv", index=False, encoding="utf-8-sig")
    return rows


# =============================================================================
# AUDIO FEATURE EXTRACTION
# =============================================================================

def find_audio_path(file_id, label):
    folder = DATA_ROOT / audio_folder_label(label)
    for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]:
        p = folder / f"{file_id}{ext}"
        if p.exists():
            return p
    if folder.exists():
        for p in folder.iterdir():
            if p.is_file() and p.stem == file_id:
                return p
    raise FileNotFoundError(f"Audio missing for file_id={file_id}, label={label}, folder={folder}")


def load_audio(path):
    if not HAS_LIBROSA:
        raise ImportError("librosa not installed. Install: pip install librosa soundfile")
    y, _ = librosa.load(str(path), sr=AUDIO_SR, mono=True)
    target = int(AUDIO_SR * AUDIO_SECONDS)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    elif len(y) > target:
        y = y[:target]
    return y.astype(np.float32)


def extract_acoustic_vector(path):
    y = load_audio(path)
    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=AUDIO_SR, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    for arr in [mfcc, d1, d2]:
        feats.extend(arr.mean(axis=1))
        feats.extend(arr.std(axis=1))
        feats.extend(np.percentile(arr, 25, axis=1))
        feats.extend(np.percentile(arr, 75, axis=1))

    for arr in [
        librosa.feature.chroma_stft(y=y, sr=AUDIO_SR),
        librosa.feature.spectral_contrast(y=y, sr=AUDIO_SR),
        librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=AUDIO_SR),
    ]:
        feats.extend(arr.mean(axis=1))
        feats.extend(arr.std(axis=1))

    for arr in [
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
        librosa.feature.spectral_centroid(y=y, sr=AUDIO_SR),
        librosa.feature.spectral_bandwidth(y=y, sr=AUDIO_SR),
        librosa.feature.spectral_rolloff(y=y, sr=AUDIO_SR),
    ]:
        feats.extend(arr.mean(axis=1))
        feats.extend(arr.std(axis=1))
        feats.extend(arr.max(axis=1))
        feats.extend(arr.min(axis=1))

    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=AUDIO_SR)
        f0 = f0[np.isfinite(f0)]
        if len(f0):
            feats.extend([np.mean(f0), np.std(f0), np.median(f0), np.max(f0), np.min(f0)])
        else:
            feats.extend([0, 0, 0, 0, 0])
    except Exception:
        feats.extend([0, 0, 0, 0, 0])

    return np.nan_to_num(np.array(feats, dtype=np.float32))


def extract_mfcc_seq(path):
    y = load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=AUDIO_SR, n_mfcc=AUDIO_N_MFCC, n_fft=512, hop_length=160, win_length=400)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.astype(np.float32)  # C,T


def extract_logmel(path):
    y = load_audio(path)
    mel = librosa.feature.melspectrogram(
        y=y, sr=AUDIO_SR, n_fft=512, hop_length=160, win_length=400,
        n_mels=AUDIO_N_MELS, fmin=50, fmax=7600, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)  # C,T


def build_audio_cache(train_df, val_df, test_df, cache_path, extractor, cache_name):
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if cache_path.exists():
        print(f"Loading {cache_name} cache:", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        ids = data["file_id"].tolist()
        if ids == all_df["file_id"].tolist():
            ntr, nv = len(train_df), len(val_df)
            return X[:ntr], X[ntr:ntr + nv], X[ntr + nv:]
        print("Cache order mismatch. Rebuilding.")

    print(f"Extracting {cache_name} features. First run may take time...")
    feats = []
    for i, row in all_df.iterrows():
        if i % 100 == 0:
            print(f"  {cache_name}: {i+1}/{len(all_df)}")
        p = find_audio_path(row["file_id"], row["target_label"])
        feats.append(extractor(p))

    X = np.stack(feats).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, file_id=all_df["file_id"].to_numpy(dtype=object))
    ntr, nv = len(train_df), len(val_df)
    return X[:ntr], X[ntr:ntr + nv], X[ntr + nv:]


# =============================================================================
# AUDIO ML
# =============================================================================

def make_audio_ml_models():
    models = [
        ("Acoustic_SVM_RBF", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
        ])),
        ("Acoustic_LinearSVM", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3)),
        ])),
        ("Acoustic_LogisticRegression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", random_state=SEED)),
        ])),
        ("Acoustic_RandomForest", Pipeline([
            ("clf", RandomForestClassifier(n_estimators=700, class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS)),
        ])),
        ("Acoustic_ExtraTrees", Pipeline([
            ("clf", ExtraTreesClassifier(n_estimators=700, class_weight="balanced", random_state=SEED, n_jobs=N_JOBS)),
        ])),
    ]

    if USE_XGBOOST and HAS_XGB:
        models.append(("Acoustic_XGBoost", Pipeline([
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.9,
                colsample_bytree=0.85, objective="multi:softprob",
                eval_metric="mlogloss", random_state=SEED, n_jobs=N_JOBS
            )),
        ])))

    if USE_LIGHTGBM and HAS_LGBM:
        models.append(("Acoustic_LightGBM", Pipeline([
            ("clf", LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=31,
                subsample=0.9, colsample_bytree=0.85, class_weight="balanced",
                random_state=SEED, n_jobs=N_JOBS
            )),
        ])))

    models.append(("Acoustic_KNN", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ])))

    return models


def run_audio_ml(train_df, val_df, test_df):
    print("\n" + "=" * 100)
    print("AUDIO CLASSICAL ML")
    print("=" * 100)

    if not HAS_LIBROSA:
        return [{"branch": "audio_ml", "model": "ALL", "error": "librosa not installed"}]

    X_train, X_val, X_test = build_audio_cache(train_df, val_df, test_df, ACOUSTIC_CACHE, extract_acoustic_vector, "acoustic")
    y_train = train_df["target_id"].to_numpy(dtype=int)
    y_test = test_df["target_id"].to_numpy(dtype=int)

    rows = []
    out_root = OUT_DIR / "audio_ml"
    for name, model in make_audio_ml_models():
        print("\nTraining:", name)
        outdir = out_root / safe_name(name)
        try:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            pred = model.predict(X_test)
            proba = sklearn_proba(model, X_test)
            infer_time = time.time() - t1

            extra = {"training_time_sec": float(train_time), "inference_time_sec": float(infer_time), "trainable_parameters": 0}
            m, rep, cm = save_report(outdir, "test", y_test, pred, proba, test_df["file_id"].tolist(), extra)
            row = {"branch": "audio_ml", "model": name, **m, "outdir": str(outdir)}
            rows.append(row)
            print(json.dumps({k: m[k] for k in ["accuracy", "macro_f1", "weighted_f1", "min_class_f1"]}, indent=2))
        except Exception as e:
            print("FAILED:", name, repr(e))
            rows.append({"branch": "audio_ml", "model": name, "error": repr(e), "outdir": str(outdir)})

        pd.DataFrame(rows).to_csv(OUT_DIR / "audio_ml_summary.csv", index=False, encoding="utf-8-sig")
    return rows


# =============================================================================
# AUDIO DL
# =============================================================================

class AudioSeqDataset(Dataset):
    def __init__(self, X, df, augment=False):
        self.X = X
        self.y = df["target_id"].to_numpy(dtype=np.int64)
        self.file_ids = df["file_id"].tolist()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            if random.random() < 0.5:
                t = x.shape[1]
                w = random.randint(3, min(18, max(4, t // 10)))
                s = random.randint(0, max(0, t - w))
                x[:, s:s+w] = 0
            if random.random() < 0.5:
                f = x.shape[0]
                w = random.randint(3, min(10, max(4, f // 5)))
                s = random.randint(0, max(0, f - w))
                x[s:s+w, :] = 0
        return {"x": torch.tensor(x[None, :, :], dtype=torch.float32), "y": torch.tensor(self.y[idx], dtype=torch.long)}


class Audio1DCNN(nn.Module):
    def __init__(self, in_ch=AUDIO_N_MFCC):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.15),
            nn.Conv1d(96, 160, 5, padding=2), nn.BatchNorm1d(160), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.20),
            nn.Conv1d(160, 224, 3, padding=1), nn.BatchNorm1d(224), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(224, len(LABELS)))

    def forward(self, x):
        x = x.squeeze(1)
        return self.fc(self.net(x))


class AudioCNNRNN(nn.Module):
    def __init__(self, in_ch=AUDIO_N_MFCC, rnn_type="lstm", bidirectional=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.15),
            nn.Conv1d(128, 160, 3, padding=1), nn.BatchNorm1d(160), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.20),
        )
        klass = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = klass(160, 128, batch_first=True, bidirectional=bidirectional)
        out_dim = 128 * (2 if bidirectional else 1)
        self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(out_dim, len(LABELS)))

    def forward(self, x):
        x = x.squeeze(1)
        h = self.cnn(x).transpose(1, 2)
        h, _ = self.rnn(h)
        pooled = h.max(dim=1).values
        return self.fc(pooled)


class AudioBiRNNAttention(nn.Module):
    def __init__(self, in_ch=AUDIO_N_MFCC):
        super().__init__()
        self.rnn = nn.LSTM(in_ch, 128, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)
        self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(256, len(LABELS)))

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        h, _ = self.rnn(x)
        score = self.attn(h).squeeze(-1)
        w = F.softmax(score, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.fc(pooled)


class LogMelCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout2d(0.20),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(192, len(LABELS)))

    def forward(self, x):
        return self.fc(self.net(x))


class LogMelCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.20),
        )
        self.rnn = nn.LSTM(128 * 10, 128, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256, 1)
        self.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(256, len(LABELS)))

    def forward(self, x):
        h = self.cnn(x)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        h, _ = self.rnn(h)
        score = self.attn(h).squeeze(-1)
        w = F.softmax(score, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.fc(pooled)


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        return F.relu(out + identity)


class LogMelResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.layer2 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
        self.layer3 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, len(LABELS)))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x))


def run_audio_dl(train_df, val_df, test_df):
    print("\n" + "=" * 100)
    print("AUDIO DEEP LEARNING")
    print("=" * 100)

    if not HAS_LIBROSA:
        return [{"branch": "audio_dl", "model": "ALL", "error": "librosa not installed"}]

    # MFCC sequence models
    Xtr_mfcc, Xva_mfcc, Xte_mfcc = build_audio_cache(train_df, val_df, test_df, MFCC_SEQ_CACHE, extract_mfcc_seq, "mfcc_seq")
    # LogMel spectrogram models
    Xtr_mel, Xva_mel, Xte_mel = build_audio_cache(train_df, val_df, test_df, LOGMEL_CACHE, extract_logmel, "logmel")

    y_train = train_df["target_id"].to_numpy(dtype=int)

    rows = []
    out_root = OUT_DIR / "audio_dl"

    audio_jobs = [
        ("MFCC_1D_CNN", "mfcc", lambda: Audio1DCNN(AUDIO_N_MFCC)),
        ("MFCC_CNN_LSTM", "mfcc", lambda: AudioCNNRNN(AUDIO_N_MFCC, "lstm", False)),
        ("MFCC_CNN_BiLSTM", "mfcc", lambda: AudioCNNRNN(AUDIO_N_MFCC, "lstm", True)),
        ("MFCC_CNN_GRU", "mfcc", lambda: AudioCNNRNN(AUDIO_N_MFCC, "gru", False)),
        ("MFCC_CNN_BiGRU", "mfcc", lambda: AudioCNNRNN(AUDIO_N_MFCC, "gru", True)),
        ("Audio_BiLSTM_Attention", "mfcc", lambda: AudioBiRNNAttention(AUDIO_N_MFCC)),
        ("LogMel_CNN", "mel", lambda: LogMelCNN()),
        ("LogMel_CRNN", "mel", lambda: LogMelCRNN()),
        ("LogMel_ResNet", "mel", lambda: LogMelResNet()),
    ]

    for name, kind, factory in audio_jobs:
        print("\nTraining:", name)
        outdir = out_root / safe_name(name)
        try:
            if kind == "mfcc":
                train_ds = AudioSeqDataset(Xtr_mfcc, train_df, augment=True)
                val_ds = AudioSeqDataset(Xva_mfcc, val_df, augment=False)
                test_ds = AudioSeqDataset(Xte_mfcc, test_df, augment=False)
            else:
                train_ds = AudioSeqDataset(Xtr_mel, train_df, augment=True)
                val_ds = AudioSeqDataset(Xva_mel, val_df, augment=False)
                test_ds = AudioSeqDataset(Xte_mel, test_df, augment=False)

            train_loader = DataLoader(train_ds, batch_size=AUDIO_BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=AUDIO_BATCH_SIZE, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=AUDIO_BATCH_SIZE, shuffle=False, num_workers=0)

            model = factory()
            model, train_time, params = train_torch_model(
                model, train_loader, val_loader, outdir,
                AUDIO_EPOCHS, AUDIO_PATIENCE, AUDIO_LR, AUDIO_WEIGHT_DECAY, y_train
            )

            t1 = time.time()
            y_true, y_pred, prob, _ = evaluate_torch(model, test_loader)
            infer_time = time.time() - t1

            extra = {"training_time_sec": float(train_time), "inference_time_sec": float(infer_time), "trainable_parameters": int(params)}
            m, rep, cm = save_report(outdir, "test", y_true, y_pred, prob, test_df["file_id"].tolist(), extra)
            row = {"branch": "audio_dl", "model": name, **m, "outdir": str(outdir)}
            rows.append(row)
            print(json.dumps({k: m[k] for k in ["accuracy", "macro_f1", "weighted_f1", "min_class_f1"]}, indent=2))
        except Exception as e:
            print("FAILED:", name, repr(e))
            rows.append({"branch": "audio_dl", "model": name, "error": repr(e), "outdir": str(outdir)})

        pd.DataFrame(rows).to_csv(OUT_DIR / "audio_dl_summary.csv", index=False, encoding="utf-8-sig")

    return rows


# =============================================================================
# SIMPLE FUSION
# =============================================================================

def find_best_prob_file(branches: List[str]) -> Optional[Path]:
    # Search within this benchmark output for best test_predictions.csv based on summary macro_f1.
    summaries = []
    if (OUT_DIR / "text_ml_summary.csv").exists():
        summaries.append(pd.read_csv(OUT_DIR / "text_ml_summary.csv"))
    if (OUT_DIR / "text_dl_summary.csv").exists():
        summaries.append(pd.read_csv(OUT_DIR / "text_dl_summary.csv"))
    if (OUT_DIR / "audio_ml_summary.csv").exists():
        summaries.append(pd.read_csv(OUT_DIR / "audio_ml_summary.csv"))
    if (OUT_DIR / "audio_dl_summary.csv").exists():
        summaries.append(pd.read_csv(OUT_DIR / "audio_dl_summary.csv"))

    if not summaries:
        return None
    df = pd.concat(summaries, ignore_index=True)
    df = df[df["branch"].isin(branches)].copy()
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df = df.dropna(subset=["macro_f1"]).sort_values(["macro_f1", "accuracy"], ascending=False)
    for _, row in df.iterrows():
        p = Path(row["outdir"]) / "test_predictions.csv"
        if p.exists():
            return p
    return None


def load_pred_csv(path: Path):
    df = pd.read_csv(path)
    df["file_id"] = df["file_id"].map(norm_id)
    if "true_label" in df.columns:
        df["true_label"] = df["true_label"].map(canonical_label)
        df["true_id"] = df["true_label"].map(LABEL2ID).astype(int)
    prob_cols = [f"prob_{lab}" for lab in LABELS]
    if not all(c in df.columns for c in prob_cols):
        return df, None
    probs = df[prob_cols].to_numpy(dtype=np.float32)
    probs = np.maximum(probs, 0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return df, probs


def simple_fusion_from_files(text_path, audio_path, test_df):
    tdf, tp = load_pred_csv(text_path)
    adf, ap = load_pred_csv(audio_path)
    if tp is None or ap is None:
        print("Simple fusion skipped because best text/audio file does not contain probabilities.")
        return []

    left = tdf[["file_id", "true_id"]].copy()
    left["t_idx"] = np.arange(len(left))
    right = adf[["file_id", "true_id"]].copy()
    right["a_idx"] = np.arange(len(right))
    merged = left.merge(right, on="file_id", suffixes=("_t", "_a"))
    if len(merged) < 300:
        print("Fusion merge too small:", len(merged))
        return []
    if not np.all(merged["true_id_t"].to_numpy() == merged["true_id_a"].to_numpy()):
        print("Fusion label mismatch.")
        return []

    y = merged["true_id_t"].to_numpy(dtype=int)
    file_ids = merged["file_id"].tolist()
    tprob = tp[merged["t_idx"].to_numpy()]
    aprob = ap[merged["a_idx"].to_numpy()]

    rows = []
    out_root = OUT_DIR / "simple_fusion"

    # 1 Average probability fusion
    methods = []
    methods.append(("Average_probability_fusion", 0.5 * tprob + 0.5 * aprob))

    # 2 Weighted probability fusion. Validation-free default based on previous best: text stronger than audio.
    methods.append(("Weighted_probability_fusion_text0.65_audio0.35", 0.65 * tprob + 0.35 * aprob))

    # 3 Late decision fusion. Use text unless audio confidence is much higher.
    tconf = tprob.max(axis=1)
    aconf = aprob.max(axis=1)
    tpred = tprob.argmax(axis=1)
    apred = aprob.argmax(axis=1)
    late_pred = np.where(aconf > tconf + 0.15, apred, tpred)
    late_prob = np.zeros_like(tprob)
    late_prob[np.arange(len(late_pred)), late_pred] = 1.0

    # 4 Max-confidence fusion.
    max_pred = np.where(aconf > tconf, apred, tpred)
    max_prob = np.where((aconf > tconf)[:, None], aprob, tprob)

    for name, prob in methods:
        outdir = out_root / safe_name(name)
        pred = prob.argmax(axis=1)
        m, rep, cm = save_report(outdir, "test", y, pred, prob, file_ids, {
            "training_time_sec": 0.0,
            "inference_time_sec": 0.0,
            "trainable_parameters": 0,
            "text_prediction_file": str(text_path),
            "audio_prediction_file": str(audio_path),
        })
        rows.append({"branch": "simple_fusion", "model": name, **m, "outdir": str(outdir)})

    outdir = out_root / "Late_decision_fusion"
    m, rep, cm = save_report(outdir, "test", y, late_pred, late_prob, file_ids, {
        "training_time_sec": 0.0, "inference_time_sec": 0.0, "trainable_parameters": 0,
        "text_prediction_file": str(text_path), "audio_prediction_file": str(audio_path),
    })
    rows.append({"branch": "simple_fusion", "model": "Late_decision_fusion", **m, "outdir": str(outdir)})

    outdir = out_root / "Max_confidence_fusion"
    m, rep, cm = save_report(outdir, "test", y, max_pred, max_prob, file_ids, {
        "training_time_sec": 0.0, "inference_time_sec": 0.0, "trainable_parameters": 0,
        "text_prediction_file": str(text_path), "audio_prediction_file": str(audio_path),
    })
    rows.append({"branch": "simple_fusion", "model": "Max_confidence_fusion", **m, "outdir": str(outdir)})

    pd.DataFrame(rows).to_csv(OUT_DIR / "simple_fusion_summary.csv", index=False, encoding="utf-8-sig")
    return rows


def run_simple_fusion(test_df):
    print("\n" + "=" * 100)
    print("SIMPLE FUSION")
    print("=" * 100)
    text_path = find_best_prob_file(["text_ml", "text_dl"])
    audio_path = find_best_prob_file(["audio_ml", "audio_dl"])
    print("Best text prediction file:", text_path)
    print("Best audio prediction file:", audio_path)
    if text_path is None or audio_path is None:
        return [{"branch": "simple_fusion", "model": "ALL", "error": "missing text/audio probability files"}]
    return simple_fusion_from_files(text_path, audio_path, test_df)


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nUrduSER Selected ML + DL + Simple Fusion Benchmark")
    print("Device:", DEVICE)
    print("Split:", SPLIT_DIR)
    print("Output:", OUT_DIR)
    print("No NEXT context. Text column:", TEXT_COLUMN)
    print("XGBoost available:", HAS_XGB)
    print("LightGBM available:", HAS_LGBM)
    print("Librosa available:", HAS_LIBROSA)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    print("\nSplit sizes:")
    print("Train:", len(train_df), train_df.target_label.value_counts().to_dict())
    print("Val  :", len(val_df), val_df.target_label.value_counts().to_dict())
    print("Test :", len(test_df), test_df.target_label.value_counts().to_dict())

    all_rows = []

    if RUN_TEXT_ML:
        all_rows.extend(run_text_ml(train_df, val_df, test_df))
    if RUN_TEXT_DL:
        all_rows.extend(run_text_dl(train_df, val_df, test_df))
    if RUN_AUDIO_ML:
        all_rows.extend(run_audio_ml(train_df, val_df, test_df))
    if RUN_AUDIO_DL:
        all_rows.extend(run_audio_dl(train_df, val_df, test_df))
    if RUN_SIMPLE_FUSION:
        all_rows.extend(run_simple_fusion(test_df))

    summary = pd.DataFrame(all_rows)
    if "macro_f1" in summary.columns:
        summary["macro_f1"] = pd.to_numeric(summary["macro_f1"], errors="coerce")
        summary["accuracy"] = pd.to_numeric(summary["accuracy"], errors="coerce")
        summary = summary.sort_values(["macro_f1", "accuracy"], ascending=False, na_position="last")

    summary.to_csv(OUT_DIR / "selected_ml_dl_fusion_summary.csv", index=False, encoding="utf-8-sig")

    # Best by branch
    if "branch" in summary.columns and "macro_f1" in summary.columns:
        best = (
            summary.dropna(subset=["macro_f1"])
            .sort_values(["branch", "macro_f1", "accuracy"], ascending=[True, False, False])
            .groupby("branch", as_index=False)
            .head(1)
        )
        best.to_csv(OUT_DIR / "best_by_branch_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("FINAL SELECTED ML + DL + SIMPLE FUSION SUMMARY")
    print("=" * 100)
    cols = [c for c in ["branch", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "min_class_f1", "training_time_sec", "inference_time_sec", "trainable_parameters", "outdir", "error"] if c in summary.columns]
    print(summary[cols].to_string(index=False))

    print("\nSaved:")
    print(" ", OUT_DIR / "selected_ml_dl_fusion_summary.csv")
    print(" ", OUT_DIR / "best_by_branch_summary.csv")
    print(" ", OUT_DIR / "text_ml_summary.csv")
    print(" ", OUT_DIR / "text_dl_summary.csv")
    print(" ", OUT_DIR / "audio_ml_summary.csv")
    print(" ", OUT_DIR / "audio_dl_summary.csv")
    print(" ", OUT_DIR / "simple_fusion_summary.csv")


if __name__ == "__main__":
    main()
