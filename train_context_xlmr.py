# urdu_ser_text_true_context_xlmr_7class.py 
# Accuracy-boosted 7-class version converted from the uploaded 4-class text-only XLM-R code:
# Neutral, Happy, Angry, Sad, Fear, Disgust, Boredum
#
# IMPORTANT:
# This script needs an unmerged 7-class label column in your CSV/XLSX.
# Accepted examples: label_7, folder_7, emotion, folder, label, target_label.
# If your file only contains folder_merged4 / label_4 / label_id_4, the original
# seven emotions cannot be recovered automatically.

import os, re, json, random, warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class CFG:
    DATA_CSV: str = "./data/full_audit_report.csv"
    TRAIN_CSV: str = ""
    VAL_CSV: str = ""
    TEST_CSV: str = ""
    ORIGINAL_FILES: tuple = ("./data/UrSEC.xlsx",)

    # Optional: put the exact 7-class column name here if automatic detection fails.
    # Example: LABEL_COL: str = "folder"
    LABEL_COL: str = ""

    TEXT_MODEL: str = "xlm-roberta-base"
    SAVE_DIR: str = "./saved_models_text_shared_prev_curr_7class"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    SEED: int = 123
    EPOCHS: int = 35
    PATIENCE: int = 5
    MAX_TEXT_LEN: int = 320
    BATCH_SIZE: int = 8
    GRAD_ACCUM_STEPS: int = 4
    NUM_WORKERS: int = 2

    LR_ENCODER: float = 2e-5
    LR_CLASSIFIER: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.08
    MAX_GRAD_NORM: float = 1.0
    LABEL_SMOOTHING: float = 0.02
    USE_AMP: bool = True
    USE_WEIGHTED_SAMPLER: bool = False

    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    TENSORBOARD_DIR: str = "./runs/text_shared_prev_curr_7class"

    # Accuracy-boost options for the harder 7-class task.
    USE_LAST4_HIDDEN: bool = True
    USE_FOCAL_LOSS: bool = True
    FOCAL_GAMMA: float = 1.35
    USE_RDROP: bool = True
    RDROP_ALPHA: float = 0.50
    CONTEXT_WITH_NEXT: bool = False
    CURRENT_ONLY_TEXT: bool = False
    SHARED_SPLIT_DIR: str = "./shared_splits_urdu_ser_7class_seed123_prev_curr"

cfg = CFG()

# The order below controls target IDs:
# 0=Neutral, 1=Happy, 2=Angry, 3=Sad, 4=Fear, 5=Disgust, 6=Boredum
LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredum"]
LABEL_MAP = {x: i for i, x in enumerate(LABELS)}
ID2LABEL = {i: x for x, i in LABEL_MAP.items()}
N_CLASSES = len(LABELS)

# Accept common spelling/case variants and normalize them to the exact label names above.
# The user requested "Boredum"; if your dataset uses "Boredom", it will be converted to "Boredum".
LABEL_ALIASES = {
    "neutral": "Neutral",
    "happy": "Happy",
    "happiness": "Happy",
    "angry": "Angry",
    "anger": "Angry",
    "sad": "Sad",
    "sadness": "Sad",
    "fear": "Fear",
    "fearful": "Fear",
    "disgust": "Disgust",
    "disgusted": "Disgust",
    "boredum": "Boredum",
    "boredom": "Boredum",
    "bored": "Boredum",
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def norm_text(x):
    x = "" if pd.isna(x) else str(x)
    x = re.sub(r"[؟?۔!،,:;؛]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def norm_id(x):
    x = "" if pd.isna(x) else str(x).strip().replace("\\", "/")
    x = x.split("/")[-1]
    x = re.sub(r"\.(wav|mp3|flac|m4a|aac)$", "", x, flags=re.I)
    return x.strip()

def order_key(x):
    nums = re.findall(r"\d+", str(x))
    return tuple(int(n) for n in nums) if nums else (10**9, str(x))

def first_file(paths):
    for p in paths:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No original transcript file found. Checked: " + str(paths))

def read_any(path):
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Unsupported file: " + path)

def choose_col(df, exact, contains):
    lower = {str(c).lower().strip(): c for c in df.columns}
    for e in exact:
        if e.lower() in lower:
            return lower[e.lower()]
    for c in df.columns:
        lc = str(c).lower()
        if any(t in lc for t in contains):
            return c
    return None

def canonical_label(x):
    """
    Convert a raw label value to one of the seven classes.
    Merged four-class labels such as Angry_Disgust or Fear_Sad are intentionally
    not converted because that would create incorrect seven-class labels.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if not s:
        return np.nan

    key = s.lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z_]", "", key)
    key = re.sub(r"_+", "_", key).strip("_")

    # Do not split merged 4-class labels. The correct seven-class label must
    # come from the original unmerged dataset column.
    merged_4_labels = {
        "angry_disgust",
        "fear_sad",
        "neutral_boredom",
        "neutral_boredum",
    }
    if key in merged_4_labels:
        return np.nan

    return LABEL_ALIASES.get(key, np.nan)

def choose_7class_label_col(df):
    """
    Select the best available unmerged seven-class label column.
    The function prefers columns that contain more unique valid seven-class labels.
    """
    if cfg.LABEL_COL:
        if cfg.LABEL_COL not in df.columns:
            raise ValueError(
                f"cfg.LABEL_COL='{cfg.LABEL_COL}' was not found. Available columns: {list(df.columns)}"
            )
        return cfg.LABEL_COL

    preferred = [
        "target_label",
        "label_7",
        "label7",
        "emotion_7",
        "emotion7",
        "emotion",
        "folder_7",
        "folder7",
        "folder",
        "class",
        "category",
        "label",
    ]

    candidate_cols = []
    lower_to_real = {str(c).lower().strip(): c for c in df.columns}

    for name in preferred:
        if name.lower() in lower_to_real:
            candidate_cols.append(lower_to_real[name.lower()])

    # Add other likely columns, but avoid 4-class/id columns.
    for c in df.columns:
        lc = str(c).lower().strip()
        if c in candidate_cols:
            continue
        if "id" in lc:
            continue
        if "merged4" in lc or "label_4" in lc or "folder_merged4" in lc:
            continue
        if any(k in lc for k in ["emotion", "label", "folder", "class", "category"]):
            candidate_cols.append(c)

    scored = []
    for c in candidate_cols:
        canon = df[c].map(canonical_label)
        valid_count = int(canon.notna().sum())
        unique_count = int(canon.dropna().nunique())
        scored.append((unique_count, valid_count, c))

    scored = sorted(scored, reverse=True)

    if scored and scored[0][0] >= 2 and scored[0][1] > 0:
        best_unique, best_valid, best_col = scored[0]
        print(f"Detected 7-class label column: {best_col} | valid rows={best_valid} | unique labels={best_unique}")
        return best_col

    raise ValueError(
        "Could not detect a valid unmerged 7-class label column. "
        "Your data may only contain 4-class labels such as folder_merged4/label_4. "
        "Please add a column containing: Neutral, Happy, Angry, Sad, Fear, Disgust, Boredum "
        "and set cfg.LABEL_COL to that column name. "
        f"Available columns: {list(df.columns)}"
    )

def prep_split(df):
    df = df.copy()

    # Convert full audit report format into training format if needed.
    # Original 4-class code used folder_merged4 here. For 7-class training,
    # we must use an original unmerged emotion column instead.
    if "audio_path" in df.columns and "norm_file_id" in df.columns:
        label_col = choose_7class_label_col(df)

        out = pd.DataFrame()
        out["path"] = df["audio_path"].astype(str)
        out["target_label"] = df[label_col].map(canonical_label)
        out["urdu_text"] = df["urdu_text"].fillna("").map(norm_text) if "urdu_text" in df.columns else ""
        out["file_id"] = df["norm_file_id"].apply(norm_id)

        for extra_col in ["duration", "rms", "final_status", "audio_status"]:
            if extra_col in df.columns:
                out[extra_col] = df[extra_col]

        df = out

    # General split/training CSV format.
    if "target_label" not in df.columns:
        label_col = choose_7class_label_col(df)
        df["target_label"] = df[label_col].map(canonical_label)
    else:
        df["target_label"] = df["target_label"].map(canonical_label)

    # Recompute target IDs from the 7-class labels.
    # This avoids accidentally reusing old 4-class target_id values.
    df["target_id"] = df["target_label"].map(LABEL_MAP)

    if "file_id" not in df.columns:
        if "path" not in df.columns:
            raise ValueError("Split missing file_id and path.")
        df["file_id"] = df["path"].apply(norm_id)

    if "urdu_text" not in df.columns:
        raise ValueError("Split missing urdu_text")

    df = df.dropna(subset=["target_id", "urdu_text", "file_id"]).copy()
    df["file_id"] = df["file_id"].apply(norm_id)
    df["target_id"] = df["target_id"].astype(int)
    df["target_label"] = df["target_id"].map(ID2LABEL)
    df["urdu_text"] = df["urdu_text"].fillna("").map(norm_text)

    # Keep only valid 7-class rows.
    df = df[df["target_id"].isin(list(ID2LABEL.keys()))].copy()

    if df.empty:
        raise ValueError("After 7-class label filtering, no valid rows remained.")

    counts = df["target_label"].value_counts().reindex(LABELS, fill_value=0)
    print("\n7-class label counts after cleaning:")
    print(counts.to_string())

    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        print("\nWARNING: These requested labels are missing from the cleaned dataset:", missing)

    return df.reset_index(drop=True)

def make_true_context_map():
    path = first_file(cfg.ORIGINAL_FILES)
    print("\nUsing original transcript source:", path)
    df = read_any(path)
    print("Original columns:", list(df.columns))

    file_col = "Unnamed: 9"
    text_col = "UrSEC Description"

    df = df.iloc[1:].reset_index(drop=True)

    if file_col is None:
        raise ValueError("Could not detect file_id column. Rename it to file_id.")
    if text_col is None:
        raise ValueError("Could not detect text column. Rename it to urdu_text.")

    print("Detected file_id column:", file_col)
    print("Detected text column :", text_col)

    w = pd.DataFrame({
        "file_id": df[file_col].map(norm_id),
        "orig_text": df[text_col].fillna("").map(norm_text)
    })

    w = w[(w.file_id.astype(str).str.len() > 0) & (w.orig_text.astype(str).str.len() > 0)].copy()

    before = len(w)
    w = w.drop_duplicates("file_id", keep="first").reset_index(drop=True)
    if before != len(w):
        print("Removed duplicate file_id rows:", before - len(w))

    w["_key"] = w.file_id.map(order_key)
    w = w.sort_values("_key").reset_index(drop=True)

    texts = w.orig_text.tolist()
    ctx = []
    for i, t in enumerate(texts):
        prev = texts[i - 1] if i > 0 else ""
        nxt = texts[i + 1] if i < len(texts) - 1 else ""

        # Publication-safe setting: use only the current utterance text.
        # This avoids using the next utterance and avoids any future-context leakage.
        if getattr(cfg, "CURRENT_ONLY_TEXT", True):
            ctx.append(f"[CURR] {t}".strip())
        elif cfg.CONTEXT_WITH_NEXT:
            ctx.append(f"[PREV] {prev} [CURR] {t} [NEXT] {nxt}".strip())
        else:
            ctx.append(f"[PREV] {prev} [CURR] {t}".strip())

    w["true_context_text"] = ctx

    print("Context mapping size:", len(w))
    print(w[["file_id", "true_context_text"]].head(3).to_string(index=False))

    return dict(zip(w.file_id, w.true_context_text))

def load_splits():
    """
    Load the full audit report and create a fresh stratified 80/10/10 split
    using the seven original emotion classes.
    """
    cmap = make_true_context_map()

    raw_path = Path(cfg.DATA_CSV)
    if not raw_path.exists():
        raise FileNotFoundError(f"Full audit dataset not found: {raw_path}")

    raw = pd.read_csv(raw_path)

    # Keep all valid audio/text rows. Do not remove too_short/too_long.
    if "audio_status" in raw.columns:
        raw = raw[raw["audio_status"].astype(str).str.lower().str.strip() == "ok"].copy()

    all_df = prep_split(raw)

    # Remove exact duplicates by file_id, if any.
    dup_count = all_df["file_id"].duplicated().sum()
    if dup_count > 0:
        print(f"Removed duplicate file_id rows: {dup_count}")
        all_df = all_df.drop_duplicates(subset=["file_id"], keep="first").reset_index(drop=True)

    # Optional safety: remove rows whose audio path is missing.
    if "path" in all_df.columns:
        exists_mask = all_df["path"].apply(lambda p: Path(str(p)).exists())
        missing_audio = int((~exists_mask).sum())
        if missing_audio > 0:
            print(f"WARNING: removing {missing_audio} rows because audio file path does not exist.")
            all_df = all_df[exists_mask].reset_index(drop=True)

    # Attach true context.
    all_df["true_context_text"] = all_df.file_id.map(cmap)
    missing = all_df.true_context_text.isna().sum()
    if missing:
        print(f"WARNING all: {missing}/{len(all_df)} missing context, using own urdu_text")
        all_df["true_context_text"] = all_df.true_context_text.fillna(all_df.urdu_text)

    # 80/10/10 split.
    train, temp = train_test_split(
        all_df,
        test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        stratify=all_df["target_id"],
        random_state=cfg.SEED,
    )

    relative_test_size = cfg.TEST_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)

    val, test = train_test_split(
        temp,
        test_size=relative_test_size,
        stratify=temp["target_id"],
        random_state=cfg.SEED,
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("\nLoaded TRUE-context 7-class 80/10/10 splits:")
    print("All  :", len(all_df), all_df.target_label.value_counts().to_dict())
    print("Train:", len(train), train.target_label.value_counts().to_dict())
    print("Val  :", len(val), val.target_label.value_counts().to_dict())
    print("Test :", len(test), test.target_label.value_counts().to_dict())
    print(f"\nActual split ratios: train={len(train)/len(all_df):.3f}, val={len(val)/len(all_df):.3f}, test={len(test)/len(all_df):.3f}")

    return train, val, test

class TextDS(Dataset):
    def __init__(self, df, tok):
        self.df = df.reset_index(drop=True)
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        e = self.tok(
            str(r.true_context_text),
            max_length=cfg.MAX_TEXT_LEN,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": e["input_ids"].squeeze(0),
            "attention_mask": e["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(r.target_id), dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }

def sampler(df):
    y = df.target_id.to_numpy()
    c = np.bincount(y, minlength=N_CLASSES)

    sw = (1 / np.maximum(c, 1))[y]

    # Confusion-aware boosts based on the common 7-class errors:
    # Boredum, Disgust, and Happy usually need stronger attention.
    boosts = np.ones(N_CLASSES, dtype=np.float64)
    boosts[LABEL_MAP["Neutral"]] *= 0.92
    boosts[LABEL_MAP["Happy"]] *= 1.12
    boosts[LABEL_MAP["Angry"]] *= 1.05
    boosts[LABEL_MAP["Sad"]] *= 1.05
    boosts[LABEL_MAP["Fear"]] *= 1.08
    boosts[LABEL_MAP["Disgust"]] *= 1.18
    boosts[LABEL_MAP["Boredum"]] *= 1.20

    sw *= boosts[y]

    return WeightedRandomSampler(
        torch.DoubleTensor(sw),
        len(sw),
        replacement=True,
    )

def loss_weights(df):
    y = df.target_id.to_numpy()
    c = np.bincount(y, minlength=N_CLASSES)

    w = c.sum() / np.maximum(c, 1)
    w = w / w.mean()

    # Confusion-aware loss weights. These are mild because your test set is balanced.
    w[LABEL_MAP["Neutral"]] *= 0.92
    w[LABEL_MAP["Happy"]] *= 1.12
    w[LABEL_MAP["Angry"]] *= 1.05
    w[LABEL_MAP["Sad"]] *= 1.05
    w[LABEL_MAP["Fear"]] *= 1.08
    w[LABEL_MAP["Disgust"]] *= 1.18
    w[LABEL_MAP["Boredum"]] *= 1.20

    w = w / w.mean()

    return torch.tensor(w, dtype=torch.float32, device=cfg.DEVICE)

class MultiPoolXLMR(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL)
        hidden = self.encoder.config.hidden_size

        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self.norm = nn.LayerNorm(hidden * 4)

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=cfg.USE_LAST4_HIDDEN,
            return_dict=True,
        )

        if cfg.USE_LAST4_HIDDEN and outputs.hidden_states is not None and len(outputs.hidden_states) >= 4:
            # Averaging the last four transformer layers usually gives a more stable
            # sentence representation for emotion classes that differ subtly.
            last_hidden = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
        else:
            last_hidden = outputs.last_hidden_state

        cls_pool = last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()

        mean_pool = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        masked_hidden = last_hidden.masked_fill(mask == 0, -1e4)
        max_pool = masked_hidden.max(dim=1).values

        attn_scores = self.attn(last_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_pool = torch.sum(last_hidden * attn_weights, dim=1)

        fused = torch.cat(
            [cls_pool, mean_pool, max_pool, attn_pool],
            dim=1,
        )

        fused = self.norm(fused)
        logits = self.classifier(fused)

        return logits

def build_model():
    return MultiPoolXLMR()

def build_opt(model):
    enc = []
    clf = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if n.startswith("classifier") or n.startswith("norm") or n.startswith("attn"):
            clf.append(p)
        else:
            enc.append(p)

    return torch.optim.AdamW(
        [
            {"params": enc, "lr": cfg.LR_ENCODER},
            {"params": clf, "lr": cfg.LR_CLASSIFIER},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )

def evaluate(model, loader, probs=False):
    model.eval()

    ys = []
    ps = []
    idx = []
    pr = []

    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(cfg.DEVICE)
            mask = b["attention_mask"].to(cfg.DEVICE)
            y = b["labels"].to(cfg.DEVICE)

            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP and "cuda" in cfg.DEVICE):
                logits = model(input_ids=ids, attention_mask=mask)

            p = torch.softmax(logits, 1)
            pred = p.argmax(1)

            ys += y.cpu().tolist()
            ps += pred.cpu().tolist()
            idx += b["index"].cpu().tolist()

            if probs:
                pr += p.cpu().numpy().tolist()

    m = {
        "accuracy": accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, ps, average="weighted", zero_division=0),
    }

    return (m, ys, ps, idx, np.array(pr)) if probs else (m, ys, ps, idx)

def save_preds(df, ys, ps, pr, outdir, prefix="test"):
    out = df.copy().reset_index(drop=True)

    out["true_id"] = ys
    out["pred_id"] = ps
    out["true_label"] = [ID2LABEL[i] for i in ys]
    out["pred_label"] = [ID2LABEL[i] for i in ps]
    out["correct"] = out.true_id == out.pred_id
    out["confidence"] = pr.max(1)

    for i, label in ID2LABEL.items():
        out[f"prob_{label}"] = pr[:, i]

    out.to_csv(
        outdir / f"{prefix}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    out[~out.correct].to_csv(
        outdir / f"{prefix}_misclassified_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        confusion_matrix(ys, ps, labels=list(range(N_CLASSES))),
        index=LABELS,
        columns=LABELS,
    ).to_csv(
        outdir / f"{prefix}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    with open(outdir / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(
            classification_report(
                ys,
                ps,
                labels=list(range(N_CLASSES)),
                target_names=LABELS,
                zero_division=0,
            )
        )

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Focal loss gives more weight to hard examples. It is useful here because
    7-class SER has many close emotion pairs.
    """
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.03):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()

def symmetric_kl_loss(logits_a, logits_b):
    """Symmetric KL used by R-Drop regularization."""
    log_prob_a = F.log_softmax(logits_a, dim=-1)
    log_prob_b = F.log_softmax(logits_b, dim=-1)
    prob_a = torch.softmax(logits_a, dim=-1)
    prob_b = torch.softmax(logits_b, dim=-1)

    kl_ab = F.kl_div(log_prob_a, prob_b, reduction="batchmean")
    kl_ba = F.kl_div(log_prob_b, prob_a, reduction="batchmean")

    return 0.5 * (kl_ab + kl_ba)

def train():
    seed_everything(cfg.SEED)

    outdir = Path(cfg.SAVE_DIR) / f"seed_{cfg.SEED}"
    outdir.mkdir(parents=True, exist_ok=True)

    tb_dir = Path(cfg.TENSORBOARD_DIR) / f"seed_{cfg.SEED}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    print("TensorBoard log directory:", tb_dir)

    with open(Path(cfg.SAVE_DIR) / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_df, val_df, test_df = load_splits()

    train_df.to_csv(outdir / "train_true_context.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(outdir / "val_true_context.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(outdir / "test_true_context.csv", index=False, encoding="utf-8-sig")

    train_df.to_csv(outdir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(outdir / "val_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(outdir / "test_split.csv", index=False, encoding="utf-8-sig")

    # Save shared splits for the audio branch so text/audio/fusion use exactly the same files.
    shared_dir = Path(cfg.SHARED_SPLIT_DIR)
    shared_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(shared_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(shared_dir / "val_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(shared_dir / "test_split.csv", index=False, encoding="utf-8-sig")
    print("Shared splits saved in:", shared_dir)

    tok = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)

    tr = DataLoader(
        TextDS(train_df, tok),
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler(train_df) if cfg.USE_WEIGHTED_SAMPLER else None,
        shuffle=False if cfg.USE_WEIGHTED_SAMPLER else True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory="cuda" in cfg.DEVICE,
        drop_last=True,
    )

    va = DataLoader(
        TextDS(val_df, tok),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory="cuda" in cfg.DEVICE,
    )

    te = DataLoader(
        TextDS(test_df, tok),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory="cuda" in cfg.DEVICE,
    )

    model = build_model().to(cfg.DEVICE)
    opt = build_opt(model)

    if cfg.USE_FOCAL_LOSS:
        crit = FocalLoss(
            weight=loss_weights(train_df),
            gamma=cfg.FOCAL_GAMMA,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
        print(f"Using FocalLoss: gamma={cfg.FOCAL_GAMMA}, label_smoothing={cfg.LABEL_SMOOTHING}")
    else:
        crit = nn.CrossEntropyLoss(
            weight=loss_weights(train_df),
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
        print(f"Using CrossEntropyLoss: label_smoothing={cfg.LABEL_SMOOTHING}")

    if cfg.USE_RDROP:
        print(f"Using R-Drop regularization: alpha={cfg.RDROP_ALPHA}")

    steps = max(1, (len(tr) // cfg.GRAD_ACCUM_STEPS) * cfg.EPOCHS)
    sched = get_cosine_schedule_with_warmup(
        opt,
        int(steps * cfg.WARMUP_RATIO),
        steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and "cuda" in cfg.DEVICE)

    best = -1
    bad = 0
    best_path = outdir / "best_model.pth"
    hist = []

    print("\nSTARTING ACCURACY-BOOSTED TRUE-CONTEXT TEXT XLM-R TRAINING - 7 CLASS")

    for ep in range(1, cfg.EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        loss_sum = 0.0
        corr = 0
        total = 0

        for step, b in enumerate(tqdm(tr, desc=f"Epoch {ep}/{cfg.EPOCHS}"), start=1):
            ids = b["input_ids"].to(cfg.DEVICE)
            mask = b["attention_mask"].to(cfg.DEVICE)
            y = b["labels"].to(cfg.DEVICE)

            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP and "cuda" in cfg.DEVICE):
                if cfg.USE_RDROP:
                    logits1 = model(input_ids=ids, attention_mask=mask)
                    logits2 = model(input_ids=ids, attention_mask=mask)
                    ce_loss = 0.5 * (crit(logits1, y) + crit(logits2, y))
                    kl_loss = symmetric_kl_loss(logits1, logits2)
                    loss = (ce_loss + cfg.RDROP_ALPHA * kl_loss) / cfg.GRAD_ACCUM_STEPS
                    logits = 0.5 * (logits1 + logits2)
                else:
                    logits = model(input_ids=ids, attention_mask=mask)
                    loss = crit(logits, y) / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % cfg.GRAD_ACCUM_STEPS == 0 or step == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

            loss_sum += loss.item() * cfg.GRAD_ACCUM_STEPS
            pred = logits.detach().argmax(1)
            corr += (pred == y).sum().item()
            total += y.size(0)

        val_m, _, _, _ = evaluate(model, va)

        score = 0.7 * val_m["weighted_f1"] + 0.3 * val_m["macro_f1"]

        row = {
            "epoch": ep,
            "train_loss": loss_sum / max(len(tr), 1),
            "train_acc": corr / max(total, 1),
            **{f"val_{k}": v for k, v in val_m.items()},
            "score": score,
        }

        hist.append(row)
        pd.DataFrame(hist).to_csv(outdir / "training_history.csv", index=False)

        print(
            f"\nEpoch {ep:02d}: "
            f"loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"val_acc={val_m['accuracy']:.4f} "
            f"macro_f1={val_m['macro_f1']:.4f} "
            f"weighted_f1={val_m['weighted_f1']:.4f}"
        )

        # TensorBoard logging.
        writer.add_scalar("Loss/train", row["train_loss"], ep)
        writer.add_scalar("Accuracy/train", row["train_acc"], ep)
        writer.add_scalar("Accuracy/val", val_m["accuracy"], ep)
        writer.add_scalar("F1_macro/val", val_m["macro_f1"], ep)
        writer.add_scalar("F1_weighted/val", val_m["weighted_f1"], ep)
        writer.add_scalar("Score/val", score, ep)
        writer.add_scalar("LearningRate/encoder", opt.param_groups[0]["lr"], ep)
        writer.add_scalar("LearningRate/classifier", opt.param_groups[1]["lr"], ep)

        if score > best:
            best = score
            bad = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "val_metrics": val_m,
                    "label_map": LABEL_MAP,
                    "id2label": ID2LABEL,
                    "labels": LABELS,
                },
                best_path,
            )

            print("Saved best model:", best_path)

        else:
            bad += 1
            print(f"No improvement. Patience: {bad}/{cfg.PATIENCE}")

            if bad >= cfg.PATIENCE:
                print("Early stopping.")
                break

    print("\nLoading best model for final test...")

    ckpt = torch.load(best_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # Save validation predictions for fusion validation.
    val_m, val_ys, val_ps, val_idx, val_pr = evaluate(model, va, probs=True)

    with open(outdir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_m, f, indent=2)

    save_preds(
        val_df.iloc[val_idx].reset_index(drop=True),
        val_ys,
        val_ps,
        val_pr,
        outdir,
        prefix="val",
    )

    print("\nValidation predictions saved:", outdir / "val_predictions.csv")

    # Test evaluation.
    test_m, ys, ps, idx, pr = evaluate(model, te, probs=True)

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION - ACCURACY-BOOSTED TRUE CONTEXT TEXT XLM-R - 7 CLASS")
    print("=" * 80)
    print(json.dumps(test_m, indent=2))
    print(
        classification_report(
            ys,
            ps,
            labels=list(range(N_CLASSES)),
            target_names=LABELS,
            zero_division=0,
        )
    )
    print(confusion_matrix(ys, ps, labels=list(range(N_CLASSES))))

    writer.add_scalar("FinalTest/accuracy", test_m["accuracy"], 0)
    writer.add_scalar("FinalTest/macro_f1", test_m["macro_f1"], 0)
    writer.add_scalar("FinalTest/weighted_f1", test_m["weighted_f1"], 0)

    with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_m, f, indent=2)

    save_preds(
        test_df.iloc[idx].reset_index(drop=True),
        ys,
        ps,
        pr,
        outdir,
        prefix="test",
    )

    writer.flush()
    writer.close()

    print("\nSaved results in:", outdir)

if __name__ == "__main__":
    train()
