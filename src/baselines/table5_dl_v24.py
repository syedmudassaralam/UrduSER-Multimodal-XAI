#!/usr/bin/env python

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from comp_dl_v24_reference import (
    AudioSeqDataset,
    Audio1DCNN,
    AudioCNNRNN,
    AudioBiRNNAttention,
    LogMelCNN,
    LogMelCRNN,
    LogMelResNet,
    train_torch_model,
)

SEED = 123
BATCH_SIZE = 16
EPOCHS = 45
PATIENCE = 7
LR = 8e-4
WD = 1e-4

MFCC_CACHE = Path(
    "saved_models_urdu_ser_selected_ml_dl_fusion_benchmark/"
    "cache_mfcc_sequence_features.npz"
)

LOGMEL_CACHE = Path(
    "saved_models_urdu_ser_deep_learning_baselines/"
    "audio_logmel_cache.npz"
)

MANIFEST = Path(
    "github_repo/data/urdu_ser_manifest_v11_portable.csv"
)

OUT = Path(
    "github_repo/outputs/baselines/table5_dl_v24"
)

OUT.mkdir(parents=True, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    print("\nCache:", path)
    print("Keys:", d.files)

    for k in d.files:
        try:
            print(k, d[k].shape)
        except:
            print(k)

    return d


mfcc = load_cache(MFCC_CACHE)
logmel = load_cache(LOGMEL_CACHE)


manifest = pd.read_csv(MANIFEST)
manifest["file_id"] = manifest["file_id"].astype(str)


def align_cache(cache):
    ids = [str(x) for x in cache["file_id"]]
    index = {x:i for i,x in enumerate(ids)}

    order = [
        index[x]
        for x in manifest["file_id"]
    ]

    return cache["X"][order]


# MFCC sequence features
X_mfcc = align_cache(mfcc)

# LogMel features
X_logmel = align_cache(logmel)


LABELS = sorted(manifest["label"].unique())
label_map = {x:i for i,x in enumerate(LABELS)}

manifest["target_id"] = manifest["label"].map(label_map)


train_df = manifest[manifest.split=="train"].reset_index(drop=True)
val_df   = manifest[manifest.split=="val"].reset_index(drop=True)
test_df  = manifest[manifest.split=="test"].reset_index(drop=True)


def split_features(X):

    mask_train = (manifest.split=="train").to_numpy()
    mask_val = (manifest.split=="val").to_numpy()
    mask_test = (manifest.split=="test").to_numpy()

    return (
        X[mask_train],
        X[mask_val],
        X[mask_test],
    )


mfcc_train, mfcc_val, mfcc_test = split_features(X_mfcc)
mel_train, mel_val, mel_test = split_features(X_logmel)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def evaluate(model, loader):

    model.eval()

    ys=[]
    ps=[]

    with torch.no_grad():
        for b in loader:
            x=b["x"].to(device)

            out=model(x)
            pred=out.argmax(1).cpu().numpy()

            ps.extend(pred)
            ys.extend(
                b["y"].numpy()
            )

    acc=accuracy_score(ys,ps)
    f1=f1_score(
        ys,
        ps,
        average="macro"
    )

    return acc,f1,ys,ps


models = [
    (
        "LogMel_CNN",
        lambda: LogMelCNN(),
        mel_train,
        mel_val,
        mel_test
    ),

    (
        "LogMel_CRNN",
        lambda: LogMelCRNN(),
        mel_train,
        mel_val,
        mel_test
    ),

    (
        "LogMel_ResNet",
        lambda: LogMelResNet(),
        mel_train,
        mel_val,
        mel_test
    ),

    (
        "Audio_BiLSTM_Attention",
        lambda: AudioBiRNNAttention(),
        mfcc_train,
        mfcc_val,
        mfcc_test
    ),

    (
        "MFCC_1D_CNN",
        lambda: Audio1DCNN(),
        mfcc_train,
        mfcc_val,
        mfcc_test
    ),

    (
        "MFCC_CNN_LSTM",
        lambda: AudioCNNRNN(
            rnn_type="lstm",
            bidirectional=False
        ),
        mfcc_train,
        mfcc_val,
        mfcc_test
    ),

    (
        "MFCC_CNN_BiLSTM",
        lambda: AudioCNNRNN(
            rnn_type="lstm",
            bidirectional=True
        ),
        mfcc_train,
        mfcc_val,
        mfcc_test
    ),

    (
        "MFCC_CNN_GRU",
        lambda: AudioCNNRNN(
            rnn_type="gru",
            bidirectional=False
        ),
        mfcc_train,
        mfcc_val,
        mfcc_test
    ),

    (
        "MFCC_CNN_BiGRU",
        lambda: AudioCNNRNN(
            rnn_type="gru",
            bidirectional=True
        ),
        mfcc_train,
        mfcc_val,
        mfcc_test
    ),
]


results=[]


for name, maker, Xtr, Xv, Xte in models:

    print("\nTraining:",name)

    train_loader=DataLoader(
        AudioSeqDataset(
            Xtr,
            train_df,
            augment=True
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader=DataLoader(
        AudioSeqDataset(
            Xv,
            val_df,
            augment=False
        ),
        batch_size=BATCH_SIZE
    )

    test_loader=DataLoader(
        AudioSeqDataset(
            Xte,
            test_df,
            augment=False
        ),
        batch_size=BATCH_SIZE
    )


    model=maker().to(device)

    outdir=OUT/name
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )


    train_torch_model(
        model,
        train_loader,
        val_loader,
        outdir,
        EPOCHS,
        PATIENCE,
        LR,
        WD,
        train_df.target_id.values
    )


    acc,f1,y,p=evaluate(
        model,
        test_loader
    )


    result={
        "model":name,
        "accuracy_percent":round(acc*100,2),
        "macro_f1_percent":round(f1*100,2),
        "accuracy":acc,
        "macro_f1":f1,
        "seed":SEED
    }


    results.append(result)


    with open(outdir/"test_metrics.json","w") as f:
        json.dump(
            result,
            f,
            indent=2
        )


summary=pd.DataFrame(results)

summary.to_csv(
    OUT/"table5_dl_summary.csv",
    index=False
)

print("\nFINAL V24 TABLE 5 DL RESULTS")
print(
    summary[
        [
            "model",
            "accuracy_percent",
            "macro_f1_percent"
        ]
    ].to_string(index=False)
)
