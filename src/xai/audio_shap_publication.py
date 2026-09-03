#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-quality Audio SHAP analysis
for V24 acoustic ensemble.

Generates:
1. Global acoustic SHAP importance
2. Sample-level SHAP explanation
3. Acoustic feature-group contribution

Model:
ExtraTrees acoustic branch

Author:
UrduSER XAI Analysis
"""

from pathlib import Path
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

warnings.filterwarnings("ignore")


# ==========================================================
# PATHS
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ROOT = REPO_ROOT

BASE_DIR = ROOT / "outputs" / "xai" / "audio"
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(
    os.environ.get(
        "EXTRA_TREES_MODEL_PATH",
        str(BASE_DIR / "extra_trees.joblib")
    )
)
FEATURE_PATH = BASE_DIR / "acoustic_features_sr16000_mfcc20.npz"

OUTPUT = BASE_DIR
OUTPUT.mkdir(parents=True, exist_ok=True)


LABELS = [
    "Neutral",
    "Happy",
    "Angry",
    "Sad",
    "Fear",
    "Disgust",
    "Boredum"
]


# ==========================================================
# FEATURE NAMES
# ==========================================================

def create_feature_names(n_features):

    names = []

    groups = [
        "duration",
        "rms",
        "zcr",
        "spectral",
        "mfcc",
        "delta_mfcc",
        "chroma",
        "pitch",
        "prosody"
    ]

    for i in range(n_features):
        names.append(
            f"acoustic_feature_{i+1}"
        )

    return names



# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading acoustic model...")

model = joblib.load(
    MODEL_PATH
)


print("Loading acoustic features...")

data = np.load(
    FEATURE_PATH,
    allow_pickle=True
)


X = data["features"]


if "feature_names" in data:
    feature_names = list(data["feature_names"])
else:
    feature_names = create_feature_names(
        X.shape[1]
    )


print(
    "Feature matrix:",
    X.shape
)



# ==========================================================
# SHAP CALCULATION
# ==========================================================

print("Computing SHAP values...")

explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    X
)


# multiclass handling

if isinstance(shap_values, list):

    shap_matrix = np.mean(
        np.abs(
            np.stack(shap_values)
        ),
        axis=0
    )

else:

    shap_matrix = np.abs(
        shap_values
    )


global_importance = (
    shap_matrix.mean(axis=0)
)



# ==========================================================
# FIGURE 1
# GLOBAL SHAP IMPORTANCE
# ==========================================================


idx = np.argsort(
    global_importance
)[-20:]


plt.figure(
    figsize=(8,6)
)


plt.barh(
    np.array(feature_names)[idx],
    global_importance[idx]
)


plt.xlabel(
    "Mean absolute SHAP value"
)

plt.ylabel(
    "Acoustic features"
)


plt.title(
    "Global SHAP Feature Importance of Acoustic Branch"
)


plt.tight_layout()


plt.savefig(
    OUTPUT /
    "audio_global_shap_importance.pdf",
    bbox_inches="tight"
)


plt.savefig(
    OUTPUT /
    "audio_global_shap_importance.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# ==========================================================
# SELECT BEST SAMPLE
# ==========================================================


pred = model.predict(X)


target = None


# preference order

for label in [
    "Fear",
    "Angry",
    "Disgust"
]:

    if label in LABELS:

        idx_label = LABELS.index(label)

        candidates = np.where(
            pred == idx_label
        )[0]

        if len(candidates):

            target = candidates[0]
            break


if target is None:

    target = 0



print(
    "Selected sample:",
    target,
    "Prediction:",
    LABELS[pred[target]]
)



# ==========================================================
# FIGURE 2
# SAMPLE SHAP WATERFALL
# ==========================================================


sample_values = shap_matrix[target]


top = np.argsort(
    np.abs(sample_values)
)[-15:]


plt.figure(
    figsize=(8,6)
)


plt.barh(
    np.array(feature_names)[top],
    sample_values[top]
)


plt.axvline(
    0
)


plt.xlabel(
    "SHAP contribution"
)


plt.ylabel(
    "Acoustic features"
)


plt.title(
    f"Sample-level SHAP Explanation: {LABELS[pred[target]]}"
)


plt.tight_layout()


plt.savefig(
    OUTPUT /
    "audio_sample_shap_explanation.pdf",
    bbox_inches="tight"
)


plt.savefig(
    OUTPUT /
    "audio_sample_shap_explanation.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# ==========================================================
# FIGURE 3
# FEATURE GROUP ANALYSIS
# ==========================================================


groups = {

    "MFCC":
    ["mfcc"],

    "Spectral":
    ["spectral"],

    "Energy":
    ["rms"],

    "Pitch":
    ["pitch"],

    "Prosody":
    ["duration","zcr"],

    "Chroma":
    ["chroma"]

}


group_scores = {}


for g, keys in groups.items():

    values = []

    for i,n in enumerate(feature_names):

        if any(
            k.lower() in n.lower()
            for k in keys
        ):

            values.append(
                global_importance[i]
            )


    if values:

        group_scores[g] = np.mean(values)



plt.figure(
    figsize=(7,4)
)


plt.bar(
    list(group_scores.keys()),
    list(group_scores.values())
)


plt.ylabel(
    "Mean SHAP importance"
)


plt.xlabel(
    "Acoustic feature groups"
)


plt.title(
    "Contribution of Acoustic Feature Groups"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


plt.savefig(
    OUTPUT /
    "audio_feature_group_shap.pdf",
    bbox_inches="tight"
)


plt.savefig(
    OUTPUT /
    "audio_feature_group_shap.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



print("\nCompleted.")
print(
    "Figures saved in:",
    OUTPUT
)


