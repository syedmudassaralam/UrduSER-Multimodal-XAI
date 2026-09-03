#!/usr/bin/env python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

SEED = 123
N_JOBS = -1

FEATURE_FILE = Path(
    "saved_models_prev_curr_v24_multibase_acoustic_gate_7class/"
    "acoustic_features_sr16000_mfcc20.npz"
)
MANIFEST = Path("github_repo/data/urdu_ser_manifest_v11_portable.csv")
OUT = Path("github_repo/outputs/baselines/table5_ml_v24")

LABELS = ["Neutral","Happy","Angry","Sad","Fear","Disgust","Boredom"]
LABEL2ID = {x:i for i,x in enumerate(LABELS)}

OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load V24 acoustic features
# ------------------------------------------------------------------
d = np.load(FEATURE_FILE, allow_pickle=True)

feat = pd.DataFrame({
    "file_id": [str(x) for x in d["file_ids"]]
})

X = np.asarray(d["features"], dtype=np.float32)

manifest = pd.read_csv(MANIFEST)
manifest["file_id"] = manifest["file_id"].astype(str)

id_to_idx = {fid:i for i,fid in enumerate(feat["file_id"])}

missing = [x for x in manifest["file_id"] if x not in id_to_idx]
if missing:
    raise RuntimeError(f"Missing feature IDs: {len(missing)}")

idx = np.array([id_to_idx[x] for x in manifest["file_id"]])
X = X[idx]

y = manifest["label"].map(LABEL2ID).to_numpy(dtype=int)

train_mask = manifest["split"].eq("train").to_numpy()
val_mask   = manifest["split"].eq("val").to_numpy()
test_mask  = manifest["split"].eq("test").to_numpy()

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

test_ids = manifest.loc[test_mask, "file_id"].tolist()

print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ------------------------------------------------------------------
# Original Table-5 ML model definitions
# ------------------------------------------------------------------
models = {
    "Acoustic KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ))
    ]),

    "Acoustic Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=SEED
            ),
            cv=3
        ))
    ]),

    "Acoustic Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=2.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED
        ))
    ]),

    "Acoustic Extra Trees": Pipeline([
        ("clf", ExtraTreesClassifier(
            n_estimators=700,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=N_JOBS
        ))
    ]),

    "Acoustic Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=700,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=N_JOBS
        ))
    ]),

    "Acoustic SVM-RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED
        ))
    ]),

    "Acoustic XGBoost": Pipeline([
        ("clf", XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=N_JOBS
        ))
    ])
}

rows = []

for name, model in models.items():
    print("\nTraining:", name)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")

    result = {
        "model": name,
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "accuracy_percent": round(acc * 100, 2),
        "macro_f1_percent": round(mf1 * 100, 2),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_dimension": int(X_train.shape[1]),
        "seed": SEED
    }

    rows.append(result)

    model_dir = OUT / name.replace(" ","_")
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "test_metrics.json","w") as f:
        json.dump(result,f,indent=2)

    pd.DataFrame({
        "file_id": test_ids,
        "true_id": y_test,
        "pred_id": pred
    }).to_csv(model_dir / "test_predictions.csv",index=False)

    print(
        f"{name}: "
        f"Accuracy={acc*100:.2f}% | "
        f"Macro-F1={mf1*100:.2f}%"
    )

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "table5_ml_summary.csv",index=False)

print("\nFINAL V24 TABLE 5 ML RESULTS")
print(
    summary[
        ["model","accuracy_percent","macro_f1_percent"]
    ].to_string(index=False)
)
