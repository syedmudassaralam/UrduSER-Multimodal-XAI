#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithfulness verification for UrduSER five-seed text XAI.

The test masks the highest positive-attribution token positions and compares the
drop in the originally predicted-class probability against equally sized random
token masks. Models are loaded one at a time to keep memory use low.

Default verified samples:
    row 328 -> 64-step XAI tokens
    row 191 -> 64-step XAI tokens
    row 196 -> 128-step XAI tokens

Example:
    python verify_text_xai_deletion.py \
      --project-root . \
      --device cpu \
      --ks 3,5 \
      --random-repeats 20 \
      --output-dir outputs/xai/text_deletion_verification
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


LABELS = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Boredum"]
N_CLASSES = 7

MODEL_SNAPSHOT = (
    "xlm-roberta-base/snapshots/"
    "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
)

CHECKPOINTS = [
    "saved_models_text_shared_prev_curr_7class/seed_123/best_model.pth",
    "saved_models_text_shared_prev_curr_seed42_7class/seed_42/best_model.pth",
    "saved_models_text_shared_prev_curr_seed777_7class/seed_777/best_model.pth",
    "saved_models_text_shared_prev_curr_seed2024_7class/seed_2024/best_model.pth",
    "saved_models_text_shared_prev_curr_seed2025_7class/seed_2025/best_model.pth",
]

TEST_CSV = (
    "saved_models_text_shared_prev_curr_7class/"
    "seed_123/test_true_context.csv"
)

SUMMARY_CSV = "UrduSER_XAI_Experiment/text/text_xai_summary.csv"

DEFAULT_XAI_SOURCES = {
    328: "text_xai_verification_steps64/text_xai_tokens.csv",
    191: "text_xai_verification_steps64/text_xai_tokens.csv",
    196: "text_xai_verification_row196_steps128/text_xai_tokens.csv",
}

# Pieces produced when XLM-R tokenizes the manually inserted [PREV]/[CURR]
# delimiters. These positions are excluded from lexical deletion tests.
MARKER_PIECES = {
    "[", "]", "PRE", "CURR",
    "P", "R", "E", "V", "C", "U",
    "▁[", "▁]",
}


@dataclass(frozen=True)
class CFG:
    model_source: str
    max_text_len: int = 320
    use_last4_hidden: bool = True


class MultiPoolXLMR(nn.Module):
    """Exact UrduSER text architecture used by the five-seed ensemble."""

    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_source)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.cfg.use_last4_hidden,
            return_dict=True,
        )

        if (
            self.cfg.use_last4_hidden
            and outputs.hidden_states is not None
            and len(outputs.hidden_states) >= 4
        ):
            last_hidden = torch.stack(
                outputs.hidden_states[-4:], dim=0
            ).mean(dim=0)
        else:
            last_hidden = outputs.last_hidden_state

        cls_pool = last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (
            (last_hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1e-9)
        )

        max_pool = (
            last_hidden.masked_fill(mask == 0, -1e4)
            .max(dim=1)
            .values
        )

        attn_scores = self.attn(last_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(
            attention_mask == 0, -1e4
        )
        attn_weights = torch.softmax(
            attn_scores, dim=1
        ).unsqueeze(-1)
        attn_pool = torch.sum(
            last_hidden * attn_weights, dim=1
        )

        fused = self.norm(
            torch.cat(
                [cls_pool, mean_pool, max_pool, attn_pool],
                dim=1,
            )
        )
        return self.classifier(fused)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    for candidate in (path, root / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Path not found: {value}")


def resolve_model_source(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    candidates = [
        path,
        root / path,
        Path.cwd() / path,
        Path.home()
        / ".cache/huggingface/hub/"
        / "models--xlm-roberta-base/snapshots/"
        / "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate.resolve())
    return value


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def extract_state_dict(
    checkpoint: Any,
) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    else:
        if checkpoint and all(
            isinstance(k, str) and torch.is_tensor(v)
            for k, v in checkpoint.items()
        ):
            state = checkpoint
        else:
            raise KeyError(
                "No model_state_dict/state_dict/model key found."
            )

    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}

    return state


def load_model(
    checkpoint_path: Path,
    cfg: CFG,
    device: torch.device,
) -> MultiPoolXLMR:
    model = MultiPoolXLMR(cfg)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    model.load_state_dict(
        extract_state_dict(checkpoint),
        strict=True,
    )
    model.to(device).eval()
    return model


def unload_model(
    model: MultiPoolXLMR,
    device: torch.device,
) -> None:
    model.to("cpu")
    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_int_list(value: str) -> List[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one integer is required.")
    return list(dict.fromkeys(values))


def parse_xai_sources(
    root: Path,
    values: Sequence[str] | None,
) -> Dict[int, Path]:
    if not values:
        return {
            row: resolve_path(root, path)
            for row, path in DEFAULT_XAI_SOURCES.items()
        }

    result: Dict[int, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                "--xai-source must use ROW=PATH, for example "
                "196=/path/to/text_xai_tokens.csv"
            )
        row_text, path_text = item.split("=", 1)
        result[int(row_text.strip())] = resolve_path(
            root, path_text.strip()
        )
    return result


def lexical_candidates(
    token_df: pd.DataFrame,
    row_index: int,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> pd.DataFrame:
    rows = token_df[
        token_df["row_index"].astype(int) == int(row_index)
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No XAI token rows found for row_index={row_index}"
        )

    if "is_special" in rows.columns:
        rows = rows[
            ~rows["is_special"].astype(bool)
        ].copy()

    rows["clean_token"] = (
        rows["display_token"].astype(str).str.strip()
    )
    rows = rows[
        ~rows["clean_token"].isin(MARKER_PIECES)
    ].copy()

    special_ids = set(tokenizer.all_special_ids)
    valid_positions: List[bool] = []

    for position in rows["position"].astype(int):
        valid = (
            0 <= position < input_ids.shape[0]
            and int(attention_mask[position]) == 1
            and int(input_ids[position]) not in special_ids
        )
        valid_positions.append(valid)

    rows = rows[np.asarray(valid_positions, dtype=bool)].copy()

    if rows.empty:
        raise ValueError(
            f"No eligible lexical positions for row_index={row_index}"
        )

    # Verify that stored token IDs still match fresh tokenization.
    mismatches = []
    for _, row in rows.iterrows():
        position = int(row["position"])
        stored_id = int(row["token_id"])
        current_id = int(input_ids[position])
        if stored_id != current_id:
            mismatches.append(
                (position, stored_id, current_id)
            )

    if mismatches:
        preview = mismatches[:5]
        raise RuntimeError(
            f"Token-position alignment failed for row {row_index}. "
            f"First mismatches: {preview}"
        )

    return rows.sort_values("position").reset_index(drop=True)


def choose_top_supporting_positions(
    candidates: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    # Positive attribution supports the explained/predicted class.
    supporting = candidates[
        candidates["attribution"].astype(float) > 0
    ].copy()

    if len(supporting) < k:
        warnings.warn(
            f"Only {len(supporting)} positive tokens are available "
            f"for requested k={k}; using the strongest {min(k, len(candidates))} "
            "absolute-attribution lexical tokens instead."
        )
        selected = candidates.nlargest(
            min(k, len(candidates)),
            "abs_attribution",
        )
    else:
        selected = supporting.nlargest(k, "attribution")

    return selected.sort_values(
        "attribution", ascending=False
    ).copy()


def masked_copy(
    original_ids: torch.Tensor,
    positions: Sequence[int],
    replacement_id: int,
) -> torch.Tensor:
    output = original_ids.clone()
    for position in positions:
        output[int(position)] = int(replacement_id)
    return output


def predict_variants_one_model(
    model: MultiPoolXLMR,
    variant_ids: torch.Tensor,
    variant_masks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    all_probs: List[np.ndarray] = []

    for start in range(0, len(variant_ids), batch_size):
        end = min(start + batch_size, len(variant_ids))

        ids = variant_ids[start:end].to(device)
        masks = variant_masks[start:end].to(device)

        with torch.inference_mode():
            probs = torch.softmax(
                model(ids, masks), dim=-1
            )

        all_probs.append(probs.cpu().numpy())

        del ids, masks, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_probs, axis=0)


def build_sample_variants(
    row_index: int,
    text: str,
    tokenizer: Any,
    cfg: CFG,
    xai_tokens: pd.DataFrame,
    ks: Sequence[int],
    repeats: int,
    seed: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    List[Dict[str, Any]],
    Dict[int, pd.DataFrame],
]:
    encoded = tokenizer(
        text,
        max_length=cfg.max_text_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    original_ids = encoded["input_ids"][0].cpu()
    original_mask = encoded["attention_mask"][0].cpu()

    replacement_id = tokenizer.mask_token_id
    if replacement_id is None:
        replacement_id = tokenizer.unk_token_id
    if replacement_id is None:
        raise RuntimeError(
            "Tokenizer has neither mask_token_id nor unk_token_id."
        )

    candidates = lexical_candidates(
        token_df=xai_tokens,
        row_index=row_index,
        tokenizer=tokenizer,
        input_ids=original_ids,
        attention_mask=original_mask,
    )

    rng = np.random.default_rng(seed + int(row_index))

    variant_ids: List[torch.Tensor] = [original_ids]
    metadata: List[Dict[str, Any]] = [
        {
            "variant_type": "original",
            "k": 0,
            "repeat": -1,
            "positions": [],
        }
    ]
    selected_by_k: Dict[int, pd.DataFrame] = {}

    candidate_positions = candidates["position"].astype(int).to_numpy()

    for k in ks:
        effective_k = min(int(k), len(candidates))
        if effective_k < 1:
            raise ValueError(
                f"No positions available for k={k}, row={row_index}"
            )

        top_rows = choose_top_supporting_positions(
            candidates,
            effective_k,
        )
        top_positions = top_rows["position"].astype(int).tolist()
        selected_by_k[int(k)] = top_rows

        variant_ids.append(
            masked_copy(
                original_ids,
                top_positions,
                replacement_id,
            )
        )
        metadata.append(
            {
                "variant_type": "top",
                "k": int(k),
                "effective_k": len(top_positions),
                "repeat": -1,
                "positions": top_positions,
            }
        )

        # Exclude top positions from the random pool so the baseline does
        # not accidentally mask the same highly ranked evidence.
        random_pool = np.asarray(
            [
                p for p in candidate_positions
                if int(p) not in set(top_positions)
            ],
            dtype=int,
        )

        if len(random_pool) < len(top_positions):
            # Fall back to all lexical positions only when necessary.
            random_pool = candidate_positions

        for repeat in range(repeats):
            chosen = rng.choice(
                random_pool,
                size=len(top_positions),
                replace=False,
            )
            chosen_positions = sorted(
                int(x) for x in chosen.tolist()
            )

            variant_ids.append(
                masked_copy(
                    original_ids,
                    chosen_positions,
                    replacement_id,
                )
            )
            metadata.append(
                {
                    "variant_type": "random",
                    "k": int(k),
                    "effective_k": len(chosen_positions),
                    "repeat": int(repeat),
                    "positions": chosen_positions,
                }
            )

    ids_tensor = torch.stack(variant_ids, dim=0)
    masks_tensor = original_mask.unsqueeze(0).repeat(
        len(variant_ids), 1
    )

    return (
        ids_tensor,
        masks_tensor,
        metadata,
        selected_by_k,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify UrduSER text-XAI faithfulness using "
            "top-token deletion versus random deletion."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--test-csv",
        default=TEST_CSV,
    )
    parser.add_argument(
        "--summary-csv",
        default=SUMMARY_CSV,
    )
    parser.add_argument(
        "--model-source",
        default=MODEL_SNAPSHOT,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoint_list",
        help="Repeat exactly five times to override defaults.",
    )
    parser.add_argument(
        "--rows",
        default="328,191,196",
        help="Comma-separated zero-based test row indices.",
    )
    parser.add_argument(
        "--ks",
        default="3,5",
        help="Comma-separated numbers of tokens to mask.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--xai-source",
        action="append",
        help=(
            "Optional repeated ROW=PATH mapping. "
            "Defaults are configured for rows 328, 191 and 196."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "xai"
            / "text"
            / "deletion_verification"
        ),
    )
    args = parser.parse_args()

    if args.random_repeats < 1:
        raise ValueError("--random-repeats must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    set_seed(args.seed)

    root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    model_source = resolve_model_source(
        root, args.model_source
    )
    cfg = CFG(model_source=model_source)

    rows = parse_int_list(args.rows)
    ks = sorted(parse_int_list(args.ks))

    test_csv = resolve_path(root, args.test_csv)
    summary_csv = resolve_path(root, args.summary_csv)
    xai_sources = parse_xai_sources(
        root, args.xai_source
    )

    missing_source_rows = [
        row for row in rows if row not in xai_sources
    ]
    if missing_source_rows:
        raise ValueError(
            "Missing --xai-source mapping for rows: "
            f"{missing_source_rows}"
        )

    checkpoint_values = (
        args.checkpoint_list or CHECKPOINTS
    )
    if len(checkpoint_values) != 5:
        raise ValueError(
            "Exactly five checkpoints are required; "
            f"received {len(checkpoint_values)}."
        )
    checkpoints = [
        resolve_path(root, value)
        for value in checkpoint_values
    ]

    print("=" * 78)
    print("UrduSER TEXT-XAI DELETION FAITHFULNESS TEST")
    print("=" * 78)
    print("Device:", device)
    print("Rows:", rows)
    print("k values:", ks)
    print("Random repetitions:", args.random_repeats)
    print("Replacement token: XLM-R mask token")
    print("Output:", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source
    )

    test_df = pd.read_csv(test_csv).reset_index(drop=True)
    summary_df = pd.read_csv(summary_csv)

    required_test_columns = {
        "true_context_text",
        "file_id",
        "target_id",
    }
    missing = required_test_columns - set(test_df.columns)
    if missing:
        raise KeyError(
            f"Test CSV missing columns: {sorted(missing)}"
        )

    required_summary_columns = {
        "row_index",
        "file_id",
        "predicted_id",
        "predicted_label",
        "confidence",
    }
    missing = required_summary_columns - set(
        summary_df.columns
    )
    if missing:
        raise KeyError(
            f"Summary CSV missing columns: {sorted(missing)}"
        )

    token_tables: Dict[Path, pd.DataFrame] = {}
    for path in set(xai_sources.values()):
        token_tables[path] = pd.read_csv(path)

    sample_data: Dict[int, Dict[str, Any]] = {}

    for row_index in rows:
        if row_index < 0 or row_index >= len(test_df):
            raise IndexError(
                f"Invalid test row index: {row_index}"
            )

        summary_rows = summary_df[
            summary_df["row_index"].astype(int)
            == int(row_index)
        ]
        if len(summary_rows) != 1:
            raise ValueError(
                f"Expected one summary row for {row_index}; "
                f"found {len(summary_rows)}."
            )

        summary_row = summary_rows.iloc[0]
        test_row = test_df.iloc[row_index]

        if str(summary_row["file_id"]) != str(
            test_row["file_id"]
        ):
            raise RuntimeError(
                f"file_id mismatch for row {row_index}: "
                f"{summary_row['file_id']} != "
                f"{test_row['file_id']}"
            )

        token_path = xai_sources[row_index]
        token_df = token_tables[token_path]

        (
            variant_ids,
            variant_masks,
            metadata,
            selected_by_k,
        ) = build_sample_variants(
            row_index=row_index,
            text=str(test_row["true_context_text"]),
            tokenizer=tokenizer,
            cfg=cfg,
            xai_tokens=token_df,
            ks=ks,
            repeats=args.random_repeats,
            seed=args.seed,
        )

        sample_data[row_index] = {
            "file_id": str(test_row["file_id"]),
            "true_id": int(test_row["target_id"]),
            "true_label": LABELS[int(test_row["target_id"])],
            "target_id": int(summary_row["predicted_id"]),
            "target_label": str(
                summary_row["predicted_label"]
            ),
            "stored_original_confidence": float(
                summary_row["confidence"]
            ),
            "variant_ids": variant_ids,
            "variant_masks": variant_masks,
            "metadata": metadata,
            "selected_by_k": selected_by_k,
            "probability_sum": np.zeros(
                (len(metadata), N_CLASSES),
                dtype=np.float64,
            ),
        }

    # Low-memory inference: one checkpoint at a time.
    for model_number, checkpoint in enumerate(
        checkpoints, start=1
    ):
        print(
            f"\nModel {model_number}/5: {checkpoint}"
        )
        model = load_model(
            checkpoint,
            cfg,
            device,
        )

        for row_index in rows:
            item = sample_data[row_index]
            probabilities = predict_variants_one_model(
                model=model,
                variant_ids=item["variant_ids"],
                variant_masks=item["variant_masks"],
                device=device,
                batch_size=args.batch_size,
            )
            item["probability_sum"] += probabilities
            print(
                f"  row {row_index}: "
                f"{len(item['metadata'])} variants"
            )

        unload_model(model, device)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for row_index in rows:
        item = sample_data[row_index]
        ensemble_probs = (
            item["probability_sum"] / len(checkpoints)
        )
        metadata = item["metadata"]
        target_id = int(item["target_id"])

        original_prob = float(
            ensemble_probs[0, target_id]
        )
        original_pred_id = int(
            np.argmax(ensemble_probs[0])
        )
        original_pred_label = LABELS[original_pred_id]

        confidence_difference = abs(
            original_prob
            - item["stored_original_confidence"]
        )

        if original_pred_id != target_id:
            raise RuntimeError(
                f"Fresh prediction mismatch for row {row_index}: "
                f"summary target={item['target_label']}, "
                f"fresh prediction={original_pred_label}."
            )

        for variant_index, (meta, probs) in enumerate(
            zip(metadata, ensemble_probs)
        ):
            variant_target_prob = float(probs[target_id])
            variant_pred_id = int(np.argmax(probs))

            detail_rows.append(
                {
                    "row_index": row_index,
                    "file_id": item["file_id"],
                    "true_label": item["true_label"],
                    "target_id": target_id,
                    "target_label": item["target_label"],
                    "variant_index": variant_index,
                    "variant_type": meta["variant_type"],
                    "k": int(meta.get("k", 0)),
                    "repeat": int(meta.get("repeat", -1)),
                    "masked_positions": json.dumps(
                        meta.get("positions", [])
                    ),
                    "target_probability": variant_target_prob,
                    "probability_drop": (
                        original_prob - variant_target_prob
                    ),
                    "predicted_id": variant_pred_id,
                    "predicted_label": LABELS[
                        variant_pred_id
                    ],
                    "prediction_changed": (
                        variant_pred_id != original_pred_id
                    ),
                }
            )

        details = pd.DataFrame(
            [
                row for row in detail_rows
                if row["row_index"] == row_index
            ]
        )

        for k in ks:
            top_row = details[
                (details["variant_type"] == "top")
                & (details["k"] == k)
            ].iloc[0]

            random_rows = details[
                (details["variant_type"] == "random")
                & (details["k"] == k)
            ].copy()

            top_drop = float(top_row["probability_drop"])
            random_drops = random_rows[
                "probability_drop"
            ].to_numpy(dtype=float)
            random_mean = float(np.mean(random_drops))
            random_std = float(
                np.std(random_drops, ddof=1)
                if len(random_drops) > 1
                else 0.0
            )

            advantage = top_drop - random_mean

            ratio = (
                top_drop / random_mean
                if random_mean > 1e-12
                else np.nan
            )

            # One-sided empirical randomisation p-value:
            # fraction of random drops at least as large as top drop.
            empirical_p = float(
                (
                    1
                    + np.sum(random_drops >= top_drop)
                )
                / (len(random_drops) + 1)
            )

            selected = item["selected_by_k"][k]
            top_positions = (
                selected["position"]
                .astype(int)
                .tolist()
            )
            top_tokens = (
                selected["display_token"]
                .astype(str)
                .tolist()
            )
            top_attributions = (
                selected["attribution"]
                .astype(float)
                .tolist()
            )

            summary_rows.append(
                {
                    "row_index": row_index,
                    "file_id": item["file_id"],
                    "true_label": item["true_label"],
                    "target_id": target_id,
                    "target_label": item["target_label"],
                    "k_requested": int(k),
                    "k_effective": len(top_positions),
                    "original_prediction": original_pred_label,
                    "original_target_probability": original_prob,
                    "stored_original_confidence": item[
                        "stored_original_confidence"
                    ],
                    "original_confidence_abs_difference": (
                        confidence_difference
                    ),
                    "top_masked_target_probability": float(
                        top_row["target_probability"]
                    ),
                    "top_probability_drop": top_drop,
                    "top_prediction_after_masking": str(
                        top_row["predicted_label"]
                    ),
                    "top_prediction_changed": bool(
                        top_row["prediction_changed"]
                    ),
                    "random_probability_drop_mean": random_mean,
                    "random_probability_drop_std": random_std,
                    "random_probability_drop_min": float(
                        np.min(random_drops)
                    ),
                    "random_probability_drop_max": float(
                        np.max(random_drops)
                    ),
                    "faithfulness_advantage": advantage,
                    "faithfulness_ratio": ratio,
                    "empirical_p_value": empirical_p,
                    "faithfulness_pass": bool(
                        top_drop > random_mean
                    ),
                    "top_positions": json.dumps(
                        top_positions
                    ),
                    "top_tokens": json.dumps(
                        top_tokens,
                        ensure_ascii=False,
                    ),
                    "top_attributions": json.dumps(
                        top_attributions
                    ),
                    "xai_token_source": str(
                        xai_sources[row_index]
                    ),
                }
            )

    summary_output = pd.DataFrame(summary_rows)
    details_output = pd.DataFrame(detail_rows)

    summary_path = (
        output_dir
        / "text_xai_deletion_summary.csv"
    )
    details_path = (
        output_dir
        / "text_xai_deletion_random_details.csv"
    )
    config_path = output_dir / "run_config.json"

    summary_output.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    details_output.to_csv(
        details_path,
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "project_root": str(root),
        "test_csv": str(test_csv),
        "summary_csv": str(summary_csv),
        "model_source": model_source,
        "checkpoints": [str(x) for x in checkpoints],
        "rows": rows,
        "ks": ks,
        "random_repeats": args.random_repeats,
        "seed": args.seed,
        "device": str(device),
        "replacement": (
            tokenizer.mask_token
            if tokenizer.mask_token_id is not None
            else tokenizer.unk_token
        ),
        "xai_sources": {
            str(k): str(v)
            for k, v in xai_sources.items()
        },
    }
    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FAITHFULNESS RESULTS")
    print("=" * 78)

    display_columns = [
        "row_index",
        "target_label",
        "k_requested",
        "original_target_probability",
        "top_probability_drop",
        "random_probability_drop_mean",
        "faithfulness_advantage",
        "faithfulness_ratio",
        "empirical_p_value",
        "top_prediction_changed",
        "faithfulness_pass",
    ]
    print(
        summary_output[display_columns].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", details_path)
    print(" ", config_path)

    failures = int(
        (~summary_output["faithfulness_pass"]).sum()
    )
    print(
        f"\nPassed comparisons: "
        f"{len(summary_output) - failures}/"
        f"{len(summary_output)}"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise