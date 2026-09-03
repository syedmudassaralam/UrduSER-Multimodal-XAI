"""
======================================================================
FINAL PUBLICATION FUSION-XAI — V1.1
UrduSER V24 Decision-Level Probability Trace
======================================================================

FINAL VISUAL CLEANUP ONLY

Changes from previous version:
    1. Fixes Panel-B title / ground-truth overlap
    2. Separates confidence and margin onto two lines
    3. Replaces "Damaged" with publication-friendly "New errors"
    4. Shortens global-change description
    5. Keeps Panel-C probability trace unchanged
    6. Does NOT change fusion methodology or probabilities

Inputs:
    fusion_explanation_summary.json
    fusion_sample_explanations.csv
    test_change_analysis.csv

Representative sample:
    8_1_1_48

Outputs:
    fusion_xai_publication_FINAL_v1_1.png
    fusion_xai_publication_FINAL_v1_1.pdf
    fusion_xai_sample_probabilities.csv
    fusion_xai_global_summary.csv
"""

# ======================================================================
# IMPORTS
# ======================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch


# ======================================================================
# PATHS
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

FUSION_XAI_DIR = (
    REPO_ROOT
    / "outputs"
    / "xai"
    / "fusion"
)

SUMMARY_JSON = (
    FUSION_XAI_DIR
    / "fusion_explanation_summary.json"
)

SAMPLE_CSV = (
    FUSION_XAI_DIR
    / "fusion_sample_explanations.csv"
)

CHANGE_CSV = (
    FUSION_XAI_DIR
    / "test_change_analysis.csv"
)

OUTPUT_DIR = (
    FUSION_XAI_DIR
    / "publication_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PNG_OUT = (
    OUTPUT_DIR
    / "fusion_xai_publication_FINAL_v1_1.png"
)

PDF_OUT = (
    OUTPUT_DIR
    / "fusion_xai_publication_FINAL_v1_1.pdf"
)

SAMPLE_PROB_OUT = (
    OUTPUT_DIR
    / "fusion_xai_sample_probabilities.csv"
)

GLOBAL_SUMMARY_OUT = (
    OUTPUT_DIR
    / "fusion_xai_global_summary.csv"
)


# ======================================================================
# REPRESENTATIVE SAMPLE
# ======================================================================

SAMPLE_ID = "8_1_1_48"


# ======================================================================
# CLASS ORDER
# ======================================================================

CLASSES = [
    "Neutral",
    "Happy",
    "Angry",
    "Sad",
    "Fear",
    "Disgust",
    "Boredom",
]


# ======================================================================
# PUBLICATION SETTINGS
# ======================================================================

DPI = 600

FIG_WIDTH = 7.2
FIG_HEIGHT = 4.45


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ======================================================================
# COLORS
# ======================================================================

DARK = "#202124"
GREY = "#5F6368"
LIGHT_GREY = "#E5E7EB"

PANEL_BG = "#FAFAFA"

PROTECTED_COLOR = "#777777"
AUDIO_COLOR = "#3274A1"
FINAL_COLOR = "#2E8B57"

WRONG_COLOR = "#B23A48"
CORRECT_COLOR = "#2E8B57"

GRID_COLOR = "#D9DDE3"


# ======================================================================
# HELPERS
# ======================================================================

def bool_series(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
            "y",
        ])
    )


def add_panel_background(ax):

    patch = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=PANEL_BG,
        edgecolor=LIGHT_GREY,
        linewidth=0.9,
        clip_on=False,
        zorder=-10,
    )

    ax.add_patch(patch)


def get_stage_probabilities(row, stage):

    values = []

    for emotion in CLASSES:

        column = f"{stage}_prob_{emotion}"

        if column not in row.index:
            raise KeyError(
                f"Missing column: {column}"
            )

        values.append(
            float(row[column])
        )

    return np.asarray(
        values,
        dtype=float
    )


# ======================================================================
# IMPROVED DECISION-STAGE CARD
# ======================================================================

def draw_stage_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    prediction,
    confidence,
    margin,
    edge_color,
    correct=None,
):

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="white",
        edgecolor=edge_color,
        linewidth=1.45,
    )

    ax.add_patch(box)

    # ----------------------------------------------------------
    # Stage title
    # ----------------------------------------------------------

    ax.text(
        x + width / 2,
        y + height * 0.80,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.3,
        color=GREY,
        fontweight="bold",
    )

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    label = prediction

    if correct is True:
        label += "  ✓"

    elif correct is False:
        label += "  ✗"

    ax.text(
        x + width / 2,
        y + height * 0.55,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.3,
        color=edge_color,
        fontweight="bold",
    )

    # ----------------------------------------------------------
    # Probability
    # ----------------------------------------------------------

    ax.text(
        x + width / 2,
        y + height * 0.31,
        f"p = {confidence:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=GREY,
    )

    # ----------------------------------------------------------
    # Margin
    # ----------------------------------------------------------

    ax.text(
        x + width / 2,
        y + height * 0.15,
        f"margin = {margin:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        color=GREY,
    )


# ======================================================================
# LOAD DATA
# ======================================================================

def load_data():

    for path in [
        SUMMARY_JSON,
        SAMPLE_CSV,
        CHANGE_CSV,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}\n"
            )

    with open(
        SUMMARY_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        summary = json.load(file)

    samples = pd.read_csv(
        SAMPLE_CSV
    )

    changes = pd.read_csv(
        CHANGE_CSV
    )

    return (
        summary,
        samples,
        changes
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 72)
    print("UrduSER V24 — FINAL FUSION XAI V1.1")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    summary, df, changes = load_data()

    # ------------------------------------------------------------------
    # Boolean columns
    # ------------------------------------------------------------------

    protected_correct = bool_series(
        df["protected_correct"]
    )

    final_correct = bool_series(
        df["final_correct"]
    )

    changed = bool_series(
        df["changed"]
    )

    corrected = bool_series(
        df["corrected"]
    )

    damaged = bool_series(
        df["damaged"]
    )

    triggered = bool_series(
        df["selective_rule_triggered"]
    )

    # ------------------------------------------------------------------
    # Global results
    # ------------------------------------------------------------------

    n_samples = len(df)

    protected_correct_count = int(
        protected_correct.sum()
    )

    final_correct_count = int(
        final_correct.sum()
    )

    changed_count = int(
        changed.sum()
    )

    corrected_count = int(
        corrected.sum()
    )

    damaged_count = int(
        damaged.sum()
    )

    triggered_count = int(
        triggered.sum()
    )

    unchanged_count = (
        n_samples
        -
        changed_count
    )

    wrong_to_wrong_count = (
        changed_count
        -
        corrected_count
        -
        damaged_count
    )

    protected_accuracy = (
        100.0
        *
        protected_correct_count
        /
        n_samples
    )

    final_accuracy = (
        100.0
        *
        final_correct_count
        /
        n_samples
    )

    accuracy_gain = (
        final_accuracy
        -
        protected_accuracy
    )

    # ------------------------------------------------------------------
    # Representative sample
    # ------------------------------------------------------------------

    sample_rows = df[
        df["file_id"].astype(str)
        ==
        SAMPLE_ID
    ]

    if len(sample_rows) != 1:

        raise ValueError(
            f"Expected exactly one row for "
            f"{SAMPLE_ID}; found {len(sample_rows)}."
        )

    row = sample_rows.iloc[0]

    true_label = str(
        row["true_label"]
    )

    protected_prediction = str(
        row["protected_prediction"]
    )

    audio_prediction = str(
        row["audio_prediction"]
    )

    final_prediction = str(
        row["final_prediction"]
    )

    protected_confidence = float(
        row["protected_confidence"]
    )

    protected_margin = float(
        row["protected_margin"]
    )

    audio_confidence = float(
        row["audio_confidence"]
    )

    audio_margin = float(
        row["audio_margin"]
    )

    final_confidence = float(
        row["final_confidence"]
    )

    final_margin = float(
        row["final_margin"]
    )

    protected_probs = get_stage_probabilities(
        row,
        "protected"
    )

    audio_probs = get_stage_probabilities(
        row,
        "audio"
    )

    final_probs = get_stage_probabilities(
        row,
        "final"
    )

    sample_triggered = (
        str(
            row["selective_rule_triggered"]
        )
        .strip()
        .lower()
        ==
        "true"
    )

    sample_corrected = (
        str(
            row["corrected"]
        )
        .strip()
        .lower()
        ==
        "true"
    )

    sample_damaged = (
        str(
            row["damaged"]
        )
        .strip()
        .lower()
        ==
        "true"
    )

    # ------------------------------------------------------------------
    # Save probability table
    # ------------------------------------------------------------------

    probability_table = pd.DataFrame({

        "emotion":
            CLASSES,

        "protected_probability":
            protected_probs,

        "acoustic_probability":
            audio_probs,

        "final_probability":
            final_probs,
    })

    probability_table.to_csv(
        SAMPLE_PROB_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------------
    # Save global summary
    # ------------------------------------------------------------------

    global_table = pd.DataFrame(
        [
            {
                "samples":
                    n_samples,

                "protected_correct":
                    protected_correct_count,

                "protected_accuracy_percent":
                    protected_accuracy,

                "final_correct":
                    final_correct_count,

                "final_accuracy_percent":
                    final_accuracy,

                "accuracy_gain_percentage_points":
                    accuracy_gain,

                "changed_predictions":
                    changed_count,

                "unchanged_predictions":
                    unchanged_count,

                "corrected_errors":
                    corrected_count,

                "new_errors":
                    damaged_count,

                "wrong_to_wrong_changes":
                    wrong_to_wrong_count,

                "selective_rule_triggered":
                    triggered_count,
            }
        ]
    )

    global_table.to_csv(
        GLOBAL_SUMMARY_OUT,
        index=False,
        encoding="utf-8-sig"
    )

    # ==================================================================
    # FIGURE
    # ==================================================================

    fig = plt.figure(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,

        width_ratios=[
            0.90,
            1.55
        ],

        height_ratios=[
            0.95,
            1.25
        ],

        left=0.055,
        right=0.985,
        bottom=0.10,
        top=0.88,

        wspace=0.22,
        hspace=0.34,
    )

    ax_global = fig.add_subplot(
        grid[0, 0]
    )

    ax_trace = fig.add_subplot(
        grid[0, 1]
    )

    ax_prob = fig.add_subplot(
        grid[1, :]
    )

    # ==================================================================
    # HEADER
    # ==================================================================

    fig.suptitle(
        "Decision-Level Explainability of Selective Multimodal Fusion",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )

    v21_weight = float(
        summary.get(
            "v21_weight",
            0.55
        )
    )

    v13_weight = float(
        summary.get(
            "v13_weight",
            0.45
        )
    )

    fig.text(
        0.055,
        0.915,
        (
            "Protected probability ensemble "
            f"({v21_weight:.2f} V21 + "
            f"{v13_weight:.2f} V13) with "
            "validation-selected selective acoustic correction"
        ),
        ha="left",
        va="center",
        fontsize=7.7,
        color=GREY,
    )

    # ==================================================================
    # PANEL A — GLOBAL BEHAVIOR
    # ==================================================================

    ax_global.set_axis_off()

    add_panel_background(
        ax_global
    )

    ax_global.text(
        0.05,
        0.89,
        "A   Global selective-fusion behavior",
        transform=ax_global.transAxes,
        ha="left",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=DARK,
    )

    # Protected
    ax_global.text(
        0.08,
        0.68,
        "Protected",
        transform=ax_global.transAxes,
        ha="left",
        fontsize=7.3,
        color=GREY,
        fontweight="bold",
    )

    ax_global.text(
        0.08,
        0.49,
        f"{protected_accuracy:.2f}%",
        transform=ax_global.transAxes,
        ha="left",
        va="center",
        fontsize=16,
        color=PROTECTED_COLOR,
        fontweight="bold",
    )

    ax_global.text(
        0.08,
        0.34,
        f"{protected_correct_count}/{n_samples} correct",
        transform=ax_global.transAxes,
        ha="left",
        fontsize=7,
        color=GREY,
    )

    # Arrow
    ax_global.annotate(
        "",
        xy=(0.72, 0.54),
        xytext=(0.46, 0.54),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
            "color": DARK,
        },
    )

    ax_global.text(
        0.59,
        0.64,
        f"+{accuracy_gain:.2f}\npp",
        transform=ax_global.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=CORRECT_COLOR,
        fontweight="bold",
    )

    # Final
    ax_global.text(
        0.74,
        0.68,
        "Final",
        transform=ax_global.transAxes,
        ha="left",
        fontsize=7.3,
        color=GREY,
        fontweight="bold",
    )

    ax_global.text(
        0.74,
        0.49,
        f"{final_accuracy:.2f}%",
        transform=ax_global.transAxes,
        ha="left",
        va="center",
        fontsize=16,
        color=FINAL_COLOR,
        fontweight="bold",
    )

    ax_global.text(
        0.74,
        0.34,
        f"{final_correct_count}/{n_samples} correct",
        transform=ax_global.transAxes,
        ha="left",
        fontsize=7,
        color=GREY,
    )

    # Compact summary
    ax_global.text(
        0.05,
        0.14,
        (
            f"Changed {changed_count}/{n_samples}"
            f"   •   Corrected {corrected_count}"
            f"   •   New errors {damaged_count}"
        ),
        transform=ax_global.transAxes,
        ha="left",
        va="center",
        fontsize=7.0,
        color=DARK,
        fontweight="bold",
    )

    ax_global.text(
        0.05,
        0.045,
        (
            f"{unchanged_count} unchanged"
            f"   •   "
            f"{wrong_to_wrong_count} incorrect → incorrect"
        ),
        transform=ax_global.transAxes,
        ha="left",
        va="center",
        fontsize=6.5,
        color=GREY,
    )

    # ==================================================================
    # PANEL B — DECISION TRACE
    # ==================================================================

    ax_trace.set_axis_off()

    add_panel_background(
        ax_trace
    )

    # Main title — no ground-truth overlap
    ax_trace.text(
        0.035,
        0.89,
        (
            "B   Representative decision trace "
            f"— {SAMPLE_ID}"
        ),
        transform=ax_trace.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=DARK,
    )

    # Ground truth moved to second line
    ax_trace.text(
        0.965,
        0.76,
        f"Ground truth: {true_label}",
        transform=ax_trace.transAxes,
        ha="right",
        va="center",
        fontsize=7.6,
        color=CORRECT_COLOR,
        fontweight="bold",
    )

    # ----------------------------------------------------------
    # Protected
    # ----------------------------------------------------------

    draw_stage_box(
        ax_trace,

        x=0.035,
        y=0.24,
        width=0.27,
        height=0.40,

        title="Protected",

        prediction=
            protected_prediction,

        confidence=
            protected_confidence,

        margin=
            protected_margin,

        edge_color=
            WRONG_COLOR,

        correct=(
            protected_prediction
            ==
            true_label
        ),
    )

    # Plus
    ax_trace.text(
        0.337,
        0.44,
        "+",
        transform=ax_trace.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        color=GREY,
    )

    # ----------------------------------------------------------
    # Acoustic
    # ----------------------------------------------------------

    draw_stage_box(
        ax_trace,

        x=0.37,
        y=0.24,
        width=0.27,
        height=0.40,

        title="Acoustic evidence",

        prediction=
            audio_prediction,

        confidence=
            audio_confidence,

        margin=
            audio_margin,

        edge_color=
            AUDIO_COLOR,

        correct=(
            audio_prediction
            ==
            true_label
        ),
    )

    # Arrow
    ax_trace.annotate(
        "",
        xy=(0.705, 0.44),
        xytext=(0.655, 0.44),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
            "color": DARK,
        },
    )

    # ----------------------------------------------------------
    # Final
    # ----------------------------------------------------------

    draw_stage_box(
        ax_trace,

        x=0.72,
        y=0.24,
        width=0.245,
        height=0.40,

        title="Final fusion",

        prediction=
            final_prediction,

        confidence=
            final_confidence,

        margin=
            final_margin,

        edge_color=
            FINAL_COLOR,

        correct=(
            final_prediction
            ==
            true_label
        ),
    )

    # ----------------------------------------------------------
    # Trigger explanation
    # ----------------------------------------------------------

    if sample_corrected:

        status_text = (
            "Selective rule triggered  •  "
            "protected error corrected"
        )

        status_color = CORRECT_COLOR

    elif sample_damaged:

        status_text = (
            "Selective rule triggered  •  "
            "new error introduced"
        )

        status_color = WRONG_COLOR

    elif sample_triggered:

        status_text = (
            "Selective rule triggered  •  "
            "prediction remained incorrect"
        )

        status_color = GREY

    else:

        status_text = (
            "Selective rule not triggered"
        )

        status_color = GREY

    ax_trace.text(
        0.50,
        0.09,
        status_text,
        transform=ax_trace.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        color=status_color,
        fontweight="bold",
    )

    # ==================================================================
    # PANEL C — PROBABILITY TRACE
    # ==================================================================

    add_panel_background(
        ax_prob
    )

    x = np.arange(
        len(CLASSES)
    )

    width = 0.23

    bars_protected = ax_prob.bar(
        x - width,
        protected_probs,
        width=width,
        color=PROTECTED_COLOR,
        alpha=0.87,
        label="Protected",
        zorder=3,
    )

    bars_audio = ax_prob.bar(
        x,
        audio_probs,
        width=width,
        color=AUDIO_COLOR,
        alpha=0.90,
        label="Acoustic",
        zorder=3,
    )

    bars_final = ax_prob.bar(
        x + width,
        final_probs,
        width=width,
        color=FINAL_COLOR,
        alpha=0.90,
        label="Final fusion",
        zorder=3,
    )

    ax_prob.set_title(
        "C   Seven-class probability trace for the corrected sample",
        loc="left",
        fontweight="bold",
        pad=7,
    )

    ax_prob.set_ylabel(
        "Class probability"
    )

    ax_prob.set_xticks(
        x
    )

    ax_prob.set_xticklabels(
        CLASSES
    )

    max_probability = max(
        protected_probs.max(),
        audio_probs.max(),
        final_probs.max(),
    )

    y_max = min(
        1.0,
        max(
            0.80,
            np.ceil(
                (
                    max_probability
                    +
                    0.05
                )
                *
                10
            )
            /
            10
        )
    )

    ax_prob.set_ylim(
        0,
        y_max
    )

    ax_prob.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45,
        color=GRID_COLOR,
        zorder=0,
    )

    ax_prob.spines[
        "top"
    ].set_visible(False)

    ax_prob.spines[
        "right"
    ].set_visible(False)

    ax_prob.spines[
        "left"
    ].set_color(LIGHT_GREY)

    ax_prob.spines[
        "bottom"
    ].set_color(LIGHT_GREY)

    ax_prob.legend(
        loc="upper right",
        frameon=False,
        ncol=3,
    )

    # ----------------------------------------------------------
    # Show labels only for meaningful probabilities
    # ----------------------------------------------------------

    for bars in [
        bars_protected,
        bars_audio,
        bars_final
    ]:

        for bar in bars:

            value = bar.get_height()

            if value >= 0.10:

                ax_prob.text(
                    bar.get_x()
                    +
                    bar.get_width()
                    /
                    2,

                    value + 0.012,

                    f"{value:.3f}",

                    ha="center",
                    va="bottom",

                    fontsize=6.1,
                    color=DARK,
                )

    # ----------------------------------------------------------
    # Highlight target / competing class
    # ----------------------------------------------------------

    for tick in ax_prob.get_xticklabels():

        label = tick.get_text()

        if label == true_label:

            tick.set_fontweight(
                "bold"
            )

            tick.set_color(
                FINAL_COLOR
            )

        elif label == protected_prediction:

            tick.set_fontweight(
                "bold"
            )

            tick.set_color(
                WRONG_COLOR
            )

    # ==================================================================
    # SAVE
    # ==================================================================

    fig.savefig(
        PNG_OUT,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )

    fig.savefig(
        PDF_OUT,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )

    plt.close(
        fig
    )

    # ==================================================================
    # TERMINAL OUTPUT
    # ==================================================================

    print("\nGLOBAL FUSION RESULTS")
    print("-" * 72)

    print(
        f"Samples                 : {n_samples}"
    )

    print(
        f"Protected correct       : "
        f"{protected_correct_count}/{n_samples}"
    )

    print(
        f"Protected accuracy      : "
        f"{protected_accuracy:.2f}%"
    )

    print(
        f"Final correct           : "
        f"{final_correct_count}/{n_samples}"
    )

    print(
        f"Final accuracy          : "
        f"{final_accuracy:.2f}%"
    )

    print(
        f"Accuracy gain           : "
        f"{accuracy_gain:+.2f} percentage points"
    )

    print(
        f"Changed predictions     : "
        f"{changed_count}"
    )

    print(
        f"Corrected errors        : "
        f"{corrected_count}"
    )

    print(
        f"New errors              : "
        f"{damaged_count}"
    )

    print(
        f"Wrong -> wrong changes  : "
        f"{wrong_to_wrong_count}"
    )

    print(
        f"Selective rule triggers : "
        f"{triggered_count}"
    )

    print("\nREPRESENTATIVE SAMPLE")
    print("-" * 72)

    print(
        f"File ID                 : "
        f"{SAMPLE_ID}"
    )

    print(
        f"Ground truth            : "
        f"{true_label}"
    )

    print(
        f"Protected               : "
        f"{protected_prediction} "
        f"(p={protected_confidence:.6f}, "
        f"margin={protected_margin:.6f})"
    )

    print(
        f"Acoustic                : "
        f"{audio_prediction} "
        f"(p={audio_confidence:.6f}, "
        f"margin={audio_margin:.6f})"
    )

    print(
        f"Final fusion            : "
        f"{final_prediction} "
        f"(p={final_confidence:.6f}, "
        f"margin={final_margin:.6f})"
    )

    print(
        f"Selective rule triggered: "
        f"{sample_triggered}"
    )

    print(
        f"Corrected               : "
        f"{sample_corrected}"
    )

    print(
        f"New error               : "
        f"{sample_damaged}"
    )

    print("\nSaved:")
    print(PNG_OUT)
    print(PDF_OUT)
    print(SAMPLE_PROB_OUT)
    print(GLOBAL_SUMMARY_OUT)

    print(
        "\nFUSION XAI V1.1 FINAL — DONE"
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()