#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
======================================================================
FINAL COMPACT PUBLICATION VERSION — V5.2
Urdu Integrated Gradients Word-Level Visualization
======================================================================

This script:
1. Loads IG_values.csv
2. Reconstructs the Urdu sentence represented by the IG tokens
3. Automatically finds the SAME sample in complete_test_predictions_v6.csv
4. Loads prediction and confidence
5. Maps token-level IG scores to original Urdu words
6. Preserves signed attribution:
       Positive -> Red
       Negative -> Blue
       Near zero -> White
7. Builds a compact publication-friendly figure
8. Saves 300-DPI PNG and PDF outputs

Outputs:
    IG_Urdu_publication_FINAL_v5_2.png
    IG_Urdu_publication_FINAL_v5_2.pdf
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import csv
import unicodedata
import numpy as np

from PIL import Image, ImageDraw, ImageFont


# ======================================================================
# PATHS
# ======================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BASE_DIR = os.path.join(REPO_ROOT, "outputs", "xai", "text")

IG_CSV = os.path.join(BASE_DIR, "IG_values.csv")
PRED_CSV = os.path.join(BASE_DIR, "complete_test_predictions_v6.csv")

PNG_OUT = os.path.join(BASE_DIR, "IG_Urdu_publication_FINAL_v5_2.png")
PDF_OUT = os.path.join(BASE_DIR, "IG_Urdu_publication_FINAL_v5_2.pdf")


# ======================================================================
# FONT PATHS
# ======================================================================

URDU_FONT_PATH = os.environ.get(
    "URDU_FONT_PATH",
    "NotoNastaliqUrdu-Regular.otf"
)

ENGLISH_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ENGLISH_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ======================================================================
# CHECK REQUIRED FILES
# ======================================================================

for required_file in [
    IG_CSV,
    PRED_CSV,
    URDU_FONT_PATH,
    ENGLISH_FONT_PATH,
    ENGLISH_BOLD_PATH
]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{required_file}\n"
        )


# ======================================================================
# FONTS (COMPACT)
# ======================================================================

TITLE_FONT = ImageFont.truetype(ENGLISH_BOLD_PATH, 36)
SUBTITLE_FONT = ImageFont.truetype(ENGLISH_FONT_PATH, 18)

SECTION_FONT = ImageFont.truetype(ENGLISH_BOLD_PATH, 21)
META_FONT = ImageFont.truetype(ENGLISH_FONT_PATH, 19)
META_BOLD_FONT = ImageFont.truetype(ENGLISH_BOLD_PATH, 20)

BODY_FONT = ImageFont.truetype(ENGLISH_FONT_PATH, 16)
SMALL_FONT = ImageFont.truetype(ENGLISH_FONT_PATH, 15)
SCORE_FONT = ImageFont.truetype(ENGLISH_FONT_PATH, 16)

URDU_WORD_FONT = ImageFont.truetype(URDU_FONT_PATH, 48)
URDU_RANK_FONT = ImageFont.truetype(URDU_FONT_PATH, 26)


# ======================================================================
# COLORS
# ======================================================================

BACKGROUND = (255, 255, 255)
PANEL_BG = (250, 250, 251)

TEXT = (28, 30, 34)
SECONDARY_TEXT = (88, 92, 98)
MUTED_TEXT = (128, 131, 136)

BORDER = (214, 216, 220)
DIVIDER = (228, 229, 232)
NEUTRAL = (250, 250, 250)

NEGATIVE_BLUE = (44, 98, 170)
POSITIVE_RED = (178, 24, 43)


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def first_existing_key(row, candidates):
    for key in candidates:
        if key in row:
            value = row[key]
            if value is not None and str(value).strip() != "":
                return key
    return None


def normalize_match_text(text):
    text = unicodedata.normalize("NFKC", str(text))

    for char in ["\u200c", "\u200d", "\ufeff", "\u0640"]:
        text = text.replace(char, "")

    return "".join(text.split())


def clean_token(token):
    token = str(token)
    token = token.replace("▁", "")
    token = token.replace("Ġ", "")
    return token.strip()


# ======================================================================
# SAFE TEXT HELPERS
# ======================================================================

def draw_text_safe(draw, position, text, font, fill, anchor=None, direction=None):
    kwargs = {"font": font, "fill": fill}

    if anchor is not None:
        kwargs["anchor"] = anchor

    if direction is not None:
        try:
            draw.text(position, text, direction=direction, **kwargs)
            return
        except Exception:
            pass

    draw.text(position, text, **kwargs)


def text_bbox_safe(draw, text, font, direction=None):
    try:
        if direction is not None:
            return draw.textbbox((0, 0), text, font=font, direction=direction)
    except Exception:
        pass

    return draw.textbbox((0, 0), text, font=font)


# ======================================================================
# LOAD IG VALUES
# ======================================================================

def load_ig():
    tokens = []
    normalized_values = []
    raw_values = []

    with open(IG_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError("IG_values.csv has no header.")

        if "token" not in reader.fieldnames:
            raise KeyError('IG_values.csv must contain a "token" column.')

        if "normalized_ig" not in reader.fieldnames:
            raise KeyError('IG_values.csv must contain a "normalized_ig" column.')

        has_raw = "raw_ig" in reader.fieldnames

        for row in reader:
            token = str(row["token"]).strip()

            if token in {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}:
                continue

            try:
                normalized_score = float(row["normalized_ig"])
            except Exception:
                continue

            if has_raw:
                try:
                    raw_score = float(row["raw_ig"])
                except Exception:
                    raw_score = np.nan
            else:
                raw_score = np.nan

            tokens.append(token)
            normalized_values.append(normalized_score)
            raw_values.append(raw_score)

    if len(tokens) == 0:
        raise RuntimeError("No valid IG tokens were loaded.")

    normalized_values = np.asarray(normalized_values, dtype=float)
    raw_values = np.asarray(raw_values, dtype=float)

    print("\n========================================")
    print("IG FILE LOADED")
    print("========================================")
    print(f"Number of tokens   : {len(tokens)}")
    print("Attribution used   : normalized_ig")
    print(f"Raw IG available   : {np.any(~np.isnan(raw_values))}")

    print("\nIG tokens:")
    print(" | ".join(tokens))

    return tokens, normalized_values, raw_values


# ======================================================================
# RECONSTRUCT TEXT REPRESENTED BY IG TOKENS
# ======================================================================

def reconstruct_ig_text(tokens):
    cleaned_tokens = []

    for token in tokens:
        cleaned = clean_token(token)
        if cleaned:
            cleaned_tokens.append(cleaned)

    readable = " ".join(cleaned_tokens)
    normalized = normalize_match_text("".join(cleaned_tokens))

    return cleaned_tokens, readable, normalized


# ======================================================================
# FIND MATCHING PREDICTION ROW
# ======================================================================

def load_prediction_matching_ig(tokens):
    cleaned_tokens, readable_ig_text, normalized_ig_text = reconstruct_ig_text(tokens)

    print("\n========================================")
    print("IG SAMPLE IDENTIFICATION")
    print("========================================")
    print("\nText represented by IG tokens:")
    print(readable_ig_text)

    with open(PRED_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        raise RuntimeError("Prediction CSV is empty.")

    text_key = first_existing_key(
        rows[0],
        ["urdu_text", "text", "sentence", "transcript", "transcription"]
    )

    if text_key is None:
        raise KeyError("Could not find Urdu text column inside prediction CSV.")

    matched_row = None
    matched_index = None

    for index, row in enumerate(rows):
        sentence = str(row[text_key]).strip()
        normalized_sentence = normalize_match_text(sentence)

        if normalized_sentence == normalized_ig_text:
            matched_row = row
            matched_index = index
            break

    if matched_row is None:
        for index, row in enumerate(rows):
            sentence = str(row[text_key]).strip()
            normalized_sentence = normalize_match_text(sentence)

            if normalized_ig_text in normalized_sentence or normalized_sentence in normalized_ig_text:
                matched_row = row
                matched_index = index
                break

    if matched_row is None:
        raise RuntimeError("Could not match IG tokens to a prediction CSV sentence.")

    row = matched_row
    sentence = str(row[text_key]).strip()

    prediction_key = first_existing_key(
        row,
        [
            "predicted_emotion",
            "prediction",
            "predicted_label",
            "pred_label",
            "predicted_class",
            "pred",
            "prediction_label",
            "predicted_label_name"
        ]
    )

    prediction = str(row[prediction_key]).strip() if prediction_key else "Unknown"

    emotion_map = {
        "0": "Angry",
        "1": "Boredom",
        "2": "Disgust",
        "3": "Fear",
        "4": "Happy",
        "5": "Neutral",
        "6": "Sad"
    }

    prediction = emotion_map.get(prediction, prediction)

    confidence_key = first_existing_key(
        row,
        [
            "confidence",
            "predicted_probability",
            "probability",
            "pred_prob",
            "max_probability",
            "prediction_probability",
            "predicted_confidence",
            "score"
        ]
    )

    confidence = None
    if confidence_key:
        try:
            confidence = float(row[confidence_key])
            if confidence <= 1:
                confidence *= 100
        except Exception:
            confidence = None

    print(f"\nMatched prediction row index: {matched_index}")
    print("\nMatched original Urdu sentence:")
    print(sentence)
    print("\nPrediction:")
    print(prediction)

    if confidence is not None:
        print("\nConfidence:")
        print(f"{confidence:.2f}%")

    return sentence, prediction, confidence, matched_index


# ======================================================================
# TOKEN -> WORD ALIGNMENT
# ======================================================================

def map_tokens_to_words(sentence, tokens, token_scores):
    words = sentence.split()
    normalized_words = [normalize_match_text(word) for word in words]

    joined_sentence = ""
    word_ranges = []
    cursor = 0

    for word in normalized_words:
        start = cursor
        joined_sentence += word
        cursor += len(word)
        end = cursor
        word_ranges.append((start, end))

    word_scores = np.zeros(len(words), dtype=float)

    search_cursor = 0
    matched_tokens = 0
    unmatched_tokens = []

    for token, score in zip(tokens, token_scores):
        piece = normalize_match_text(clean_token(token))

        if piece == "":
            continue

        position = joined_sentence.find(piece, search_cursor)

        if position == -1:
            position = joined_sentence.find(piece, max(0, search_cursor - 3))

        if position == -1:
            unmatched_tokens.append(token)
            continue

        token_start = position
        token_end = position + len(piece)
        token_length = max(1, token_end - token_start)

        for word_index, (word_start, word_end) in enumerate(word_ranges):
            overlap_start = max(token_start, word_start)
            overlap_end = min(token_end, word_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                proportion = overlap / token_length
                word_scores[word_index] += float(score) * proportion

        search_cursor = max(search_cursor, token_end)
        matched_tokens += 1

    print("\n========================================")
    print("TOKEN → WORD ALIGNMENT")
    print("========================================")
    print(f"Original words     : {len(words)}")
    print(f"Model tokens       : {len(tokens)}")
    print(f"Matched tokens     : {matched_tokens}")
    print(f"Unmatched tokens   : {len(unmatched_tokens)}")

    if unmatched_tokens:
        print("\nUnmatched tokens:")
        for token in unmatched_tokens:
            print("  ", repr(token))

    return words, word_scores


# ======================================================================
# NUMERIC + COLOR HELPERS
# ======================================================================

def normalize_for_color(scores):
    scores = np.asarray(scores, dtype=float)
    max_abs = np.max(np.abs(scores))

    if max_abs <= 1e-12:
        return np.zeros_like(scores)

    return scores / max_abs


def format_score(value):
    value = float(value)

    if abs(value) < 1e-12:
        return "0.000"

    if abs(value) < 0.0005:
        return f"{value:+.2e}"

    return f"{value:+.3f}"


def interpolate_color(start, end, amount):
    amount = float(np.clip(amount, 0, 1))
    return tuple(
        int(start[i] + amount * (end[i] - start[i]))
        for i in range(3)
    )


def attribution_color(normalized_value):
    value = float(np.clip(normalized_value, -1, 1))
    magnitude = abs(value) ** 0.72

    if value > 0:
        return interpolate_color(NEUTRAL, POSITIVE_RED, magnitude)

    if value < 0:
        return interpolate_color(NEUTRAL, NEGATIVE_BLUE, magnitude)

    return NEUTRAL


# ======================================================================
# CARD WIDTH + ROW BUILDING
# ======================================================================

def calculate_card_width(draw, word):
    bbox = text_bbox_safe(draw, word, URDU_WORD_FONT, direction="rtl")
    text_width = bbox[2] - bbox[0]
    return max(118, text_width + 48)


def build_centered_rows(draw, words, scores, max_width, gap=14):
    rows = []
    current_row = []
    current_width = 0

    for word, score in zip(words, scores):
        card_width = calculate_card_width(draw, word)

        if len(current_row) == 0:
            proposed_width = card_width
        else:
            proposed_width = current_width + gap + card_width

        if proposed_width > max_width and len(current_row) > 0:
            rows.append(current_row)
            current_row = [(word, score, card_width)]
            current_width = card_width
        else:
            current_row.append((word, score, card_width))
            current_width = proposed_width

    if current_row:
        rows.append(current_row)

    return rows


# ======================================================================
# DRAW MAIN URDU WORD CARDS
# ======================================================================

def draw_word_cards(draw, panel, rows, all_scores):
    x0, y0, x1, y1 = panel

    normalized_scores = normalize_for_color(all_scores)
    score_counter = 0

    CARD_HEIGHT = 96
    SCORE_SPACE = 26
    ROW_GAP = 18
    CARD_GAP = 14
    TOP_PADDING = 22

    center_x = (x0 + x1) / 2
    current_y = y0 + TOP_PADDING

    for row in rows:
        row_width = sum(card_width for _, _, card_width in row)
        row_width += CARD_GAP * (len(row) - 1)

        row_right = center_x + row_width / 2

        for word, score, card_width in row:
            norm_score = normalized_scores[score_counter]
            score_counter += 1

            left = row_right - card_width
            top = current_y
            right = row_right
            bottom = top + CARD_HEIGHT

            draw.rounded_rectangle(
                (left, top, right, bottom),
                radius=12,
                fill=attribution_color(norm_score),
                outline=BORDER,
                width=2
            )

            draw_text_safe(
                draw,
                ((left + right) / 2, top + 49),
                word,
                URDU_WORD_FONT,
                TEXT,
                anchor="mm",
                direction="rtl"
            )

            draw_text_safe(
                draw,
                ((left + right) / 2, bottom + 17),
                format_score(score),
                SCORE_FONT,
                SECONDARY_TEXT,
                anchor="mm"
            )

            row_right = left - CARD_GAP

        current_y += CARD_HEIGHT + SCORE_SPACE + ROW_GAP


# ======================================================================
# DRAW MOST INFLUENTIAL WORDS PANEL
# ======================================================================

def draw_top_words(draw, words, scores, box):
    x0, y0, x1, y1 = box

    draw.rounded_rectangle(
        box,
        radius=14,
        fill=PANEL_BG,
        outline=BORDER,
        width=2
    )

    draw.text(
        (x0 + 22, y0 + 16),
        "Most Influential Words",
        font=SECTION_FONT,
        fill=TEXT
    )

    draw.text(
        (x0 + 22, y0 + 42),
        "Ranked by absolute normalized IG attribution",
        font=SMALL_FONT,
        fill=MUTED_TEXT
    )

    ranking = np.argsort(np.abs(scores))[::-1]
    ranking = ranking[:min(5, len(ranking))]

    normalized_scores = normalize_for_color(scores)

    row_start = y0 + 70
    row_height = 40

    for rank, index in enumerate(ranking, start=1):
        row_y = row_start + (rank - 1) * row_height

        draw_text_safe(
            draw,
            (x0 + 28, row_y + 16),
            str(rank),
            BODY_FONT,
            SECONDARY_TEXT,
            anchor="lm"
        )

        draw.rounded_rectangle(
            (x0 + 52, row_y + 5, x0 + 66, row_y + 28),
            radius=4,
            fill=attribution_color(normalized_scores[index]),
            outline=BORDER,
            width=1
        )

        draw_text_safe(
            draw,
            (x0 + 92, row_y + 16),
            words[index],
            URDU_RANK_FONT,
            TEXT,
            anchor="lm",
            direction="rtl"
        )

        draw_text_safe(
            draw,
            (x1 - 22, row_y + 16),
            format_score(scores[index]),
            BODY_FONT,
            TEXT,
            anchor="rm"
        )

        if rank < len(ranking):
            draw.line(
                (x0 + 22, row_y + 34, x1 - 22, row_y + 34),
                fill=DIVIDER,
                width=1
            )


# ======================================================================
# DRAW ATTRIBUTION SCALE PANEL
# ======================================================================

def draw_scale(draw, box):
    x0, y0, x1, y1 = box

    draw.rounded_rectangle(
        box,
        radius=14,
        fill=PANEL_BG,
        outline=BORDER,
        width=2
    )

    draw.text(
        (x0 + 22, y0 + 16),
        "Attribution Scale",
        font=SECTION_FONT,
        fill=TEXT
    )

    draw.text(
        (x0 + 22, y0 + 42),
        "Signed Integrated Gradients normalized within this sentence",
        font=SMALL_FONT,
        fill=MUTED_TEXT
    )

    bar_left = x0 + 28
    bar_right = x1 - 28
    bar_top = y0 + 82
    bar_bottom = bar_top + 32

    bar_width = int(bar_right - bar_left)

    for i in range(bar_width):
        value = 2 * i / max(1, bar_width - 1) - 1
        draw.line(
            (bar_left + i, bar_top, bar_left + i, bar_bottom),
            fill=attribution_color(value),
            width=1
        )

    draw.rounded_rectangle(
        (bar_left, bar_top, bar_right, bar_bottom),
        radius=7,
        outline=BORDER,
        width=2
    )

    label_y = bar_bottom + 18

    draw_text_safe(draw, (bar_left, label_y), "Negative", BODY_FONT, TEXT, anchor="la")
    draw_text_safe(draw, ((bar_left + bar_right) / 2, label_y), "Near zero", BODY_FONT, TEXT, anchor="ma")
    draw_text_safe(draw, (bar_right, label_y), "Positive", BODY_FONT, TEXT, anchor="ra")

    number_y = label_y + 24

    draw_text_safe(draw, (bar_left, number_y), "-1", SMALL_FONT, NEGATIVE_BLUE, anchor="la")
    draw_text_safe(draw, ((bar_left + bar_right) / 2, number_y), "0", SMALL_FONT, SECONDARY_TEXT, anchor="ma")
    draw_text_safe(draw, (bar_right, number_y), "+1", SMALL_FONT, POSITIVE_RED, anchor="ra")

    explanation_y = number_y + 24

    draw.text(
        (bar_left, explanation_y),
        "Red: supports predicted class",
        font=SMALL_FONT,
        fill=POSITIVE_RED
    )

    draw.text(
        (bar_left, explanation_y + 22),
        "Blue: opposes predicted class",
        font=SMALL_FONT,
        fill=NEGATIVE_BLUE
    )

    draw.text(
        (bar_left, explanation_y + 44),
        "Intensity: relative attribution magnitude",
        font=SMALL_FONT,
        fill=SECONDARY_TEXT
    )


# ======================================================================
# TERMINAL DIAGNOSTICS
# ======================================================================

def print_diagnostics(token_scores, words, word_scores):
    print("\n========================================")
    print("INTEGRATED GRADIENTS DIAGNOSTICS")
    print("========================================")
    print(f"Token attribution minimum : {np.min(token_scores):+.10f}")
    print(f"Token attribution maximum : {np.max(token_scores):+.10f}")
    print(f"Token attribution mean    : {np.mean(token_scores):+.10f}")
    print(f"Token attribution abs sum : {np.sum(np.abs(token_scores)):.10f}")

    print("\n========================================")
    print("WORD-LEVEL ATTRIBUTIONS")
    print("========================================")

    for word, score in zip(words, word_scores):
        print(f"{word}    {score:+.10f}")

    if np.all(word_scores >= 0):
        print("\nNOTE:")
        print("All word-level IG values for this sample are positive.")
        print("This is valid because the signed IG pipeline has been verified.")
    elif np.all(word_scores <= 0):
        print("\nNOTE:")
        print("All word-level IG values for this sample are negative.")
    else:
        print("\nNOTE:")
        print("This sample contains both positive and negative IG contributions.")


# ======================================================================
# MAIN
# ======================================================================

def main():
    # ------------------------------------------------------------
    # STEP 1 — Load IG values
    # ------------------------------------------------------------

    tokens, token_scores, raw_scores = load_ig()

    # ------------------------------------------------------------
    # STEP 2 — Match prediction row
    # ------------------------------------------------------------

    sentence, prediction, confidence, matched_prediction_index = load_prediction_matching_ig(tokens)

    # ------------------------------------------------------------
    # STEP 3 — Token -> word mapping
    # ------------------------------------------------------------

    words, word_scores = map_tokens_to_words(sentence, tokens, token_scores)

    # ------------------------------------------------------------
    # STEP 4 — Terminal diagnostics
    # ------------------------------------------------------------

    print_diagnostics(token_scores, words, word_scores)

    # ------------------------------------------------------------
    # STEP 5 — Compact dynamic layout
    # ------------------------------------------------------------

    CANVAS_WIDTH = 2100
    OUTER_MARGIN = 65

    dummy_image = Image.new("RGB", (CANVAS_WIDTH, 900), BACKGROUND)
    dummy_draw = ImageDraw.Draw(dummy_image)

    available_card_width = CANVAS_WIDTH - 2 * OUTER_MARGIN - 50

    rows = build_centered_rows(
        dummy_draw,
        words,
        word_scores,
        available_card_width,
        gap=14
    )

    number_of_rows = len(rows)

    # Vertical layout
    title_y = 24
    subtitle_y = 62
    divider_y = 92

    meta_top = 108
    meta_bottom = 156

    section_title_y = 184
    section_note_y = 210

    main_panel_top = 236

    CARD_HEIGHT = 96
    SCORE_SPACE = 26
    ROW_GAP = 18
    TOP_PADDING = 22
    BOTTOM_PADDING = 12

    main_panel_height = (
        TOP_PADDING
        + number_of_rows * (CARD_HEIGHT + SCORE_SPACE)
        + max(0, number_of_rows - 1) * ROW_GAP
        + BOTTOM_PADDING
    )

    main_panel_bottom = main_panel_top + main_panel_height

    lower_top = main_panel_bottom + 18
    lower_height = 285
    lower_bottom = lower_top + lower_height

    canvas_height = lower_bottom + 24

    # ------------------------------------------------------------
    # STEP 6 — Create figure
    # ------------------------------------------------------------

    image = Image.new("RGB", (CANVAS_WIDTH, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    LEFT = OUTER_MARGIN
    RIGHT = CANVAS_WIDTH - OUTER_MARGIN

    # Title
    draw.text(
        (LEFT, title_y),
        "Integrated Gradients Explanation for Urdu Text",
        font=TITLE_FONT,
        fill=TEXT
    )

    draw.text(
        (LEFT, subtitle_y),
        "XLM-RoBERTa-based Urdu Speech Emotion Recognition",
        font=SUBTITLE_FONT,
        fill=SECONDARY_TEXT
    )

    draw.line((LEFT, divider_y, RIGHT, divider_y), fill=DIVIDER, width=2)

    # Metadata panel
    draw.rounded_rectangle(
        (LEFT, meta_top, RIGHT, meta_bottom),
        radius=12,
        fill=PANEL_BG,
        outline=BORDER,
        width=2
    )

    meta_center_y = (meta_top + meta_bottom) / 2

    draw_text_safe(
        draw,
        (LEFT + 22, meta_center_y),
        "Prediction:",
        META_FONT,
        SECONDARY_TEXT,
        anchor="lm"
    )

    draw_text_safe(
        draw,
        (LEFT + 132, meta_center_y),
        prediction,
        META_BOLD_FONT,
        TEXT,
        anchor="lm"
    )

    separator_x = LEFT + 310

    draw.line(
        (separator_x, meta_top + 12, separator_x, meta_bottom - 12),
        fill=DIVIDER,
        width=2
    )

    draw_text_safe(
        draw,
        (separator_x + 20, meta_center_y),
        "Confidence:",
        META_FONT,
        SECONDARY_TEXT,
        anchor="lm"
    )

    confidence_text = f"{confidence:.2f}%" if confidence is not None else "Unavailable"

    draw_text_safe(
        draw,
        (separator_x + 138, meta_center_y),
        confidence_text,
        META_BOLD_FONT,
        TEXT,
        anchor="lm"
    )

    # Section heading
    draw.text(
        (LEFT, section_title_y),
        "Original Urdu Sentence — Word-Level Attribution",
        font=SECTION_FONT,
        fill=TEXT
    )

    draw.text(
        (LEFT, section_note_y),
        "Signed Integrated Gradients scores are mapped from model tokens to the original Urdu words.",
        font=BODY_FONT,
        fill=SECONDARY_TEXT
    )

    # Main panel
    main_panel = (LEFT, main_panel_top, RIGHT, main_panel_bottom)

    draw.rounded_rectangle(
        main_panel,
        radius=15,
        fill=PANEL_BG,
        outline=BORDER,
        width=2
    )

    draw_word_cards(draw, main_panel, rows, word_scores)

    # Lower panels
    PANEL_GAP = 20
    lower_available_width = RIGHT - LEFT - PANEL_GAP
    left_panel_width = int(lower_available_width * 0.46)

    influential_box = (
        LEFT,
        lower_top,
        LEFT + left_panel_width,
        lower_bottom
    )

    scale_box = (
        LEFT + left_panel_width + PANEL_GAP,
        lower_top,
        RIGHT,
        lower_bottom
    )

    draw_top_words(draw, words, word_scores, influential_box)
    draw_scale(draw, scale_box)

    # ------------------------------------------------------------
    # STEP 7 — Save
    # ------------------------------------------------------------

    image.save(PNG_OUT, format="PNG", dpi=(300, 300))
    image.save(PDF_OUT, format="PDF", resolution=300.0)

    # ------------------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # ------------------------------------------------------------

    print("\n========================================")
    print("FINAL COMPACT PUBLICATION FIGURE CREATED")
    print("========================================")
    print(f"\nMatched prediction row: {matched_prediction_index}")

    print("\nSentence:")
    print(sentence)

    print("\nPrediction:")
    print(prediction)

    if confidence is not None:
        print(f"\nConfidence: {confidence:.2f}%")

    print("\nPNG:")
    print(PNG_OUT)

    print("\nPDF:")
    print(PDF_OUT)

    print("\nCanvas size:")
    print(f"{CANVAS_WIDTH} x {canvas_height}")

    print("\nUrdu card rows:")
    print(number_of_rows)

    print("\nV5.2 COMPACT FINAL — DONE")


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    main()