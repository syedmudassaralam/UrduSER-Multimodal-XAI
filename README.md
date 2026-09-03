# Explainable Context-Aware Multimodal Urdu Speech Emotion Recognition

Code and reproducibility resources for an explainable context-aware multimodal Urdu Speech Emotion Recognition (Urdu SER) framework.

## Dataset

Experiments use UrduSER / UrSEC with seven emotion classes: Neutral, Happy, Angry, Sad, Fear, Disgust, and Boredom.

Fixed split:
- Train: 2,800
- Validation: 350
- Test: 350
- Total: 3,500

Raw audio is not distributed in this repository. Portable metadata and fixed split files are provided under `data/` and `data_splits/`.

## Main Results

| Configuration | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| Current-utterance text | 30.29 | 30.36 |
| Context-aware text | 79.14 | 79.03 |
| Pretrained speech-model ensemble | 54.00 | 54.07 |
| Equal-weight audio-text fusion | 82.29 | 82.30 |
| Protected validation-selected fusion | 88.57 | 88.56 |
| Final selective multimodal fusion | **89.14** | **89.10** |

## Repository

- `src/text/` - text models and five-seed ensemble
- `src/audio/` - pretrained speech model training
- `src/baselines/` - acoustic ML/DL baselines
- `src/fusion/` - final V24 multimodal fusion
- `src/xai/` - Integrated Gradients, SHAP, fusion XAI, and faithfulness verification
- `outputs/` - verified predictions, metrics, baseline tables, statistical tests, and XAI figures

## Explainable AI

Text explanations use word-level Integrated Gradients with deletion-based faithfulness verification. Acoustic explanations use TreeSHAP over 187 acoustic/prosodic descriptors. Fusion explanations use decision-level probability traces and selective-rule analysis.

The large ExtraTrees checkpoint is not included. Its path can be supplied with the `EXTRA_TREES_MODEL_PATH` environment variable.

## Reproducibility

Fixed train/validation/test partitions and verified test outputs are included. Machine-specific absolute paths have been removed. Raw audio and large trained checkpoints are excluded.

## Citation

Please cite the associated Urdu SER manuscript when using this repository. Full citation information will be added after publication.

## License

Dataset usage remains subject to the original UrduSER / UrSEC terms. A code license will be specified separately before public release.
