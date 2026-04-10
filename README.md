# Transformer Project (From Scratch, PyTorch)

This project implements a Transformer architecture from scratch for:

1. Sentiment analysis
2. Text summarization

Everything runs from a single file: `run_project.py`.

## Requirements

- Python 3.9+
- PyTorch

Install PyTorch:

```bash
pip install torch
```

## Project structure

```text
transformer_project/
+-- data/
+-- models/
+-- tasks/
+-- utils/
+-- checkpoints/
+-- config.py
+-- run_project.py
+-- README.md
```

## Terminal workflow

Open a terminal in `transformer_project`.

### Train and save a sentiment model once

Toy sentiment model:

```bash
python run_project.py --task sentiment --save-checkpoint
```

Sentiment model from your raw text file:

```bash
python run_project.py --task sentiment --input-text "C:\Users\preet\Downloads\data.txt" --save-checkpoint
```

Sentiment model from an existing labeled CSV:

```bash
python run_project.py --task sentiment --data-csv "data\auto_labeled_sentiment.csv" --save-checkpoint
```

### Predict sentiment without retraining

If a checkpoint exists, this now auto-loads it:

```bash
python run_project.py --task sentiment --predict-text "movie i saw was ugly"
```

You can also force explicit predict-only mode:

```bash
python run_project.py --task sentiment --predict-only --predict-text "movie i saw was ugly"
```

If you saved to a custom checkpoint path, pass it too:

```bash
python run_project.py --task sentiment --predict-only --checkpoint "checkpoints\my_sentiment.pt" --predict-text "movie i saw was ugly"
```

### Train and save a summarization model once

```bash
python run_project.py --task summarization --save-checkpoint
```

### Predict summary without retraining

If a summarization checkpoint exists, this now auto-loads it:

```bash
python run_project.py --task summarization --predict-text "heavy rain caused traffic delays across the city this morning"
```

Explicit predict-only mode also works:

```bash
python run_project.py --task summarization --predict-only --predict-text "heavy rain caused traffic delays across the city this morning"
```

### Force training again

If you want to ignore the saved checkpoint and rerun epochs:

```bash
python run_project.py --task summarization --predict-text "your text here" --force-train
```

## Notes

- Default checkpoint paths are:
  - `checkpoints/sentiment_model.pt`
  - `checkpoints/summarization_model.pt`
- `--predict-text` now auto-loads an existing checkpoint when possible.
- `--predict-only` always skips training and requires a saved checkpoint.
- Auto-labeling is rule-based, so custom sentiment quality depends on the generated labels.
- The negative keyword rules were expanded to better catch words like `ugly`, `bad`, and `terrible`.
