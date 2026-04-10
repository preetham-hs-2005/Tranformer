import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from models.transformer import SentimentTransformer, TransformerModel
from tasks.sentiment import build_sentiment_components, predict_sentiment
from tasks.summarization import build_summarization_components, generate_summary
from utils.dataset import SentimentDataset
from utils.tokenizer import SimpleTokenizer


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}
DEFAULT_SENTIMENT_CHECKPOINT = "checkpoints/sentiment_model.pt"
DEFAULT_SUMMARIZATION_CHECKPOINT = "checkpoints/summarization_model.pt"

POSITIVE_WORDS = {
    "amazing", "awesome", "benchmark", "best", "brilliant", "calm", "celebrating",
    "character", "clear", "composure", "credit", "defining", "delivered", "enjoyed",
    "excellent", "experience", "fantastic", "formidable", "good", "great", "helpful",
    "important", "improved", "impressive", "luxury", "masterclass", "nice", "outstanding",
    "phenomenal", "plan", "pleasure", "positive", "prepare", "prepared", "proud",
    "remarkable", "resilience", "safe", "special", "star", "stood tall", "strength",
    "strong", "support", "terrific", "unbeaten", "unbelievable", "utilize", "well",
    "win", "winning", "won", "wonderful"
}

NEGATIVE_WORDS = {
    "awful", "bad", "booing", "boring", "couldn't", "defeat", "defeats", "delay",
    "delays", "disappointing", "disappointed", "difficult", "did not", "didn't", "down",
    "drama", "dropped", "failure", "fell", "firepower", "horrible", "injured", "lost",
    "lopsided", "mistake", "negative", "not easy", "painful", "pill to swallow", "poor",
    "pressure was immense", "rough", "slow", "terrible", "tough", "traffic", "ugly",
    "unfortunately", "waiting", "weak", "wickets", "worse", "worst"
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer_from_mapping(token_to_id):
    tokenizer = SimpleTokenizer()
    tokenizer.token_to_id = dict(token_to_id)
    tokenizer.id_to_token = {idx: token for token, idx in tokenizer.token_to_id.items()}
    return tokenizer


def normalize_text(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00e2\u20ac\u2122", "'")
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text):
    text = normalize_text(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    sentences = [part.strip() for part in parts if part.strip()]
    return [sentence for sentence in sentences if len(sentence.split()) >= 4]


def label_sentence(sentence):
    lowered = sentence.lower()
    positive = sum(1 for phrase in POSITIVE_WORDS if phrase in lowered)
    negative = sum(1 for phrase in NEGATIVE_WORDS if phrase in lowered)
    if positive == 0 and negative == 0:
        return None
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return None


def build_labeled_dataset_from_text(input_path, output_path):
    raw_text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    sentences = split_into_sentences(raw_text)

    labeled_rows = []
    for sentence in sentences:
        label = label_sentence(sentence)
        if label is not None:
            labeled_rows.append((sentence, label))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(labeled_rows)

    counts = Counter(label for _, label in labeled_rows)
    print(f"Sentences scanned: {len(sentences)}")
    print(f"Labeled rows written: {len(labeled_rows)}")
    print(f"Positive rows: {counts.get('positive', 0)}")
    print(f"Negative rows: {counts.get('negative', 0)}")
    print(f"Labeled CSV: {output_file.resolve()}")

    return output_file


def load_custom_samples(csv_path):
    samples = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip().lower()
            if text and label in LABEL_TO_ID:
                samples.append((text, label))

    if len(samples) < 4:
        raise ValueError("Need at least 4 labeled rows to train and validate the model.")
    return samples


def stratified_split(samples, train_ratio=0.8):
    grouped = {"positive": [], "negative": []}
    for text, label in samples:
        grouped[label].append((text, label))

    train_samples = []
    val_samples = []
    for _, items in grouped.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * train_ratio))
        split_idx = min(split_idx, len(items) - 1)
        train_samples.extend(items[:split_idx])
        val_samples.extend(items[split_idx:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    return train_samples, val_samples


def evaluate_sentiment(model, loader, tokenizer, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            src_mask = TransformerModel.create_src_mask(input_ids, tokenizer.pad_id)
            logits = model(input_ids, src_mask)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

            tp += int(((preds == 1) & (labels == 1)).sum().item())
            fp += int(((preds == 1) & (labels == 0)).sum().item())
            fn += int(((preds == 0) & (labels == 1)).sum().item())

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    avg_loss = total_loss / max(len(loader), 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_checkpoint(checkpoint_path, task, model, tokenizer, extra=None):
    payload = {
        "task": task,
        "model_state": model.state_dict(),
        "token_to_id": tokenizer.token_to_id,
        "config": {
            "d_model": config.D_MODEL,
            "num_heads": config.NUM_HEADS,
            "d_ff": config.D_FF,
            "num_layers": config.NUM_LAYERS,
            "max_len": config.MAX_LEN if task == "sentiment" else max(config.MAX_SRC_LEN, config.MAX_TGT_LEN),
        },
        "extra": extra or {},
    }
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_file)
    print(f"Checkpoint saved: {checkpoint_file.resolve()}")


def load_checkpoint(checkpoint_path, device):
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    tokenizer = build_tokenizer_from_mapping(checkpoint["token_to_id"])
    cfg = checkpoint["config"]
    task = checkpoint["task"]

    if task == "sentiment":
        model = SentimentTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            num_layers=cfg["num_layers"],
            max_len=cfg["max_len"],
            num_classes=len(LABEL_TO_ID),
        ).to(device)
    elif task == "summarization":
        model = TransformerModel(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
            num_layers=cfg["num_layers"],
            max_len=cfg["max_len"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint task: {task}")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return task, model, tokenizer, checkpoint.get("extra", {})


def print_sentiment_prediction(model, tokenizer, text, device):
    result = predict_sentiment(model, tokenizer, text, config.MAX_LEN, device)
    print("\nCustom sentiment prediction:")
    print(f"Text: {text}")
    print(f"Predicted sentiment: {result['label']}")
    print(f"Probabilities: {result['probabilities']}")


def print_summary_prediction(model, tokenizer, text, device):
    summary = generate_summary(
        model=model,
        tokenizer=tokenizer,
        text=text,
        max_src_len=config.MAX_SRC_LEN,
        max_tgt_len=config.MAX_TGT_LEN,
        device=device,
    )
    print("\nCustom summarization output:")
    print(f"Input: {text}")
    print(f"Summary: {summary}")


def run_toy_sentiment(show_samples=True):
    device = config.DEVICE
    tokenizer, dataset, model = build_sentiment_components(device)

    loader = DataLoader(dataset, batch_size=config.SENTIMENT_BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SENTIMENT_LR)

    for epoch in range(1, config.SENTIMENT_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            src_mask = TransformerModel.create_src_mask(input_ids, tokenizer.pad_id)
            logits = model(input_ids, src_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        print(f"Epoch {epoch:02d}/{config.SENTIMENT_EPOCHS} - Loss: {running_loss / len(loader):.4f}")

    if show_samples:
        print("\nSample predictions:")
        for text in [
            "this movie was fantastic",
            "the story was boring",
            "i enjoyed the acting",
            "this film was awful",
        ]:
            result = predict_sentiment(model, tokenizer, text, config.MAX_LEN, device)
            print(f"Text: {text} -> Predicted sentiment: {result['label']} | probs={result['probabilities']}")

    return model, tokenizer


def run_custom_sentiment(data_csv, epochs, batch_size, lr):
    device = config.DEVICE
    samples = load_custom_samples(data_csv)
    train_samples, val_samples = stratified_split(samples)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([text for text, _ in train_samples + val_samples])

    train_dataset = SentimentDataset(train_samples, tokenizer, LABEL_TO_ID, config.MAX_LEN)
    val_dataset = SentimentDataset(val_samples, tokenizer, LABEL_TO_ID, config.MAX_LEN)

    model = SentimentTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_LEN,
        num_classes=len(LABEL_TO_ID),
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_counts = Counter(label for _, label in samples)
    print(f"Device: {device}")
    print(f"Total samples: {len(samples)}")
    print(f"Positive samples: {class_counts.get('positive', 0)}")
    print(f"Negative samples: {class_counts.get('negative', 0)}")
    print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    best_metrics = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            src_mask = TransformerModel.create_src_mask(input_ids, tokenizer.pad_id)
            logits = model(input_ids, src_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate_sentiment(model, val_loader, tokenizer, device)
        if best_metrics is None or val_metrics["f1"] > best_metrics["f1"]:
            best_metrics = val_metrics
            best_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

    print("\nBest validation metrics:")
    print(f"epoch={best_epoch}")
    print(f"loss={best_metrics['loss']:.4f}")
    print(f"accuracy={best_metrics['accuracy']:.4f}")
    print(f"precision={best_metrics['precision']:.4f}")
    print(f"recall={best_metrics['recall']:.4f}")
    print(f"f1={best_metrics['f1']:.4f}")

    return model, tokenizer


def run_toy_summarization(show_samples=True):
    device = config.DEVICE
    tokenizer, dataset, model = build_summarization_components(device)

    loader = DataLoader(dataset, batch_size=config.SUMMARIZATION_BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SUMMARIZATION_LR)

    for epoch in range(1, config.SUMMARIZATION_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            tgt_in_ids = batch["tgt_in_ids"].to(device)
            tgt_out_ids = batch["tgt_out_ids"].to(device)
            src_mask = TransformerModel.create_src_mask(src_ids, tokenizer.pad_id)
            tgt_mask = TransformerModel.create_tgt_mask(tgt_in_ids, tokenizer.pad_id)
            logits = model(src_ids, tgt_in_ids, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        print(f"Epoch {epoch:02d}/{config.SUMMARIZATION_EPOCHS} - Loss: {running_loss / len(loader):.4f}")

    if show_samples:
        print("\nSample generations:")
        for text in [
            "the weather is warm and sunny",
            "students studied hard and improved results",
            "heavy rain caused delays in the city",
        ]:
            summary = generate_summary(
                model=model,
                tokenizer=tokenizer,
                text=text,
                max_src_len=config.MAX_SRC_LEN,
                max_tgt_len=config.MAX_TGT_LEN,
                device=device,
            )
            print(f"Input: {text} -> Summary: {summary}")

    return model, tokenizer


def resolve_checkpoint_path(task, checkpoint_path):
    if checkpoint_path:
        return checkpoint_path
    return DEFAULT_SENTIMENT_CHECKPOINT if task == "sentiment" else DEFAULT_SUMMARIZATION_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(description="Run the Transformer project from one entrypoint.")
    parser.add_argument("--task", choices=["sentiment", "summarization"], required=True, help="Which task to run.")
    parser.add_argument("--input-text", help="Optional raw .txt file for custom sentiment training. If provided, the file is auto-labeled first.")
    parser.add_argument("--data-csv", help="Optional labeled CSV for custom sentiment training.")
    parser.add_argument("--output-csv", default="data/auto_labeled_sentiment.csv", help="Where to write the auto-labeled CSV when --input-text is used.")
    parser.add_argument("--predict-text", help="Custom text to test in the terminal.")
    parser.add_argument("--predict-only", action="store_true", help="Skip training and load a saved checkpoint for prediction.")
    parser.add_argument("--checkpoint", help="Checkpoint path to save or load.")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save the trained model checkpoint after training.")
    parser.add_argument("--force-train", action="store_true", help="Force training even if a checkpoint exists.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = config.DEVICE
    checkpoint_path = resolve_checkpoint_path(args.task, args.checkpoint)
    checkpoint_exists = Path(checkpoint_path).exists()

    should_auto_predict_from_checkpoint = (
        args.predict_text is not None
        and not args.predict_only
        and not args.force_train
        and checkpoint_exists
        and not args.save_checkpoint
        and args.input_text is None
        and args.data_csv is None
    )

    if should_auto_predict_from_checkpoint:
        loaded_task, model, tokenizer, _ = load_checkpoint(checkpoint_path, device)
        if loaded_task != args.task:
            raise ValueError(f"Checkpoint task mismatch. Expected {args.task}, found {loaded_task}.")
        if args.task == "sentiment":
            print_sentiment_prediction(model, tokenizer, args.predict_text, device)
        else:
            print_summary_prediction(model, tokenizer, args.predict_text, device)
        return

    if args.predict_only:
        if not args.predict_text:
            raise ValueError("--predict-only requires --predict-text.")
        loaded_task, model, tokenizer, _ = load_checkpoint(checkpoint_path, device)
        if loaded_task != args.task:
            raise ValueError(f"Checkpoint task mismatch. Expected {args.task}, found {loaded_task}.")
        if args.task == "sentiment":
            print_sentiment_prediction(model, tokenizer, args.predict_text, device)
        else:
            print_summary_prediction(model, tokenizer, args.predict_text, device)
        return

    if args.task == "summarization":
        model, tokenizer = run_toy_summarization(show_samples=args.predict_text is None)
        if args.save_checkpoint:
            save_checkpoint(checkpoint_path, "summarization", model, tokenizer)
        if args.predict_text:
            print_summary_prediction(model, tokenizer, args.predict_text, device)
        return

    if args.input_text:
        csv_path = build_labeled_dataset_from_text(args.input_text, args.output_csv)
        model, tokenizer = run_custom_sentiment(csv_path, args.epochs, args.batch_size, args.lr)
        if args.save_checkpoint:
            save_checkpoint(checkpoint_path, "sentiment", model, tokenizer, {"dataset": str(csv_path)})
        if args.predict_text:
            print_sentiment_prediction(model, tokenizer, args.predict_text, device)
        return

    if args.data_csv:
        model, tokenizer = run_custom_sentiment(args.data_csv, args.epochs, args.batch_size, args.lr)
        if args.save_checkpoint:
            save_checkpoint(checkpoint_path, "sentiment", model, tokenizer, {"dataset": str(args.data_csv)})
        if args.predict_text:
            print_sentiment_prediction(model, tokenizer, args.predict_text, device)
        return

    model, tokenizer = run_toy_sentiment(show_samples=args.predict_text is None)
    if args.save_checkpoint:
        save_checkpoint(checkpoint_path, "sentiment", model, tokenizer, {"dataset": "toy"})
    if args.predict_text:
        print_sentiment_prediction(model, tokenizer, args.predict_text, device)


if __name__ == "__main__":
    main()
