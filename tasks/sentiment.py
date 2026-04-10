import torch

import config
from models.transformer import SentimentTransformer, TransformerModel
from utils.dataset import SentimentDataset
from utils.tokenizer import SimpleTokenizer


def build_sentiment_components(device):
    tokenizer = SimpleTokenizer()
    texts = [text for text, _ in config.SENTIMENT_DATA]
    tokenizer.build_vocab(texts)

    dataset = SentimentDataset(
        samples=config.SENTIMENT_DATA,
        tokenizer=tokenizer,
        label_to_id=config.SENTIMENT_LABELS,
        max_len=config.MAX_LEN,
    )

    model = SentimentTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_LEN,
        num_classes=len(config.SENTIMENT_LABELS),
    ).to(device)

    return tokenizer, dataset, model


@torch.no_grad()
def predict_sentiment(model, tokenizer, text, max_len, device):
    model.eval()

    input_ids = tokenizer.encode(text, add_eos=True, max_len=max_len, pad_to_max=True)
    src = torch.tensor([input_ids], dtype=torch.long, device=device)
    src_mask = TransformerModel.create_src_mask(src, tokenizer.pad_id)

    logits = model(src, src_mask)
    probs = torch.softmax(logits, dim=-1)
    pred_id = int(torch.argmax(probs, dim=-1).item())

    return {
        "label": config.SENTIMENT_ID_TO_LABEL[pred_id],
        "probabilities": probs.squeeze(0).cpu().tolist(),
    }
