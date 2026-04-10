import torch

import config
from models.transformer import TransformerModel
from utils.dataset import SummarizationDataset
from utils.tokenizer import SimpleTokenizer


def build_summarization_components(device):
    tokenizer = SimpleTokenizer()
    all_texts = []
    for source, summary in config.SUMMARIZATION_DATA:
        all_texts.append(source)
        all_texts.append(summary)
    tokenizer.build_vocab(all_texts)

    dataset = SummarizationDataset(
        pairs=config.SUMMARIZATION_DATA,
        tokenizer=tokenizer,
        max_src_len=config.MAX_SRC_LEN,
        max_tgt_len=config.MAX_TGT_LEN,
    )

    model = TransformerModel(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        max_len=max(config.MAX_SRC_LEN, config.MAX_TGT_LEN),
    ).to(device)

    return tokenizer, dataset, model


@torch.no_grad()
def generate_summary(model, tokenizer, text, max_src_len, max_tgt_len, device):
    model.eval()

    src_ids = tokenizer.encode(text, add_eos=True, max_len=max_src_len, pad_to_max=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = TransformerModel.create_src_mask(src, tokenizer.pad_id)

    generated = [tokenizer.sos_id]

    for _ in range(max_tgt_len):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_mask = TransformerModel.create_tgt_mask(tgt, tokenizer.pad_id)

        logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())

        if next_token == tokenizer.eos_id:
            break

        generated.append(next_token)

    return tokenizer.decode(generated[1:], skip_special=True)
