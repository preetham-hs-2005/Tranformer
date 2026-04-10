import math

import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder
from utils.positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    def forward(self, input_ids):
        return self.weight[input_ids]


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        pe = PositionalEncoding.create(max_len, d_model)
        self.register_buffer("positional_encoding", pe)

    @staticmethod
    def create_src_mask(src_ids, pad_id):
        return (src_ids != pad_id).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def create_causal_mask(seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(1)

    @classmethod
    def create_tgt_mask(cls, tgt_ids, pad_id):
        batch_size, seq_len = tgt_ids.shape
        pad_mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = cls.create_causal_mask(seq_len, tgt_ids.device)
        _ = batch_size
        return pad_mask & causal_mask

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        src_len = src_ids.size(1)
        tgt_len = tgt_ids.size(1)

        src_emb = self.src_embedding(src_ids) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)

        src_emb = src_emb + self.positional_encoding[:, :src_len, :]
        tgt_emb = tgt_emb + self.positional_encoding[:, :tgt_len, :]

        memory = self.encoder(src_emb, src_mask)
        decoder_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, src_mask=src_mask)
        logits = self.output_projection(decoder_output)
        return logits


class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, num_classes):
        super().__init__()
        self.d_model = d_model

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

        pe = PositionalEncoding.create(max_len, d_model)
        self.register_buffer("positional_encoding", pe)

    def forward(self, src_ids, src_mask=None):
        seq_len = src_ids.size(1)

        x = self.embedding(src_ids) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len, :]

        encoded = self.encoder(x, src_mask)

        if src_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            token_mask = src_mask.squeeze(1).squeeze(1).float()
            pooled = (encoded * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        return self.classifier(pooled)
