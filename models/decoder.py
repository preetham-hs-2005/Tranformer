import torch.nn as nn

from models.attention import MultiHeadAttention
from models.encoder import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_output)

        cross_attn_output, _ = self.cross_attention(x, memory, memory, src_mask)
        x = self.norm2(x + cross_attn_output)

        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x
