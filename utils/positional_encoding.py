import math

import torch


class PositionalEncoding:
    @staticmethod
    def create(max_len, d_model, device=None):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if device is not None:
            pe = pe.to(device)

        return pe.unsqueeze(0)
