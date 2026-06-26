import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        weights: Optional[Tensor] = None,
        freeze: bool = False,
    ):
        super(GeneEncoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, _freeze=freeze
        )

        if weights is not None:
            self.embedding.weight.data.copy_(torch.Tensor(weights))

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int,
        token_to_pos: dict[str, int],
        maxval=10000.0,
    ):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)


        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(maxval) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        arr = []
        for _, v in token_to_pos.items():
            arr.append(pe[v - 1].numpy())
        pe = torch.Tensor(np.array(arr))
        self.register_buffer("pe", pe)

    def forward(self, gene_pos: Tensor) -> Tensor:
        return torch.index_select(self.pe, 0, gene_pos.view(-1)).view(
            gene_pos.shape + (-1,)
        )


class DPositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len_x: int,
        max_len_y: int,
        maxvalue_x=10000.0,
        maxvalue_y=10000.0,
    ):
        super(DPositionalEncoding, self).__init__()
        position2 = torch.arange(max_len_y).unsqueeze(1)
        position1 = torch.arange(max_len_x).unsqueeze(1)

        half_n = d_model // 2

        div_term2 = torch.exp(
            torch.arange(0, half_n, 2) * (-math.log(maxvalue_y) / d_model)
        )
        div_term1 = torch.exp(
            torch.arange(0, half_n, 2) * (-math.log(maxvalue_x) / d_model)
        )
        pe1 = torch.zeros(max_len_x, 1, d_model)
        pe2 = torch.zeros(max_len_y, 1, d_model)
        pe1[:, 0, 0:half_n:2] = torch.sin(position1 * div_term1)
        pe1[:, 0, 1:half_n:2] = torch.cos(position1 * div_term1)
        pe2[:, 0, half_n::2] = torch.sin(position2 * div_term2)
        pe2[:, 0, 1 + half_n :: 2] = torch.cos(position2 * div_term2)
        self.register_buffer("pe1", pe1)
        self.register_buffer("pe2", pe2)


    def forward(self, x: Tensor, pos_x: Tensor, pos_y: Tensor) -> Tensor:
        x = x + self.pe1[pos_x]
        x = x + self.pe2[pos_y]
        return x


class ContinuousValueEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 100_000,
        layers: int = 1,
        size: int = 1,
    ):
        super(ContinuousValueEncoder, self).__init__()
        self.max_value = max_value
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(size, d_model))
        for _ in range(layers - 1):
            self.encoder.append(nn.LayerNorm(d_model))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(p=dropout))
            self.encoder.append(nn.Linear(d_model, d_model))

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, min=0, max=self.max_value)
        for val in self.encoder:
            x = val(x)
        if mask is not None:
            x = x.masked_fill_(mask.unsqueeze(-1), 0)
        return x


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super(CategoryValueEncoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x.long())
