from typing import Callable, Dict, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphSDEExprDecoder(nn.Module):
    def __init__(self, d_model: int, drift: nn.Module, diffusion: nn.Module):
        super().__init__()
        self.d_model = d_model
        self.drift = drift
        self.diffusion = diffusion

    def forward(self, x: Tensor, dt: float) -> Tensor:
        drift = self.drift(x)
        diffusion = self.diffusion(x)
        dW = torch.randn_like(x) * torch.sqrt(dt)
        return x + drift * dt + diffusion * dW


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nfirst_tokens_to_skip: int = 0,
        dropout: float = 0.1,
        zinb: bool = True,
    ):
        super(ExprDecoder, self).__init__()
        self.nfirst_tokens_to_skip = nfirst_tokens_to_skip
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
        )
        self.pred_var_zero = nn.Linear(d_model, 3 if zinb else 1)
        self.zinb = zinb

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.fc(x[:, self.nfirst_tokens_to_skip :, :])
        if self.zinb:
            pred_value, var_value, zero_logits = self.pred_var_zero(x).split(
                1, dim=-1
            )
            return dict(
                mean=F.softmax(pred_value.squeeze(-1), dim=-1),
                disp=torch.exp(torch.clamp(var_value.squeeze(-1), max=15)),
                zero_logits=zero_logits.squeeze(-1),
            )
        else:
            pred_value = self.pred_var_zero(x)
            return dict(mean=F.softmax(pred_value.squeeze(-1), dim=-1))


class MVCDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        tot_labels: int = 1,
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
    ) -> None:
        super(MVCDecoder, self).__init__()
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)
            self.query_activation = query_activation()
            self.pred_var_zero = nn.Linear(d_model, d_model * 3, bias=False)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model * (1 + tot_labels), d_model / 2)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(d_model / 2, 3)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 3)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.d_model = d_model

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(self.norm(self.gene2query(gene_embs)))
            pred, var, zero_logits = self.pred_var_zero(query_vecs).split(
                self.d_model, dim=-1
            )
            cell_emb = cell_emb.unsqueeze(2)
            pred, var, zero_logits = (
                torch.bmm(pred, cell_emb).squeeze(2),
                torch.bmm(var, cell_emb).squeeze(2),
                torch.bmm(zero_logits, cell_emb).squeeze(2),
            )
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            pred, var, zero_logits = self.fc2(h).split(1, dim=-1)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            pred, var, zero_logits = self.fc2(h).split(1, dim=-1)
        return dict(
            mvc_mean=F.softmax(pred, dim=-1),
            mvc_disp=torch.exp(torch.clamp(var, max=15)),
            mvc_zero_logits=zero_logits,
        )


class ClsDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        layers: list[int] = [256, 128],
        activation: Callable = nn.ReLU,
        dropout: float = 0.1,
    ):
        super(ClsDecoder, self).__init__()
        layers = [d_model] + layers
        self.decoder = nn.Sequential()
        for i, l in enumerate(layers[1:]):
            self.decoder.append(nn.Linear(layers[i], l))
            self.decoder.append(nn.LayerNorm(l))
            self.decoder.append(activation())
            self.decoder.append(nn.Dropout(dropout))
        self.out_layer = nn.Linear(layers[-1], n_cls)

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder(x)
        return self.out_layer(x)
