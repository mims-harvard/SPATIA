from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.distributions import NegativeBinomial


def mse(input: Tensor, target: Tensor) -> Tensor:
    input = torch.log2(input + 1)
    input = (input / torch.sum(input, dim=1, keepdim=True)) * 10000
    target = torch.log2(target + 1)
    target = target / torch.sum(target, dim=1, keepdim=True) * 10000
    return F.mse_loss(input, target, reduction="mean")


def masked_mse(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_mae(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.float()
    loss = F.l1_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_nb(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.float()
    nb = torch.distributions.NegativeBinomial(total_count=target, probs=input)
    masked_log_probs = nb.log_prob(target) * mask
    return -masked_log_probs.sum() / mask.sum()


def nb(target: Tensor, mu: Tensor, theta: Tensor, eps=1e-8):
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )

    return -res.mean()


def nb_dist(x: Tensor, mu: Tensor, theta: Tensor, eps=1e-8):
    loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x)
    return loss


def zinb(
    target: Tensor,
    mu: Tensor,
    theta: Tensor,
    pi: Tensor,
    eps=1e-8,
):
    softplus_pi = F.softplus(-pi)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (torch.log(theta + eps) - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((target < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )
    mul_case_non_zero = torch.mul((target > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return -res.mean()


def criterion_neg_log_bernoulli(input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: Tensor, target: Tensor, mask: torch.LongTensor
) -> Tensor:
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()


def similarity(x: Tensor, y: Tensor, temp: float) -> Tensor:
    res = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0)) / temp
    labels = torch.arange(res.size(0)).long().to(device=res.device)
    return F.cross_entropy(res, labels)


def ecs(cell_emb: Tensor, ecs_threshold: float = 0.5) -> Tensor:
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())

    mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
    cos_sim = cos_sim.masked_fill(mask, 0.0)
    cos_sim = F.relu(cos_sim)
    return torch.mean(1 - (cos_sim - ecs_threshold) ** 2)


def classification(
    clsname: str,
    pred: torch.Tensor,
    cl: torch.Tensor,
    maxsize: int,
    labels_hierarchy: Optional[Dict[str, Dict[int, list[int]]]] = {},
) -> torch.Tensor:
    newcl = torch.zeros(
        (cl.shape[0], maxsize), device=cl.device
    )
    valid_indices = (cl != -1) & (cl < maxsize)
    valid_cl = cl[valid_indices]
    newcl[valid_indices, valid_cl] = 1

    weight = torch.ones_like(newcl, device=cl.device)
    weight[cl == -1, :] = 0
    inv = cl >= maxsize
    if inv.any():
        if clsname in labels_hierarchy.keys():
            clhier = labels_hierarchy[clsname]

            inv_weight = weight[inv]
            inv_weight[clhier[cl[inv] - maxsize]] = 0
            weight[inv] = inv_weight

            addnewcl = torch.ones(
                weight.shape[0], device=pred.device
            )
            addweight = torch.zeros(weight.shape[0], device=pred.device)
            addweight[inv] = 1
            addpred = pred.clone()
            inv_addpred = addpred[inv]
            inv_addpred[inv_weight.to(bool)] = torch.finfo(pred.dtype).min
            addpred[inv] = inv_addpred

            addpred = torch.logsumexp(addpred, dim=-1)

            newcl = torch.cat([newcl, addnewcl.unsqueeze(1)], dim=1)
            pred = torch.cat([pred, addpred.unsqueeze(1)], dim=1)
            weight = torch.cat([weight, addweight.unsqueeze(1)], dim=1)
        else:
            raise ValueError("need to use labels_hierarchy for this usecase")

    myloss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target=newcl, weight=weight
    )
    return myloss


class AdversarialDiscriminatorLoss(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = True,
    ):
        super().__init__()
        self.decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self.decoder.append(nn.Linear(d_model, d_model))
            self.decoder.append(nn.LayerNorm(d_model))
            self.decoder.append(activation())
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor, batch_labels: Tensor) -> Tensor:
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self.decoder:
            x = layer(x)
        x = self.out_layer(x)
        return F.cross_entropy(x, batch_labels)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    return GradReverse.apply(x, lambd)
