import gc
import json
import math
from collections import Counter
from typing import Dict, List, Optional, Union

import bionty as bt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Gamma, Poisson

from ..tasks import cell_emb as embbed_task
from ..tasks import denoise as denoise_task
from ..tasks import grn as grn_task


def make_adata(
    embs: Tensor,
    labels: List[str],
    pred: Tensor = None,
    attention: Optional[Tensor] = None,
    step: int = 0,
    label_decoders: Optional[Dict] = None,
    labels_hierarchy: Dict = {},
    gtclass: Optional[Tensor] = None,
    name: str = "",
    mdir: str = "/tmp",
    doplot: bool = True,
):
    colname = ["pred_" + i for i in labels]
    if pred is not None:
        obs = np.array(pred.to(device="cpu", dtype=torch.int32))
        if label_decoders is not None:
            obs = np.array(
                [
                    [label_decoders[labels[i]][n] for n in name]
                    for i, name in enumerate(obs.T)
                ]
            ).T

        if gtclass is not None:
            colname += labels
            nobs = np.array(gtclass.to(device="cpu", dtype=torch.int32))
            if label_decoders is not None:
                nobs = np.array(
                    [
                        [label_decoders[labels[i]][n] for n in name]
                        for i, name in enumerate(nobs.T)
                    ]
                ).T
            obs = np.hstack([obs, nobs])

        adata = AnnData(
            np.array(embs.to(device="cpu", dtype=torch.float32)),
            obs=pd.DataFrame(
                obs,
                columns=colname,
            ),
        )
        accuracy = {}
        for label in labels:
            if gtclass is not None:
                tr = translate(adata.obs[label].tolist(), label)
                if tr is not None:
                    adata.obs["conv_" + label] = adata.obs[label].replace(tr)
            tr = translate(adata.obs["pred_" + label].tolist(), label)
            if tr is not None:
                adata.obs["conv_pred_" + label] = adata.obs["pred_" + label].replace(tr)
            res = []
            if label_decoders is not None and gtclass is not None:
                class_topred = label_decoders[label].values()
                if label in labels_hierarchy:
                    cur_labels_hierarchy = {
                        label_decoders[label][k]: [label_decoders[label][i] for i in v]
                        for k, v in labels_hierarchy[label].items()
                    }
                else:
                    cur_labels_hierarchy = {}
                for pred, true in adata.obs[["pred_" + label, label]].values:
                    if pred == true:
                        res.append(True)
                        continue
                    if len(labels_hierarchy) > 0:
                        if true in cur_labels_hierarchy:
                            res.append(pred in cur_labels_hierarchy[true])
                        elif true not in class_topred:
                            raise ValueError(
                                f"true label {true} not in available classes"
                            )
                        elif true != "unknown":
                            res.append(False)
                    elif true not in class_topred:
                        raise ValueError(f"true label {true} not in available classes")
                    elif true != "unknown":
                        res.append(False)
                    else:
                        pass
                accuracy["pred_" + label] = sum(res) / len(res) if len(res) > 0 else 0
        adata.obs = adata.obs.astype("category")
    else:
        adata = AnnData(
            np.array(embs.to(device="cpu", dtype=torch.float32)),
        )
    if False:
        adata.varm["Qs"] = (
            attention[:, :, 0, :, :]
            .permute(1, 3, 0, 2)
            .view(
                attention.shape[0],
                attention.shape[1],
                attention.shape[3] * attention.shape[4],
            )
            .detach()
            .cpu()
            .numpy()
        )
        adata.varm["Ks"] = (
            attention[:, :, 1, :, :]
            .permute(1, 3, 0, 2)
            .view(
                attention.shape[0],
                attention.shape[1],
                attention.shape[3] * attention.shape[4],
            )
            .detach()
            .cpu()
            .numpy()
        )
    print(adata)
    if doplot and adata.shape[0] > 100 and pred is not None:
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)
        sc.tl.leiden(adata, key_added="sprint_leiden")
        if gtclass is not None:
            color = [
                i
                for pair in zip(
                    [
                        "conv_" + i if "conv_" + i in adata.obs.columns else i
                        for i in labels
                    ],
                    [
                        (
                            "conv_pred_" + i
                            if "conv_pred_" + i in adata.obs.columns
                            else "pred_" + i
                        )
                        for i in labels
                    ],
                )
                for i in pair
            ]
            fig, axs = plt.subplots(
                int(len(color) / 2), 2, figsize=(24, len(color) * 4)
            )
            plt.subplots_adjust(wspace=1)
            for i, col in enumerate(color):
                sc.pl.umap(
                    adata,
                    color=col,
                    ax=axs[i // 2, i % 2],
                    show=False,
                )
                acc = ""
                if "_pred_" in col and col.split("conv_")[-1] in accuracy:
                    acc = " (accuracy: {:.2f})".format(accuracy[col.split("conv_")[-1]])
                axs[i // 2, i % 2].set_title(col + " UMAP" + acc)
                if "cell_type" in col:
                    axs[i // 2, i % 2].legend(fontsize="x-small")
                axs[i // 2, i % 2].set_xlabel("UMAP1")
                axs[i // 2, i % 2].set_ylabel("UMAP2")
        else:
            color = [
                (
                    "conv_pred_" + i
                    if "conv_pred_" + i in adata.obs.columns
                    else "pred_" + i
                )
                for i in labels
            ]
            fig, axs = plt.subplots(len(color), 1, figsize=(16, len(color) * 8))
            for i, col in enumerate(color):
                sc.pl.umap(
                    adata,
                    color=col,
                    ax=axs[i],
                    show=False,
                )
                acc = ""
                if "_pred_" in col and col.split("conv_")[-1] in accuracy:
                    acc = " (accuracy: {:.2f})".format(accuracy[col.split("conv_")[-1]])
                axs[i].set_title(col + " UMAP" + acc)
                axs[i].set_xlabel("UMAP1")
                axs[i].set_ylabel("UMAP2")
        plt.show()
    else:
        fig = None
    if mdir is not None:
        adata.write(mdir + "/step_" + str(step) + "_" + name + ".h5ad")
    return adata, fig


def _init_weights(
    module: torch.nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    mup_width_scale: float = 1.0,
    rescale_prenorm_residual: bool = True,
):
    mup_init_scale = math.sqrt(mup_width_scale)
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, std=initializer_range * mup_init_scale)
        optim_cfg = getattr(module.weight, "_optim", {})
        optim_cfg.update({"lr_multiplier": mup_width_scale})
        setattr(module.weight, "_optim", optim_cfg)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        pass

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=initializer_range * mup_init_scale / math.sqrt(2 * n_layer),
                )


def downsample_profile(mat: Tensor, dropout: float, method="new"):
    if method == "old":
        totcounts = mat.sum(1)
        batch = mat.shape[0]
        ngenes = mat.shape[1]
        tnoise = 1 - (1 - dropout) ** (1 / 2)
        res = torch.poisson(
            torch.rand((batch, ngenes)).to(device=mat.device)
            * ((tnoise * totcounts.unsqueeze(1)) / (0.5 * ngenes))
        ).int()
        drop = (torch.rand((batch, ngenes)) > tnoise).int().to(device=mat.device)

        mat = (mat - res) * drop
        return torch.maximum(mat, torch.Tensor([[0]]).to(device=mat.device)).int()
    elif method == "jules":
        scaler = (1 - dropout) ** (1 / 2)
        notdrop = (
            torch.rand(
                mat.shape,
                device=mat.device,
            )
            < scaler
        ).int()
        notdrop[mat == 0] = 0
        return notdrop * torch.poisson(mat * scaler)
    elif method == "new":
        batch = mat.shape[0]
        ngenes = mat.shape[1]
        dropout = dropout * 1.1
        res = torch.poisson((mat * (dropout / 2))).int()
        notdrop = (
            torch.rand((batch, ngenes), device=mat.device) >= (dropout / 2)
        ).int()
        mat = (mat - res) * notdrop
        return torch.maximum(
            mat, torch.zeros((1, 1), device=mat.device, dtype=torch.int)
        )
    else:
        raise ValueError(f"method {method} not recognized")


def simple_masker(
    shape: list[int],
    mask_ratio: float = 0.15,
) -> torch.Tensor:
    return torch.rand(shape) < mask_ratio


def weighted_masker(
    shape: list[int],
    mask_ratio: float = 0.15,
    mask_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mask_value: int = 1,
) -> torch.Tensor:
    mask = []
    for _ in range(shape[0]):
        m = np.zeros(shape[1])
        loc = np.random.choice(
            a=shape[1], size=int(shape[1] * mask_ratio), replace=False, p=mask_prob
        )
        m[loc] = mask_value
        mask.append(m)

    return torch.Tensor(np.array(mask)).to(torch.bool)


def zinb_sample(
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_probs: torch.Tensor,
    sample_shape: torch.Size = torch.Size([]),
):
    concentration = theta
    rate = theta / mu
    gamma_d = Gamma(concentration=concentration, rate=rate)
    p_means = gamma_d.sample(sample_shape)

    l_train = torch.clamp(p_means, max=1e8)
    samp = Poisson(l_train).sample()
    is_zero = torch.rand_like(samp) <= zi_probs
    samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
    return samp_


def translate(
    val: Union[str, list, set, dict, Counter], t: str = "cell_type_ontology_term_id"
):
    if t == "cell_type_ontology_term_id":
        obj = bt.CellType.filter().df().set_index("ontology_id")
    elif t == "assay_ontology_term_id":
        obj = bt.ExperimentalFactor.filter().df().set_index("ontology_id")
    elif t == "tissue_ontology_term_id":
        obj = bt.Tissue.filter().df().set_index("ontology_id")
    elif t == "disease_ontology_term_id":
        obj = bt.Disease.filter().df().set_index("ontology_id")
    elif t == "self_reported_ethnicity_ontology_term_id":
        obj = bt.Ethnicity.filter().df().set_index("ontology_id")
    else:
        return None
    if type(val) is str:
        if val == "unknown":
            return {val: val}
        return {val: obj.loc[val]["name"]}
    elif type(val) is list or type(val) is set:
        return {i: obj.loc[i]["name"] if i != "unknown" else i for i in set(val)}
    elif type(val) is dict or type(val) is Counter:
        return {obj.loc[k]["name"] if k != "unknown" else k: v for k, v in val.items()}


class Attention:
    def __init__(
        self,
        gene_dim: int,
        comp_attn: bool = False,
        apply_softmax: bool = False,
        sum_heads: bool = True,
    ):
        self.data: Optional[Tensor] = None
        self.gene_dim: int = gene_dim
        self.div: Optional[Tensor] = None
        self.comp_attn: bool = comp_attn
        self.apply_softmax: bool = apply_softmax
        self.sum_heads: bool = sum_heads
        self.shared_qk: bool = True

    def add(self, *args, **kwargs) -> None:
        if self.shared_qk:
            self.add_qk(*args, **kwargs)
        else:
            self.add_attn(*args, **kwargs)

    def add_attn(
        self, x: List[Tensor], pos: Tensor, expr: Optional[Tensor] = None
    ) -> None:
        if self.data is None:
            self.data = torch.zeros(
                [self.gene_dim, self.gene_dim, len(x) * x[0].shape[3]],
                device=pos.device,
                dtype=torch.float32,
            )
            self.div = torch.zeros(1, device=pos.device, dtype=torch.float32)

        for i, elem in enumerate(x):
            batch, seq_len, _, heads, _ = elem.shape
            if self.apply_softmax:
                attn = torch.nn.functional.softmax(
                    elem[:, :, 0, :, :].permute(0, 2, 1, 3)
                    @ elem[:, :, 1, :, :].permute(0, 2, 3, 1),
                    dim=-1,
                )
                if expr is not None:
                    attn = attn * (expr > 0).float()
                self.data[:, :, heads * i : heads * (i + 1)] += (
                    attn.sum(0).permute(1, 2, 0) / batch
                )
            else:
                self.data[:, :, heads * i : heads * (i + 1)] += (
                    elem[:, :, 0, :, :].permute(0, 2, 1, 3)
                    @ elem[:, :, 1, :, :].permute(0, 2, 3, 1)
                ).sum(0).permute(1, 2, 0) / batch
        self.div += 1

    def add_qk(
        self, x: List[Tensor], pos: Tensor, expr: Optional[Tensor] = None
    ) -> None:
        if self.data is None:
            self.data = torch.zeros(
                [len(x), self.gene_dim] + list(x[0].shape[2:]), device=pos.device
            )
            self.div = torch.zeros(self.gene_dim, device=pos.device)
        for i in range(x[0].shape[0]):
            loc = torch.cat([torch.arange(8, device=pos.device), pos[i] + 8]).int()
            for j in range(len(x)):
                self.data[j, loc, :, :, :] += x[j][i]
            self.div[loc] += 1

    def get(self) -> Optional[np.ndarray]:
        if self.shared_qk:
            if self.data is None:
                return None
            return self.data / self.div.view(1, self.div.shape[0], 1, 1, 1)
        else:
            if self.data is None:
                return None
            self.data.div_(self.div)
            return self.data


def test(model: torch.nn.Module, name: str, filedir: str) -> None:
    metrics = {}
    print(metrics)

    """
    gc.collect()
    res = denoise_task.default_benchmark(
        model, filedir + "/../../data/gNNpgpo6gATjuxTE7CCp.h5ad"
    )
    metrics.update(
        {
            "denoise/reco2full_vs_noisy2full": float(
                res["reco2full"] - res["noisy2full"]
            ),
        }
    )
    gc.collect()
    print(metrics)
    f = open("metrics_" + name + ".json", "a")
    f.write(json.dumps({"denoise": res}, indent=4))
    f.close()
    res = grn_task.default_benchmark(
        model, "gwps", batch_size=32 if model.d_model <= 512 else 8
    )
    f = open("metrics_" + name + ".json", "a")
    f.write(json.dumps({"grn_gwps": res}, default=lambda o: str(o), indent=4))
    f.close()
    metrics.update(
        {
            "grn_gwps/auprc_self": float(res["self"]["auprc"]),
            "grn_gwps/epr_self": float(res["self"]["epr"]),
            "grn_gwps/auprc_omni": float(res["omni"]["auprc"]),
            "grn_gwps/epr_omni": float(res["omni"]["epr"]),
            "grn_gwps/auprc": float(res["mean"]["auprc"]),
            "grn_gwps/epr": float(res["mean"]["epr"]),
        }
    )
    print(metrics)
    gc.collect()
    res = grn_task.default_benchmark(
        model, "sroy", batch_size=32 if model.d_model <= 512 else 8
    )
    f = open("metrics_" + name + ".json", "a")
    f.write(json.dumps({"grn_sroy": res}, default=lambda o: str(o), indent=4))
    f.close()
    metrics.update(
        {
            "grn_sroy/auprc_self": float(
                np.mean(
                    [
                        i["auprc"]
                        for k, i in res.items()
                        if k.startswith("self_")
                        and not any(
                            x in k for x in ["chip_", "ko_", "classifier", "_base"]
                        )
                    ]
                )
            ),
            "grn_sroy/epr_self": float(
                np.mean(
                    [
                        i["epr"]
                        for k, i in res.items()
                        if k.startswith("self_")
                        and not any(
                            x in k for x in ["chip_", "ko_", "classifier", "_base"]
                        )
                    ]
                )
            ),
            "grn_sroy/auprc_omni": float(
                np.mean(
                    [
                        i["auprc"]
                        for k, i in res.items()
                        if k.startswith("omni_")
                        and not any(
                            x in k for x in ["chip_", "ko_", "classifier", "_base"]
                        )
                    ]
                )
            ),
            "grn_sroy/epr_omni": float(
                np.mean(
                    [
                        i["epr"]
                        for k, i in res.items()
                        if k.startswith("omni_")
                        and not any(
                            x in k for x in ["chip_", "ko_", "classifier", "_base"]
                        )
                    ]
                )
            ),
            "grn_sroy/auprc": float(
                np.mean(
                    [
                        i["auprc"]
                        for k, i in res.items()
                        if k.startswith("mean_")
                        and not any(
                            x in k for x in ["chip_", "ko_", "classifier", "_base"]
                        )
                    ]
                )
            ),
            "grn_sroy/epr": float(
                np.mean(
                    [
                        i["epr"]
                        for k, i in res.items()
                        if k.startswith("mean_")
                        and not any(
                            x in k for x in ["chip_", "ko_", "classifier", "_base"]
                        )
                    ]
                )
            ),
        }
    )
    print(metrics)
    gc.collect()
    res = grn_task.default_benchmark(
        model,
        filedir + "/../../data/yBCKp6HmXuHa0cZptMo7.h5ad",
        batch_size=32 if model.d_model <= 512 else 8,
    )
    f = open("metrics_" + name + ".json", "a")
    f.write(json.dumps({"grn_omni": res}, default=lambda o: str(o), indent=4))
    f.close()
    metrics.update(
        {
            "grn_omni/auprc_class": float(
                np.mean([i["auprc"] for k, i in res.items() if "_class" in k])
            ),
            "grn_omni/epr_class": float(
                np.mean([i["epr"] for k, i in res.items() if "_class" in k])
            ),
            "grn_omni/tf_enr_class": float(
                np.sum(
                    [i.get("TF_enr", False) for k, i in res.items() if "_class" in k]
                )
            ),
            "grn_omni/tf_targ_enr_class": float(
                np.mean(
                    [
                        i["significant_enriched_TFtargets"]
                        for k, i in res.items()
                        if "_class" in k
                    ]
                )
            ),
            "grn_omni/auprc": float(
                np.mean([i["auprc"] for k, i in res.items() if "_mean" in k])
            ),
            "grn_omni/epr": float(
                np.mean([i["epr"] for k, i in res.items() if "_mean" in k])
            ),
            "grn_omni/tf_enr": float(
                np.sum([i.get("TF_enr", False) for k, i in res.items() if "_mean" in k])
            ),
            "grn_omni/tf_targ_enr": float(
                np.mean(
                    [
                        i["significant_enriched_TFtargets"]
                        for k, i in res.items()
                        if "_mean" in k
                    ]
                )
            ),
            # 'grn_omni/ct': res['classif']['cell_type_ontology_term_id']['accuracy'],
        }
    )
    """
    return metrics
