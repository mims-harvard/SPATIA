import copy
import os
from functools import partial

from math import factorial
from typing import *

import ipdb
import lightning as L
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.tuner.lr_finder import _LRCallback
from scipy.sparse import load_npz
from scprint.model.esm_layers import *
from torch import Tensor, nn, optim
from transformers import (
    AutoModel,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    ViTMAEForPreTraining,
)

from . import decoders, encoders, loss, utils
from .flash_attn import FlashTransformerEncoder
from .loss import grad_reverse
from .utils import simple_masker

FILEDIR = os.path.dirname(os.path.realpath(__file__))


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


class scPrint(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        genes: list,
        organisms: list = ["NCBITaxon:9606"],
        precpt_gene_emb: Optional[str] = None,
        gene_pos_enc: Optional[list] = None,
        normalization: str = "sum",
        d_model: int = 512,
        nhead: int = 8,
        attn_bias: str = "none",
        d_hid: int = 512,
        clip_model_type: Literal[
            "openai/clip-vit-base-patch32", "facebook/vit-mae-base"
        ] = "openai/clip-vit-base-patch32",
        combine_weight: float = 1,
        image_combine_weight: float = 1,
        image_recon_loss_weight: float = 1,
        edge_dim: int = 12,
        nlayers: int = 6,
        expr_encoder_layers: int = 2,
        layers_cls: list[int] = [],
        classes: Dict[str, int] = {},
        labels_hierarchy: Dict[str, Dict[int, list[int]]] = {},
        dropout: float = 0.2,
        transformer: str = "fast",
        expr_emb_style: str = "continuous",
        domain_spec_batchnorm: str = "None",
        n_input_bins: int = 0,
        num_batch_labels: int = 0,
        mvc_decoder: str = "None",
        pred_embedding: list[str] = [],
        cell_emb_style: str = "cls",
        freeze_embeddings: bool = True,
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        zinb: bool = True,
        lr: float = 0.0001,
        optim="adamW",
        weight_decay=0.01,
        ckpt_path=None,
        **flash_attention_kwargs,
    ):
        super().__init__()

        self.do_denoise = True
        self.noise = [0.6]
        self.do_cce = False
        self.cce_sim = 0.5
        self.cce_scale = 0.002
        self.do_ecs = False
        self.ecs_threshold = 0.3
        self.ecs_scale = 0.05
        self.do_mvc = False
        self.mvc_scale = 1.0
        self.class_embd_diss_scale = 0.2
        self.do_adv_cls = False
        self.adv_class_scale = 0.1
        self.do_cls = False
        self.mean_attn_tot = None
        self.mean_attn_tot_c = 0
        self.do_adv_batch = False
        self.run_full_forward = True
        self.class_scale = 0.4
        self.do_next_tp = False
        self.do_generate = False
        self.mask_ratio = []
        self.warmup_duration = 500
        self.weight_decay = 0.01
        self.optim = "adamW"
        self.fused_adam = False
        self.lr_reduce_patience = 1
        self.lr_reduce_factor = 0.6
        self.test_every = 1
        self.lr_reduce_monitor = "val_loss"
        self.name = ""
        self.lr = lr
        self.lrfinder_steps = 0
        self.doplot = True
        self.get_attention_layer = []
        self.embs = None
        self.pred_log_adata = True
        self.attn = utils.Attention(len(classes) + 2 + len(genes))
        self.predict_depth_mult = 3
        self.predict_mode = "none"
        self.keep_all_cls_pred = False
        self.d_model = d_model
        self.normalization = normalization
        self.organisms = organisms
        self.edge_dim = edge_dim
        self.attn_bias = attn_bias
        self.nlayers = nlayers
        self.gene_pos_enc = gene_pos_enc
        self.mvc_decoder = mvc_decoder
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.n_input_bins = n_input_bins
        self.transformer = transformer
        self.label_counts = classes
        self.classes = list(classes.keys())
        self.cell_emb_style = cell_emb_style
        self.label_decoders = label_decoders
        self.pred_embedding = pred_embedding
        self.mat_labels_hierarchy = {}
        self.labels_hierarchy = labels_hierarchy
        if "strict_loading" in flash_attention_kwargs:
            flash_attention_kwargs.pop("strict_loading")

        for k, v in labels_hierarchy.items():
            tens = torch.zeros((len(v), classes[k]))
            for k2, v2 in v.items():
                tens[k2 - classes[k], v2] = 1
            self.mat_labels_hierarchy[k] = tens.to(bool)
        self.expr_emb_style = expr_emb_style

        if self.expr_emb_style not in ["category", "continuous", "none"]:
            raise ValueError(
                f"expr_emb_style should be one of category, continuous, scaling, "
                f"got {expr_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.genes = genes
        self.vocab = {i: n for i, n in enumerate(genes)}

        if precpt_gene_emb is not None:
            embeddings = pd.read_parquet(precpt_gene_emb).loc[self.genes]
            if len(embeddings) == 0:
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(embeddings) < len(self.genes):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(embeddings))
            sembeddings = torch.nn.AdaptiveAvgPool1d(d_model)(
                torch.tensor(embeddings.values)
            )

            self.gene_encoder = encoders.GeneEncoder(
                len(self.vocab), d_model, weights=sembeddings, freeze=freeze_embeddings
            )
        else:
            self.gene_encoder = encoders.GeneEncoder(len(self.vocab), d_model)

        if expr_emb_style in ["continuous", "full_pos"]:
            self.expr_encoder = encoders.ContinuousValueEncoder(
                d_model, dropout, layers=expr_encoder_layers
            )
        elif expr_emb_style == "binned_pos":
            assert n_input_bins > 0
            self.expr_encoder = encoders.CategoryValueEncoder(n_input_bins, d_model)
        else:
            self.expr_encoder = torch.nn.Identity()

        if self.gene_pos_enc is not None:
            max_len = max(gene_pos_enc)
            token_to_pos = {token: pos for token, pos in enumerate(self.gene_pos_enc)}
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, max_len=max_len, token_to_pos=token_to_pos
            )

        self.cell_embs_count = len(self.classes) + 2
        self.class_encoder = encoders.CategoryValueEncoder(
            self.cell_embs_count - 1, d_model
        )
        self.depth_encoder = encoders.ContinuousValueEncoder(
            d_model, dropout, layers=expr_encoder_layers
        )

        if transformer == "linear":
            raise NotImplementedError("Linear transformer is not implemented")
        else:
            self.transformer = FlashTransformerEncoder(
                d_model,
                nhead,
                nlayers,
                dropout=dropout,
                use_flash_attn=(transformer == "flash"),
                **flash_attention_kwargs,
            )

        self.clip_model_type = clip_model_type
        if "clip" in clip_model_type:
            self.clip_model = CLIPVisionModel.from_pretrained(clip_model_type).train()
        else:
            self.clip_model = ViTMAEForPreTraining.from_pretrained(
                clip_model_type
            ).train()
        self.combine_weight = combine_weight

        if isinstance(self.clip_model, ViTMAEForPreTraining):
            self.image_combine_weight = image_combine_weight
            self.image_recon_loss_weight = image_recon_loss_weight
            self.expression2image = nn.Linear(
                d_model, self.clip_model.config.hidden_size
            )
            self.image_fusion_layer = FusionLayer(
                self.clip_model.config.hidden_size, nhead=nhead
            )
        else:
            self.image_combine_weight = None
            self.image_recon_loss_weight = None
            self.expression2image = None
        self.fusion_layer = FusionLayer(d_model, nhead=nhead)
        self.projection_layer = nn.Linear(
            self.clip_model.config.hidden_size, self.d_model
        )

        self.expr_decoder = decoders.ExprDecoder(
            d_model,
            nfirst_tokens_to_skip=self.cell_embs_count,
            dropout=dropout,
            zinb=zinb,
        )
        self.cls_decoders = torch.nn.ModuleDict()
        for clss, n_cls in classes.items():
            self.cls_decoders[clss] = decoders.ClsDecoder(
                d_model, n_cls, layers=layers_cls, dropout=dropout
            )

        if num_batch_labels > 0:
            self.grad_reverse_discriminator_loss = loss.AdversarialDiscriminatorLoss(
                d_model,
                n_cls=num_batch_labels,
            )
        else:
            self.grad_reverse_discriminator_loss = None

        if mvc_decoder != "None":
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder,
            )
        else:
            self.mvc_decoder = None

        self.apply(
            partial(
                utils._init_weights,
                n_layer=nlayers,
            )
        )
        for i, dec in self.cls_decoders.items():
            torch.nn.init.constant_(dec.out_layer.bias, -0.13)

        if ckpt_path is not None:
            self.init_scprint(ckpt_path)
        print(self)
        self.save_hyperparameters()

    def init_scprint(self, ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}, pwd: {os.getcwd()}")
        checkpoints = torch.load(ckpt_path, map_location="cpu")

        ckpt_genes = checkpoints["hyper_parameters"]["genes"]
        if len(ckpt_genes) != len(self.genes):
            ckpt_gene2index = {gene: i for i, gene in enumerate(ckpt_genes)}
            model_gene2index = {gene: i for i, gene in enumerate(self.genes)}

            model_index = []
            ckpt_index = []
            missing_genes = []
            for i, gene in enumerate(self.genes):
                if gene not in ckpt_gene2index:
                    missing_genes.append(gene)
                    continue
                model_index.append(i)
                ckpt_index.append(ckpt_gene2index[gene])
            print(f"missing genes: {missing_genes}")
            print(f"number of genes found in the checkpoint: {len(ckpt_genes)}")

            device = self.gene_encoder.embedding.weight.device
            model_index = torch.tensor(model_index, device=device)
            ckpt_index = torch.tensor(ckpt_index, device=device)

            self.gene_encoder.embedding.weight.data[model_index, ...] = checkpoints[
                "state_dict"
            ]["gene_encoder.embedding.weight"][ckpt_index, ...]
            self.pos_encoder.pe[model_index, ...] = checkpoints["state_dict"][
                "pos_encoder.pe"
            ][ckpt_index, ...]
            del checkpoints["state_dict"]["gene_encoder.embedding.weight"]
            del checkpoints["state_dict"]["pos_encoder.pe"]
            print(
                "Deleted gene_encoder.embedding.weight and pos_encoder.pe from state_dict."
            )

        if (
            self.class_encoder.embedding.weight.shape[0]
            != checkpoints["state_dict"]["class_encoder.embedding.weight"].shape[0]
        ):
            del checkpoints["state_dict"]["class_encoder.embedding.weight"]
            print("Deleted class_encoder.embedding.weight from state_dict.")

        state_dict = checkpoints["state_dict"]
        for key in list(state_dict.keys()):
            if key.startswith("cls_decoders") or key.startswith(
                "grad_reverse_discriminator_loss"
            ):
                del state_dict[key]
        result = self.load_state_dict(state_dict, strict=False)

        print(f"Loaded checkpoint from {ckpt_path}")

    def on_load_checkpoint(self, checkpoints):

        ckpt_genes = checkpoints["hyper_parameters"]["genes"]
        if len(ckpt_genes) != len(self.genes):
            ckpt_gene2index = {gene: i for i, gene in enumerate(ckpt_genes)}
            model_gene2index = {gene: i for i, gene in enumerate(self.genes)}

            model_index = []
            ckpt_index = []
            missing_genes = []
            for i, gene in enumerate(self.genes):
                if gene not in ckpt_gene2index:
                    missing_genes.append(gene)
                    continue
                model_index.append(i)
                ckpt_index.append(ckpt_gene2index[gene])
            print(f"missing genes: {missing_genes}")
            print(f"number of genes found in the checkpoint: {len(ckpt_genes)}")

            device = self.gene_encoder.embedding.weight.device
            model_index = torch.tensor(model_index, device=device)
            ckpt_index = torch.tensor(ckpt_index, device=device)

            self.gene_encoder.embedding.weight.data[model_index, ...] = checkpoints[
                "state_dict"
            ]["gene_encoder.embedding.weight"][ckpt_index, ...]
            self.pos_encoder.pe[model_index, ...] = checkpoints["state_dict"][
                "pos_encoder.pe"
            ][ckpt_index, ...]
            del checkpoints["state_dict"]["gene_encoder.embedding.weight"]
            del checkpoints["state_dict"]["pos_encoder.pe"]
            print(
                "Deleted gene_encoder.embedding.weight and pos_encoder.pe from state_dict."
            )

        if (
            self.class_encoder.embedding.weight.shape[0]
            != checkpoints["state_dict"]["class_encoder.embedding.weight"].shape[0]
        ):
            del checkpoints["state_dict"]["class_encoder.embedding.weight"]
            print("Deleted class_encoder.embedding.weight from state_dict.")

        for name, clss in self.cls_decoders.items():
            key = f"cls_decoders.{name}.out_layer.weight"
            if key in checkpoints["state_dict"]:
                size = checkpoints["state_dict"][key].shape[0]
            else:
                print(
                    f"Warning: {key} not found in checkpoint when loading cls_decoders"
                )
                size = None
            if size is not None and size != clss.out_layer.bias.shape[0]:
                self.cls_decoders[name].out_layer = torch.nn.Linear(
                    clss.out_layer.weight.shape[1], size
                )
        key = "grad_reverse_discriminator_loss.out_layer.bias"
        if key in checkpoints["state_dict"]:
            size = checkpoints["state_dict"][key].shape[0]
        else:
            print(
                f"Warning: {key} not found in checkpoint when loading grad_reverse_discriminator_loss"
            )
            size = None
        if (size is not None) and (
            self.grad_reverse_discriminator_loss is None
            or size > self.grad_reverse_discriminator_loss.out_layer.bias.shape[0]
        ):
            self.grad_reverse_discriminator_loss = loss.AdversarialDiscriminatorLoss(
                self.d_model,
                n_cls=size,
            )
            print(f"grad_reverse_discriminator_loss size: {size}")
        else:
            print(f"unchanged grad_reverse_discriminator_loss size: {size}")
        classes = checkpoints["hyper_parameters"]["classes"]
        self.normalization = checkpoints["hyper_parameters"]["normalization"]
        self.label_decoders = checkpoints["hyper_parameters"]["label_decoders"]
        self.labels_hierarchy = checkpoints["hyper_parameters"]["labels_hierarchy"]
        for k, v in self.labels_hierarchy.items():
            tens = torch.zeros((len(v), classes[k]))
            for k2, v2 in v.items():
                tens[k2 - classes[k], v2] = 1
            self.mat_labels_hierarchy[k] = tens.to(bool)

        mencoders = {}
        try:
            if self.trainer.datamodule.decoders != self.label_decoders:
                for k, v in checkpoints["hyper_parameters"]["label_decoders"].items():
                    mencoders[k] = {va: ke for ke, va in v.items()}
                self.trainer.datamodule.dataset.mapped_dataset.encoders = mencoders
                if (
                    self.trainer.datamodule.kwargs["collate_fn"].organism_name
                    in mencoders
                ):
                    self.trainer.datamodule.kwargs["collate_fn"]._setup(
                        org_to_id=mencoders[
                            self.trainer.datamodule.kwargs["collate_fn"].organism_name
                        ],
                        valid_genes=self.genes,
                    )
        except RuntimeError as e:
            import traceback

            if "scPrint is not attached to a `Trainer`." in str(e):
                print("RuntimeError caught: scPrint is not attached to a `Trainer`.")
        if not is_interactive():
            self.save_hyperparameters()

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,
    ):
        enc = self.gene_encoder(gene_pos)
        self.cur_gene_token_embs = enc.clone()

        if expression is not None:
            if self.normalization == "sum":
                enc += self.expr_encoder(
                    (expression / torch.clamp(expression.sum(1).unsqueeze(1), min=1)),
                    mask,
                )
            elif self.normalization == "log":
                enc += self.expr_encoder(
                    torch.log2(1 + expression), mask
                )
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")

        if self.gene_pos_enc:
            enc += self.pos_encoder(gene_pos)
        cell_embs = (
            self.class_encoder(
                torch.Tensor(
                    [list(range(self.cell_embs_count - 1))] * gene_pos.shape[0]
                )
                .int()
                .to(gene_pos.device)
            )
            if cell_embs is None
            else cell_embs
        )
        if timepoint is not None:
            pass
        if full_depth is not None:
            depth_encoded = self.depth_encoder(torch.log2(1 + full_depth)).unsqueeze(1)
            cell_embs = torch.cat(
                (cell_embs[:, :1, :], depth_encoded, cell_embs[:, 1:, :]), dim=1
            )

        enc = torch.cat([cell_embs, enc], dim=1)
        return enc

    def _decoder(
        self,
        transformer_output,
        depth_mult,
        get_gene_emb=False,
        do_sample=False,
        do_mvc=False,
        do_class=False,
    ):
        output = self.expr_decoder(transformer_output)

        output["mean"] = depth_mult.unsqueeze(1) * output["mean"]
        if do_sample:
            pass

        output["cell_embs"] = self.get_cell_embs(transformer_output)
        output["cell_emb"] = torch.mean(output["cell_embs"].clone(), dim=1)
        if len(self.classes) > 0 and do_class:
            output.update(
                {
                    "cls_output_"
                    + clsname: self.cls_decoders[clsname](
                        output["cell_embs"][
                            :, 2 + i, :
                        ]
                    )
                    for i, clsname in enumerate(self.classes)
                }
            )
        if do_mvc:
            output.update(
                self.mvc_decoder(output["cell_emb"], self.cur_gene_token_embs)
            )
            output["mvc_mean"] = (
                depth_mult.unsqueeze(1) * output["mvc_mean"]
            )

        if get_gene_emb:
            output["gene_embedding"] = transformer_output[
                :, self.cell_embs_count :, :
            ]
        return output

    def _get_image_outputs(self, image):
        if isinstance(self.clip_model, CLIPVisionModel):
            outputs = self.clip_model(image)
        elif isinstance(self.clip_model, ViTMAEForPreTraining):
            outputs = self.clip_model.vit(image)
        else:
            raise ValueError(f"Unknown clip model type: {type(self.clip_model)}")
        return outputs

    def _get_recon_loss(self, image, outputs):
        assert isinstance(
            self.clip_model, ViTMAEForPreTraining
        ), "Only ViTMAEForPreTraining supports reconstruction loss"

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.clip_model.decoder(
            latent, ids_restore, interpolate_pos_encoding=False
        )
        logits = (
            decoder_outputs.logits
        )

        loss = self.clip_model.forward_loss(
            image, logits, mask, interpolate_pos_encoding=False
        )
        return loss

    def _fuse(self, query, keyvalue, combine_weight, layer=None, mask=None):
        if layer is None:
            layer = self.fusion_layer
        keyvalue = layer(
            query,
            src_key_padding_mask=mask,
            structure_feature=keyvalue,
        )
        query = query + combine_weight * keyvalue
        return query

    def _fuse_image(self, expression_latent, image, combine_weight=None, mask=None):
        if combine_weight is None:
            combine_weight = self.combine_weight

        if expression_latent.shape[-1] != image.shape[-1]:
            image_outputs = self._get_image_outputs(image)
            image_latent = image_outputs.last_hidden_state
            image_latent = self.projection_layer(image_latent)
        else:
            image_latent = image

        expression_latent = self._fuse(
            expression_latent,
            image_latent,
            combine_weight,
        )
        return expression_latent

    def forward(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,
        get_gene_emb: bool = False,
        depth_mult: Optional[Tensor] = None,
        do_sample: bool = False,
        do_mvc: bool = False,
        do_class: bool = False,
        get_attention_layer: list = [],
    ):


        encoding = self._encoder(gene_pos, expression, mask, full_depth, timepoint)

        if self.attn_bias != "none":
            if not hasattr(self, "nbias"):
                self.nbias = torch.Tensor(
                    load_npz(FILEDIR + "/../../data/bias_sparse.npz").todense()
                ).to(device=gene_pos.device, dtype=torch.float16)
            num = len(self.classes) + 2
            bias = torch.zeros(
                (
                    gene_pos.shape[0],
                    gene_pos.shape[1] + num,
                    gene_pos.shape[1] + num,
                ),
                device=gene_pos.device,
                dtype=torch.float16,
            )
            bias[:, num:, :num] = -10_000
            bias[:, num:, num:] = self.nbias[gene_pos[:, :, None], gene_pos[:, None, :]]
        transformer_output = self.transformer(
            encoding,
            return_qkv=get_attention_layer,
            bias=bias if self.attn_bias != "none" else None,
            bias_layer=list(range(self.nlayers - 1)),
        )

        depth_mult = expression.sum(1) if depth_mult is None else depth_mult
        qkvs = None
        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output

        image_outputs = self._get_image_outputs(image)

        image_latent = image_outputs.last_hidden_state
        image_latent = self.projection_layer(image_latent)
        expression_latent = transformer_output

        transformer_output = self._fuse(
            expression_latent,
            image_latent,
            self.combine_weight,
        )

        if isinstance(self.clip_model, ViTMAEForPreTraining):
            image_outputs.last_hidden_state = self._fuse(
                image_outputs.last_hidden_state,
                self.expression2image(expression_latent),
                self.image_combine_weight,
                self.image_fusion_layer,
            )
            recon_loss = self._get_recon_loss(image, image_outputs)
        else:
            recon_loss = None

        decoder_output = self._decoder(
            transformer_output,
            depth_mult,
            get_gene_emb,
            do_sample,
            do_mvc,
            do_class,
        )
        decoder_output["image_embedding"] = image_latent

        if recon_loss is not None:
            decoder_output["recon_loss"] = recon_loss

        if qkvs is not None:
            return decoder_output, qkvs
        return decoder_output

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "adamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.weight_decay,
                amsgrad=False,
                fused=self.fused_adam,
            )
        elif self.optim == "galore":
            raise NotImplementedError("Galore optimizer not implemented")
        else:
            raise ValueError(f"Unknown optimizer: {self.optim}")
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_reduce_patience,
            factor=self.lr_reduce_factor,
            verbose=True,
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": self.lr_reduce_monitor,
        }
        self.lrfinder_steps = 0
        for val in self.trainer.callbacks:
            if type(val) is _LRCallback:
                self.lrfinder_steps = val.num_training
            if type(val) is LearningRateFinder:
                self.lrfinder_steps = val._num_training_steps
        return [optimizer], [lr_dict]

    def on_fit_start(self):
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(True)
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx,
    ):
        total_loss, losses = self._full_training(
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_sim=self.cce_sim,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_adv_batch=self.do_adv_batch,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log_dict(losses, prog_bar=True, sync_dist=True)
        return total_loss

    def _full_training(
        self,
        batch: Dict[str, Tensor],
        do_denoise: bool = False,
        noise: list[float] = [],
        do_next_tp: bool = False,
        do_cce: bool = False,
        cce_sim: float = 0.5,
        do_ecs: bool = False,
        do_mvc: bool = False,
        do_adv_cls: bool = False,
        do_adv_batch: bool = False,
        do_cls: bool = False,
        do_generate: bool = False,
        run_full_forward: bool = True,
        mask_ratio: list[float] = [0.15],
    ):
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio]

        expression = batch["x"]
        gene_pos = batch["genes"]
        total_count = batch["depth"]
        clss = batch.get("class", None)
        batch_idx = batch.get("dataset", None)
        image = batch.get("image", None)

        total_loss = 0
        losses = {}
        cell_embs = []
        if run_full_forward:
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                mask=None,
                full_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            output.pop("disp")
            output.pop("zero_logits")
            output.pop("mean")
            l, tot = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            cell_embs.append(output["cell_emb"].clone())
            full_cell_embs = output["cell_embs"].clone()
            total_loss += tot
            losses.update({"full_forward_" + k: v for k, v in l.items()})
            if "recon_loss" in output:
                losses["recon_loss"] = output["recon_loss"]
                total_loss += output["recon_loss"] * self.image_recon_loss_weight
            do_mvc = False if do_mvc else do_mvc
            do_cls = False if do_cls else do_cls

        for i in mask_ratio:
            mask = simple_masker(
                shape=gene_pos.shape,
                mask_ratio=i,
            ).to(gene_pos.device)
            output = self.forward(
                gene_pos,
                expression=expression,
                image=image,
                mask=mask,
                full_depth=total_count,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            l, tot = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            do_mvc = False if do_mvc else do_mvc
            do_cls = False if do_cls else do_cls

            cell_embs.append(output["cell_emb"].clone())
            total_loss += tot
            losses.update(
                {"mask_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
            )
            if "recon_loss" in output:
                losses["recon_loss"] = output["recon_loss"]
                total_loss += output["recon_loss"] * self.image_recon_loss_weight

        if do_denoise:
            for i in noise:
                expr = utils.downsample_profile(expression, dropout=i)
                output = self.forward(
                    gene_pos,
                    expression=expr,
                    image=image,
                    mask=None,
                    depth_mult=expression.sum(1),
                    full_depth=total_count,
                    do_mvc=do_mvc,
                    do_class=do_cls,
                )
                l, tot = self._compute_loss(
                    output,
                    expression,
                    clss,
                    batch_idx,
                    do_ecs,
                    do_adv_cls & do_cls,
                    do_adv_batch & do_cls,
                )
                do_mvc = False if do_mvc else do_mvc
                do_cls = False if do_cls else do_cls

                cell_embs.append(output["cell_emb"].clone())
                total_loss += tot
                losses.update(
                    {"denoise_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
                )
                if "recon_loss" in output:
                    losses["recon_loss"] = output["recon_loss"]
                    total_loss += output["recon_loss"] * self.image_recon_loss_weight

        if do_generate:
            output = self._generate(
                output["cell_embs"] if not run_full_forward else full_cell_embs,
                gene_pos,
                depth_mult=expression.sum(1),
                image=image,
                full_depth=None,
                do_mvc=do_mvc,
                do_class=do_cls,
            )
            cell_embs.append(output["cell_emb"].clone())
            l, tloss = self._compute_loss(
                output,
                expression,
                clss,
                batch_idx,
                do_ecs,
                do_adv_cls & do_cls,
                do_adv_batch & do_cls,
            )
            losses.update({"gen_" + k: v for k, v in l.items()})
            total_loss += tloss

        if do_next_tp:
            pass

        if do_cce:
            loss_cce = 0
            for i, cell_emb1 in enumerate(cell_embs[:-1]):
                for cell_emb2 in cell_embs[(i + 1) :]:
                    loss_cce += loss.similarity(
                        cell_emb1, cell_emb2, cce_sim
                    )
            fact = factorial(len(cell_embs))
            total_loss += loss_cce * self.cce_scale / fact
            losses.update({"cce": loss_cce / fact})

        return total_loss, losses

    def _compute_loss(
        self,
        output,
        expression,
        clss,
        batch_idx,
        do_ecs=False,
        do_cls=False,
        do_adv_cls=False,
        do_adv_batch=False,
        do_mse=0,
    ):
        total_loss = 0
        losses = {}
        if "zero_logits" in output:
            loss_expr = loss.zinb(
                theta=output["disp"],
                pi=output["zero_logits"],
                mu=output["mean"],
                target=expression,
            )
            if do_mse:
                loss_expr += loss.mse(
                    input=torch.log(output["mean"] + 1)
                    * (1 - torch.sigmoid(output["zero_logits"])),
                    target=torch.log(expression + 1),
                )
        elif "disp" in output:
            loss_expr = loss.nb(
                theta=output["disp"],
                mu=output["mean"],
                target=expression,
            )
        elif "mean" in output:
            loss_expr = loss.mse(
                input=output["mean"],
                target=expression,
            )
        else:
            loss_expr = 0
        total_loss += loss_expr
        losses.update({"expr": loss_expr})

        if do_cls and len(self.classes) > 0:
            cos_sim_matrix = (
                torch.nn.functional.cosine_similarity(
                    output["cell_embs"].unsqueeze(2),
                    output["cell_embs"].unsqueeze(1),
                    dim=3,
                )
                .abs()
                .mean(0)
            )
            loss_class_emb_diss = cos_sim_matrix.fill_diagonal_(0).mean()
            losses.update({"class_emb_sim": loss_class_emb_diss})
            total_loss += self.class_embd_diss_scale * loss_class_emb_diss
            loss_cls = 0
            loss_adv_cls = 0
            for j, clsname in enumerate(self.classes):
                if "cls_output_" + clsname not in output:
                    continue
                loss_cls += loss.classification(
                    clsname,
                    pred=output["cls_output_" + clsname],
                    cl=clss[:, j],
                    maxsize=self.label_counts[clsname],
                    labels_hierarchy=self.mat_labels_hierarchy,
                )
            total_loss += self.class_scale * loss_cls
            if loss_cls != 0:
                losses.update({"cls": loss_cls})
            if do_adv_cls:
                embs = output["cell_embs"][:, 2:, :].clone()
                for j, adv_cls in enumerate(self.classes):
                    ind = torch.arange(len(self.classes))
                    mean_embs = torch.mean(embs[:, ind != j, :], dim=1)
                    mean_embs = grad_reverse(mean_embs, lambd=1.0)
                    adv_pred = self.cls_decoders[adv_cls](mean_embs)
                    loss_adv_cls += loss.classification(
                        adv_cls,
                        pred=adv_pred,
                        cl=clss[:, j],
                        maxsize=self.label_counts[adv_cls],
                        labels_hierarchy=self.mat_labels_hierarchy,
                    )

                total_loss += self.adv_class_scale * loss_adv_cls
                losses.update({"adv_cls": loss_adv_cls})

        if (
            do_adv_batch
            and self.grad_reverse_discriminator_loss is not None
            and batch_idx is not None
        ):
            mean_emb = torch.mean(output["cell_embs"][:, 2:, :].clone(), dim=1)
            loss_adv = self.grad_reverse_discriminator_loss(mean_emb, batch_idx)
            total_loss += loss_adv * self.class_scale / 16
            losses.update({"adv_batch": loss_adv})
        if "mvc_disp" in output:
            loss_expr_mvc = loss.zinb(
                theta=output["mvc_disp"],
                pi=output["mvc_zero_logits"],
                mu=output["mvc_mean"],
                target=expression,
            )
            total_loss += loss_expr_mvc * self.mvc_scale
            losses.update({"expr_mvc": loss_expr_mvc})
        if do_ecs:
            loss_ecs = loss.ecs(output["cell_emb"], ecs_threshold=self.ecs_threshold)
            total_loss += self.ecs_scale * loss_ecs
            losses.update({"ecs": loss_ecs})
        return losses, total_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)

        for i, pg in enumerate(optimizer.param_groups):
            if (
                self.global_step < self.warmup_duration + self.lrfinder_steps
            ) and self.lrfinder_steps < self.global_step:
                lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_duration)
                pg["lr"] = lr_scale * self.hparams.lr
        for i, pg in enumerate(optimizer.param_groups):
            self.log("lr_" + str(i), pg["lr"])

    def on_validation_start(self):
        for k, v in self.mat_labels_hierarchy.items():
            self.mat_labels_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):
        self.embs = None
        self.counter = 0

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        val_loss, losses = self._full_training(
            batch=batch,
            do_denoise=self.do_denoise,
            noise=self.noise,
            do_next_tp=self.do_next_tp,
            do_cce=self.do_cce,
            cce_sim=self.cce_sim,
            do_ecs=self.do_ecs,
            do_mvc=self.do_mvc,
            do_adv_cls=self.do_adv_cls,
            do_adv_batch=self.do_adv_batch,
            do_cls=self.do_cls,
            do_generate=self.do_generate,
            run_full_forward=self.run_full_forward,
            mask_ratio=self.mask_ratio,
        )
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        image = batch["image"]
        if self.embs is not None:
            if self.embs.shape[0] < 100_000:
                self.info = torch.cat([self.info, batch["class"]])
                self._predict(
                    gene_pos,
                    expression,
                    depth,
                    image=image,
                    pred_embedding=self.pred_embedding,
                    max_size_in_mem=1_000_000,
                )
        else:
            self.info = batch["class"]
            self._predict(
                gene_pos,
                expression,
                depth,
                image=image,
                pred_embedding=self.pred_embedding,
                max_size_in_mem=1_000_000,
            )
        self.log("val_loss", val_loss, sync_dist=True)
        self.log_dict(losses, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        self.embs = self.all_gather(self.embs).view(-1, self.embs.shape[-1])
        self.info = self.all_gather(self.info).view(-1, self.info.shape[-1])
        self.pred = (
            self.all_gather(self.pred).view(-1, self.pred.shape[-1])
            if self.pred is not None
            else None
        )
        self.pos = self.all_gather(self.pos).view(-1, self.pos.shape[-1])
        if not self.trainer.is_global_zero:
            return
        if self.trainer.state.stage != "sanity_check":
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics["val_loss"])
            self.log_adata(
                gtclass=self.info, name="validation_part_" + str(self.counter)
            )
            if (self.current_epoch + 1) % self.test_every == 0:
                self.on_test_epoch_end()

    def test_step(self, *args, **kwargs):
        print("step")
        pass

    def on_test_epoch_end(self):
        print("start test")
        model_copy = copy.deepcopy(self)
        name = self.name + "_step" + str(self.global_step)
        metrics = utils.test(model_copy, name, filedir=FILEDIR)
        print(metrics)
        print("done test")
        self.log_dict(metrics, sync_dist=True, rank_zero_only=True)

    def on_predict_epoch_start(self):
        self.embs = None
        self.attn.data = None
        self.attn.attn = None
        self.counter = 0
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)

    def predict_step(self, batch, batch_idx):
        return self._predict(
            batch["genes"],
            batch["x"],
            batch["depth"],
            batch["image"],
            self.predict_mode,
            self.pred_embedding,
            self.get_attention_layer,
            self.predict_depth_mult,
        )

    def _predict(
        self,
        gene_pos,
        expression,
        depth,
        image=None,
        predict_mode="none",
        pred_embedding=[],
        get_attention_layer=[],
        depth_mult=6,
        keep_output=True,
        max_size_in_mem=1_000_000,
    ):
        if predict_mode == "none":
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                depth_mult=expression.sum(1),
                full_depth=depth,
                get_attention_layer=get_attention_layer,
                do_class=True,
            )
            if len(get_attention_layer) > 0:
                self.attn.add([i[:, :, :2, :] for i in output[1]], gene_pos)
                output = output[0]
            cell_embs = output["cell_embs"]
            image_embs = output.get("image_embedding", None)
        elif predict_mode == "denoise":
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                depth_mult=expression.sum(1) * depth_mult,
                full_depth=depth * depth_mult,
                get_attention_layer=get_attention_layer,
                do_class=True,
            )
            if len(get_attention_layer) > 0:
                self.attn.add([i[:, :, :2, :] for i in output[1]], gene_pos)
                output = output[0]
            cell_embs = output["cell_embs"]
            image_embs = None
        elif predict_mode == "generate":
            output = self.forward(
                gene_pos,
                expression,
                image=image,
                full_depth=depth,
                do_mvc=False,
                do_class=False,
            )
            cell_embs = output["cell_embs"]
            output = self._generate(
                output["cell_embs"],
                gene_pos,
                image=image,
                full_depth=None,
                depth_mult=expression.sum(1),
                do_class=self.do_cls,
                do_mvc=False,
            )
            image_embs = None
        else:
            raise ValueError(
                "predict_mode needs to be one of ['none', 'denoise', 'generate']"
            )

        if len(pred_embedding) == 0:
            pred_embedding = self.classes
        ind = [self.classes.index(i) + 2 for i in pred_embedding]
        embs = (
            torch.mean(cell_embs, dim=1)
            if image_embs is None
            else (
                torch.concat(
                    [
                        torch.mean(cell_embs, dim=1),
                        (
                            image_embs
                            if image_embs.ndim == 2
                            else torch.mean(image_embs, dim=1)
                        ),
                    ],
                    dim=1,
                )
            )
        )
        if not keep_output:
            return {
                "embs": embs,
                "class": (
                    torch.stack(
                        [
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            for clsname in self.classes
                        ]
                    ).transpose(0, 1)
                    if len(self.classes) > 0
                    else None
                ),
                "pos": gene_pos,
                "expr": (
                    [output["mean"], output["disp"], output["zero_logits"]]
                    if "disp" in output
                    else [output["mean"]]
                ),
            }
        if self.embs is None:
            self.embs = embs
            self.pred = (
                torch.stack(
                    [
                        (
                            torch.argmax(output["cls_output_" + clsname], dim=1)
                            if not self.keep_all_cls_pred
                            else output["cls_output_" + clsname]
                        )
                        for clsname in self.classes
                    ]
                ).transpose(0, 1)
                if len(self.classes) > 0
                else None
            )
            self.pos = gene_pos
            self.expr_pred = (
                [output["mean"], output["disp"], output["zero_logits"]]
                if "disp" in output
                else [output["mean"]]
            )
        else:
            self.embs = torch.cat(
                [self.embs, embs]
            )
            self.pred = torch.cat(
                [
                    self.pred,
                    (
                        torch.stack(
                            [
                                (
                                    torch.argmax(output["cls_output_" + clsname], dim=1)
                                    if not self.keep_all_cls_pred
                                    else output["cls_output_" + clsname]
                                )
                                for clsname in self.classes
                            ]
                        ).transpose(0, 1)
                        if len(self.classes) > 0
                        else None
                    ),
                ],
            )
            self.pos = torch.cat([self.pos, gene_pos])
            self.expr_pred = (
                [
                    torch.cat([self.expr_pred[0], output["mean"]]),
                    torch.cat([self.expr_pred[1], output["disp"]]),
                    torch.cat([self.expr_pred[2], output["zero_logits"]]),
                ]
                if "disp" in output
                else [torch.cat([self.expr_pred[0], output["mean"]])]
            )
        if self.embs is not None:
            if self.embs.shape[0] > max_size_in_mem:
                raise MemoryError(
                    "The size of the embeddings exceeds the maximum limit. Please reduce the batch size or use a smaller model."
                )
                print("logging")
                self.log_adata(name="predict_part_" + str(self.counter))
                self.counter += 1
                self.pos = None
                self.expr_pred = None
                self.pred = None
                self.embs = None

    def on_predict_epoch_end(self):
        if self.pos.shape[0] < 100:
            return
        if self.pred_log_adata:
            print("adding on disk")
            return self.log_adata(name="predict_part_" + str(self.counter))

    def get_cell_embs(self, layer_output):
        if self.cell_emb_style == "cls" and self.classes is not None:
            cell_emb = layer_output[:, : 2 + len(self.classes)]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")
        return cell_emb

    def _generate(
        self,
        cell_embs: Tensor,
        gene_pos: Tensor,
        depth_mult: Tensor,
        image: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        tp: Optional[Tensor] = None,
        gen_iters: int = 1,
        **decoder_kwargs,
    ):
        if tp is not None:
            tp = tp / gen_iters
        for i in range(gen_iters):
            encoding = self._encoder(
                cell_embs=cell_embs,
                gene_pos=gene_pos,
                full_depth=full_depth,
                timepoint=tp * (i + 1) if tp is not None else None,
            )
            transformer_output = self.transformer(encoding)
            transformer_output = self._fuse_image(transformer_output, image)
            cell_embs = self.get_cell_embs(transformer_output)
        output = self._decoder(
            transformer_output, depth_mult=depth_mult, **decoder_kwargs
        )
        return output

    def get_adata(self, doplot=False):
        if doplot is None:
            doplot = self.doplot
        adata, fig = utils.make_adata(
            self.embs,
            self.classes,
            self.pred if not self.keep_all_cls_pred else None,
            self.attn.get(),
            self.global_step,
            self.label_decoders,
            self.labels_hierarchy,
            gtclass=None,
            name=None,
            mdir=None,
            doplot=doplot,
        )
        if doplot:
            try:
                self.logger.experiment.add_figure(fig)
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig])
            except:
                print("couldn't log to wandb")

        return adata

    def log_adata(self, gtclass=None, name=""):
        try:
            mdir = self.logger.save_dir if self.logger.save_dir is not None else "/tmp"
        except:
            mdir = "data/"
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        adata, fig = utils.make_adata(
            self.embs,
            self.classes,
            self.pred if not self.keep_all_cls_pred else None,
            self.attn.get(),
            self.global_step,
            self.label_decoders,
            self.labels_hierarchy,
            gtclass,
            self.name + "_" + name + "_" + str(self.global_rank),
            mdir,
            self.doplot,
        )
        if self.doplot:
            try:
                self.logger.experiment.add_figure(fig)
            except:
                print("couldn't log to tensorboard")
            try:
                self.logger.log_image(key="umaps", images=[fig])
            except:
                print("couldn't log to wandb")

        return adata

    def _predict_denoised_expression(self, gene_pos, expression, depth):
        output = self.forward(gene_pos, expression, full_depth=depth)
        return output
