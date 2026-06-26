import gc
import math
import warnings
from typing import *

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
try:
    from flash_attn.flash_attention import FlashMHA
    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash attention available")
except ImportError as e:
    print(f"⚠️  Flash attention not available: {e}")
    print("Using standard PyTorch attention (flash-attn disabled for compatibility)")
    FLASH_ATTN_AVAILABLE = False
    FlashMHA = None
from torch import Tensor, nn
from torch.distributions import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import trange
from transformers import CLIPVisionModel, ViTMAEForPreTraining

from .esm_layers import FusionLayer
try:
    from .flash_layers import FlashscGPTGenerator, FlashscGPTLayer
    FLASH_LAYERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Flash layers not available: {e}")
    FLASH_LAYERS_AVAILABLE = False
    FlashscGPTGenerator = None
    FlashscGPTLayer = None
from .grad_reverse import grad_reverse
from .MoE import MoELayer


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.8,
        explicit_zero_prob: bool = False,
        use_generative_training=False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        use_MVC_impute: bool = False,
        impute_MVC_knn_k: Optional[int] = None,
        use_moe_dec: bool = False,
        image_encoder_cls: Optional[
            Literal["openai/clip-vit-base-patch32", "facebook/vit-mae-base"]
        ] = None,
        combine_weight: float = 1,
        image_combine_weight: float = 1,
        image_recon_loss_weight: float = 1,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        self.use_MVC_impute = use_MVC_impute
        self.impute_MVC_knn_k = impute_MVC_knn_k
        self.use_moe_dec = use_moe_dec

        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.flag_encoder = nn.Embedding(2, d_model)

        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        if use_generative_training:
            if FLASH_LAYERS_AVAILABLE:
                encoder_layers = FlashscGPTLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = FlashscGPTGenerator(encoder_layers, nlayers)
            else:
                print("⚠️  Flash layers not available, falling back to standard transformer for generative training")
                encoder_layers = TransformerEncoderLayer(
                    d_model, nhead, d_hid, dropout, batch_first=True
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        elif use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                if FLASH_ATTN_AVAILABLE:
                    encoder_layers = FlashTransformerEncoderLayer(
                        d_model,
                        nhead,
                        d_hid,
                        dropout,
                        batch_first=True,
                        norm_scheme=self.norm_scheme,
                    )
                    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
                else:
                    print("⚠️  Flash layers not available, falling back to standard transformer")
                    encoder_layers = TransformerEncoderLayer(
                        d_model, nhead, d_hid, dropout, batch_first=True
                    )
                    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if self.use_moe_dec:
            self.decoder = MoeDecoder(
                d_model,
                num_experts=4,
                use_batch_labels=use_batch_labels,
            )
        else:
            self.decoder = ExprDecoder(
                d_model,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if n_cls > 1:
            self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )

        if use_MVC_impute:
            self.impute_mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        self.init_weights()

        self.image_encoder_cls = image_encoder_cls
        if self.image_encoder_cls is not None:
            if "clip" in self.image_encoder_cls:
                self.image_encoder = CLIPVisionModel.from_pretrained(
                    self.image_encoder_cls
                ).train()
            elif "mae" in self.image_encoder_cls:
                self.image_encoder = ViTMAEForPreTraining.from_pretrained(
                    self.image_encoder_cls
                ).train()
            else:
                raise ValueError(
                    f"Unknown image encoder class: {self.image_encoder_cls}"
                )

            self.combine_weight = combine_weight
            if isinstance(self.image_encoder, ViTMAEForPreTraining):
                self.image_combine_weight = image_combine_weight
                self.image_recon_loss_weight = image_recon_loss_weight
                self.expression2image = nn.Linear(
                    d_model, self.image_encoder.config.hidden_size
                )
                self.image_fusion_layer = FusionLayer(
                    self.image_encoder.config.hidden_size, nhead=nhead
                )
            elif isinstance(self.image_encoder, CLIPVisionModel):
                self.image_combine_weight = None
                self.image_recon_loss_weight = None
                self.expression2image = None
            self.fusion_layer = FusionLayer(d_model, nhead=nhead)
            self.projection_layer = nn.Linear(
                self.image_encoder.config.hidden_size, self.d_model
            )

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _get_image_outputs(self, image):
        if isinstance(self.image_encoder, CLIPVisionModel):
            outputs = self.image_encoder(image)
        elif isinstance(self.image_encoder, ViTMAEForPreTraining):
            outputs = self.image_encoder.vit(image)
        else:
            raise ValueError(f"Unknown clip model type: {type(self.image_encoder)}")
        return outputs

    def _get_recon_loss(self, image, outputs):
        assert isinstance(
            self.image_encoder, ViTMAEForPreTraining
        ), "Only ViTMAEForPreTraining supports reconstruction loss"

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        try:
            decoder_outputs = self.image_encoder.decoder(
                latent, ids_restore, interpolate_pos_encoding=False
            )
        except TypeError:
            decoder_outputs = self.image_encoder.decoder(latent, ids_restore)
        logits = (
            decoder_outputs.logits
        )

        try:
            loss = self.image_encoder.forward_loss(
                image, logits, mask, interpolate_pos_encoding=False
            )
        except TypeError:
            loss = self.image_encoder.forward_loss(image, logits, mask)
        return loss * self.image_recon_loss_weight

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
            mask=mask,
        )
        return expression_latent

    def _process_transformer_output(
        self,
        transformer_output: Tensor,
        image: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        if self.image_encoder_cls is not None:
            assert image is not None, "image is required when image_encoder_cls is set"
            image_outputs = self._get_image_outputs(image)

            image_latent = image_outputs.last_hidden_state
            image_latent = self.projection_layer(image_latent)
            expression_latent = transformer_output

            transformer_output = self._fuse(
                expression_latent,
                image_latent,
                self.combine_weight,
            )

            if isinstance(self.image_encoder, ViTMAEForPreTraining):
                image_outputs.last_hidden_state = self._fuse(
                    image_outputs.last_hidden_state,
                    self.expression2image(expression_latent),
                    self.image_combine_weight,
                    self.image_fusion_layer,
                )
                image_latent = image_outputs.last_hidden_state
                recon_loss = self._get_recon_loss(image, image_outputs)
            else:
                recon_loss = 0.0
            return {
                "expression_latent": transformer_output,
                "image_latent": image_latent,
                "recon_loss": recon_loss,
            }
        else:
            return {
                "expression_latent": transformer_output,
                "image_latent": None,
                "recon_loss": 0.0,
            }

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
    ) -> Tensor:
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if FlashscGPTGenerator is not None and isinstance(self.transformer_encoder, FlashscGPTGenerator):
            output, _ = self.transformer_encoder(
                pcpt_total_embs=total_embs,
                gen_total_embs=None,
                pcpt_key_padding_mask=src_key_padding_mask,
                gen_key_padding_mask=None,
            )
        else:
            output = self.transformer_encoder(
                total_embs, src_key_padding_mask=src_key_padding_mask
            )
        return output

    def _encode_spatial(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        image: Optional[Tensor] = None,
    ) -> Tensor:
        transformer_output = self._encode(src, values, src_key_padding_mask)
        outputs = self._process_transformer_output(transformer_output, image=image)
        return outputs

    def transformer_generate(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        self._check_batch_labels(batch_labels)

        pcpt_token_embs = self.encoder(pcpt_genes)
        pcpt_values = self.value_encoder(pcpt_values)
        pcpt_total_embs = pcpt_token_embs + pcpt_values

        assert self.input_emb_style != "scaling"
        if gen_genes is not None:
            gen_token_embs = self.encoder(gen_genes)
            self.cur_gene_token_embs = torch.cat(
                [pcpt_token_embs, gen_token_embs], dim=1
            )
            gen_flags = self.flag_encoder(
                torch.tensor(1).to(pcpt_values.device)
            ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)

            gen_total_embs = gen_token_embs + gen_flags
        else:
            self.cur_gene_token_embs = pcpt_token_embs
            gen_total_embs = None

        if input_cell_emb is not None:
            pcpt_total_embs[:, 0, :] = input_cell_emb

        pcpt_output, gen_output = self.transformer_encoder(
            pcpt_total_embs,
            gen_total_embs,
            pcpt_key_padding_mask=pcpt_key_padding_mask,
            gen_key_padding_mask=gen_key_padding_mask,
        )

        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels` is True"
            )

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        image: Optional[Tensor],
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,
    ) -> Tensor:
        try:
            self._check_batch_labels(batch_labels)
        except:
            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.encoder(src)

        if values is not None:
            values = self.value_encoder(values)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        outputs = self._process_transformer_output(transformer_output, image=image)
        transformer_output = outputs["expression_latent"]

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)
        mlm_output = self.decoder(
            (
                transformer_output
                if not self.use_batch_labels
                else torch.cat(
                    [
                        transformer_output,
                        batch_emb.unsqueeze(1).repeat(
                            1, transformer_output.shape[1], 1
                        ),
                    ],
                    dim=2,
                )
            ),
        )
        output = mlm_output["pred"]
        outputs["pred"] = output

        return outputs

    def _extend_output(
        self,
        output: Mapping[str, Tensor],
        transformer_output: Tensor,
        batch_emb: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        MVC_impute: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:

        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)
        if MVC:
            mvc_output = self.mvc_decoder(
                (
                    cell_emb
                    if not self.use_batch_labels
                    else torch.cat([cell_emb, batch_emb], dim=1)
                ),
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())

            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        if MVC_impute:
            coordinates = output["coordinates"]
            K_NN = self.impute_MVC_knn_k
            dist = torch.cdist(coordinates, coordinates, p=2)

            topk_index = torch.topk(
                dist, k=K_NN + 1, dim=-1, largest=False, sorted=True
            )[1]
            topk_index = topk_index[:, 1:]

            NN_cells = transformer_output[
                topk_index, 0, :
            ]

            cell_emb_mean = NN_cells.mean(1)

            if self.use_batch_labels:
                batch_emb = batch_emb

            out_mvc = self.impute_mvc_decoder(
                (
                    cell_emb_mean
                    if not self.use_batch_labels
                    else torch.cat([cell_emb_mean, batch_emb], dim=1)
                ),
                self.cur_gene_token_embs,
            )

            output["impute_pred"] = out_mvc["pred"]

        return output

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Mapping[str, Tensor]:
        if "generative_training" not in kwargs:
            warnings.warn(
                "generative_training kwarg is required but not provided! "
                "Using False and calling perceptual_forward instead"
            )
            return self.perceptual_forward(*args, **kwargs)

        do_generative_training = kwargs.pop("generative_training")
        if do_generative_training:
            return self.generative_forward(*args, **kwargs)
        else:
            return self.perceptual_forward(*args, **kwargs)

    def generative_forward(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        image: Optional[Tensor],
        batch_labels: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        MVC_impute: bool = False,
        do_sample: bool = False,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:

        pcpt_output, gen_output = self.transformer_generate(
            pcpt_genes,
            pcpt_values,
            pcpt_key_padding_mask,
            gen_genes,
            gen_key_padding_mask,
            batch_labels,
            input_cell_emb=input_cell_emb,
        )
        if gen_output is None:
            transformer_output = pcpt_output
        else:
            transformer_output = torch.cat([pcpt_output, gen_output], dim=1)

        output = self._process_transformer_output(transformer_output, image=image)
        transformer_output = output["expression_latent"]

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)

        decoder_output = self.decoder(
            (
                transformer_output
                if not self.use_batch_labels
                else torch.cat(
                    [
                        transformer_output,
                        batch_emb.unsqueeze(1).repeat(
                            1, transformer_output.shape[1], 1
                        ),
                    ],
                    dim=2,
                )
            ),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=decoder_output["zero_probs"])
            full_preds = bernoulli.sample() * decoder_output["pred"]
            output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
            output["gen_preds"] = full_preds[:, pcpt_genes.shape[1] :]
        else:
            full_preds = decoder_output["pred"]
            output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
            output["gen_preds"] = full_preds[:, pcpt_genes.shape[1] :]
        if self.explicit_zero_prob:
            output["zero_probs"] = decoder_output["zero_probs"]
        if MVC_impute:
            output["coordinates"] = coordinates

        output = self._extend_output(
            output,
            transformer_output,
            batch_emb=batch_emb if self.use_batch_labels else None,
            CLS=CLS,
            MVC=MVC,
            ECS=ECS,
            MVC_impute=MVC_impute,
            do_sample=do_sample,
        )

        return output

    def perceptual_forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        image: Optional[Tensor],
        batch_labels: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        MVC_impute: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        transformer_output = self._encode(
            src, values, src_key_padding_mask, batch_labels
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)

        output = self._process_transformer_output(transformer_output, image=image)
        transformer_output = output["expression_latent"]

        mlm_output = self.decoder(
            (
                transformer_output
                if not self.use_batch_labels
                else torch.cat(
                    [
                        transformer_output,
                        batch_emb.unsqueeze(1).repeat(
                            1, transformer_output.shape[1], 1
                        ),
                    ],
                    dim=2,
                )
            ),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        if MVC_impute:
            output["values"] = values
            output["coordinates"] = coordinates

        output = self._extend_output(
            output,
            transformer_output,
            batch_emb=batch_emb if self.use_batch_labels else None,
            CLS=CLS,
            MVC=MVC,
            ECS=ECS,
            MVC_impute=MVC_impute,
            do_sample=do_sample,
        )

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        image: Optional[Tensor],
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        N = src.size(0)
        device = next(self.parameters()).device

        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                (
                    batch_labels[i : i + batch_size].to(device)
                    if batch_labels is not None
                    else None
                ),
            )

            raw_output = self._process_transformer_output(raw_output, image=image)[
                "expression_latent"
            ]

            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class FastTransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ) -> nn.Module:

        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, "
                f"got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":

        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} "
                f"does not match first two dims of src shape {src.shape[:2]}"
            )

        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask needs to be of type torch.bool, "
                f"got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        output = self.fast_transformer_encoder(src, length_mask=length_mask)
        return output


class FlashTransformerEncoderLayer(nn.Module):

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )

        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pred_value = self.fc(x).squeeze(-1)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)


class MoeDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.moe = MoELayer(d_in, d_model, 1, num_experts)

    def forward(self, x: Tensor, topn: int = 2) -> Dict[str, Tensor]:
        pred_value = self.moe(x, topn).squeeze(-1)
        return dict(pred=pred_value)


class ClsDecoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)


class AdversarialDiscriminator(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
