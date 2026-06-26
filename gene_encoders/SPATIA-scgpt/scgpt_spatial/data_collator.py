from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import ipdb
import numpy as np
import torch

from .preprocess import binning


@dataclass
class DataCollator:

    do_padding: bool = True
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True
    n_bins: int = 51
    mlm_probability: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    reserve_keys: List[str] = field(default_factory=lambda: [])
    append_tokens: List[Callable] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1
    data_style: str = "pcpt"
    cell_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")

        if self.do_binning:
            if self.n_bins < 2:
                raise ValueError("`n_bins` must be greater than 1.")

        if isinstance(self.mlm_probability, float):
            if self.mlm_probability <= 0 or self.mlm_probability >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        elif isinstance(self.mlm_probability, (list, tuple)):
            if min(self.mlm_probability) <= 0 or max(self.mlm_probability) >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        else:
            raise ValueError("`mlm_probability` must be a float or iterable of floats.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if len(self.append_tokens) > 0:
            self.original_prefix_n_tokens = self.keep_first_n_tokens
            self.keep_first_n_tokens = self.keep_first_n_tokens + len(
                self.append_tokens
            )

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

        if self.data_style not in ["pcpt", "gen", "both"]:
            raise ValueError("`data_style` must be one of 'pcpt', 'gen', 'both'.")

        self.cell_types = (
            {key: idx for idx, key in enumerate(self.cell_types)}
            if self.cell_types
            else {"unknown": 0}
        )
        assert (
            self.cell_types is not None
        ), "Please use unknown cell types if not provided."
        assert (
            "unknown" in self.cell_types
        ), "Please ensure 'unknown' is included in the cell types."

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:

        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )
        if "image" in examples[0]:
            self.reserve_keys.append("image")
        if "cell_type" in examples[0]:
            self.reserve_keys.append("cell_type")

        if len(self.append_tokens) > 0:
            for i, func in enumerate(self.append_tokens):
                examples = self._append_token(
                    examples, func, prefix_n_tokens=self.original_prefix_n_tokens + i
                )

        if self.data_style == "pcpt":
            data_dict = self._call_pcpt(examples)
        elif self.data_style == "gen":
            data_dict = self._call_gen(examples)
        elif self.data_style == "both":
            data_dict = self._call_both(examples)

        device = examples[0]["genes"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            if isinstance(data_[0], torch.Tensor):
                data_dict[key] = torch.stack(data_, dim=0).to(device)
            if key == "cell_type":
                data_ = [
                    self.cell_types.get(item, self.cell_types["unknown"])
                    for item in data_
                ]
                data_dict[key] = torch.tensor(data_, dtype=torch.long, device=device)

        return data_dict

    def _append_token(
        self,
        examples: List[Dict[str, torch.Tensor]],
        func: Callable,
        prefix_n_tokens: int,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int]:
        for i in range(len(examples)):
            examples[i] = func(examples[i], prefix_n_tokens)
        return examples

    def _call_pcpt(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        padded_genes = []
        padded_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                try:
                    expressions[self.keep_first_n_tokens :] = binning(
                        row=expressions[self.keep_first_n_tokens :],
                        n_bins=self.n_bins,
                    )
                except ValueError:
                    pass

            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )

            padded_genes.append(genes)
            padded_expressions.append(expressions)

        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)

        data_dict = {
            "gene": padded_genes,
            "expr": padded_expressions,
        }

        if self.do_mlm:
            masked_expressions = self._mask(
                padded_expressions, self.keep_first_n_tokens
            )
        else:
            masked_expressions = padded_expressions
        data_dict["masked_expr"] = masked_expressions

        return data_dict

    def _call_gen(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=self.n_bins,
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )
            padded_pcpt_genes.append(genes)
            padded_pcpt_expressions.append(expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0).to(device)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0).to(device)

        data_dict = {
            "pcpt_gene": padded_pcpt_genes,
            "pcpt_expr": padded_pcpt_expressions,
        }
        return data_dict

    def _call_both(
        self,
        examples: List[Dict[str, torch.Tensor]],
        gen_prob: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(examples[0], Mapping):
            return NotImplementedError

        if not self.do_mlm:
            return self._call_gen(examples)

        if gen_prob is None:
            gen_prob = self.get_mlm_probability()


        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len

        num_genes_to_split_from = _max_length - self.keep_first_n_tokens
        if num_genes_to_split_from < 0:
            num_genes_to_split_from = 0

        gen_length = int(num_genes_to_split_from * gen_prob)
        pcpt_length = _max_length - gen_length

        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        padded_gen_genes = []
        padded_gen_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=self.n_bins,
                )

            genes_to_split = genes[self.keep_first_n_tokens :]
            expressions_to_split = expressions[self.keep_first_n_tokens :]

            current_gen_genes = torch.empty(
                (0,) + genes_to_split.shape[1:],
                dtype=genes_to_split.dtype,
                device=device,
            )
            current_gen_expressions = torch.empty(
                (0,) + expressions_to_split.shape[1:],
                dtype=expressions_to_split.dtype,
                device=device,
            )
            current_pcpt_genes_suffix = torch.empty(
                (0,) + genes_to_split.shape[1:],
                dtype=genes_to_split.dtype,
                device=device,
            )
            current_pcpt_expressions_suffix = torch.empty(
                (0,) + expressions_to_split.shape[1:],
                dtype=expressions_to_split.dtype,
                device=device,
            )

            if not genes_to_split.numel():
                pass
            elif gen_prob == 0:
                current_pcpt_genes_suffix = genes_to_split
                current_pcpt_expressions_suffix = expressions_to_split
            elif gen_prob == 1:
                current_gen_genes = genes_to_split
                current_gen_expressions = expressions_to_split
            else:
                (
                    current_gen_genes,
                    current_gen_expressions,
                    current_pcpt_genes_suffix,
                    current_pcpt_expressions_suffix,
                ) = self._random_split(
                    genes_to_split,
                    expressions_to_split,
                    ratio=gen_prob,
                )

            pcpt_genes_combined = torch.cat(
                (genes[: self.keep_first_n_tokens], current_pcpt_genes_suffix), dim=0
            )
            pcpt_expressions_combined = torch.cat(
                (
                    expressions[: self.keep_first_n_tokens],
                    current_pcpt_expressions_suffix,
                ),
                dim=0,
            )

            g_pcpt, e_pcpt = self._sample_or_truncate_plus_pad(
                pcpt_genes_combined, pcpt_expressions_combined, pcpt_length
            )
            padded_pcpt_genes.append(g_pcpt)
            padded_pcpt_expressions.append(e_pcpt)

            g_gen, e_gen = self._sample_or_truncate_plus_pad(
                current_gen_genes, current_gen_expressions, gen_length
            )
            padded_gen_genes.append(g_gen)
            padded_gen_expressions.append(e_gen)

        stacked_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0).to(device)
        stacked_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0).to(
            device
        )
        stacked_gen_genes = torch.stack(padded_gen_genes, dim=0).to(device)

        if self.do_mlm:
            masked_pcpt_expressions = self._mask(
                stacked_pcpt_expressions, self.keep_first_n_tokens
            )
        else:
            masked_pcpt_expressions = stacked_pcpt_expressions

        stacked_gen_expressions = torch.stack(padded_gen_expressions, dim=0).to(device)

        data_dict = {
            "pcpt_gene": stacked_pcpt_genes,
            "pcpt_expr": masked_pcpt_expressions,
            "gen_gene": stacked_gen_genes,
            "pcpt_key_padding_mask": stacked_pcpt_genes.eq(self.pad_token_id),
            "gen_key_padding_mask": stacked_gen_genes.eq(self.pad_token_id),
            "gen_expr_target": stacked_gen_expressions,
            "pcpt_expr_target": stacked_pcpt_expressions,
        }
        return data_dict

    def _random_split(
        self,
        *arrays: torch.Tensor,
        ratio: float,
    ) -> Tuple[torch.Tensor, ...]:
        assert len(arrays) > 0
        assert 0 < ratio < 1
        if len(arrays) > 1:
            assert all(
                array.shape[0] == arrays[0].shape[0] for array in arrays
            ), "The arrays must have the same length."

        length = arrays[0].shape[0]
        split_index = int(length * ratio)

        indices = torch.randperm(length, device=arrays[0].device)
        first_part_indices = indices[:split_index]
        second_part_indices = indices[split_index:]

        first_parts = tuple(array[first_part_indices] for array in arrays)
        second_parts = tuple(array[second_part_indices] for array in arrays)

        return first_parts + second_parts

    def get_mlm_probability(self) -> float:
        if isinstance(self.mlm_probability, float):
            return self.mlm_probability
        elif isinstance(self.mlm_probability, list):
            return np.random.choice(self.mlm_probability)
        else:
            raise ValueError(
                "mlm_probability must be a float or a list of floats, "
                f"but got {self.mlm_probability}."
            )

    def _mask(
        self, expressions: torch.Tensor, keep_first_n_tokens: int = 0
    ) -> torch.Tensor:
        if keep_first_n_tokens > 0:
            result_ = self._mask(
                expressions[:, keep_first_n_tokens:],
                keep_first_n_tokens=0,
            )
            return torch.cat([expressions[:, :keep_first_n_tokens], result_], dim=1)

        device = expressions.device
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.get_mlm_probability())
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0

        mask = torch.bernoulli(probability_matrix).bool()
        mask = mask.to(device)

        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return masked_expressions

    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)

        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            else:
                return genes[:max_length], expressions[:max_length]
        else:
            return self._pad(genes, expressions, max_length)

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        device = genes.device
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes), device=device)[:max_length]
            return genes[indices], expressions[indices]

        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n, device=device)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return genes[indices], expressions[indices]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        device = genes.device
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token_id,
                    dtype=genes.dtype,
                    device=device,
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions
