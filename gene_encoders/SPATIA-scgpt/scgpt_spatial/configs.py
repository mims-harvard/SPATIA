from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DataConfig:

    data_source: str
    vocab_path: str
    gene_stats_file: str
    gene_col: str = "gene_name"
    n_hvg: Optional[int] = None
    valid_size_or_ratio: float = 0.0001
    input_style: Literal["normed_raw", "log1p", "binned"] = "binned"
    input_emb_style: Literal["category", "continuous", "scaling"] = "continuous"
    n_bins: int = 51
    max_seq_len: int = 1200
    trunc_by_sample: bool = True
    spatial_datadir: Optional[str] = None
    fix_missing_images: bool = False
    preprocessor_cls: Optional[str] = None
    test_h5ad_path: Optional[str] = None
    cell_type_col: Optional[str] = "auto"


@dataclass_json
@dataclass
class ModelConfig:

    load_model_path: Optional[str] = None
    embsize: int = 64
    d_hid: int = 64
    nlayers: int = 4
    nheads: int = 4
    n_layers_cls: int = 3
    dropout: float = 0.2
    use_fast_transformer: bool = True
    no_cls: bool = False
    no_cce: bool = True
    image_encoder_cls: Optional[
        Literal["openai/clip-vit-base-patch32", "facebook/vit-mae-base"]
    ] = None
    combine_weight: float = 1
    image_combine_weight: float = 1
    image_recon_loss_weight: float = 1


@dataclass_json
@dataclass
class OptimizerConfig:

    lr: float = 1e-4
    warmup_ratio_or_step: float = 0.1
    scheduler_interval: int = 100
    scheduler_factor: float = 0.99


@dataclass_json
@dataclass
class TrainerConfig:

    save_dir: str
    epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 64
    grad_accu_steps: int = 1
    fp16: bool = True
    training_tasks: Literal["pcpt", "gen", "both"] = "both"
    mask_ratio: Union[float, List[float]] = 0.40
    pad_token: str = "<pad>"
    log_interval: int = 100
    save_interval_steps: int = 10000
    project_name: str = "scGPT-spatial"
    exp_name: Optional[str] = None
    clustering_method: Literal["kmeans", "leiden", "louvain"] = "kmeans"
    clustering_resolution: float = 0.5
    eval_interval_steps: int = 1000


@dataclass_json
@dataclass
class InferenceConfig:
    seed: int
    spatial_config_path: str
    spatial_weight_path: str
    h5ad_file: str
    output_path: str


@dataclass_json
@dataclass
class MainConfig:

    data: DataConfig
    model: ModelConfig
    optim: OptimizerConfig
    trainer: TrainerConfig
