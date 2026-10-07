import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch


DATASET_NAMES = ("ct_dma", "florida_gulf_2025")
EVAL_SEEDS = (92026, 93035, 94044, 95053, 96062)


@dataclass
class MASTKANConfig:
    data_dir: Path = Path("data/ct_dma")
    out_root: Path = Path("runs")
    dataset_name: str = "ct_dma"
    seed: int = 42005
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    lat_min: float = 55.5
    lat_max: float = 58.0
    lon_min: float = 10.3
    lon_max: float = 13.0
    lat_size: int = 250
    lon_size: int = 270
    sog_size: int = 30
    cog_size: int = 72
    init_seqlen: int = 18
    max_seqlen: int = 144
    min_seqlen: int = 36
    dt_minutes: float = 10.0
    sog_max_knots: float = 30.0
    moving_threshold: float = 0.05
    n_lat_embd: int = 128
    n_lon_embd: int = 128
    n_sog_embd: int = 64
    n_cog_embd: int = 64
    d_model: int = 384
    d_state: int = 64
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    n_heads_fusion: int = 8
    n_heads_cross: int = 8
    d_freq: int = 128
    d_band: int = 64
    dwt_levels: int = 2
    dwt_wavelet: str = "db4"
    effkan_grid_size: int = 7
    effkan_spline_order: int = 3
    effkan_scale_noise: float = 0.1
    effkan_scale_base: float = 1.0
    effkan_scale_spline: float = 1.0
    kan_reg_weight: float = 0.0
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    drop_path: float = 0.1
    label_smoothing: float = 0.05
    lambda_phys_final: float = 0.1
    lambda_phys_warmup_start: int = 5
    lambda_phys_warmup_end: int = 20
    lambda_smooth: float = 0.01
    max_epochs: int = 50
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_norm_clip: float = 1.0
    use_amp: bool = True
    ema_decay: float = 0.999
    early_stop_patience: int = 5
    warmup_tokens: int = 10240
    final_tokens: int = 0
    min_lr_scale: float = 0.25
    n_samples: int = 16
    eval_repeats: int = 5
    eval_seed: int = 91017
    top_k: int = 10
    temperature: float = 1.0
    sample_mode: str = "pos_vicinity"
    r_vicinity: int = 40

    @property
    def full_size(self):
        return sum(self.att_sizes)

    @property
    def att_sizes(self):
        return self.lat_size, self.lon_size, self.sog_size, self.cog_size


def make_config(dataset="ct_dma", data_dir=None, device=None):
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    cfg = MASTKANConfig(dataset_name=dataset, data_dir=Path("data") / dataset)
    if dataset == "florida_gulf_2025":
        cfg.lat_min, cfg.lat_max = 26.0, 28.5
        cfg.lon_min, cfg.lon_max, cfg.lon_size = -85.0, -82.0, 300
    if data_dir is not None:
        cfg.data_dir = Path(data_dir).expanduser().resolve()
    if device is not None:
        cfg.device = torch.device(device)
    return cfg


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("config", payload)
    cfg = make_config(values.get("dataset_name", "ct_dma"))
    allowed = {f.name for f in fields(cfg)}
    for key, value in values.items():
        if key in allowed:
            setattr(cfg, key, value)
    cfg.data_dir, cfg.out_root = Path(cfg.data_dir), Path(cfg.out_root)
    cfg.device, cfg.betas = torch.device(cfg.device), tuple(cfg.betas)
    return cfg


def config_dict(cfg):
    return {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in asdict(cfg).items()}
