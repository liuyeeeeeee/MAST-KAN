import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_trajectories(data, min_seqlen: int, moving_threshold: float):
    out = []
    for item in data:
        traj = item["traj"]
        try:
            start = np.where(traj[:, 2] > moving_threshold)[0][0]
        except IndexError:
            start = len(traj) - 1
        traj = traj[start:, :]
        if np.isnan(traj).any() or len(traj) <= min_seqlen:
            continue
        out.append({"mmsi": item["mmsi"], "traj": traj})
    return out


class AISDataset(Dataset):
    def __init__(self, data, max_seqlen: int):
        self.data = data
        self.max_seqlen = max_seqlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        values = item["traj"][:, :4].copy()
        values[values > 0.9999] = 0.9999

        seqlen = min(len(values), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4), dtype=np.float32)
        mask = np.zeros(self.max_seqlen, dtype=np.float32)
        seq[:seqlen] = values[:seqlen]
        mask[:seqlen] = 1.0
        t0 = int(item["traj"][0, 4]) if item["traj"].shape[1] > 4 else 0

        return (
            torch.from_numpy(seq),
            torch.from_numpy(mask),
            torch.tensor(seqlen, dtype=torch.int32),
            torch.tensor(item["mmsi"], dtype=torch.int64),
            torch.tensor(t0, dtype=torch.int64),
        )


def build_datasets(cfg):
    paths = {
        "train": cfg.data_dir / f"{cfg.dataset_name}_train.pkl",
        "valid": cfg.data_dir / f"{cfg.dataset_name}_valid.pkl",
        "test": cfg.data_dir / f"{cfg.dataset_name}_test.pkl",
    }
    datasets, counts = {}, {}
    for split, path in paths.items():
        raw = _load_pkl(path)
        cleaned = clean_trajectories(raw, cfg.min_seqlen, cfg.moving_threshold)
        counts[split] = {"raw": len(raw), "kept": len(cleaned)}
        datasets[split] = AISDataset(cleaned, cfg.max_seqlen + 1)
    return datasets["train"], datasets["valid"], datasets["test"], counts


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(cfg):
    train_ds, valid_ds, test_ds, counts = build_datasets(cfg)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)

    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=_worker_init,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, valid_loader, test_loader, counts


def build_dataset(cfg, split):
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    path = Path(cfg.data_dir) / f"{cfg.dataset_name}_{split}.pkl"
    raw = _load_pkl(path)
    cleaned = clean_trajectories(raw, cfg.min_seqlen, cfg.moving_threshold)
    return AISDataset(cleaned, cfg.max_seqlen + 1), {
        "raw": len(raw), "kept": len(cleaned)
    }


def build_loader(cfg, split):
    dataset, counts = build_dataset(cfg, split)
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=_worker_init,
    )
    if split == "train":
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)
        kwargs["generator"] = generator
    loader = DataLoader(dataset, shuffle=(split == "train"), **kwargs)
    return loader, counts


def _dataset_data(dataset_or_loader):
    dataset = getattr(dataset_or_loader, "dataset", dataset_or_loader)
    return dataset.data


def validate_data_contract(dataset_or_loader, cfg, split="data"):
    data = _dataset_data(dataset_or_loader)
    if not data:
        raise ValueError(f"{split} has no trajectories after cleaning")
    feature_min = np.full(4, np.inf)
    feature_max = np.full(4, -np.inf)
    intervals = []
    point_count = 0
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "traj" not in item or "mmsi" not in item:
            raise ValueError(f"{split}[{index}] needs traj and mmsi fields")
        trajectory = np.asarray(item["traj"])
        if trajectory.ndim != 2 or trajectory.shape[1] != 6:
            raise ValueError(f"{split}[{index}] must have trajectory shape [N, 6]")
        if len(trajectory) < cfg.init_seqlen:
            raise ValueError(f"{split}[{index}] has fewer than init_seqlen observations")
        if not np.isfinite(trajectory[:, :5]).all():
            raise ValueError(f"{split}[{index}] contains a non-finite feature or timestamp")
        if not np.all(trajectory[:, 5] == item["mmsi"]):
            raise ValueError(f"{split}[{index}] has inconsistent MMSI values")
        feature_min = np.minimum(feature_min, trajectory[:, :4].min(axis=0))
        feature_max = np.maximum(feature_max, trajectory[:, :4].max(axis=0))
        delta_minutes = np.diff(trajectory[:, 4]) / 60.0
        if np.any(delta_minutes <= 0):
            raise ValueError(f"{split}[{index}] timestamps are not strictly increasing")
        if delta_minutes.size:
            intervals.append(delta_minutes)
        point_count += len(trajectory)

    if np.any(feature_min < -1e-6) or np.any(feature_max > 1.0 + 1e-6):
        raise ValueError(f"{split} normalized columns 0-3 fall outside [0, 1]")
    if not intervals:
        raise ValueError(f"{split} has no timestamp intervals to validate")
    median_interval = float(np.median(np.concatenate(intervals)))
    if not np.isclose(median_interval, cfg.dt_minutes, atol=0.5):
        raise ValueError(
            f"{split} median interval is {median_interval:.3f} min; "
            f"expected {cfg.dt_minutes:.1f} min"
        )
    return {
        "trajectory_columns": [
            "lat_norm", "lon_norm", "sog_norm", "cog_norm", "timestamp", "mmsi"
        ],
        "cleaned_points": point_count,
        "normalized_min": feature_min.tolist(),
        "normalized_max": feature_max.tolist(),
        "median_interval_minutes": median_interval,
    }


def dataset_coverage(dataset_or_loader, cfg):
    lengths = [
        min(len(item["traj"]), cfg.max_seqlen + 1)
        for item in _dataset_data(dataset_or_loader)
    ]
    coverage = {}
    for hours in (1, 3, 5, 10, 15):
        steps = int(round(hours * 60.0 / cfg.dt_minutes))
        required = cfg.init_seqlen + steps
        coverage[f"{hours}h"] = {
            "required_points": required,
            "trajectory_count": sum(length >= required for length in lengths),
        }
    return coverage
