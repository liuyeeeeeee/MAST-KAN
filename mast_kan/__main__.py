import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .config import DATASET_NAMES, EVAL_SEEDS, config_dict, load_config, make_config
from .data import build_loader, build_loaders, dataset_coverage, validate_data_contract
from .engine import autoregressive_sample, evaluate_best_of_n, json_safe, make_generator, set_seed, train_model
from .model import MASTKAN


def parser():
    root = argparse.ArgumentParser(prog="python -m mast_kan")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "infer", "check", "example"):
        p = commands.add_parser(name)
        p.add_argument("--dataset", choices=DATASET_NAMES, default="ct_dma")
        p.add_argument("--data-dir", type=Path)
        p.add_argument("--device")
        p.add_argument("--out-dir", type=Path)
        if name in ("train", "evaluate", "check"):
            p.add_argument("--batch-size", type=int)
            p.add_argument("--num-workers", type=int)
        if name in ("train", "infer", "example"):
            p.add_argument("--seed", type=int, default=42005)
        if name == "train":
            p.add_argument("--max-epochs", type=int)
            p.add_argument("--no-amp", action="store_true")
        if name in ("evaluate", "infer"):
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--config", type=Path)
            p.add_argument("--n-samples", type=int)
            p.add_argument("--greedy", action="store_true")
        if name == "evaluate":
            p.add_argument("--eval-seeds", nargs="+", type=int, default=list(EVAL_SEEDS))
        if name == "infer":
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--steps", type=int, default=90)
    return root


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_safe, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_coverage(loader, cfg):
    coverage = dataset_coverage(loader, cfg)
    missing = [key for key, value in coverage.items() if value["trajectory_count"] == 0]
    if missing:
        raise ValueError("No valid target trajectories at: " + ", ".join(missing))
    return coverage


def configuration(args):
    if getattr(args, "checkpoint", None):
        path = args.config or args.checkpoint.parent / "config.json"
        if not path.is_file():
            raise FileNotFoundError("Checkpoint configuration is required; use --config.")
        cfg = load_config(path)
    else:
        cfg = make_config(args.dataset)
    for name in ("data_dir", "device", "batch_size", "num_workers", "n_samples", "max_epochs", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, torch.device(value) if name == "device" else value)
    if getattr(args, "no_amp", False):
        cfg.use_amp = False
    for name in ("batch_size", "n_samples", "max_epochs"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    return cfg


def load_model(cfg, checkpoint=None):
    if cfg.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official Mamba execution requires a working CUDA device.")
    set_seed(cfg.seed)
    model = MASTKAN(cfg).to(cfg.device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=cfg.device, weights_only=True), strict=True)
    return model


def output_dir(args, cfg):
    directory = args.out_dir or Path("runs") / cfg.dataset_name / args.command
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == "train" and (directory / "model.pt").exists():
        raise FileExistsError(f"Training output already contains model.pt: {directory}")
    return directory


def train(args, cfg):
    model = load_model(cfg)
    train_loader, valid_loader, test_loader, counts = build_loaders(cfg)
    contract = {name: validate_data_contract(loader, cfg, name) for name, loader in (("train", train_loader), ("valid", valid_loader), ("test", test_loader))}
    coverage = {name: require_coverage(loader, cfg) for name, loader in (("train", train_loader), ("valid", valid_loader), ("test", test_loader))}
    directory = output_dir(args, cfg)
    cfg.out_root = directory
    cfg.final_tokens = 2 * len(train_loader.dataset) * cfg.max_seqlen
    write_json(directory / "config.json", config_dict(cfg))
    logger = logging.getLogger("mast-kan")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(directory / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    try:
        epoch, score, _ = train_model(model, train_loader, valid_loader, cfg, directory, logger)
    finally:
        for item in list(logger.handlers):
            item.close()
            logger.removeHandler(item)
    write_json(directory / "training.json", {"model": "MAST-KAN", "version": "1.0.0", "train_seed": cfg.seed, "best_epoch": epoch, "validation_total": score, "ema": cfg.ema_decay > 0, "checkpoint_sha256": checkpoint_hash(directory / "model.pt"), "backend": model.check_backend(), "torch_version": torch.__version__, "data_counts": counts, "data_contract": contract, "coverage": coverage})
    print(f"Saved {directory / 'model.pt'}")


def evaluate(args, cfg):
    model = load_model(cfg, args.checkpoint)
    loader, counts = build_loader(cfg, "test")
    validate_data_contract(loader, cfg, "test")
    coverage = require_coverage(loader, cfg)
    directory = output_dir(args, cfg)
    rows = []
    for seed in args.eval_seeds:
        metrics = evaluate_best_of_n(model, loader, cfg, seed, sample=not args.greedy)
        rows.append({"seed": seed, **metrics})
    summary = {}
    for key in rows[0]:
        if key == "seed" or key.startswith("_"):
            continue
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0} if values else {"mean": None, "std": None}
    write_json(directory / "evaluation.json", {"model": "MAST-KAN", "version": "1.0.0", "config": config_dict(cfg), "checkpoint": args.checkpoint.name, "checkpoint_sha256": checkpoint_hash(args.checkpoint), "selection": "greedy" if args.greedy else "pointwise_best_of_N", "backend": model.check_backend(), "torch_version": torch.__version__, "data_counts": counts, "coverage": coverage, "n_samples": 1 if args.greedy else cfg.n_samples, "repeats": rows, "summary": summary})
    print(json.dumps(summary["ADE_km"], indent=2))


def infer(args, cfg):
    if not 1 <= args.steps <= 90:
        raise ValueError("steps must be between 1 and 90")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("dataset", cfg.dataset_name) != cfg.dataset_name:
        raise ValueError("Input dataset does not match checkpoint configuration")
    history = np.asarray(payload.get("history", payload) if isinstance(payload, dict) else payload, dtype=np.float32)
    if history.shape != (cfg.init_seqlen, 4) or not np.isfinite(history).all() or (history < 0).any() or (history > 1).any():
        raise ValueError(f"Input must contain {cfg.init_seqlen} normalized [lat, lon, sog, cog] points")
    model = load_model(cfg, args.checkpoint)
    tensor = torch.as_tensor(history, device=cfg.device).unsqueeze(0)
    generator = make_generator(args.seed, cfg.device)
    count = 1 if args.greedy else cfg.n_samples
    futures = [autoregressive_sample(model, tensor, args.steps, cfg, sample=not args.greedy, generator=generator)[0, cfg.init_seqlen:].cpu().numpy() for _ in range(count)]
    normalized = np.stack(futures)
    physical = normalized.copy()
    physical[..., 0] = physical[..., 0] * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
    physical[..., 1] = physical[..., 1] * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
    physical[..., 2] *= cfg.sog_max_knots
    physical[..., 3] *= 360.0
    directory = output_dir(args, cfg)
    write_json(directory / "prediction.json", {"dataset": cfg.dataset_name, "seed": args.seed, "checkpoint_sha256": checkpoint_hash(args.checkpoint), "selection": "greedy" if args.greedy else "sampled", "dt_minutes": cfg.dt_minutes, "columns": ["latitude_deg", "longitude_deg", "sog_knots", "cog_deg"], "trajectories": physical.tolist()})
    print(f"Saved {directory / 'prediction.json'}")


def example(args, cfg):
    import pickle

    rng = np.random.default_rng(args.seed)
    directory = args.out_dir or Path("data") / "synthetic" / cfg.dataset_name
    directory.mkdir(parents=True, exist_ok=True)
    destinations = [directory / f"{cfg.dataset_name}_{split}.pkl" for split in ("train", "valid", "test")]
    if any(path.exists() for path in destinations) or (directory / "history.json").exists():
        raise FileExistsError("Synthetic output already exists; choose a new --out-dir")
    for split in ("train", "valid", "test"):
        entries = []
        for index in range(4):
            steps = np.arange(120)
            trajectory = np.column_stack((0.4 + steps * 0.001 + rng.uniform(-0.0001, 0.0001, 120), 0.3 + steps * 0.001, np.full(120, 0.3), np.full(120, 0.125), 1609459200 + steps * 600, np.full(120, index + 1)))
            entries.append({"mmsi": index + 1, "traj": trajectory})
        path = directory / f"{cfg.dataset_name}_{split}.pkl"
        with path.open("wb") as handle:
            pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if split == "test":
            write_json(directory / "history.json", {"dataset": cfg.dataset_name, "synthetic": True, "history": entries[0]["traj"][:cfg.init_seqlen, :4].tolist()})
    print(f"Saved synthetic examples to {directory}")


def check(args, cfg):
    model = load_model(cfg)
    info = {"model": "MAST-KAN", "backend": model.check_backend(), "parameters": sum(p.numel() for p in model.parameters()), "torch_version": torch.__version__, "device": str(cfg.device)}
    if args.data_dir:
        loader, counts = build_loader(cfg, "test")
        info.update(data_counts=counts, data_contract=validate_data_contract(loader, cfg, "test"), coverage=dataset_coverage(loader, cfg))
    print(json.dumps(json_safe(info), indent=2))


def main():
    args = parser().parse_args()
    cfg = configuration(args)
    {"train": train, "evaluate": evaluate, "infer": infer, "example": example, "check": check}[args.command](args, cfg)


if __name__ == "__main__":
    main()
