import csv
import json
import math
import random
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def class_centers(n, device, dtype):
    return (torch.arange(n, device=device, dtype=dtype) + 0.5) / n


def expected_value(logits):
    centers = class_centers(logits.size(-1), logits.device, logits.dtype)
    return (F.softmax(logits, dim=-1) * centers).sum(dim=-1)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def ce_loss(logits, targets, mask, cfg):
    total = 0.0
    detail = {}
    specs = [
        ("lat", cfg.lat_size, 0),
        ("lon", cfg.lon_size, 1),
        ("sog", cfg.sog_size, 2),
        ("cog", cfg.cog_size, 3),
    ]
    for name, size, idx in specs:
        loss = F.cross_entropy(
            logits[name].reshape(-1, size),
            targets[..., idx].reshape(-1),
            reduction="none",
            label_smoothing=cfg.label_smoothing,
        ).view_as(mask)
        item = masked_mean(loss, mask)
        detail[f"ce_{name}"] = item
        total = total + item
    detail["ce"] = total
    return total, detail


def kinematic_consistency_loss(logits, mask, cfg):
    lat_n = expected_value(logits["lat"])
    lon_n = expected_value(logits["lon"])
    sog_n = expected_value(logits["sog"])
    cog_n = expected_value(logits["cog"])
    if lat_n.size(1) < 2:
        return lat_n.new_zeros(())

    R = 6371.0
    dt_h = cfg.dt_minutes / 60.0
    knots_to_kmh = 1.852
    lat_deg = lat_n * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
    lon_deg = lon_n * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
    sog_kmh = sog_n * cfg.sog_max_knots * knots_to_kmh
    cog_rad = cog_n * 2.0 * math.pi

    dlat_pred = (lat_deg[:, 1:] - lat_deg[:, :-1]) * math.pi / 180.0
    dlon_pred = (lon_deg[:, 1:] - lon_deg[:, :-1]) * math.pi / 180.0
    lat_rad = lat_deg[:, :-1] * math.pi / 180.0
    dlat_kin = sog_kmh[:, :-1] * torch.cos(cog_rad[:, :-1]) * dt_h / R
    dlon_kin = sog_kmh[:, :-1] * torch.sin(cog_rad[:, :-1]) * dt_h / (R * torch.cos(lat_rad).clamp(min=0.1))
    err = (dlat_pred - dlat_kin).square() + (dlon_pred - dlon_kin).square()
    return masked_mean(err, mask[:, 1:])


def smooth_loss(logits, mask, cfg):
    lat = expected_value(logits["lat"]) * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
    lon = expected_value(logits["lon"]) * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
    if lat.size(1) < 3:
        return lat.new_zeros(())
    d2lat = lat[:, :-2] - 2 * lat[:, 1:-1] + lat[:, 2:]
    d2lon = lon[:, :-2] - 2 * lon[:, 1:-1] + lon[:, 2:]
    return masked_mean(d2lat.square() + d2lon.square(), mask[:, 1:-1])


def lambda_phys_schedule(epoch, cfg):
    if epoch < cfg.lambda_phys_warmup_start:
        return 0.0
    if epoch >= cfg.lambda_phys_warmup_end:
        return cfg.lambda_phys_final
    span = max(1, cfg.lambda_phys_warmup_end - cfg.lambda_phys_warmup_start)
    return cfg.lambda_phys_final * (epoch - cfg.lambda_phys_warmup_start) / span


def compute_losses(model, logits, targets_idx, target_norm, mask, cfg, epoch=0):
    ce, detail = ce_loss(logits, targets_idx, mask, cfg)
    lam_phys = lambda_phys_schedule(epoch, cfg)
    phys = kinematic_consistency_loss(logits, mask, cfg) if lam_phys > 0 else ce.new_zeros(())
    smt = smooth_loss(logits, mask, cfg) if cfg.lambda_smooth > 0 else ce.new_zeros(())
    kan = model.kan_regularization_loss() * cfg.kan_reg_weight
    total = ce + lam_phys * phys + cfg.lambda_smooth * smt + kan
    detail.update({
        "phys": phys,
        "smooth": smt,
        "kan": kan,
        "lambda_phys": ce.new_tensor(lam_phys),
        "total": total,
    })
    return detail


def _top_k(logits, k):
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    vals, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < vals[:, [-1]]] = -float("inf")
    return out


def _vicinity(logits, center, radius):
    idx = torch.arange(logits.size(-1), device=logits.device).view(1, -1)
    out = logits.clone()
    out[(idx - center).abs() >= radius / 2] = -float("inf")
    return out


def make_generator(seed, device):
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))
    return gen


@torch.no_grad()
def autoregressive_sample(model, seqs, steps, cfg, sample=True, generator=None):
    model.eval()
    device = seqs.device
    current = model.to_indexes(seqs)
    hist_norm = seqs[:, :cfg.init_seqlen]
    hist_idxs = current[:, :cfg.init_seqlen]
    enc_ctx = model.encode(hist_idxs, hist_norm)

    for _ in range(steps):
        hidden = model.decode(current, enc_ctx)
        logits = model.logits_from_hidden(hidden[:, -1:, :])

        lat = logits["lat"][:, -1] / cfg.temperature
        lon = logits["lon"][:, -1] / cfg.temperature
        sog = logits["sog"][:, -1] / cfg.temperature
        cog = logits["cog"][:, -1] / cfg.temperature

        if cfg.sample_mode == "pos_vicinity":
            lat = _vicinity(lat, current[:, -1, 0:1], cfg.r_vicinity)
            lon = _vicinity(lon, current[:, -1, 1:2], cfg.r_vicinity)

        lat = _top_k(lat, cfg.top_k)
        lon = _top_k(lon, cfg.top_k)
        sog = _top_k(sog, cfg.top_k)
        cog = _top_k(cog, cfg.top_k)

        probs = [F.softmax(x, dim=-1) for x in (lat, lon, sog, cog)]
        if sample:
            nxt = [torch.multinomial(p, 1, generator=generator) for p in probs]
        else:
            nxt = [torch.argmax(p, dim=-1, keepdim=True) for p in probs]
        current = torch.cat([current, torch.cat(nxt, dim=-1).unsqueeze(1)], dim=1)

    sizes = model.att_sizes.to(device).float()
    return (current.float() + 0.5) / sizes.view(1, 1, -1)


HORIZONS_HOURS = (1, 3, 5, 10, 15)


def horizon_to_step(hours, dt_min=10.0):
    return int(round(hours * 60.0 / dt_min))


def denorm_lat_lon(x, cfg):
    lat = x[..., 0] * (cfg.lat_max - cfg.lat_min) + cfg.lat_min
    lon = x[..., 1] * (cfg.lon_max - cfg.lon_min) + cfg.lon_min
    return torch.stack([lat, lon], dim=-1)


def haversine_km(pred_ll, true_ll):
    R = 6371.0
    lat1 = pred_ll[..., 0] * math.pi / 180.0
    lon1 = pred_ll[..., 1] * math.pi / 180.0
    lat2 = true_ll[..., 0] * math.pi / 180.0
    lon2 = true_ll[..., 1] * math.pi / 180.0
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2.0 * torch.atan2(torch.sqrt(a.clamp(min=0.0)), torch.sqrt((1.0 - a).clamp(min=0.0)))
    return R * c


class MetricAccumulator:
    def __init__(self, max_steps, cfg):
        self.max_steps = max_steps
        self.cfg = cfg
        self.sum_hav = torch.zeros(max_steps, dtype=torch.float64)
        self.sum_sq = torch.zeros(max_steps, dtype=torch.float64)
        self.count = torch.zeros(max_steps, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds, target, mask):
        T = min(preds.size(2), target.size(1), self.max_steps)
        preds = preds[:, :, :T]
        target = target[:, :T]
        mask = mask[:, :T]
        pred_ll = denorm_lat_lon(preds, self.cfg)
        true_ll = denorm_lat_lon(target, self.cfg).unsqueeze(1)
        hav = haversine_km(pred_ll, true_ll)
        best, _ = hav.min(dim=1)
        m = mask.float()
        self.sum_hav += (best * m).sum(dim=0).double().cpu()
        self.sum_sq += (best.square() * m).sum(dim=0).double().cpu()
        self.count += m.sum(dim=0).double().cpu()

    def compute(self):
        count = self.count.numpy()
        valid = count > 0
        mae = np.full(self.max_steps, np.nan, dtype=np.float64)
        rmse = np.full(self.max_steps, np.nan, dtype=np.float64)
        mae[valid] = (self.sum_hav.numpy()[valid] / count[valid])
        rmse[valid] = np.sqrt(self.sum_sq.numpy()[valid] / count[valid])
        times = (np.arange(self.max_steps) + 1) * (self.cfg.dt_minutes / 60.0)
        out = OrderedDict()
        for h in HORIZONS_HOURS:
            i = horizon_to_step(h, self.cfg.dt_minutes) - 1
            if 0 <= i < len(mae) and valid[i]:
                out[f"Haversine@{h}h_km"] = float(mae[i])
                out[f"MAE_km@{h}h"] = float(mae[i])
                out[f"RMSE_km@{h}h"] = float(rmse[i])
        out["ADE_km"] = float(np.nanmean(mae)) if valid.any() else None
        out["ADE_weighted_km"] = float(self.sum_hav.sum() / self.count.sum()) if valid.any() else None
        out["FDE_km"] = float(mae[-1]) if valid[-1] else None
        out["RMSE_km_overall"] = float(np.nanmean(rmse)) if valid.any() else None
        out["_per_step_haversine_km"] = [None if np.isnan(x) else float(x) for x in mae]
        out["_per_step_rmse_km"] = [None if np.isnan(x) else float(x) for x in rmse]
        out["_per_step_count"] = count.astype(int).tolist()
        out["_per_step_time_hours"] = times.tolist()
        return out


try:
    from tqdm.auto import tqdm, trange
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

    def trange(*args, **kwargs):
        return range(*args)


HEADLINE_KEYS = (
    *(f"Haversine@{h}h_km" for h in HORIZONS_HOURS),
    *(f"MAE_km@{h}h" for h in HORIZONS_HOURS),
    *(f"RMSE_km@{h}h" for h in HORIZONS_HOURS),
    "ADE_km", "ADE_weighted_km", "FDE_km", "RMSE_km_overall",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(state[k].detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        backup = {}
        state = model.state_dict()
        for k, v in self.shadow.items():
            backup[k] = state[k].detach().clone()
            state[k].copy_(v)
        return backup

    def restore(self, model, backup):
        state = model.state_dict()
        for k, v in backup.items():
            state[k].copy_(v)


def lr_at(tokens, cfg):
    if tokens < cfg.warmup_tokens:
        return cfg.learning_rate * max(1, tokens) / max(1, cfg.warmup_tokens)
    span = max(1, cfg.final_tokens - cfg.warmup_tokens)
    progress = min(1.0, (tokens - cfg.warmup_tokens) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.learning_rate * (cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine)


def _float_dict(d):
    return {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in d.items()}


def run_epoch(model, loader, optimizer, cfg, epoch, token_state=None, ema=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    sums, count = {}, 0
    amp_on = cfg.use_amp and cfg.device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_on else nullcontext()

    for batch in loader:
        seqs, masks, *_ = batch
        seqs = seqs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        target_idx = model.to_indexes(seqs[:, 1:].contiguous())
        target_norm = seqs[:, 1:].contiguous()
        target_mask = masks[:, :-1].contiguous()

        if is_train:
            token_state["tokens"] += int((seqs >= 0).sum().item())
            lr = lr_at(token_state["tokens"], cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        with amp_ctx:
            logits = model(seqs)
            losses = compute_losses(model, logits, target_idx, target_norm, target_mask, cfg, epoch=epoch)
            loss = losses["total"]

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        vals = _float_dict(losses)
        bs = seqs.size(0)
        for k, v in vals.items():
            sums[k] = sums.get(k, 0.0) + v * bs
        count += bs

    return {k: v / max(1, count) for k, v in sums.items()}


def train_model(model, train_loader, valid_loader, cfg, savedir, logger):
    optimizer = model.configure_optimizers(cfg)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    cfg.final_tokens = 2 * len(train_loader.dataset) * cfg.max_seqlen
    best_score = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    ckpt_path = Path(savedir) / "model.pt"
    token_state = {"tokens": 0}

    for epoch in trange(cfg.max_epochs, desc=f"train {Path(savedir).name}", unit="epoch"):
        train_stats = run_epoch(model, train_loader, optimizer, cfg, epoch, token_state, ema=ema)
        if ema is not None:
            backup = ema.apply_to(model)
            valid_stats = run_epoch(model, valid_loader, None, cfg, epoch)
            ema.restore(model, backup)
        else:
            valid_stats = run_epoch(model, valid_loader, None, cfg, epoch)

        row = {"epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"valid_{k}": v for k, v in valid_stats.items()})
        history.append(row)

        score = valid_stats["total"]
        logger.info(
            "epoch %03d | train total=%.4f ce=%.4f phys=%.6f smooth=%.6f | "
            "valid total=%.4f ce=%.4f phys=%.6f smooth=%.6f",
            epoch + 1,
            train_stats["total"], train_stats["ce"], train_stats["phys"], train_stats["smooth"],
            valid_stats["total"], valid_stats["ce"], valid_stats["phys"], valid_stats["smooth"],
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch + 1
            wait = 0
            if ema is not None:
                backup = ema.apply_to(model)
                torch.save(model.state_dict(), ckpt_path)
                ema.restore(model, backup)
            else:
                torch.save(model.state_dict(), ckpt_path)
            logger.info("saved best epoch %d valid=%.6f", best_epoch, best_score)
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                logger.info("early stop at epoch %d", epoch + 1)
                break

    with open(Path(savedir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return best_epoch, best_score, history


@torch.no_grad()
def evaluate_best_of_n(model, loader, cfg, seed, sample=True):
    model.eval()
    max_steps = max(horizon_to_step(h, cfg.dt_minutes) for h in HORIZONS_HOURS)
    acc = MetricAccumulator(max_steps, cfg)
    gen = make_generator(seed, cfg.device)

    for batch in loader:
        seqs, masks, *_ = batch
        seqs = seqs.to(cfg.device, non_blocking=True)
        masks = masks.to(cfg.device, non_blocking=True)
        init = seqs[:, :cfg.init_seqlen].contiguous()
        target = seqs[:, cfg.init_seqlen:cfg.init_seqlen + max_steps].contiguous()
        target_mask = masks[:, cfg.init_seqlen:cfg.init_seqlen + max_steps].contiguous()
        if target.size(1) == 0:
            continue
        preds = []
        n_samples = cfg.n_samples if sample else 1
        for _ in range(n_samples):
            out = autoregressive_sample(model, init, target.size(1), cfg, sample=sample, generator=gen)
            preds.append(out[:, cfg.init_seqlen:cfg.init_seqlen + target.size(1)])
        acc.update(torch.stack(preds, dim=1), target, target_mask)
    return acc.compute()


def save_metrics(metrics, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serial = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial.keys()))
        writer.writeheader()
        writer.writerow(serial)
    with open(out_dir / "per_step_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_hours", "haversine_km", "rmse_km", "count"])
        for row in zip(
            metrics["_per_step_time_hours"],
            metrics["_per_step_haversine_km"],
            metrics["_per_step_rmse_km"],
            metrics["_per_step_count"],
        ):
            writer.writerow(row)


def write_rows_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_json(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def json_safe(obj):
    if isinstance(obj, (Path, torch.device)):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj
