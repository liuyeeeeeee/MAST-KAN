import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan import KANLinear

try:
    from mamba_ssm import Mamba as _Mamba1
except Exception as exc:
    _Mamba1 = None
    _Mamba1_ERR = repr(exc)
else:
    _Mamba1_ERR = None

try:
    from mamba_ssm import Mamba2 as _Mamba2
except Exception as exc:
    _Mamba2 = None
    _Mamba2_ERR = repr(exc)
else:
    _Mamba2_ERR = None


def backend_availability():
    return {
        "mamba1": _Mamba1 is not None,
        "mamba2": _Mamba2 is not None,
        "mamba1_import_error": _Mamba1_ERR,
        "mamba2_import_error": _Mamba2_ERR,
    }


def _mamba2_headdim(d_inner):
    for hd in (64, 128, 96, 48, 32):
        if d_inner % hd == 0 and (d_inner // hd) % 8 == 0:
            return hd
    return 64


def make_mamba(d_model, d_state=64, d_conv=4, expand=2, kind="mamba2"):
    if kind == "mamba2":
        backend, import_error = _Mamba2, _Mamba2_ERR
    elif kind == "mamba1":
        backend, import_error = _Mamba1, _Mamba1_ERR
    else:
        raise ValueError(f"Unsupported Mamba backend: {kind}")
    if backend is None:
        raise RuntimeError(f"Official {kind} is unavailable: {import_error}")
    kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if kind == "mamba2":
        kwargs["headdim"] = _mamba2_headdim(expand * d_model)
    try:
        return backend(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Official {kind} initialization failed: {exc}") from exc


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class OutputHeads(nn.Module):
    def __init__(self, d_model, cfg):
        super().__init__()
        sizes = {
            "lat": cfg.lat_size,
            "lon": cfg.lon_size,
            "sog": cfg.sog_size,
            "cog": cfg.cog_size,
        }
        self.heads = nn.ModuleDict()
        for name, size in sizes.items():
            self.heads[name] = KANLinear(
                d_model,
                size,
                grid_size=cfg.effkan_grid_size,
                spline_order=cfg.effkan_spline_order,
                scale_noise=cfg.effkan_scale_noise,
                scale_base=cfg.effkan_scale_base,
                scale_spline=cfg.effkan_scale_spline,
            )

    def forward(self, x):
        shape = x.shape[:-1]
        flat = x.reshape(-1, x.size(-1))
        out = {}
        for name, head in self.heads.items():
            out[name] = head(flat).reshape(*shape, -1)
        return out

    def regularization_loss(self, device):
        return sum(head.regularization_loss(1.0, 1.0) for head in self.heads.values())


class DiscreteEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lat = nn.Embedding(cfg.lat_size, cfg.n_lat_embd)
        self.lon = nn.Embedding(cfg.lon_size, cfg.n_lon_embd)
        self.sog = nn.Embedding(cfg.sog_size, cfg.n_sog_embd)
        self.cog = nn.Embedding(cfg.cog_size, cfg.n_cog_embd)

    def forward(self, idxs):
        return torch.cat(
            [
                self.lat(idxs[..., 0]),
                self.lon(idxs[..., 1]),
                self.sog(idxs[..., 2]),
                self.cog(idxs[..., 3]),
            ],
            dim=-1,
        )


class SequenceBlock(nn.Module):
    def __init__(self, d_model, d_state, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = make_mamba(d_model, d_state=d_state, kind="mamba2")
        self.mix = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.mix(self.mixer(self.norm1(x))))
        return x + self.drop_path(self.ffn(self.norm2(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_state, n_heads, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = make_mamba(d_model, d_state=d_state, kind="mamba2")
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x, enc_ctx):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        query = self.norm_ca(x)
        context, _ = self.cross_attn(query, enc_ctx, enc_ctx, need_weights=False)
        x = x + self.drop_path(context)
        return x + self.drop_path(self.ffn(self.norm2(x)))


_FILTERS = {
    "db2": (
        [
            -0.12940952255092145,
            0.22414386804185735,
            0.836516303737469,
            0.48296291314469025,
        ],
        [
            -0.48296291314469025,
            0.836516303737469,
            -0.22414386804185735,
            -0.12940952255092145,
        ],
    ),
    "db4": (
        [
            -0.010597401784997278,
            0.032883011666982945,
            0.030841381835986965,
            -0.18703481171888114,
            -0.027983769416983849,
            0.6308807679295904,
            0.7148465705525415,
            0.23037781330885523,
        ],
        [
            -0.23037781330885523,
            0.7148465705525415,
            -0.6308807679295904,
            -0.027983769416983849,
            0.18703481171888114,
            0.030841381835986965,
            -0.032883011666982945,
            -0.010597401784997278,
        ],
    ),
}


class DWT1D(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        lo, hi = _FILTERS[wavelet]
        self.register_buffer(
            "lo", torch.tensor(lo, dtype=torch.float32).flip(0).view(1, 1, -1)
        )
        self.register_buffer(
            "hi", torch.tensor(hi, dtype=torch.float32).flip(0).view(1, 1, -1)
        )
        self.pad = len(lo) - 1

    def forward(self, x):
        channels = x.size(1)
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        lo = self.lo.expand(channels, 1, -1)
        hi = self.hi.expand(channels, 1, -1)
        return (
            F.conv1d(x, lo, stride=2, groups=channels),
            F.conv1d(x, hi, stride=2, groups=channels),
        )


class BandBlock(nn.Module):
    def __init__(self, d_band):
        super().__init__()
        self.lift = nn.Linear(2, d_band)
        self.mixer = make_mamba(d_band, d_state=16, kind="mamba1")
        self.norm = nn.LayerNorm(d_band)

    def forward(self, x, target_len):
        hidden = self.lift(x)
        hidden = hidden + self.mixer(hidden)
        hidden = self.norm(hidden)
        hidden = F.interpolate(
            hidden.transpose(1, 2), size=target_len, mode="linear", align_corners=False
        )
        return hidden.transpose(1, 2)


class DWTBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.levels = cfg.dwt_levels
        self.dwt = DWT1D(cfg.dwt_wavelet)
        self.names = [f"cA{cfg.dwt_levels}"] + [
            f"cD{level}" for level in range(1, cfg.dwt_levels + 1)
        ]
        self.blocks = nn.ModuleDict(
            {name: BandBlock(cfg.d_band) for name in self.names}
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.d_band * len(self.names), cfg.d_freq),
            nn.GELU(),
            nn.Dropout(cfg.resid_pdrop),
        )

    def forward(self, x):
        target_len = x.size(1)
        coefficients = {}
        current = x.transpose(1, 2)
        for level in range(1, self.levels + 1):
            current, detail = self.dwt(current)
            coefficients[f"cD{level}"] = detail
        coefficients[f"cA{self.levels}"] = current
        features = [
            self.blocks[name](coefficients[name].transpose(1, 2), target_len)
            for name in self.names
        ]
        return self.out(torch.cat(features, dim=-1))


class FrequencyFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kv_proj = nn.Linear(cfg.d_model + cfg.d_freq, cfg.d_model)
        self.norm_q = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads_fusion, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, time_features, frequency_features):
        query = self.norm_q(time_features)
        key_value = self.norm_kv(
            self.kv_proj(torch.cat([time_features, frequency_features], dim=-1))
        )
        context, _ = self.attn(query, key_value, key_value, need_weights=False)
        hidden = time_features + context
        return hidden + self.ffn(self.norm(hidden))


class MASTKAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.init_seqlen = cfg.init_seqlen
        self.max_seqlen = cfg.max_seqlen
        self.register_buffer("att_sizes", torch.tensor(cfg.att_sizes, dtype=torch.long))

        self.embed = DiscreteEmbedding(cfg)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        enc_drops = [
            cfg.drop_path * i / max(1, cfg.n_enc_layers - 1)
            for i in range(cfg.n_enc_layers)
        ]
        self.encoder = nn.ModuleList(
            [
                SequenceBlock(cfg.d_model, cfg.d_state, enc_drops[i])
                for i in range(cfg.n_enc_layers)
            ]
        )

        self.dwt = DWTBranch(cfg)
        self.fusion = FrequencyFusion(cfg)

        dec_drops = [
            cfg.drop_path * i / max(1, cfg.n_dec_layers - 1)
            for i in range(cfg.n_dec_layers)
        ]
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    cfg.d_model,
                    cfg.d_state,
                    cfg.n_heads_cross,
                    dec_drops[i],
                )
                for i in range(cfg.n_dec_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.heads = OutputHeads(cfg.d_model, cfg)
        check_backend(self)

    def check_backend(self):
        return check_backend(self)

    def to_indexes(self, x):
        sizes = self.att_sizes.to(x.device)
        indexes = (x * sizes.float()).long().clamp(min=0)
        return torch.minimum(indexes, (sizes - 1).long())

    def encode(self, hist_idxs, hist_norm):
        hidden = self.drop(self.embed(hist_idxs))
        for block in self.encoder:
            hidden = block(hidden)
        return self.fusion(hidden, self.dwt(hist_norm[..., :2]))

    def decode(self, dec_idxs, enc_ctx):
        if dec_idxs.size(1) > self.max_seqlen:
            dec_idxs = dec_idxs[:, -self.max_seqlen :]
        hidden = self.drop(self.embed(dec_idxs))
        for block in self.decoder:
            hidden = block(hidden, enc_ctx)
        return self.norm(hidden)

    def logits_from_hidden(self, hidden):
        return self.heads(hidden)

    def forward(self, seqs):
        indexes = self.to_indexes(seqs)
        inputs = indexes[:, :-1].contiguous()
        hist_idxs = inputs[:, : self.init_seqlen].contiguous()
        hist_norm = seqs[:, : self.init_seqlen].contiguous()
        enc_ctx = self.encode(hist_idxs, hist_norm)
        return self.logits_from_hidden(self.decode(inputs, enc_ctx))

    def kan_regularization_loss(self):
        return self.heads.regularization_loss(next(self.parameters()).device)

    def configure_optimizers(self, cfg):
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            lower = name.lower()
            if (
                parameter.ndim < 2
                or lower.endswith("bias")
                or "norm" in lower
                or "embed" in lower
                or lower.endswith(".d")
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=cfg.betas,
        )



def check_backend(model):
    temporal = [block.mixer for block in model.encoder] + [
        block.mixer for block in model.decoder
    ]
    spectral = [block.mixer for block in model.dwt.blocks.values()]
    expected_temporal = model.cfg.n_enc_layers + model.cfg.n_dec_layers
    expected_spectral = model.cfg.dwt_levels + 1
    if len(temporal) != expected_temporal or len(spectral) != expected_spectral:
        raise RuntimeError("The MAST-KAN mixer counts do not match its configuration")
    if _Mamba2 is None or any(not isinstance(mixer, _Mamba2) for mixer in temporal):
        raise RuntimeError("Every temporal mixer must use the official Mamba2 backend")
    if _Mamba1 is None or any(not isinstance(mixer, _Mamba1) for mixer in spectral):
        raise RuntimeError("Every spectral mixer must use the official Mamba1 backend")
    return {"mamba2": len(temporal), "mamba1": len(spectral)}
