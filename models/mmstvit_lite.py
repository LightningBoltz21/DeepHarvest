"""MMST-ViT-Lite — a compact spatial-temporal transformer for county yield.

Architecture-inspired by, NOT a reproduction of:

    Lin et al., "MMST-ViT: Climate Change-aware Crop Yield Prediction via
    Multi-Modal Spatial-Temporal Vision Transformer", ICCV 2023.
    https://github.com/fudong03/MMST-ViT   (CC-BY-NC 4.0)

The paper's model consumes 384x384 Sentinel-2 tiles plus WRF-HRRR grids, needs the
2.61 TB CropNet corpus and multi-GPU self-supervised pretraining, ships no weights,
and covers ~200 counties. This keeps its three-stage structure while operating on
per-county time-series tensors, so it trains on a CPU in minutes across every county
with USDA data. It uses NO satellite imagery.

Stages, mirroring the paper:
  1. Multi-modal fusion   — weather channels + static county context projected into a
                            shared embedding at each timestep.
  2. Temporal transformer — self-attention across the 14 biweekly growing-season steps.
  3. Spatial attention    — cross-attention from a county to its k nearest neighbours
                            in the same year, the cheap analogue of the paper's
                            spatial transformer over neighbouring grids.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MMSTViTLite(nn.Module):
    def __init__(
        self,
        n_weather: int,
        n_static: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_temporal_layers: int = 3,
        n_timesteps: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Stage 1: multi-modal fusion -------------------------------------
        # Static county context is broadcast across timesteps and fused with the
        # per-timestep weather channels, so temporal attention sees both modalities.
        self.weather_proj = nn.Linear(n_weather, d_model)
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU()
        )
        self.pos = nn.Parameter(torch.zeros(1, n_timesteps, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # --- Stage 2: temporal transformer -----------------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_temporal_layers)
        self.temporal_norm = nn.LayerNorm(d_model)

        # --- Stage 3: spatial attention over k nearest neighbours -------------
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode(
        self, weather: torch.Tensor, static: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """[B, T, Fw] + [B, Fs] -> [B, D] pooled season embedding."""
        b, t, _ = weather.shape
        w = self.weather_proj(weather)
        s = self.static_proj(static).unsqueeze(1).expand(-1, t, -1)
        h = self.fuse(torch.cat([w, s], dim=-1)) + self.pos[:, :t]

        # mask is True where a timestep has NO data (an unfinished season).
        h = self.temporal(h, src_key_padding_mask=mask)
        h = self.temporal_norm(h)

        if mask is None:
            return h.mean(dim=1)
        # Mean over observed timesteps only, guarding the all-masked case.
        keep = (~mask).float().unsqueeze(-1)
        return (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        weather: torch.Tensor,        # [B, T, Fw]
        static: torch.Tensor,         # [B, Fs]
        nbr_weather: torch.Tensor,    # [B, K, T, Fw]
        nbr_static: torch.Tensor,     # [B, K, Fs]
        mask: torch.Tensor | None = None,        # [B, T]
        nbr_mask: torch.Tensor | None = None,    # [B, K, T]
        nbr_valid: torch.Tensor | None = None,   # [B, K] True where padding
    ) -> torch.Tensor:
        self_emb = self.encode(weather, static, mask)  # [B, D]

        b, k, t, fw = nbr_weather.shape
        flat_mask = nbr_mask.reshape(b * k, t) if nbr_mask is not None else None
        nbr_emb = self.encode(
            nbr_weather.reshape(b * k, t, fw),
            nbr_static.reshape(b * k, -1),
            flat_mask,
        ).reshape(b, k, self.d_model)

        attended, _ = self.spatial_attn(
            self_emb.unsqueeze(1), nbr_emb, nbr_emb, key_padding_mask=nbr_valid
        )
        spatial = self.spatial_norm(attended.squeeze(1))
        return self.head(torch.cat([self_emb, spatial], dim=-1)).squeeze(-1)


def build_knn(coords: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest counties for each county, by great-circle distance.

    Plain lat/lon Euclidean distance would distort badly across CONUS, so this uses
    a haversine metric on the unit sphere.
    """
    lat = np.deg2rad(coords[:, 0])
    lon = np.deg2rad(coords[:, 1])
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )
    # Chord distance is monotonic in great-circle distance, so ranking is identical.
    out = np.empty((len(xyz), k), dtype=np.int64)
    step = 512
    for start in range(0, len(xyz), step):
        block = xyz[start : start + step]
        d = ((block[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
        # Exclude self (distance 0) by taking k+1 and dropping the first column.
        idx = np.argpartition(d, kth=min(k, len(xyz) - 1), axis=1)[:, : k + 1]
        rows = np.arange(len(block))[:, None]
        order = np.argsort(d[rows, idx], axis=1)
        idx = idx[rows, order]
        for i in range(len(block)):
            row = idx[i][idx[i] != start + i][:k]
            if len(row) < k:  # pad by repeating the nearest available
                row = np.pad(row, (0, k - len(row)), mode="edge")
            out[start + i] = row
    return out
