# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling Titan vision + audio towers.

The vision tower (``InklingVision`` / ``HMLPPatchEncoder``) emits one token per
image patch; the audio tower (``InklingAudio``) emits one token per audio frame.
Both use vLLM's standard ``RMSNorm`` (CPU-friendly, with a native fallback).
"""

from __future__ import annotations

from itertools import combinations
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.layernorm import RMSNorm

from ..configs import InklingAudioConfig, InklingVisionConfig

# ===========================================================================
# Vision tower (HMLPPatchEncoder / InklingVision)
# ===========================================================================


def _prime_factors(n: int) -> list[int]:
    """Return the prime factors of *n* in ascending order."""
    if n < 1:
        raise ValueError("n must be a positive integer")

    factors: list[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 2
    if n > 1:
        factors.append(n)
    return factors


def linear_sum_assignment(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Implement SciPy's assignment for Inkling's ordered L1 cost matrix."""
    rows = np.arange(cost_matrix.shape[0])
    cols = np.array(
        min(
            combinations(range(cost_matrix.shape[1]), len(rows)),
            key=lambda candidate: cost_matrix[rows, candidate].sum(),
        )
    )
    return rows, cols


def plan_out_scales(
    temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int = 3
) -> list[tuple[int, int, int, int]]:
    """Plan the (t, h, w, c) dimensions for each HMLP layer.

    Spatial dims expand first, then temporal; channel counts round up to
    multiples of 64.
    """
    if patch_size <= 1:
        raise ValueError(
            "patch_size must be greater than 1, otherwise this doesn't make sense"
        )

    def _round_up(x: int) -> int:
        return int(np.ceil(x / 64)) * 64

    last_h_scale = 1
    scales: list[tuple[int, int, int, int]] = [(1, 1, 1, n_channels)]
    for pscale in _prime_factors(patch_size)[::-1]:
        last_h_scale *= pscale
        scales.append(
            (
                1,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels),
            )
        )
    last_t_scale = 1
    for tscale in _prime_factors(temporal_patch_size)[::-1]:
        last_t_scale *= tscale
        scales.append(
            (
                last_t_scale,
                last_h_scale,
                last_h_scale,
                _round_up((last_h_scale**2) * n_channels * last_t_scale),
            )
        )

    size_reduction = np.prod(np.array(scales)[:, :-1], 1)

    log_ideal_scales = np.linspace(
        0,
        np.log(patch_size * patch_size * temporal_patch_size * n_channels),
        n_layers + 1,
    )
    cost_matrix = np.abs(log_ideal_scales[:, None] - np.log(size_reduction)[None])

    if n_layers >= len(scales):
        idxs = np.argmin(cost_matrix, axis=1)
    else:
        idxs = linear_sum_assignment(cost_matrix)[1]

    assert len(idxs) >= 2
    idxs[0] = 0
    idxs[-1] = len(scales) - 1

    return [scales[i] for i in idxs]


def fold_timespace_to_depth(
    vision_patches_bthwc: torch.Tensor, t_fold: int, hw_fold: int
) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, T//t, H//hw, W//hw, C*(t*hw**2))."""
    B, T, H, W, C = vision_patches_bthwc.shape

    assert T % t_fold == 0, f"Temporal dimension {T} must be divisible by {t_fold}"
    assert H % hw_fold == 0, f"Height dimension {H} must be divisible by {hw_fold}"
    assert W % hw_fold == 0, f"Width dimension {W} must be divisible by {hw_fold}"

    t_new = T // t_fold
    h_new = H // hw_fold
    w_new = W // hw_fold

    x = vision_patches_bthwc.reshape(
        B, t_new, t_fold, h_new, hw_fold, w_new, hw_fold, C
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    x = x.reshape(B, t_new, h_new, w_new, t_fold * hw_fold * hw_fold * C)
    return x


class HMLPPatchEncoder(nn.Module):
    def __init__(self, config: InklingVisionConfig):
        super().__init__()
        self.decoder_dmodel = config.decoder_dmodel
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.n_channels = config.n_channels
        self.n_layers = config.n_layers
        self.use_vision_norm = config.use_vision_norm

        self.scales: list[tuple[int, int, int, int]] = plan_out_scales(
            self.temporal_patch_size, self.patch_size, self.n_layers, self.n_channels
        )
        self.layers: nn.ModuleDict = nn.ModuleDict()
        for i, (start_scale, end_scale) in enumerate(
            zip(self.scales[:-1], self.scales[1:])
        ):
            shuffle_mult = (
                (end_scale[0] // start_scale[0])
                * (end_scale[1] // start_scale[1])
                * (end_scale[2] // start_scale[2])
            )
            if i == self.n_layers - 1:
                self.layers[f"linear_{i}"] = nn.Linear(
                    start_scale[3] * shuffle_mult, self.decoder_dmodel, bias=False
                )
            else:
                self.layers[f"linear_{i}"] = nn.Linear(
                    start_scale[3] * shuffle_mult, end_scale[3], bias=False
                )
                self.layers[f"norm_{i}"] = RMSNorm(end_scale[3])

        self.final_norm: RMSNorm | None = None
        if self.use_vision_norm:
            assert self.decoder_dmodel is not None
            self.final_norm = RMSNorm(self.decoder_dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused norm+gelu on CUDA (same fp32-accum/bf16-rounding structure as
        # the generic path below; differs only by reduction order).
        fused = None
        if x.is_cuda and x.dtype == torch.bfloat16:
            from vllm.models.inkling.nvidia.ops.mm_towers import rmsnorm_gelu

            fused = rmsnorm_gelu

        num_patches, T, H, W, C = x.shape
        prefolded = False
        for i, (start_scale, end_scale) in enumerate(
            zip(self.scales[:-1], self.scales[1:])
        ):
            t_fold = end_scale[0] // start_scale[0]
            hw_fold = end_scale[1] // start_scale[1]
            if (hw_fold > 1 or t_fold > 1) and not prefolded:
                x = fold_timespace_to_depth(x, t_fold, hw_fold)
            prefolded = False
            assert x.shape[1:-1] == (
                T // end_scale[0],
                H // end_scale[1],
                W // end_scale[2],
            )
            x = self.layers[f"linear_{i}"](x)
            if i < self.n_layers - 1:
                norm = cast(RMSNorm, self.layers[f"norm_{i}"])
                if fused is not None:
                    # If the NEXT layer starts with a copying fold (spatial
                    # dims still > 1 after folding), store this layer's
                    # output directly in the folded layout instead.
                    nxt = self.scales[i + 2]
                    ntf = nxt[0] // end_scale[0]
                    nhf = nxt[1] // end_scale[1]
                    copy_fold = (ntf > 1 or nhf > 1) and (
                        x.shape[2] // nhf > 1 or x.shape[3] // nhf > 1
                    )
                    x = fused(
                        x,
                        norm.weight,
                        norm.variance_epsilon,
                        gelu=True,
                        fold=(ntf, nhf) if copy_fold else None,
                    )
                    prefolded = copy_fold
                else:
                    # rms_norm kernel only supports rank 2-4; x is 5-D here.
                    orig = x.shape
                    x = norm(x.reshape(-1, x.shape[-1])).reshape(orig)
                    x = F.gelu(x)

        if self.final_norm is not None:
            if fused is not None:
                x = fused(
                    x,
                    self.final_norm.weight,
                    self.final_norm.variance_epsilon,
                    gelu=False,
                )
            else:
                orig = x.shape
                x = self.final_norm(x.reshape(-1, x.shape[-1])).reshape(orig)

        x = x.reshape(num_patches, -1)
        return x


class InklingVision(nn.Module):
    def __init__(self, config: InklingVisionConfig, prefix: str = ""):
        del prefix
        super().__init__()
        assert config.vision_encoder_type == "hmlp"
        self.vision_encoder = HMLPPatchEncoder(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(vision_features)


# ===========================================================================
# Audio tower (InklingAudio)
# ===========================================================================


class InklingAudio(nn.Module):
    def __init__(self, config: InklingAudioConfig, prefix: str = ""):
        del prefix
        super().__init__()
        assert config.audio_mode == "dmel"
        self.n_mel_bins = config.n_mel_bins
        self.mel_vocab_size = config.mel_vocab_size
        self.use_audio_norm = config.use_audio_norm
        self.encoder = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size, config.decoder_dmodel
        )
        self.final_norm: RMSNorm | None = None
        if self.use_audio_norm:
            assert config.decoder_dmodel is not None
            self.final_norm = RMSNorm(config.decoder_dmodel, eps=1e-6)

    @property
    def dtype(self) -> torch.dtype:
        return self.encoder.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.encoder.weight.device

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        assert audio_features.shape[1] == self.n_mel_bins

        # dMel bins are integer indices; cast once to int32 on the right device
        # (no float round-trip).
        audio_features = audio_features.to(
            device=self.encoder.weight.device, dtype=torch.int32
        )

        weight = self.encoder.weight
        if audio_features.is_cuda and weight.dtype == torch.bfloat16:
            # One kernel: per-bin offset + embedding gather + fp32 sum + norm.
            # Skips the [T, n_mel_bins, D] intermediate entirely (bit-exact).
            from vllm.models.inkling.nvidia.ops.mm_towers import dmel_embed_sum_norm

            return dmel_embed_sum_norm(
                audio_features.contiguous(),
                weight,
                self.final_norm.weight if self.final_norm is not None else None,
                self.final_norm.variance_epsilon if self.final_norm else 0.0,
            )

        embedding_indices = (
            torch.arange(self.n_mel_bins, device=audio_features.device)
            * self.mel_vocab_size
        ).unsqueeze(0) + audio_features

        hidden_states = (
            self.encoder(embedding_indices.reshape(-1))
            .reshape(audio_features.shape[0], audio_features.shape[1], -1)
            .sum(axis=1)
        )

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states
