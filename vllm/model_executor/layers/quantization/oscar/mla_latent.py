# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR INT2 "fake-quant" for the MLA shared latent (``c_kv``).

On every MLA KV-cache write we optionally

  1. project ``c_kv`` onto a high-precision subspace (top-k most
     sensitivity-weighted latent directions, from the ``kv_b_proj``
     Hessian) and keep that component in full precision,
  2. rotate the residual by the calibrated per-layer ``R``,
  3. quantize it groupwise to INT2 and immediately dequantize,
  4. inverse-rotate and add the high-precision component back.

``k_pe`` (the 64-dim RoPE part) and, for GLM-5.2, the DSA/NSA indexer
cache are left untouched, matching the reference.

This is a *quality-only* measurement path. The value stored in the KV
cache is still BF16/FP8, so the memory footprint is unchanged -- it
measures the accuracy cost of an INT2 latent cache without requiring an
INT2-storage pool and an in-kernel dequant rewrite of the MLA backends.
For the real-storage path see ``v1/attention/ops/triton_oscar_mla.py``.

For GLM-5.2 (``kv_lora_rank=512``) the rotations are per-layer
``[512, 512]`` float32 orthogonal matrices.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import TYPE_CHECKING

import torch

from vllm import envs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lloyd-Max INT2 constants (MSE-optimal 4-level quantizer for N(0, 1)).
# Decision boundaries between levels 0-1, 1-2, 2-3.
_LM_THRESHOLDS = (-0.9810652732849121, 0.0, 0.9810652732849121)
# Centroid of each level.
_LM_CENTROIDS = (
    -1.5095585584640503,
    -0.4527800381183624,
    0.4527800381183624,
    1.5095585584640503,
)
_LM_SPAN = _LM_CENTROIDS[3] - _LM_CENTROIDS[0]  # ~3.019
# Empirical rescale that keeps the uniform dequant path stable in context.
_LM_RATIO = 1.16


def _make_hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Random-sign Haar-orthogonal matrix, shared by every layer.

    The uncalibrated fallback: incoherence without any calibration data.
    """
    rng = torch.Generator()
    rng.manual_seed(0)
    signs = torch.where(
        torch.rand(d, generator=rng) > 0.5, torch.ones(d), -torch.ones(d)
    ).view(d, 1)
    # QR of a Gaussian gives a Haar-uniform orthogonal matrix.
    rot, _ = torch.linalg.qr(torch.randn(d, d, generator=rng))
    return (rot * signs).to(dtype=torch.float32, device=device).contiguous()


def fake_quant_int2_groupwise(
    x: torch.Tensor,
    group_size: int,
    lloyd_max: bool = False,
) -> torch.Tensor:
    """INT2 quantize-then-dequantize in groups along the last dim.

    ``lloyd_max`` selects the MSE-optimal 4-level quantizer for N(0, 1)
    instead of uniform affine min-max. The rotation makes the latent
    approximately Gaussian, so Lloyd-Max is the recommended setting.
    """
    orig_shape = x.shape
    x = x.reshape(-1, group_size).to(torch.float32)

    if lloyd_max:
        # Normalize per group, bucketize, then map back.
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        std = (diff.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()
        z = diff / std

        t0, t1, t2 = _LM_THRESHOLDS
        q = (
            (z >= t0).to(torch.uint8)
            + (z >= t1).to(torch.uint8)
            + (z >= t2).to(torch.uint8)
        )  # 0..3

        scale = (_LM_SPAN / 3.0) * _LM_RATIO * std
        zero = -_LM_CENTROIDS[0] / (_LM_SPAN / 3.0) - mean / scale
        x_deq = (q.to(torch.float32) - zero) * scale
    else:
        # Uniform affine min-max over 4 levels.
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = torch.where(
            (x_max - x_min).abs() > 1e-8,
            (x_max - x_min) / 3.0,
            torch.ones_like(x_min),
        )
        q = ((x - x_min) / scale).round().clamp_(0.0, 3.0)
        x_deq = q * scale + x_min

    return x_deq.reshape(orig_shape)


class OscarMlaLatentQuantizer:
    """Applies the OSCAR latent transform to ``c_kv`` on each cache write.

    Constructed lazily on first use so that ``kv_lora_rank``, device and
    dtype can be taken from the first tensor we see.
    """

    def __init__(self, kv_lora_rank: int, device: torch.device) -> None:
        group_size = envs.VLLM_OSCAR_MLA_KV_GROUP_SIZE
        if kv_lora_rank % group_size != 0:
            raise ValueError(
                f"VLLM_OSCAR_MLA_KV_GROUP_SIZE={group_size} must divide "
                f"kv_lora_rank={kv_lora_rank}"
            )
        self.kv_lora_rank = kv_lora_rank
        self.group_size = group_size
        self.lloyd_max = envs.VLLM_OSCAR_LLOYD_MAX
        self.device = device

        self.rotation_path = envs.VLLM_OSCAR_MLA_KV_ROTATION_PATH
        self.hp_subspace_path = envs.VLLM_OSCAR_MLA_KV_HP_SUBSPACE_PATH
        if self.rotation_path and not (
            self.rotation_path == "hadamard" or os.path.isdir(self.rotation_path)
        ):
            raise ValueError(
                f"VLLM_OSCAR_MLA_KV_ROTATION_PATH={self.rotation_path!r} is "
                'neither a directory nor "hadamard"'
            )
        # Rotations and subspaces are loaded on first use per layer; fp32 so
        # the hot path never re-casts.
        self._rotations: dict[int, torch.Tensor | None] = {}
        self._hp: dict[int, torch.Tensor | None] = {}

        self.dump_dir = envs.VLLM_OSCAR_MLA_KV_DUMP_DIR
        self.dump_max_tokens = envs.VLLM_OSCAR_MLA_KV_DUMP_MAX_TOKENS
        self._dump_counts: dict[int, int] = {}
        self._dump_buffers: dict[int, list[torch.Tensor]] = {}
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            logger.info(
                "[OSCAR-MLA] dumping up to %d c_kv tokens/layer to %s",
                self.dump_max_tokens,
                self.dump_dir,
            )
            atexit.register(self.flush_dumps)

    @property
    def quantizes(self) -> bool:
        """Whether a write actually gets transformed (vs dump-only)."""
        return bool(self.rotation_path)

    def _rotation(self, layer_idx: int) -> torch.Tensor | None:
        """Per-layer ``[d, d]`` rotation, loaded on first use.

        A layer with no checkpoint falls back to identity, i.e. plain
        groupwise INT2 for that layer.
        """
        if layer_idx in self._rotations:
            return self._rotations[layer_idx]

        d = self.kv_lora_rank
        if not self.rotation_path:
            rot = None
        elif self.rotation_path == "hadamard":
            rot = _make_hadamard(d, self.device)
        else:
            path = os.path.join(self.rotation_path, f"layer_{layer_idx}.pt")
            if os.path.exists(path):
                loaded = torch.load(path, map_location="cpu")
                if tuple(loaded.shape) != (d, d):
                    raise ValueError(
                        f"[OSCAR-MLA] {path}: expected [{d}, {d}] "
                        f"(kv_lora_rank={d}), got {tuple(loaded.shape)}"
                    )
                rot = loaded.to(dtype=torch.float32, device=self.device).contiguous()
                logger.info("[OSCAR-MLA] layer %d rotation <- %s", layer_idx, path)
            else:
                rot = None
                logger.warning(
                    "[OSCAR-MLA] no rotation for layer %d in %s - using identity",
                    layer_idx,
                    self.rotation_path,
                )
        self._rotations[layer_idx] = rot
        return rot

    def _hp_subspace(self, layer_idx: int) -> torch.Tensor | None:
        """Per-layer ``[k, kv_lora_rank]`` orthonormal subspace, or None."""
        if layer_idx in self._hp:
            return self._hp[layer_idx]
        sub = None
        if self.hp_subspace_path and os.path.isdir(self.hp_subspace_path):
            path = os.path.join(self.hp_subspace_path, f"layer_{layer_idx}.pt")
            if os.path.exists(path):
                sub = (
                    torch.load(path, map_location="cpu")
                    .to(dtype=torch.float32, device=self.device)
                    .contiguous()
                )
                logger.info(
                    "[OSCAR-MLA] layer %d HP subspace (k=%d) <- %s",
                    layer_idx,
                    sub.shape[0],
                    path,
                )
        self._hp[layer_idx] = sub
        return sub

    def apply(self, layer_idx: int, c_kv: torch.Tensor) -> torch.Tensor:
        """Rotate -> INT2 fake-quant -> inverse-rotate a ``c_kv`` write."""
        out_dtype = c_kv.dtype
        x = c_kv.to(torch.float32)

        hp = None
        uk = self._hp_subspace(layer_idx)
        if uk is not None:
            # Projection onto the sensitive subspace, kept full precision.
            hp = (x @ uk.T) @ uk
            x = x - hp

        rot = self._rotation(layer_idx)
        if rot is not None:
            x = torch.matmul(x, rot)

        x = fake_quant_int2_groupwise(x, self.group_size, self.lloyd_max)

        if rot is not None:
            x = torch.matmul(x, rot.T)
        if hp is not None:
            x = x + hp
        return x.to(out_dtype)

    def maybe_dump(self, layer_idx: int, c_kv: torch.Tensor) -> None:
        """Accumulate ``c_kv`` for offline rotation fitting."""
        if not self.dump_dir:
            return
        count = self._dump_counts.get(layer_idx, 0)
        if count >= self.dump_max_tokens:
            return
        # GLM-5.2 emits c_kv 3D on some paths and 2D on others; normalize so
        # the per-layer buffer does not mix ranks.
        c_kv = c_kv.reshape(-1, c_kv.shape[-1])
        n = min(c_kv.shape[0], self.dump_max_tokens - count)
        self._dump_buffers.setdefault(layer_idx, []).append(
            c_kv[:n].detach().to(torch.float32).cpu()
        )
        self._dump_counts[layer_idx] = count + n
        if self._dump_counts[layer_idx] >= self.dump_max_tokens:
            self._flush_layer(layer_idx)

    def _flush_layer(self, layer_idx: int) -> None:
        buf = self._dump_buffers.pop(layer_idx, None)
        if not buf:
            return
        path = os.path.join(self.dump_dir, f"layer_{layer_idx}.pt")
        torch.save(torch.cat(buf, dim=0), path)
        logger.info(
            "[OSCAR-MLA] flushed layer %d c_kv dump (%d tokens) -> %s",
            layer_idx,
            self._dump_counts[layer_idx],
            path,
        )

    def flush_dumps(self) -> None:
        for layer_idx in list(self._dump_buffers):
            self._flush_layer(layer_idx)


_QUANTIZER: OscarMlaLatentQuantizer | None = None
_QUANTIZER_INIT = False


def oscar_mla_enabled() -> bool:
    """True when either the eval path or the calibration dump is requested."""
    return bool(envs.VLLM_OSCAR_MLA_KV_ROTATION_PATH or envs.VLLM_OSCAR_MLA_KV_DUMP_DIR)


def get_mla_latent_quantizer(
    kv_lora_rank: int,
    device: torch.device,
) -> OscarMlaLatentQuantizer | None:
    """Process-wide lazily-built quantizer, or ``None`` when disabled."""
    global _QUANTIZER, _QUANTIZER_INIT
    if not _QUANTIZER_INIT:
        _QUANTIZER_INIT = True
        if oscar_mla_enabled():
            _QUANTIZER = OscarMlaLatentQuantizer(kv_lora_rank, device)
            logger.info(
                "[OSCAR-MLA] INT2 latent fake-quant active "
                "(kv_lora_rank=%d, group_size=%d, lloyd_max=%s, rotation=%s). "
                "Memory footprint is unchanged; this measures accuracy only.",
                kv_lora_rank,
                _QUANTIZER.group_size,
                _QUANTIZER.lloyd_max,
                _QUANTIZER.rotation_path or "<none>",
            )
    return _QUANTIZER
