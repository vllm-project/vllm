# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-the-fly rank-based key compression for KV cache."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_compression.config import (
    KVCacheCompressionConfig,
    get_kv_compression_config,
)

logger = init_logger(__name__)

_LAYER_IDX_PATTERN = re.compile(r"layers\.(\d+)")
_manager: KVCompressionManager | None = None


@dataclass
class _HeadStats:
    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None  # Sum of squares of differences
    basis: torch.Tensor | None = None
    rank: int = 0
    last_logged: int = 0


class RunningHeadCompressor:
    """Tracks running PCA stats and projects keys to a low-rank subspace."""

    def __init__(self, dim: int, config: KVCacheCompressionConfig):
        self.dim = dim
        self.config = config
        self.stats = _HeadStats()
        self.device: torch.device | None = None

    def _ensure_buffers(self, device: torch.device) -> None:
        if self.device == device:
            return
        self.device = device
        self.stats = _HeadStats(
            mean=torch.zeros(self.dim, device=device, dtype=torch.float32),
            m2=torch.zeros(
                (self.dim, self.dim), device=device, dtype=torch.float32
            ),
        )

    def _recompute_basis(self) -> None:
        stats = self.stats
        if stats.mean is None or stats.m2 is None or stats.count < 2:
            return

        cov = stats.m2 / max(stats.count - 1, 1)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError as exc:
            logger.warning(f"[KV Compress] eigh failed: {exc}")
            return

        eigvals = eigvals.clamp_min(0.0)
        # Sort descending
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        total_energy = eigvals.sum()
        if total_energy <= 0:
            return

        cumsum = eigvals.cumsum(0) / total_energy
        rank = int((cumsum < self.config.energy_threshold).sum().item()) + 1
        rank = min(rank, eigvecs.shape[1])
        if self.config.max_rank is not None:
            rank = min(rank, self.config.max_rank)

        stats.rank = rank
        stats.basis = eigvecs[:, :rank]

    def compress_vector(self, key_vec: torch.Tensor) -> torch.Tensor:
        """
        Update running PCA stats with a single key vector and return
        its projection onto the current basis (shape preserved).
        """
        if key_vec.ndim != 1 or key_vec.shape[0] != self.dim:
            return key_vec

        with torch.no_grad():
            self._ensure_buffers(key_vec.device)
            stats = self.stats
            assert stats.mean is not None and stats.m2 is not None

            key_f = key_vec.float()
            stats.count += 1

            # Welford update for mean and covariance accumulator
            delta = key_f - stats.mean
            stats.mean += delta / stats.count
            delta2 = key_f - stats.mean
            stats.m2 += delta.unsqueeze(1) * delta2.unsqueeze(0)

            if stats.count >= self.config.min_tokens_before_svd and (
                stats.basis is None
                or stats.count % self.config.recompute_every == 0
            ):
                self._recompute_basis()

            # No basis yet, skip projection
            if stats.basis is None or stats.rank == 0:
                return key_vec

            centered = key_f - stats.mean
            proj = torch.matmul(centered, stats.basis)  # [rank]
            recon = torch.matmul(proj, stats.basis.T) + stats.mean
            out = recon.to(key_vec.dtype)

            if (
                self.config.log_every > 0
                and stats.count % self.config.log_every == 0
            ):
                energy_captured = float(
                    torch.sum(proj**2) / (torch.sum(centered**2) + 1e-6)
                )
                logger.info(
                    "[KV Compress] head rank=%d/%d tokens=%d energy≈%.3f",
                    stats.rank,
                    self.dim,
                    stats.count,
                    energy_captured,
                )

            return out


class LayerCompressor:
    """Per-layer compressor over KV heads."""

    def __init__(self, num_heads: int, head_dim: int, config: KVCacheCompressionConfig):
        self.heads = [
            RunningHeadCompressor(head_dim, config) for _ in range(num_heads)
        ]

    def compress(self, keys: torch.Tensor) -> torch.Tensor:
        # keys: [T, H_kv, d]
        if keys.ndim != 3:
            return keys
        T, H, d = keys.shape
        if H != len(self.heads) or d != self.heads[0].dim:
            return keys

        output = torch.empty_like(keys)
        for t in range(T):
            for h in range(H):
                output[t, h] = self.heads[h].compress_vector(keys[t, h])
        return output


class KVCompressionManager:
    """Owns per-layer compressors and orchestrates parsing + routing."""

    def __init__(self, config: KVCacheCompressionConfig):
        self.config = config
        self.layers: dict[int, LayerCompressor] = {}

    def _parse_layer_idx(self, layer_name: str) -> int | None:
        match = _LAYER_IDX_PATTERN.search(layer_name)
        if match:
            return int(match.group(1))
        return None

    def compress_decode_keys(
        self, key: torch.Tensor, layer_name: str, num_actual_tokens: int
    ) -> torch.Tensor:
        if not self.config.enabled or key is None:
            return key

        layer_idx = self._parse_layer_idx(layer_name)
        if layer_idx is None or not self.config.should_compress_layer(layer_idx):
            return key

        # Slice to actual tokens if padded
        key_actual = key[:num_actual_tokens] if num_actual_tokens else key
        if key_actual.numel() == 0:
            return key

        if layer_idx not in self.layers:
            num_heads = key_actual.shape[1]
            head_dim = key_actual.shape[2]
            self.layers[layer_idx] = LayerCompressor(
                num_heads=num_heads, head_dim=head_dim, config=self.config
            )

        compressed = self.layers[layer_idx].compress(key_actual)

        if key_actual.shape[0] == key.shape[0]:
            return compressed

        # Preserve padding tail if present
        out = key.clone()
        out[: key_actual.shape[0]] = compressed
        return out


def reset_kv_compression_state() -> None:
    """Reset the global compression manager (e.g., between model loads)."""
    global _manager
    _manager = None


def get_manager() -> KVCompressionManager | None:
    """Return the global compression manager (None if disabled)."""
    global _manager
    config = get_kv_compression_config()
    if not config.enabled:
        return None
    if _manager is None:
        _manager = KVCompressionManager(config)
    return _manager


def maybe_compress_kv_decode(
    key: torch.Tensor, layer_name: str, num_actual_tokens: int
) -> torch.Tensor:
    """
    Optionally compress decode-time keys to a low-rank subspace before caching.

    Args:
        key: Input key tensor [T, H_kv, d]
        layer_name: Name used to parse layer index
        num_actual_tokens: Number of real tokens (excludes padding)
    """
    manager = get_manager()
    if manager is None:
        return key
    return manager.compress_decode_keys(key, layer_name, num_actual_tokens)
