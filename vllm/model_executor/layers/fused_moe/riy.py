# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RIY: Runtime expert masking and statistics for MoE models.

Provides two capabilities:
1. Per-(layer, expert) activation statistics (frequency + weight magnitude)
2. Expert masking with weight renormalization

Statistics are accumulated without filtering — the operator decides
what to do with them via the admin API or offline tooling.
"""

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RiyLayerStats:
    """Per-layer expert statistics accumulator."""
    num_experts: int
    # Token count per expert (how often selected by router)
    freq: torch.Tensor = field(init=False)
    # Sum of routing weights per expert (contribution magnitude)
    weight_sum: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.freq = torch.zeros(self.num_experts, dtype=torch.int64)
        self.weight_sum = torch.zeros(self.num_experts, dtype=torch.float64)

    def reset(self):
        self.freq.zero_()
        self.weight_sum.zero_()


class RiyState:
    """Global RIY state: statistics + expert mask.

    Thread-safe for admin API access. The hot path (apply_mask,
    record_stats) uses pre-computed tensors without locks.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._collecting = False
        self._num_layers = 0
        self._num_experts = 0
        self._layer_stats: list[RiyLayerStats] = []
        # Mask: set of (layer, expert) tuples to deactivate
        self._mask: set[tuple[int, int]] = set()
        # Pre-computed per-layer mask tensors on device (for hot path)
        self._mask_tensors: dict[int, torch.Tensor] = {}
        self._profile_loaded = False

    def initialize(self, num_layers: int, num_experts: int):
        """Called once during model init."""
        with self._lock:
            self._num_layers = num_layers
            self._num_experts = num_experts
            self._layer_stats = [
                RiyLayerStats(num_experts) for _ in range(num_layers)
            ]
            self._enabled = True
            logger.info("RIY initialized: %d layers, %d experts/layer",
                        num_layers, num_experts)

    def register_layer(self, layer_idx: int, num_experts: int):
        """Register a MoE layer. Called from FusedMoE.__init__."""
        with self._lock:
            if num_experts > self._num_experts:
                self._num_experts = num_experts
            if layer_idx >= self._num_layers:
                # Grow stats list
                while len(self._layer_stats) <= layer_idx:
                    self._layer_stats.append(
                        RiyLayerStats(num_experts))
                self._num_layers = len(self._layer_stats)
            self._enabled = True
            # Auto-load profile from config on first registration
            if not self._profile_loaded:
                self._try_load_profile_from_config()

    def _try_load_profile_from_config(self):
        """Load RIY profile if configured via CLI."""
        self._profile_loaded = True
        try:
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
            profile_path = cfg.parallel_config.riy_expert_profile
            if profile_path:
                self.load_profile(profile_path)
        except Exception:
            pass  # Config not available yet during early init

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def collecting(self) -> bool:
        return self._collecting

    def start_collection(self):
        with self._lock:
            self._collecting = True
            logger.info("RIY stats collection started")

    def stop_collection(self):
        with self._lock:
            self._collecting = False
            logger.info("RIY stats collection stopped")

    def reset_stats(self):
        with self._lock:
            for s in self._layer_stats:
                s.reset()
            logger.info("RIY stats reset")

    def get_stats(self) -> dict:
        """Export raw statistics as dict."""
        with self._lock:
            layers = []
            for i, s in enumerate(self._layer_stats):
                layers.append({
                    "layer": i,
                    "freq": s.freq.tolist(),
                    "weight_sum": s.weight_sum.tolist(),
                })
            return {
                "num_layers": self._num_layers,
                "num_experts": self._num_experts,
                "collecting": self._collecting,
                "mask_size": len(self._mask),
                "layers": layers,
            }

    def set_mask(self, pruned_experts: list[tuple[int, int]]):
        """Set expert mask. Experts in the list will be deactivated."""
        with self._lock:
            self._mask = set(pruned_experts)
            self._rebuild_mask_tensors()
            logger.info("RIY mask set: %d experts masked", len(self._mask))

    def clear_mask(self):
        with self._lock:
            self._mask.clear()
            self._mask_tensors.clear()
            logger.info("RIY mask cleared")

    def get_mask(self) -> list[list[int]]:
        with self._lock:
            return sorted([list(t) for t in self._mask])

    def load_profile(self, path: str):
        """Load mask from a RIY profile JSON."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"RIY profile not found: {path}")
        with open(p) as f:
            profile = json.load(f)
        experts = [tuple(x) for x in profile["pruned_experts"]]
        self.set_mask(experts)
        logger.info("RIY profile loaded: %s (%s, %d experts)",
                     path, profile.get("workload", "unknown"), len(experts))

    def _rebuild_mask_tensors(self):
        """Rebuild per-layer boolean mask tensors from the mask set."""
        self._mask_tensors.clear()
        for layer_idx, expert_idx in self._mask:
            if layer_idx not in self._mask_tensors:
                self._mask_tensors[layer_idx] = torch.zeros(
                    self._num_experts, dtype=torch.bool)
            self._mask_tensors[layer_idx][expert_idx] = True

    def get_mask_tensor(self, layer_idx: int) -> torch.Tensor | None:
        """Get pre-computed mask tensor for a layer. None = no mask."""
        return self._mask_tensors.get(layer_idx)

    def record_stats(self, layer_idx: int, topk_ids: torch.Tensor,
                     topk_weights: torch.Tensor):
        """Record activation stats for a layer. Called from hot path."""
        if not self._collecting or layer_idx >= len(self._layer_stats):
            return
        stats = self._layer_stats[layer_idx]
        # Frequency: count per expert
        ids_flat = topk_ids.flatten().long().cpu()
        stats.freq.scatter_add_(
            0, ids_flat,
            torch.ones_like(ids_flat, dtype=torch.int64))
        # Weight magnitude: sum of routing weights per expert
        weights_flat = topk_weights.flatten().double().cpu()
        stats.weight_sum.scatter_add_(0, ids_flat, weights_flat)


def apply_riy_mask(topk_weights: torch.Tensor,
                   topk_ids: torch.Tensor,
                   mask_tensor: torch.Tensor) -> torch.Tensor:
    """Zero out masked experts and renormalize weights.

    Args:
        topk_weights: (num_tokens, top_k) routing weights
        topk_ids: (num_tokens, top_k) expert indices
        mask_tensor: (num_experts,) bool tensor, True = masked

    Returns:
        Modified topk_weights with masked experts zeroed and renormalized.
    """
    # Find which selections hit masked experts
    hit = mask_tensor[topk_ids.long()]  # (num_tokens, top_k)
    # Zero their weights
    topk_weights = topk_weights.masked_fill(hit, 0.0)
    # Renormalize per token
    denom = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    topk_weights = topk_weights / denom
    return topk_weights


# Global singleton
_riy_state = RiyState()


def get_riy_state() -> RiyState:
    return _riy_state
