# SPDX-License-Identifier: Apache-2.0
"""Shared data structures and constants for pairwise_fp4 quantization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODES = ("weight_only", "activation_only", "joint")
VALID_RISK_METHODS = ("max_abs", "dynamic_range")
VALID_PAIR_POLICIES = (
    "adjacent_sorted",
    "high_high",
    "high_low",
    "random_baseline",
    "joint_compatible",
)
VALID_ANGLE_SOLVERS = ("heuristic", "small_search")

# ---------------------------------------------------------------------------
# RotationPlan – the shared data structure across all modules
# ---------------------------------------------------------------------------


@dataclass
class RotationPlan:
    """Static rotation plan consumed by RotationApplier at inference time.

    Attributes:
        mode: One of "weight_only", "activation_only", "joint".
        layer_index: Fully-qualified layer name,
            e.g. "model.layers.0.self_attn.qkv_proj".
        pairs: int64 Tensor of shape ``(N, 2)`` — channel index pairs.
        angles: float32 Tensor of shape ``(N,)`` — rotation angles in radians.
        pair_meta: Metadata about pair construction (policy, top_ratio, …).
        angle_meta: Metadata about angle solving (solver name, …).
    """

    mode: str
    layer_index: str
    pairs: torch.Tensor  # (N, 2), int64
    angles: torch.Tensor  # (N,), float32
    pair_meta: dict = field(default_factory=dict)
    angle_meta: dict = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------

    @property
    def num_pairs(self) -> int:
        return self.pairs.shape[0]

    @property
    def is_empty(self) -> bool:
        return self.pairs.numel() == 0

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}', expected one of {VALID_MODES}"
            )
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError(
                f"pairs must have shape (N, 2), got {tuple(self.pairs.shape)}"
            )
        if self.angles.ndim != 1:
            raise ValueError(
                f"angles must be 1-D, got ndim={self.angles.ndim}"
            )
        if self.pairs.shape[0] != self.angles.shape[0]:
            raise ValueError(
                f"pairs ({self.pairs.shape[0]}) and angles "
                f"({self.angles.shape[0]}) length mismatch"
            )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "layer_index": self.layer_index,
            "num_pairs": self.num_pairs,
            "pairs": self.pairs.tolist(),
            "angles": self.angles.tolist(),
            "pair_meta": self.pair_meta,
            "angle_meta": self.angle_meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RotationPlan:
        pairs = torch.tensor(d["pairs"], dtype=torch.int64)
        if pairs.ndim == 1 and pairs.numel() == 0:
            pairs = pairs.reshape(0, 2)
        angles = torch.tensor(d["angles"], dtype=torch.float32)
        return cls(
            mode=d["mode"],
            layer_index=d["layer_index"],
            pairs=pairs,
            angles=angles,
            pair_meta=d.get("pair_meta", {}),
            angle_meta=d.get("angle_meta", {}),
        )


# ---------------------------------------------------------------------------
# Plan save / load helpers (used by rotation_plan.py and tests)
# ---------------------------------------------------------------------------


def save_plan(plan: RotationPlan, path: str) -> None:
    plan.validate()
    with open(path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2)


def load_plan(path: str) -> RotationPlan:
    with open(path) as f:
        d = json.load(f)
    plan = RotationPlan.from_dict(d)
    plan.validate()
    return plan


# ---------------------------------------------------------------------------
# Empty-tensor factories (avoid shape ambiguity across modules)
# ---------------------------------------------------------------------------


def empty_pairs(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a canonical empty pairs tensor ``(0, 2), int64``."""
    return torch.empty(0, 2, dtype=torch.int64, device=device)


def empty_angles(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a canonical empty angles tensor ``(0,), float32``."""
    return torch.empty(0, dtype=torch.float32, device=device)
