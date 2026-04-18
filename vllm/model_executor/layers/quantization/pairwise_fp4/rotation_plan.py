# SPDX-License-Identifier: Apache-2.0
"""RotationPlanBuilder – orchestrates Monitor → PairConstructor → AngleSolver."""

from __future__ import annotations

import json
import os

import torch

from vllm.model_executor.layers.quantization.pairwise_fp4.angle_solver import (
    solve_angles,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.channel_monitor import (
    compute_risk_scores,
    load_risk_scores,
    save_risk_scores,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.pair_constructor import (
    construct_pairs,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.utils import (
    RotationPlan,
    empty_angles,
    empty_pairs,
    load_plan,
    save_plan,
)


class RotationPlanBuilder:
    """Build a static RotationPlan for a single layer / partition.

    Parameters in *config* dict (all optional, with defaults):
        risk_method   : str   – "max_abs" | "dynamic_range"  (default "max_abs")
        pair_policy   : str   – pairing strategy              (default "high_high")
        angle_solver  : str   – "heuristic" | "small_search"  (default "heuristic")
        top_ratio     : float – fraction of channels to pair   (default 0.05)
        risk_cache_dir: str   – directory for risk-score JSON  (default "")
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.risk_method: str = cfg.get("risk_method", "max_abs")
        self.pair_policy: str = cfg.get("pair_policy", "high_high")
        self.angle_solver: str = cfg.get("angle_solver", "heuristic")
        self.top_ratio: float = cfg.get("top_ratio", 0.05)
        self.risk_cache_dir: str = cfg.get("risk_cache_dir", "")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        layer_index: str,
        mode: str,
        weight: torch.Tensor | None = None,
        activation: torch.Tensor | None = None,
    ) -> RotationPlan:
        """Construct a plan for the given layer.

        Depending on *mode*, one or both of *weight* / *activation* may be
        required to compute risk scores and angles.
        """
        weight_risk = None
        act_risk = None

        if mode in ("weight_only", "joint"):
            if weight is None:
                raise ValueError(
                    f"mode={mode!r} requires a weight tensor"
                )
            weight_risk = self._get_risk_scores(
                weight, layer_index, target="weight",
            )

        if mode in ("activation_only", "joint"):
            if activation is None:
                raise ValueError(
                    f"mode={mode!r} requires an activation tensor"
                )
            act_risk = self._get_risk_scores(
                activation, layer_index, target="activation",
            )

        # ---- pair construction ----------------------------------------
        if mode == "weight_only":
            primary_risk = weight_risk
            secondary_risk = None
            policy = self.pair_policy
        elif mode == "activation_only":
            primary_risk = act_risk
            secondary_risk = None
            policy = self.pair_policy
        else:  # joint
            primary_risk = weight_risk
            secondary_risk = act_risk
            policy = ("joint_compatible"
                      if self.pair_policy not in ("random_baseline",)
                      else self.pair_policy)

        pairs = construct_pairs(
            risk_scores=primary_risk,
            policy=policy,
            top_ratio=self.top_ratio,
            secondary_risk_scores=secondary_risk,
        )

        if pairs.shape[0] == 0:
            return RotationPlan(
                mode=mode,
                layer_index=layer_index,
                pairs=empty_pairs(),
                angles=empty_angles(),
                pair_meta={"policy": policy, "top_ratio": self.top_ratio},
                angle_meta={"solver": self.angle_solver},
            )

        # ---- angle solving --------------------------------------------
        # Choose the risk scores to pass to the solver (combined for joint)
        solver_risk = primary_risk
        if secondary_risk is not None:
            solver_risk = primary_risk + secondary_risk

        angles = solve_angles(
            pairs=pairs,
            solver=self.angle_solver,
            weight=weight,
            activation=activation,
            risk_scores=solver_risk,
        )

        plan = RotationPlan(
            mode=mode,
            layer_index=layer_index,
            pairs=pairs,
            angles=angles,
            pair_meta={"policy": policy, "top_ratio": self.top_ratio},
            angle_meta={"solver": self.angle_solver},
        )
        plan.validate()
        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_risk_scores(
        self,
        tensor: torch.Tensor,
        layer_index: str,
        target: str,
    ) -> torch.Tensor:
        """Load cached risk scores or compute + save them."""
        cache_path = self._risk_cache_path(layer_index, target)

        if cache_path and os.path.isfile(cache_path):
            scores, _ = load_risk_scores(cache_path)
            return scores.to(device=tensor.device)

        scores = compute_risk_scores(
            tensor, method=self.risk_method, channel_dim=-1,
        )

        if cache_path:
            num_channels = scores.shape[0]
            save_risk_scores(
                scores,
                cache_path,
                metadata={
                    "layer_index": layer_index,
                    "target": target,
                    "method": self.risk_method,
                    "shard_id": 0,
                    "num_channels": num_channels,
                },
            )

        return scores

    def _risk_cache_path(self, layer_index: str, target: str) -> str:
        """Return the file path for caching risk scores, or empty string."""
        if not self.risk_cache_dir:
            return ""
        safe_name = layer_index.replace("/", "_").replace(".", "_")
        fname = f"{safe_name}__{target}__{self.risk_method}.json"
        return os.path.join(self.risk_cache_dir, fname)
