# SPDX-License-Identifier: Apache-2.0
"""PairwiseFP4Config – QuantizationConfig subclass for pairwise_fp4."""

from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.pairwise_fp4.utils import (
    VALID_ANGLE_SOLVERS,
    VALID_MODES,
    VALID_PAIR_POLICIES,
    VALID_RISK_METHODS,
)


class PairwiseFP4Config(QuantizationConfig):
    """Quantization config for pairwise Givens-rotation + FP4 prototype."""

    def __init__(
        self,
        fp4_format: str = "nvfp4",
        mode: str = "weight_only",
        risk_method: str = "max_abs",
        pair_policy: str = "high_high",
        angle_solver: str = "heuristic",
        top_ratio: float = 0.05,
        group_size: int = 16,
        risk_cache_dir: str = "/tmp/pairwise_fp4_cache",
        rotation_plan_path: str = "",
        use_prebuilt_plan: bool = False,
    ) -> None:
        super().__init__()
        # --- validation ---
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}, expected one of {VALID_MODES}")
        if risk_method not in VALID_RISK_METHODS:
            raise ValueError(
                f"Invalid risk_method {risk_method!r}, "
                f"expected one of {VALID_RISK_METHODS}")
        if pair_policy not in VALID_PAIR_POLICIES:
            raise ValueError(
                f"Invalid pair_policy {pair_policy!r}, "
                f"expected one of {VALID_PAIR_POLICIES}")
        if angle_solver not in VALID_ANGLE_SOLVERS:
            raise ValueError(
                f"Invalid angle_solver {angle_solver!r}, "
                f"expected one of {VALID_ANGLE_SOLVERS}")
        if fp4_format != "nvfp4":
            raise ValueError(
                f"Only 'nvfp4' fp4_format is supported, got {fp4_format!r}")

        self.fp4_format = fp4_format
        self.mode = mode
        self.risk_method = risk_method
        self.pair_policy = pair_policy
        self.angle_solver = angle_solver
        self.top_ratio = top_ratio
        self.group_size = group_size
        self.risk_cache_dir = risk_cache_dir
        self.rotation_plan_path = rotation_plan_path
        self.use_prebuilt_plan = use_prebuilt_plan

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return "pairwise_fp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Return empty so LLM(quantization="pairwise_fp4") works with
        # defaults (no config file required).  Custom params can be passed
        # via hf_overrides={"quantization_config_dict_json": ...}.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PairwiseFP4Config":
        return cls(
            fp4_format=cls.get_from_keys_or(
                config, ["fp4_format"], "nvfp4"),
            mode=cls.get_from_keys_or(
                config, ["mode"], "weight_only"),
            risk_method=cls.get_from_keys_or(
                config, ["risk_method"], "max_abs"),
            pair_policy=cls.get_from_keys_or(
                config, ["pair_policy"], "high_high"),
            angle_solver=cls.get_from_keys_or(
                config, ["angle_solver"], "heuristic"),
            top_ratio=cls.get_from_keys_or(
                config, ["top_ratio"], 0.05),
            group_size=cls.get_from_keys_or(
                config, ["group_size"], 16),
            risk_cache_dir=cls.get_from_keys_or(
                config, ["risk_cache_dir"], "/tmp/pairwise_fp4_cache"),
            rotation_plan_path=cls.get_from_keys_or(
                config, ["rotation_plan_path"], ""),
            use_prebuilt_plan=cls.get_from_keys_or(
                config, ["use_prebuilt_plan"], False),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            from vllm.model_executor.layers.quantization.pairwise_fp4.linear_method import (  # noqa: E501
                PairwiseFP4LinearMethod,
            )
            return PairwiseFP4LinearMethod(self)
        return None

    @classmethod
    def from_config_dict_json(cls, json_str: str) -> "PairwiseFP4Config":
        """Create config from a JSON string (used by hf_overrides)."""
        import json
        config = json.loads(json_str)
        return cls.from_config(config)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_plan_builder_config(self) -> dict:
        """Return config dict suitable for ``RotationPlanBuilder``."""
        return {
            "risk_method": self.risk_method,
            "pair_policy": self.pair_policy,
            "angle_solver": self.angle_solver,
            "top_ratio": self.top_ratio,
            "risk_cache_dir": self.risk_cache_dir,
        }
