# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch


@dataclass(frozen=True)
class RecirculationConfig:
    """Fixed-coefficient residual-stream recirculation configuration."""

    source_layer: int
    destination_layer: int
    alpha: float = 0.15
    beta: float | None = None
    ramp_tokens: int = 0

    @classmethod
    def from_hf_config(cls, hf_config: object) -> "RecirculationConfig | None":
        raw_config = getattr(hf_config, "recirculation_config", None)
        if raw_config is None:
            return None
        if not isinstance(raw_config, dict):
            raise ValueError("recirculation_config must be a dictionary")

        valid_keys = {
            "source_layer",
            "destination_layer",
            "alpha",
            "beta",
            "ramp_tokens",
        }
        unknown_keys = raw_config.keys() - valid_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown recirculation_config fields: {unknown}")

        missing_keys = {"source_layer", "destination_layer"} - raw_config.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Missing recirculation_config fields: {missing}")

        config = cls(
            source_layer=raw_config["source_layer"],
            destination_layer=raw_config["destination_layer"],
            alpha=raw_config.get("alpha", 0.15),
            beta=raw_config.get("beta"),
            ramp_tokens=raw_config.get("ramp_tokens", 0),
        )
        config.validate(hf_config.num_hidden_layers)
        if config.alpha == 0.0 and config.beta in (None, 1.0):
            return None
        return config

    def validate(self, num_hidden_layers: int) -> None:
        for name, value in (
            ("source_layer", self.source_layer),
            ("destination_layer", self.destination_layer),
            ("ramp_tokens", self.ramp_tokens),
        ):
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")

        if not 0 <= self.destination_layer < self.source_layer < num_hidden_layers:
            raise ValueError(
                "recirculation layers must satisfy 0 <= destination_layer < "
                "source_layer < num_hidden_layers"
            )
        if self.ramp_tokens < 0:
            raise ValueError("ramp_tokens must be non-negative")

        self._validate_coefficient("alpha", self.alpha)
        if self.beta is not None:
            self._validate_coefficient("beta", self.beta)

    @staticmethod
    def _validate_coefficient(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a real number")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    def mix(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Mix a norm-matched source residual into the destination residual."""
        source_float = source.float()
        destination_float = destination.float()
        source_norm = torch.linalg.vector_norm(
            source_float, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).tiny)
        destination_norm = torch.linalg.vector_norm(
            destination_float, dim=-1, keepdim=True
        )
        normalized_source = source_float * (destination_norm / source_norm)

        alpha: float | torch.Tensor = self.alpha
        if self.ramp_tokens:
            ramp = positions.to(dtype=torch.float32).clamp(max=self.ramp_tokens)
            alpha = (ramp / self.ramp_tokens).unsqueeze(-1) * self.alpha
        beta = 1.0 - alpha if self.beta is None else self.beta
        mixed = beta * destination_float + alpha * normalized_source
        return mixed.to(dtype=destination.dtype)
