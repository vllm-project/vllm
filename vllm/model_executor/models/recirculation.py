# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RecirculationConfig:
    """Fixed-coefficient residual-stream recirculation configuration."""

    source_layer: int
    destination_layer: int
    alpha: float = 0.15
    beta: float | None = None
    ramp_tokens: int = 0
    wavefront: bool = False

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
            "wavefront",
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
            wavefront=raw_config.get("wavefront", False),
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
        if type(self.wavefront) is not bool:
            raise ValueError("wavefront must be a boolean")

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


@dataclass(frozen=True)
class RecirculationCapabilities:
    """Execution modes implemented by a concrete model-family adapter."""

    adapter: str
    serial: bool = True
    wavefront: bool = True


class RecirculationDecoderMixin:
    """Shared Recirculation execution for residual-stream decoder stacks."""

    recirculation_capabilities: ClassVar[RecirculationCapabilities | None] = None
    recirculation_config: RecirculationConfig | None
    start_layer: int
    end_layer: int
    layers: nn.ModuleList
    norm: nn.Module

    def _init_recirculation(
        self,
        hf_config: object,
        start_layer: int,
        end_layer: int,
    ) -> None:
        capabilities = type(self).__dict__.get("recirculation_capabilities")
        if capabilities is None:
            self.recirculation_config = None
            return

        config = RecirculationConfig.from_hf_config(hf_config)
        if config is not None and not getattr(hf_config, "is_causal", True):
            raise ValueError("Recirculation requires causal attention")
        if config is not None and (
            start_layer != 0 or end_layer != hf_config.num_hidden_layers
        ):
            raise ValueError("Recirculation does not support pipeline parallelism")
        self.recirculation_config = config

    def has_recirculation_adapter(self) -> bool:
        return self.get_recirculation_capabilities() is not None

    def get_recirculation_capabilities(
        self,
    ) -> RecirculationCapabilities | None:
        return type(self).__dict__.get("recirculation_capabilities")

    def _forward_recirculation(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        wavefront_warmup: bool | None,
        wavefront_positions: torch.Tensor | None,
        wavefront_pending: torch.Tensor | None,
        **layer_kwargs: Any,
    ) -> torch.Tensor:
        config = self.recirculation_config
        assert config is not None
        if config.wavefront and wavefront_warmup is not None:
            return self._forward_recirculation_wavefront(
                positions,
                hidden_states,
                residual,
                config,
                wavefront_warmup,
                wavefront_positions,
                wavefront_pending,
                **layer_kwargs,
            )
        return self._forward_recirculation_serial(
            positions,
            hidden_states,
            residual,
            config,
            **layer_kwargs,
        )

    def _forward_recirculation_serial(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        config: RecirculationConfig,
        **layer_kwargs: Any,
    ) -> torch.Tensor:
        hidden_states, residual, destination_states, source_states = (
            self._forward_recirculation_normal_stack(
                positions,
                hidden_states,
                residual,
                config,
                **layer_kwargs,
            )
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        recirculated_states = config.mix(source_states, destination_states, positions)
        recirculated_residual = None
        for layer_idx in range(config.destination_layer + 1, self.end_layer):
            recirculated_states, recirculated_residual = self.layers[layer_idx](
                positions,
                recirculated_states,
                recirculated_residual,
                **layer_kwargs,
            )
        return hidden_states

    def _forward_recirculation_normal_stack(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        config: RecirculationConfig,
        **layer_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        destination_states = None
        source_states = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions,
                hidden_states,
                residual,
                **layer_kwargs,
            )
            if layer_idx == config.destination_layer:
                destination_states = self._materialize_residual(hidden_states, residual)
            if layer_idx == config.source_layer:
                source_states = self._materialize_residual(hidden_states, residual)
        assert destination_states is not None and source_states is not None
        return hidden_states, residual, destination_states, source_states

    def _forward_recirculation_wavefront(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        config: RecirculationConfig,
        warmup: bool,
        wavefront_positions: torch.Tensor | None,
        wavefront_pending: torch.Tensor | None,
        **layer_kwargs: Any,
    ) -> torch.Tensor:
        assert positions.shape[0] == 1
        destination_states = None
        source_states = None
        if warmup:
            hidden_states, residual, destination_states, source_states = (
                self._forward_recirculation_normal_stack(
                    positions,
                    hidden_states,
                    residual,
                    config,
                    **layer_kwargs,
                )
            )
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            assert wavefront_positions is not None
            assert wavefront_positions.shape[0] == 2
            assert wavefront_pending is not None
            assert wavefront_pending.shape == hidden_states.shape
            for layer_idx in range(self.start_layer, config.destination_layer + 1):
                hidden_states, residual = self.layers[layer_idx](
                    positions,
                    hidden_states,
                    residual,
                    **layer_kwargs,
                )

            destination_states = self._materialize_residual(hidden_states, residual)
            hidden_states = torch.cat((wavefront_pending, destination_states), dim=0)
            residual = None
            for layer_idx in range(config.destination_layer + 1, self.end_layer):
                hidden_states, residual = self.layers[layer_idx](
                    wavefront_positions,
                    hidden_states,
                    residual,
                    **layer_kwargs,
                )
                if layer_idx == config.source_layer:
                    source_states = self._materialize_residual(hidden_states, residual)[
                        1:
                    ]
            hidden_states, _ = self.norm(hidden_states, residual)
            hidden_states = hidden_states[1:]

        assert destination_states is not None and source_states is not None
        next_pending = config.mix(source_states, destination_states, positions)
        return torch.cat((hidden_states, next_pending), dim=0)

    @staticmethod
    def _materialize_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            return hidden_states
        return hidden_states + residual
