# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from vllm.config.recirculation import RecirculationConfig


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
            start_layer != 0
            or end_layer
            != RecirculationConfig._get_text_config(hf_config).num_hidden_layers
        ):
            raise ValueError("Recirculation does not support pipeline parallelism")
        if config is not None:
            self._validate_recirculation_model_config(hf_config, config)
        self.recirculation_config = config

    def _validate_recirculation_model_config(
        self,
        hf_config: object,
        config: RecirculationConfig,
    ) -> None:
        """Validate family-specific constraints before model execution."""

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
        hidden_states, residual, destination_states, source_states, layer_states = (
            self._forward_recirculation_normal_stack(
                positions,
                hidden_states,
                residual,
                config,
                **layer_kwargs,
            )
        )
        hidden_states = self._finalize_recirculation(hidden_states, residual)
        recirculated_states = config.mix(source_states, destination_states, positions)
        recirculated_residual = None
        for layer_idx in range(config.destination_layer + 1, self.end_layer):
            self._restore_recirculation_layer_state(
                layer_idx, layer_states.get(layer_idx)
            )
            recirculated_states, recirculated_residual = (
                self._forward_recirculation_layer(
                    layer_idx,
                    positions,
                    recirculated_states,
                    recirculated_residual,
                    **layer_kwargs,
                )
            )
        return hidden_states

    def _forward_recirculation_normal_stack(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        config: RecirculationConfig,
        **layer_kwargs: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        dict[int, Any],
    ]:
        destination_states = None
        source_states = None
        layer_states: dict[int, Any] = {}
        for layer_idx in range(self.start_layer, self.end_layer):
            if layer_idx > config.destination_layer:
                layer_states[layer_idx] = self._capture_recirculation_layer_state(
                    layer_idx
                )
            hidden_states, residual = self._forward_recirculation_layer(
                layer_idx,
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
        return hidden_states, residual, destination_states, source_states, layer_states

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
            hidden_states, residual, destination_states, source_states, _ = (
                self._forward_recirculation_normal_stack(
                    positions,
                    hidden_states,
                    residual,
                    config,
                    **layer_kwargs,
                )
            )
            hidden_states = self._finalize_recirculation(hidden_states, residual)
        else:
            assert wavefront_positions is not None
            assert wavefront_positions.shape[0] == 2
            assert wavefront_pending is not None
            assert wavefront_pending.shape == hidden_states.shape
            for layer_idx in range(self.start_layer, config.destination_layer + 1):
                hidden_states, residual = self._forward_recirculation_layer(
                    layer_idx,
                    positions,
                    hidden_states,
                    residual,
                    **layer_kwargs,
                )

            destination_states = self._materialize_residual(hidden_states, residual)
            hidden_states = torch.cat((wavefront_pending, destination_states), dim=0)
            residual = None
            for layer_idx in range(config.destination_layer + 1, self.end_layer):
                hidden_states, residual = self._forward_recirculation_layer(
                    layer_idx,
                    wavefront_positions,
                    hidden_states,
                    residual,
                    **layer_kwargs,
                )
                if layer_idx == config.source_layer:
                    source_states = self._materialize_residual(hidden_states, residual)[
                        1:
                    ]
            hidden_states = self._finalize_recirculation(hidden_states, residual)
            hidden_states = hidden_states[1:]

        assert destination_states is not None and source_states is not None
        next_pending = config.mix(source_states, destination_states, positions)
        return torch.cat((hidden_states, next_pending), dim=0)

    def _forward_recirculation_layer(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **layer_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.layers[layer_idx](
            positions,
            hidden_states,
            residual,
            **layer_kwargs,
        )

    def _capture_recirculation_layer_state(self, layer_idx: int) -> Any:
        """Capture non-overwritable state before an upper-layer first pass.

        Attention KV entries are naturally overwritten by the rerun. Recurrent
        families override this hook to retain the pre-block state that their
        in-place update kernels would otherwise consume twice.
        """

    def _restore_recirculation_layer_state(self, layer_idx: int, state: Any) -> None:
        """Restore family state captured by `_capture_recirculation_layer_state`."""

    def _finalize_recirculation(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = (
            self.norm(hidden_states, residual)
            if residual is not None
            else self.norm(hidden_states)
        )
        if isinstance(normalized, tuple):
            return normalized[0]
        return normalized

    @staticmethod
    def _materialize_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            return hidden_states
        return hidden_states + residual
