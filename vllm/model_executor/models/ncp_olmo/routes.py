# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NCP-ArchPreview encoder, decoder, concept, and residual-flow routes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .contract import NCPOlmo3BackendConfig
from .hlm import (
    ConceptLMDepthDD,
    ConceptLMDiagResidualRoute,
    ConceptLMSelfDD,
    apply_ncp_layer_norm,
)
from .weights import resolve_checkpoint_weight


class ConceptLMFinalConceptRoute(nn.Module):
    """LayerNorm plus diagonal projection for the final predicted concept."""

    def __init__(self, *, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.concept_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.final_diag = nn.Parameter(torch.empty(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        final_concept: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        update = apply_ncp_layer_norm(self.concept_norm, final_concept)
        update = update * self.final_diag.to(update.dtype)
        return hidden_states + update * scale.to(update.dtype)


class ConceptLMDDTwoRouteAdd(nn.Module):
    """Decoder self-DD and final-concept routing modules."""

    def __init__(
        self,
        *,
        hidden_size: int,
        decoder_layers: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.decoder_dds = nn.ModuleList(
            [
                ConceptLMDepthDD(
                    hidden_size=hidden_size,
                    layer_index=layer_index,
                    use_softmax=True,
                )
                for layer_index in range(decoder_layers)
            ]
        )
        self.concept_routes = nn.ModuleList(
            [
                ConceptLMFinalConceptRoute(
                    hidden_size=hidden_size,
                    epsilon=epsilon,
                )
                for _ in range(decoder_layers)
            ]
        )


class NCPOlmo3Routes(nn.Module):
    """All non-HLM NCP-ArchPreview routes and fusion parameters."""

    def __init__(
        self,
        *,
        backend_config: NCPOlmo3BackendConfig,
        epsilon: float,
    ) -> None:
        super().__init__()
        hidden_size = backend_config.hidden_size
        self.backend_config = backend_config
        self.dd_encoder_self_dd = ConceptLMSelfDD(
            hidden_size=hidden_size,
            num_layers=backend_config.encoder_layers,
        )
        self.dd_two_route_add = ConceptLMDDTwoRouteAdd(
            hidden_size=hidden_size,
            decoder_layers=backend_config.decoder_layers,
            epsilon=epsilon,
        )
        self.decoder_read_encoder_routes = nn.ModuleList(
            [
                ConceptLMDiagResidualRoute(
                    hidden_size=hidden_size,
                    num_sources=backend_config.encoder_layers,
                )
                for _ in range(backend_config.decoder_layers)
            ]
        )
        self.decoder_read_concept_routes = nn.ModuleList(
            [
                ConceptLMDiagResidualRoute(
                    hidden_size=hidden_size,
                    num_sources=backend_config.hlm_layers,
                )
                for _ in range(backend_config.decoder_layers)
            ]
        )
        self.decoder_read_encoder_shared_source_norm = nn.LayerNorm(
            hidden_size,
            eps=epsilon,
        )
        self.decoder_read_concept_shared_source_norm = nn.LayerNorm(
            hidden_size,
            eps=epsilon,
        )
        self.fusion_tok_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.fusion_hl_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.fusion_norm_alpha = nn.Parameter(torch.empty(()))
        self.final_read_concept_gate_logits = nn.Parameter(
            torch.empty(backend_config.decoder_layers, 2)
        )

    def encoder_after_layer(
        self,
        layer_index: int,
        raw_output: torch.Tensor,
        history_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply encoder self-DD after saving the raw layer output."""

        return self.dd_encoder_self_dd.depth_dds[layer_index](
            raw_output,
            history_states,
        )

    def fuse(
        self,
        encoder_hidden: torch.Tensor,
        predicted_concepts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the trained normed-add fusion."""

        return apply_ncp_layer_norm(
            self.fusion_tok_norm,
            encoder_hidden,
        ) + self.fusion_norm_alpha.to(encoder_hidden.dtype) * apply_ncp_layer_norm(
            self.fusion_hl_norm,
            predicted_concepts,
        )

    def decoder_after_layer(
        self,
        *,
        layer_index: int,
        raw_output: torch.Tensor,
        history_states: torch.Tensor,
        final_concepts: torch.Tensor,
        encoder_sources: torch.Tensor,
        concept_sources: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply decoder DD, concept route, and both residual-flow reads."""

        hidden_states = self.dd_two_route_add.decoder_dds[layer_index](
            raw_output,
            history_states,
        )
        hidden_states = self.dd_two_route_add.concept_routes[layer_index](
            hidden_states,
            final_concepts,
            gate[0],
        )
        hidden_states = self.decoder_read_encoder_routes[layer_index](
            hidden_states,
            encoder_sources,
        )
        hidden_states = self.decoder_read_concept_routes[layer_index](
            hidden_states,
            concept_sources,
            residual_scale=gate[1],
        )
        return hidden_states

    def decoder_gates(self) -> torch.Tensor:
        """Compute all two-way decoder gates in one softmax launch."""

        return self.final_read_concept_gate_logits.float().softmax(dim=-1)

    def normalize_decoder_sources(
        self,
        encoder_raw_layers: Sequence[torch.Tensor],
        concept_raw_layers: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack and normalize route sources once before all decoder layers."""

        encoder_sources = apply_ncp_layer_norm(
            self.decoder_read_encoder_shared_source_norm,
            torch.stack(tuple(encoder_raw_layers), dim=-2),
        )
        concept_sources = apply_ncp_layer_norm(
            self.decoder_read_concept_shared_source_norm,
            torch.stack(tuple(concept_raw_layers), dim=-2),
        )
        return encoder_sources, concept_sources

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load all NCP-ArchPreview route and fusion parameters."""

        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        parameter_names = frozenset(params)
        for checkpoint_name, loaded_weight in weights:
            target = resolve_checkpoint_weight(checkpoint_name, parameter_names)
            if target is None:
                continue
            if target.shard_id is not None:
                raise ValueError(
                    f"route tensor unexpectedly resolved as a shard: {checkpoint_name}"
                )
            if target.parameter_name in loaded:
                raise ValueError(
                    "duplicate NCP-ArchPreview route checkpoint target: "
                    f"{checkpoint_name} -> {target.parameter_name}"
                )
            parameter = params[target.parameter_name]
            weight_loader = getattr(
                parameter,
                "weight_loader",
                default_weight_loader,
            )
            weight_loader(parameter, loaded_weight)
            loaded.add(target.parameter_name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(
                "missing NCP-ArchPreview route checkpoint parameters: "
                + ", ".join(missing)
            )
        return loaded
