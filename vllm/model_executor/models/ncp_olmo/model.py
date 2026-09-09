# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM model entry point for ConceptLM V2.2-VQ."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState

from .contract import ConceptLMBackendConfig
from .hlm import ConceptLMHighLevelBranch
from .routes import ConceptLMStage3Routes
from .state import (
    ConceptRequestState,
    ConceptRequestStateStore,
    ScheduledRequestSegment,
    active_tensor_buffer,
    append_tensor_buffer,
    clear_tensor_buffer,
)
from .token_tower import ConceptLMTokenBackbone, _config_value
from .weights import (
    Stage3WeightConfig,
    expected_stage3_weight_shapes,
    required_checkpoint_shards,
    resolve_checkpoint_weight,
)

_HLMAdvance = tuple[
    ConceptRequestState,
    torch.Tensor,
    tuple[torch.Tensor, ...],
]


def _chunk_mean(values: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Pool token chunks and restore the activation dtype.

    Batch-invariant mode accumulates reductions in FP32. The HLM consumes model
    activations, so pooling must return to the input dtype before LayerNorm and
    QKV projections.
    """

    hidden_size = int(values.shape[-1])
    pooled = values.reshape(-1, chunk_size, hidden_size).mean(dim=1)
    return pooled.to(values.dtype)


class NCPOlmo3ForCausalLM(nn.Module, HasInnerState):
    """Native NCP-OLMo model backed by vLLM paged token attention."""

    backend_status = "full_forward_parity_gated"

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        if prefix:
            raise NotImplementedError("the first ConceptLM backend requires PP=1")
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("the first ConceptLM backend requires TP=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("the first ConceptLM backend requires PP=1")
        if not vllm_config.use_v2_model_runner:
            raise ValueError("NCP-OLMo request state requires the V2 model runner")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError(
                "the first NCP-OLMo V2 backend requires enforce_eager=True; "
                "CUDA graph capture has not been validated with request-scoped "
                "HLM state"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError(
                "ConceptLM request-scoped HLM requires prefix caching to be disabled"
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "speculative decoding is not enabled before target-model parity"
            )
        if vllm_config.quant_config is not None:
            raise NotImplementedError(
                "quantized NCP-OLMo checkpoints are not enabled in the first backend"
            )

        hf_config = vllm_config.model_config.hf_config
        raw_config = (
            hf_config.to_dict()
            if hasattr(hf_config, "to_dict")
            else dict(vars(hf_config))
        )
        self.backend_config = ConceptLMBackendConfig.from_mapping(raw_config)
        self.stage3_weight_config = Stage3WeightConfig.from_mapping(raw_config)
        if self.backend_config.chunk_merge_method != "meanpooling":
            raise NotImplementedError(
                "the first Stage3 backend requires meanpooling chunks"
            )
        epsilon = float(_config_value(hf_config, "layernorm_epsilon"))
        self.token_backbone = ConceptLMTokenBackbone(
            vllm_config=vllm_config,
            backend_config=self.backend_config,
        )
        self.highlevel = ConceptLMHighLevelBranch(
            vllm_config=vllm_config,
            backend_config=self.backend_config,
        )
        self.routes = ConceptLMStage3Routes(
            backend_config=self.backend_config,
            epsilon=epsilon,
        )
        self.request_states = ConceptRequestStateStore(
            encoder_layers=self.backend_config.encoder_layers,
            hlm_layers=self.backend_config.hlm_layers,
        )
        self._logits_trace_index = 0
        self._pending_stage_trace: dict[str, torch.Tensor] | None = None
        trace_dir = os.environ.get("CONCEPTLM_VLLM_LOGITS_TRACE_DIR")
        self._logits_trace_dir = Path(trace_dir) if trace_dir else None

    @staticmethod
    def get_model_state_cls() -> type[Any]:
        """Return the V2 runner adapter for request-scoped HLM state."""

        from .model_state import NCPOlmoModelState

        return NCPOlmoModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings."""

        return self.token_backbone.embed_input_ids(input_ids)

    def _advance_completed_chunks(
        self,
        segment: ScheduledRequestSegment,
        encoder_hidden: torch.Tensor,
        encoder_raw_layers: Sequence[torch.Tensor],
    ) -> _HLMAdvance | None:
        state = segment.state
        chunk_size = self.backend_config.chunk_size
        advance: _HLMAdvance | None = None
        new_final = encoder_hidden[segment.flat_start : segment.flat_end]
        new_layers = tuple(
            raw_layer[segment.flat_start : segment.flat_end]
            for raw_layer in encoder_raw_layers
        )
        if state.pending_encoder_final.length:
            pending_final = active_tensor_buffer(state.pending_encoder_final)
            if pending_final is None:
                raise RuntimeError("missing pending encoder final tensor")
            combined_final = torch.cat(
                (pending_final, new_final),
                dim=0,
            )
            combined_layers = tuple(
                torch.cat(
                    (
                        active_tensor_buffer(pending),
                        new_layer,
                    ),
                    dim=0,
                )
                for pending, new_layer in zip(
                    state.pending_encoder_layers,
                    new_layers,
                )
            )
        else:
            combined_final = new_final
            combined_layers = new_layers

        num_completed = int(combined_final.shape[0]) // chunk_size
        completed_tokens = num_completed * chunk_size
        if num_completed:
            encoder_chunks = _chunk_mean(
                combined_final[:completed_tokens],
                chunk_size,
            )
            layer_chunks = tuple(
                _chunk_mean(
                    values[:completed_tokens],
                    chunk_size,
                )
                for values in combined_layers
            )
            if num_completed == 1:
                advance = (
                    state,
                    encoder_chunks[0],
                    tuple(values[0] for values in layer_chunks),
                )
            elif envs.VLLM_BATCH_INVARIANT:
                # Chunked prefill may present the same prompt as one multi-chunk
                # segment or as several smaller segments depending on unrelated
                # requests sharing the scheduler budget. Keep the HLM arithmetic
                # independent of that partitioning in correctness-first mode.
                for chunk_index in range(num_completed):
                    self.highlevel.advance(
                        state,
                        encoder_chunks[chunk_index],
                        tuple(values[chunk_index] for values in layer_chunks),
                    )
            else:
                self.highlevel.prefill(
                    state,
                    encoder_chunks,
                    layer_chunks,
                )

        clear_tensor_buffer(state.pending_encoder_final)
        for layer_values in state.pending_encoder_layers:
            clear_tensor_buffer(layer_values)
        if completed_tokens < int(combined_final.shape[0]):
            append_tensor_buffer(
                state.pending_encoder_final,
                combined_final[completed_tokens:],
                minimum_capacity=chunk_size,
            )
            for pending, values in zip(
                state.pending_encoder_layers,
                combined_layers,
            ):
                append_tensor_buffer(
                    pending,
                    values[completed_tokens:],
                    minimum_capacity=chunk_size,
                )
        return advance

    def _advance_hlm_batches(
        self,
        advances: Sequence[_HLMAdvance],
    ) -> None:
        """Advance one-chunk HLM updates, batching only when permitted."""

        batches: dict[int, list[_HLMAdvance]] = {}
        for advance in advances:
            concept_position = advance[0].predicted_concepts.length
            batches.setdefault(concept_position, []).append(advance)

        for batch in batches.values():
            if len(batch) == 1 or envs.VLLM_BATCH_INVARIANT:
                # The HLM is request-local. In deterministic mode, preserve the
                # exact single-request numerical path across admission,
                # preemption, and refill instead of changing GEMM/attention
                # geometry when equal-position requests happen to co-schedule.
                for state, encoder_chunk, layer_chunks in batch:
                    self.highlevel.advance(
                        state,
                        encoder_chunk,
                        layer_chunks,
                    )
                continue
            self.highlevel.advance_batch(
                [advance[0] for advance in batch],
                torch.stack(
                    [advance[1] for advance in batch],
                    dim=0,
                ),
                tuple(
                    torch.stack(
                        [advance[2][layer_index] for advance in batch],
                        dim=0,
                    )
                    for layer_index in range(self.backend_config.encoder_layers)
                ),
            )

    def _concept_index_for_position(self, position: int) -> int | None:
        if self.backend_config.shift_feature:
            shifted_chunk = (position + 1) // self.backend_config.chunk_size
        else:
            shifted_chunk = position // self.backend_config.chunk_size
        return shifted_chunk - 1 if shifted_chunk > 0 else None

    def _request_concept_at(
        self,
        state: ConceptRequestState,
        *,
        position: int,
        layer_index: int | None,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        concept_index = self._concept_index_for_position(position)
        if concept_index is None:
            return zero
        if concept_index >= state.predicted_concepts.length:
            raise RuntimeError(
                f"request {state.req_id!r} is missing concept {concept_index} "
                f"at token position {position}"
            )
        if layer_index is None:
            concepts = state.predicted_concepts.data
            if concepts is None:
                raise RuntimeError("missing predicted concept storage")
            return concepts[concept_index]
        layer_states = state.hlm_raw_layer_states[layer_index]
        if concept_index >= layer_states.length:
            raise RuntimeError(
                f"request {state.req_id!r} is missing HLM layer {layer_index} "
                f"concept {concept_index}"
            )
        if layer_states.data is None:
            raise RuntimeError("missing HLM raw layer storage")
        return layer_states.data[concept_index]

    def _build_decoder_concept_inputs(
        self,
        segments: Sequence[ScheduledRequestSegment],
        encoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        zero = encoder_hidden.new_zeros(self.backend_config.hidden_size)
        final_values: list[torch.Tensor] = []
        layer_values: list[list[torch.Tensor]] = [
            [] for _ in range(self.backend_config.hlm_layers)
        ]
        for segment in segments:
            for local_index in range(segment.flat_end - segment.flat_start):
                position = segment.position_start + local_index
                final_values.append(
                    self._request_concept_at(
                        segment.state,
                        position=position,
                        layer_index=None,
                        zero=zero,
                    )
                )
                for layer_index in range(self.backend_config.hlm_layers):
                    layer_values[layer_index].append(
                        self._request_concept_at(
                            segment.state,
                            position=position,
                            layer_index=layer_index,
                            zero=zero,
                        )
                    )
        target_length = int(encoder_hidden.shape[0])
        row_counts = [len(final_values), *(len(values) for values in layer_values)]
        if any(row_count != target_length for row_count in row_counts):
            raise RuntimeError(
                "request-owned ConceptLM rows do not match the model input batch: "
                f"{row_counts!r} != {target_length}"
            )
        return (
            torch.stack(final_values, dim=0),
            tuple(torch.stack(values, dim=0) for values in layer_values),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        request_segments: Sequence[ScheduledRequestSegment] = (),
    ) -> torch.Tensor:
        """Run token encoder, chunk HLM, fusion, and routed token decoder."""

        if intermediate_tensors is not None:
            raise NotImplementedError("the first ConceptLM backend requires PP=1")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.token_backbone.embed_input_ids(input_ids)
        )
        profile_mode = not request_segments
        stage_trace: dict[str, torch.Tensor] | None = (
            {} if not profile_mode and self._logits_trace_dir is not None else None
        )

        def record_stage(name: str, value: torch.Tensor) -> None:
            if stage_trace is not None:
                stage_trace[name] = value[-1].detach().float().cpu()

        segments = request_segments

        encoder_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.encoder_layers + 1,
                hidden_states.shape[-1],
            )
        )
        encoder_history[..., 0, :].copy_(hidden_states)
        encoder_cumsum_state = (
            hidden_states if self.backend_config.dd_self_mode == "cumsum" else None
        )
        for layer_index, layer in enumerate(self.token_backbone.encoder.layers):
            record_stage(f"encoder.input.{layer_index}", hidden_states)
            raw_output = layer(positions, hidden_states)
            record_stage(f"encoder.raw.{layer_index}", raw_output)
            encoder_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, encoder_cumsum_state = self.routes.encoder_after_layer(
                layer_index,
                raw_output,
                encoder_history[..., : layer_index + 2, :],
                encoder_cumsum_state,
            )
            record_stage(f"encoder.routed.{layer_index}", hidden_states)
        encoder_hidden = hidden_states
        record_stage("encoder_hidden", encoder_hidden)
        encoder_raw_layers = encoder_history[..., 1:, :].unbind(dim=-2)

        if profile_mode:
            final_concepts = torch.zeros_like(encoder_hidden)
            concept_raw_layers = tuple(
                torch.zeros_like(encoder_hidden)
                for _ in range(self.backend_config.hlm_layers)
            )
        else:
            hlm_advances = []
            for segment in segments:
                advance = self._advance_completed_chunks(
                    segment, encoder_hidden, encoder_raw_layers
                )
                if advance is not None:
                    hlm_advances.append(advance)
            self._advance_hlm_batches(hlm_advances)
            final_concepts, concept_raw_layers = self._build_decoder_concept_inputs(
                segments,
                encoder_hidden,
            )
        hidden_states = self.routes.fuse(encoder_hidden, final_concepts)
        record_stage("final_concepts", final_concepts)
        record_stage("fusion_output", hidden_states)
        encoder_sources, concept_sources = self.routes.normalize_decoder_sources(
            encoder_raw_layers,
            concept_raw_layers,
        )
        decoder_gates = self.routes.decoder_gates()

        decoder_history = hidden_states.new_empty(
            (
                *hidden_states.shape[:-1],
                self.backend_config.decoder_layers + 1,
                hidden_states.shape[-1],
            )
        )
        decoder_history[..., 0, :].copy_(hidden_states)
        decoder_cumsum_state = (
            hidden_states if self.backend_config.dd_self_mode == "cumsum" else None
        )
        for layer_index, layer in enumerate(self.token_backbone.decoder.layers):
            record_stage(f"decoder.input.{layer_index}", hidden_states)
            raw_output = layer(positions, hidden_states)
            record_stage(f"decoder.raw.{layer_index}", raw_output)
            decoder_history[..., layer_index + 1, :].copy_(raw_output)
            hidden_states, decoder_cumsum_state = self.routes.decoder_after_layer(
                layer_index=layer_index,
                raw_output=raw_output,
                history_states=decoder_history[..., : layer_index + 2, :],
                final_concepts=final_concepts,
                encoder_sources=encoder_sources,
                concept_sources=concept_sources,
                gate=decoder_gates[layer_index],
                cumsum_state=decoder_cumsum_state,
            )
            record_stage(f"decoder.routed.{layer_index}", hidden_states)
        final_layernorm = self.token_backbone.decoder.final_layernorm
        if final_layernorm is None:
            raise RuntimeError("ConceptLM decoder final_layernorm is missing")
        record_stage("decoder.pre_final_norm", hidden_states)
        hidden_states = final_layernorm(hidden_states)
        record_stage("decoder_final", hidden_states)
        if not profile_mode:
            for segment in segments:
                self.request_states.commit(segment)
        if stage_trace is not None:
            stage_trace.update(self.highlevel.stage_trace())
            self._pending_stage_trace = stage_trace
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Project decoder states to vocabulary logits."""

        logits = self.token_backbone.compute_logits(hidden_states)
        if (
            logits is not None
            and self._logits_trace_dir is not None
            and self._pending_stage_trace is not None
        ):
            output_dir = self._logits_trace_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"logits-{self._logits_trace_index:05d}.pt"
            torch.save(logits.detach().float().cpu(), output_path)
            stage_path = output_dir / f"stages-{self._logits_trace_index:05d}.pt"
            torch.save(self._pending_stage_trace, stage_path)
            self._pending_stage_trace = None
            self._logits_trace_index += 1
        return logits

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load every parameter from pure-HF SafeTensors keys."""

        params: dict[str, tuple[str, nn.Parameter]] = {}
        for module_name, module in (
            ("token_backbone", self.token_backbone),
            ("highlevel", self.highlevel),
            ("routes", self.routes),
        ):
            for name, parameter in module.named_parameters(remove_duplicate=False):
                if name in params:
                    raise RuntimeError(f"duplicate internal parameter name: {name}")
                params[name] = (f"{module_name}.{name}", parameter)

        stage3_weight_config = getattr(self, "stage3_weight_config", None)
        expected_parameter_count = (
            len(expected_stage3_weight_shapes(stage3_weight_config))
            if stage3_weight_config is not None
            else len(params)
        )
        if len(params) != expected_parameter_count:
            raise RuntimeError(
                "constructed Stage3 model parameter count does not match its "
                f"{self.backend_config.dd_self_mode} weight contract: "
                f"expected {expected_parameter_count}, got {len(params)}"
            )

        loaded_checkpoint_names: set[str] = set()
        loaded_targets: dict[str, set[str | int | None]] = {}
        ignored_metadata = set()
        parameter_names = frozenset(params)
        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name.endswith("._extra_state"):
                ignored_metadata.add(checkpoint_name)
                continue
            if checkpoint_name in loaded_checkpoint_names:
                raise ValueError(
                    f"duplicate ConceptLM checkpoint tensor: {checkpoint_name}"
                )
            target = resolve_checkpoint_weight(checkpoint_name, parameter_names)
            if target is None:
                raise ValueError(
                    f"unexpected ConceptLM checkpoint tensor: {checkpoint_name}"
                )
            target_shards = loaded_targets.setdefault(target.parameter_name, set())
            if target.shard_id in target_shards:
                raise ValueError(
                    "duplicate ConceptLM checkpoint target: "
                    f"{checkpoint_name} -> {target.parameter_name}"
                    f"[{target.shard_id}]"
                )
            if target.shard_id is None and target_shards:
                raise ValueError(
                    "cannot mix full and split ConceptLM checkpoint tensors for "
                    f"{target.parameter_name}"
                )
            if target.shard_id is not None and None in target_shards:
                raise ValueError(
                    "cannot mix split and full ConceptLM checkpoint tensors for "
                    f"{target.parameter_name}"
                )
            _, parameter = params[target.parameter_name]
            weight_loader = getattr(
                parameter,
                "weight_loader",
                default_weight_loader,
            )
            if target.shard_id is not None:
                weight_loader(parameter, loaded_weight, target.shard_id)
            else:
                weight_loader(parameter, loaded_weight)
            loaded_checkpoint_names.add(checkpoint_name)
            target_shards.add(target.shard_id)

        complete_parameters: set[str] = set()
        incomplete = []
        for parameter_name, loaded_shards in loaded_targets.items():
            if loaded_shards == {None}:
                complete_parameters.add(parameter_name)
                continue
            required = required_checkpoint_shards(parameter_name)
            if required is not None and loaded_shards == required:
                complete_parameters.add(parameter_name)
                continue
            incomplete.append(
                f"{parameter_name}: loaded={sorted(map(str, loaded_shards))}, "
                f"required={sorted(map(str, required or ()))}"
            )
        if incomplete:
            raise ValueError(
                "incomplete split ConceptLM checkpoint parameters: "
                + "; ".join(sorted(incomplete))
            )
        missing = sorted(set(params) - complete_parameters)
        if missing:
            raise ValueError(
                "missing ConceptLM checkpoint parameters: " + ", ".join(missing)
            )
        if len(ignored_metadata) not in (0, 2):
            raise ValueError(
                "Stage3 checkpoint must contain zero or two _extra_state tensors, "
                f"got {sorted(ignored_metadata)}"
            )
        loaded_model_names = {params[name][0] for name in complete_parameters}
        expected_model_names = {name for name, _ in self.named_parameters()}
        if loaded_model_names != expected_model_names:
            raise ValueError(
                "loaded vLLM parameter names do not match the constructed model: "
                f"missing={sorted(expected_model_names - loaded_model_names)} "
                f"unexpected={sorted(loaded_model_names - expected_model_names)}"
            )
        return loaded_model_names
