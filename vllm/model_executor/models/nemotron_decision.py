# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bidirectional Ministral backbone for NemotronForDecision checkpoints, and the
packed read of its engine: FlashAttention over the sequences of
`nemotron_packed_layout`, owned by the model through
`EncoderOnlyAttention(attn_backend=...)`, and the model state that feeds it."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from vllm.model_executor.models.mistral import MistralAttention
from vllm.model_executor.models.nemotron_packed_layout import (
    MAX_KEY_ROWS,
    packed_read,
    packed_sequences,
    segment_lengths,
    segment_positions,
    sequence_groups,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends import flash_attn
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState


@dataclass(slots=True)
class PackedAttention:
    """One attention call of a step that holds packed reads: consecutive
    sequences, each with its own keys."""

    rows: slice
    """The call's query rows."""
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    kv_rows: torch.Tensor
    """The key and value rows of every sequence, one sequence after another."""
    max_seqlen_q: int
    max_seqlen_k: int


def packed_attention(
    query_start_loc: list[int],
    layouts: list[list[int] | None],
    device: torch.device,
    max_key_rows: int = MAX_KEY_ROWS,
) -> list[PackedAttention]:
    """The attention calls of a step: all its sequences, in as many calls as
    `max_key_rows` asks for."""
    q_lens, k_lens, kv_rows = packed_sequences(query_start_loc, layouts)
    q_ends = np.cumsum([0, *q_lens])
    k_ends = np.cumsum([0, *k_lens])
    calls = []
    for first, last in sequence_groups(k_lens, max_key_rows):
        calls.append(
            PackedAttention(
                rows=slice(int(q_ends[first]), int(q_ends[last])),
                cu_seqlens_q=_cu_seqlens(q_lens[first:last], device),
                cu_seqlens_k=_cu_seqlens(k_lens[first:last], device),
                kv_rows=async_tensor_h2d(kv_rows[k_ends[first] : k_ends[last]], device),
                max_seqlen_q=max(q_lens[first:last]),
                max_seqlen_k=max(k_lens[first:last]),
            )
        )
    return calls


def _cu_seqlens(lens: list[int], device: torch.device) -> torch.Tensor:
    return async_tensor_h2d(np.cumsum([0, *lens], dtype=np.int32), device)


class NemotronPackedReadImpl(FlashAttentionImpl):
    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        packed: list[PackedAttention] | None = getattr(attn_metadata, "packed", None)
        if packed is None:
            return super()._forward_encoder_attention(
                query, key, value, output, attn_metadata, layer
            )
        assert self.vllm_flash_attn_version is not None
        # The call of the base class, with each sequence's own keys.
        for call in packed:
            descale_shape = (call.cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
            flash_attn.flash_attn_varlen_func(
                q=query[call.rows],
                k=key.index_select(0, call.kv_rows),
                v=value.index_select(0, call.kv_rows),
                out=output[call.rows],
                cu_seqlens_q=call.cu_seqlens_q,
                cu_seqlens_k=call.cu_seqlens_k,
                max_seqlen_q=call.max_seqlen_q,
                max_seqlen_k=call.max_seqlen_k,
                softmax_scale=self.scale,
                causal=False,
                alibi_slopes=self.alibi_slopes,
                window_size=None,
                softcap=self.logits_soft_cap,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape)  # type: ignore[operator]
                if self.supports_quant_query_input
                else None,
                k_descale=layer._k_scale.expand(descale_shape),  # type: ignore[operator]
                v_descale=layer._v_scale.expand(descale_shape),  # type: ignore[operator]
                num_splits=1 if self.batch_invariant_enabled else 0,
                s_aux=self.sinks,
            )
        return output


class NemotronPackedReadBackend(FlashAttentionBackend):
    """FlashAttention whose encoder attention also runs packed reads."""

    @staticmethod
    def get_impl_cls() -> type[FlashAttentionImpl]:
        return NemotronPackedReadImpl


class NemotronPackedReadModelState(EncoderOnlyModelState):
    """Encoder-only state that also serves packed reads: restarts each
    question's positions and gives the attention its sequences."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # req_id -> segment lengths and the position of every prompt token
        self.layouts: dict[str, tuple[list[int], torch.Tensor]] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        lengths = segment_lengths(
            new_req_data.sampling_params, new_req_data.prompt_token_ids
        )
        if lengths is not None:
            self.layouts[new_req_data.req_id] = (lengths, segment_positions(lengths))

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.layouts.pop(req_id, None)

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        support, backend = super().get_additional_cg_support()
        if support == AttentionCGSupport.ALWAYS:
            # A full graph replays the sequences of its capture, so a step
            # with prompts must not run in one.
            return AttentionCGSupport.UNIFORM_BATCH, NemotronPackedReadBackend.__name__
        return support, backend

    def _scheduled_layouts(
        self, input_batch: InputBatch
    ) -> list[tuple[list[int], torch.Tensor] | None] | None:
        """The layout of each request of the step that runs a whole packed
        prompt; None when the step has no such request."""
        if not self.layouts:
            return None
        layouts = [
            layout
            if (layout := self.layouts.get(req_id)) is not None
            and input_batch.num_computed_tokens_np[i] == 0
            and input_batch.num_scheduled_tokens[i] == len(layout[1])
            else None
            for i, req_id in enumerate(input_batch.req_ids)
        ]
        return layouts if any(layout is not None for layout in layouts) else None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        layouts = self._scheduled_layouts(input_batch)
        if layouts is not None:
            # In place: CUDA graphs replay the address of the runner's buffer.
            for start, layout in zip(input_batch.query_start_loc_np, layouts):
                if layout is not None:
                    positions = layout[1]
                    async_tensor_h2d(
                        positions,
                        out=input_batch.positions[start : start + len(positions)],
                    )
        return model_inputs

    def _build_encoder_attn_metadata(
        self, input_batch: InputBatch, cudagraph_mode: CUDAGraphMode, for_capture: bool
    ) -> dict[str, Any]:
        attn_metadata = super()._build_encoder_attn_metadata(
            input_batch, cudagraph_mode, for_capture
        )
        layouts = None if for_capture else self._scheduled_layouts(input_batch)
        if layouts is None:
            return attn_metadata
        assert cudagraph_mode != CUDAGraphMode.FULL, "a packed read in a full graph"
        packed = packed_attention(
            input_batch.query_start_loc_np[: input_batch.num_reqs + 1].tolist(),
            [None if layout is None else layout[0] for layout in layouts],
            self.device,
        )
        for metadata in attn_metadata.values():
            metadata.packed = packed
        return attn_metadata


class NemotronDecisionAttention(MistralAttention):
    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("attn_type") == AttentionType.ENCODER_ONLY and packed_read(
            kwargs["config"]
        ):
            # The questions of a request are read in one forward. Otherwise
            # the layer keeps the backend the selector chooses.
            kwargs["attn_backend"] = NemotronPackedReadBackend
        super().__init__(**kwargs)

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        # Ministral3 rounds the positional scale to the query dtype before
        # multiplying. Beyond the original context length this matters for BF16.
        return super()._get_llama_4_attn_scale(positions).to(self.rotary_emb.dtype)


class NemotronDecisionDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config, prefix=prefix, attn_layer_type=NemotronDecisionAttention
        )


class NemotronForDecision(LlamaForCausalLM):
    """Score the final prompt position with the checkpoint's vocabulary head."""

    hf_to_vllm_mapper = LlamaForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=NemotronDecisionDecoderLayer,
        )
        if packed_read(vllm_config.model_config.hf_config):
            # Only then does the model give its model state; otherwise the
            # model runner resolves the stock encoder-only one.
            self.get_model_state_cls = self._packed_read_model_state_cls

    @staticmethod
    def _packed_read_model_state_cls() -> type[ModelState]:
        return NemotronPackedReadModelState

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
