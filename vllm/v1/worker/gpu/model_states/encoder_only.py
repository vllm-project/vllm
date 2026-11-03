# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import PIN_MEMORY, STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


def segment_layout_rows(
    segment_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment id and position of every prompt token of a segment layout: the
    prefix is segment 0 at positions 0..P-1; candidate k is segment k, and its
    positions restart at P."""
    prefix_len = segment_lengths[0]
    segment_ids = [torch.zeros(prefix_len, dtype=torch.int64)]
    positions = [torch.arange(prefix_len, dtype=torch.int64)]
    for k, length in enumerate(segment_lengths[1:], start=1):
        segment_ids.append(torch.full((length,), k, dtype=torch.int64))
        positions.append(torch.arange(prefix_len, prefix_len + length))
    return torch.cat(segment_ids), torch.cat(positions)


class EncoderOnlyModelState(DefaultModelState):
    """ModelState for encoder-only (BERT/RoBERTa) models.

    Encoder attention needs no KV cache: it runs full, bidirectional
    self-attention over each request's tokens in a single forward. Such layers
    return no ``get_kv_cache_spec`` and therefore never enter
    ``kv_cache_config.kv_cache_groups``, so the KV-backed attention path never
    builds their metadata. We build their (non-causal) metadata here, keeping
    the normal KV-backed path untouched.
    """

    # The V2 pooling path is not wired for prompt embeds.
    supports_prompt_embeds = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)

        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = self.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Build an attention group (and its non-causal metadata builder) for the
        # encoder-only layers, grouped by backend + query/KV head config. Models
        # are typically uniform, yielding a single group.
        attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
        groups: dict[tuple[tuple[str, str], int, int, int], AttentionGroup] = {}
        for name, layer in attn_layers.items():
            if layer.attn_type != AttentionType.ENCODER_ONLY:
                continue
            # No KV cache is bound for these layers; give them an (empty)
            # device tensor so the attention forward context is well-formed.
            layer.kv_cache = torch.empty(0, dtype=kv_cache_dtype, device=device)
            backend = layer.get_attn_backend()
            # Metadata builders require a uniform query-head count per group.
            key = (
                backend.full_cls_name(),
                layer.num_heads,
                layer.num_kv_heads,
                layer.head_size,
            )
            group = groups.get(key)
            if group is None:
                spec = EncoderOnlyAttentionSpec(
                    block_size=cache_config.block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                )
                group = AttentionGroup(backend, [], spec, kv_cache_group_id=len(groups))
                groups[key] = group
            group.layer_names.append(name)

        self.encoder_attn_groups = list(groups.values())
        for group in self.encoder_attn_groups:
            group.create_metadata_builders(vllm_config, device)

        # Encoder attention reads neither the block table nor the slot mapping
        # (full varlen self-attention over the batch's q/k/v), but the metadata
        # builder expects both tensors to be present.
        self._dummy_block_table = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int32, device=device
        )
        self._dummy_slot_mapping = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.token_type_ids: dict[str, torch.Tensor] = {}
        # Segment layouts (one-pass multi-candidate scoring): per request, the
        # segment id and the position of every prompt token.
        self.segment_layouts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        pooling_params = new_req_data.pooling_params
        if pooling_params is not None and (
            layout := pooling_params.segment_layout_params
        ):
            self.segment_layouts[new_req_data.req_id] = segment_layout_rows(
                layout.segment_lengths
            )
        if pooling_params is None or pooling_params.extra_kwargs is None:
            return

        token_type_range = pooling_params.extra_kwargs.get("compressed_token_type_ids")
        if token_type_range is not None:
            assert new_req_data.prompt_token_ids is not None
            token_type_start, token_type_end = token_type_range
            positions = torch.arange(
                len(new_req_data.prompt_token_ids), dtype=torch.int32
            )
            self.token_type_ids[new_req_data.req_id] = (
                (positions >= token_type_start) & (positions < token_type_end)
            ).to(torch.int32)

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self.token_type_ids.pop(req_id, None)
        self.segment_layouts.pop(req_id, None)

    def _segment_rows(
        self, input_batch: InputBatch, column: int
    ) -> torch.Tensor | None:
        """Column 0 (segment ids) or 1 (positions) of this step's tokens for
        requests with a segment layout, as a CPU tensor over the padded batch;
        None when no scheduled request has a layout. Rows of other requests
        are left at 0 (segment ids) or -1 (positions: keep the default)."""
        if not self.segment_layouts:
            return None
        rows = None
        offset = 0
        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = int(input_batch.num_scheduled_tokens[i])
            layout = self.segment_layouts.get(req_id)
            if layout is not None:
                if rows is None:
                    rows = torch.full(
                        (input_batch.num_tokens_after_padding,),
                        0 if column == 0 else -1,
                        dtype=torch.int64,
                        pin_memory=PIN_MEMORY,
                    )
                start = int(input_batch.num_computed_tokens_np[i])
                rows[offset : offset + num_tokens].copy_(
                    layout[column][start : start + num_tokens]
                )
            offset += num_tokens
        return rows

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        layout_positions = self._segment_rows(input_batch, column=1)
        if layout_positions is not None:
            # Candidates restart their positions right after the prefix.
            default = model_inputs.get("positions")
            if default is None:
                default = input_batch.positions[: input_batch.num_tokens_after_padding]
            override = layout_positions.to(self.device, non_blocking=True)
            model_inputs["positions"] = torch.where(override >= 0, override, default)
        if not self.token_type_ids:
            return model_inputs

        token_type_ids_cpu = torch.zeros(
            input_batch.num_tokens_after_padding,
            dtype=torch.int32,
            pin_memory=PIN_MEMORY,
        )
        offset = 0
        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = int(input_batch.num_scheduled_tokens[i])
            request_token_type_ids = self.token_type_ids.get(req_id)
            if request_token_type_ids is not None:
                start = int(input_batch.num_computed_tokens_np[i])
                token_type_ids_cpu[offset : offset + num_tokens].copy_(
                    request_token_type_ids[start : start + num_tokens]
                )
            offset += num_tokens
        model_inputs["token_type_ids"] = token_type_ids_cpu.to(
            self.device, non_blocking=True
        )
        return model_inputs

    def get_additional_cg_support(self) -> tuple[AttentionCGSupport, str | None]:
        # Encoder groups are built here rather than in init_attn_backend, so
        # their cudagraph support must be surfaced to the runner separately.
        support = AttentionCGSupport.ALWAYS
        backend: str | None = None
        for group in self.encoder_attn_groups:
            builder = group.get_metadata_builder(0)
            cg_support = builder.get_cudagraph_support(
                self.vllm_config, group.kv_cache_spec
            )
            if cg_support.value < support.value:
                support = cg_support
                backend = group.backend.__name__
        return support, backend

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        attn_metadata.update(
            self._build_encoder_attn_metadata(input_batch, cudagraph_mode, for_capture)
        )
        return attn_metadata

    def _build_encoder_attn_metadata(
        self, input_batch: InputBatch, cudagraph_mode: CUDAGraphMode, for_capture: bool
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max())

        segment_ids = self._segment_rows(input_batch, column=0)
        # The encoder builder forces ``causal=False`` regardless of this value.
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            seq_lens=input_batch.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=self._dummy_block_table[:num_reqs],
            slot_mapping=self._dummy_slot_mapping[:num_tokens],
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            positions=input_batch.positions,
            encoder_segment_ids=(
                None
                if segment_ids is None
                else segment_ids[:num_tokens].to(self.device, non_blocking=True)
            ),
        )

        attn_metadata: dict[str, Any] = {}
        for group in self.encoder_attn_groups:
            builder = group.get_metadata_builder(0)
            if for_capture:
                metadata = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                metadata = builder.build(
                    common_prefix_len=0, common_attn_metadata=common_attn_metadata
                )
            for name in group.layer_names:
                attn_metadata[name] = metadata
        return attn_metadata
