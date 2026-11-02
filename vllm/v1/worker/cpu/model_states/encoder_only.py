# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec, KVCacheConfig
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBatch
from vllm.v1.worker.cpu.model_states.default import DefaultModelState
from vllm.v1.worker.cpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.cpu.states import CPURequestState
from vllm.v1.worker.utils import AttentionGroup


class EncoderOnlyModelState(DefaultModelState):
    """ModelState for encoder-only (BERT/RoBERTa) models.

    Encoder attention needs no KV cache: it runs full, bidirectional
    self-attention over each request's tokens in a single forward. Such layers
    return no ``get_kv_cache_spec`` and therefore never enter
    ``kv_cache_config.kv_cache_groups``, so the KV-backed attention path never
    builds their metadata. We build their (non-causal) metadata here, keeping
    the normal KV-backed path untouched.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
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

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        pooling_params = new_req_data.pooling_params
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

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: CPURequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.token_type_ids:
            return model_inputs

        token_type_ids = torch.zeros(
            input_batch.num_tokens_after_padding,
            dtype=torch.int32,
            device=self.device,
        )
        offset = 0
        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = int(input_batch.num_scheduled_tokens[i])
            request_token_type_ids = self.token_type_ids.get(req_id)
            if request_token_type_ids is not None:
                start = int(input_batch.num_computed_tokens_np[i])
                token_type_ids[offset : offset + num_tokens].copy_(
                    request_token_type_ids[start : start + num_tokens]
                )
            offset += num_tokens
        model_inputs["token_type_ids"] = token_type_ids
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        attn_metadata = super().prepare_attn(
            input_batch,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        attn_metadata.update(self._build_encoder_attn_metadata(input_batch))
        return attn_metadata

    def _build_encoder_attn_metadata(self, input_batch: InputBatch) -> dict[str, Any]:
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max())

        # The encoder builder forces ``causal=False`` regardless of this value.
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=input_batch.query_start_loc,
            query_start_loc_cpu=input_batch.query_start_loc,
            seq_lens=input_batch.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=self._dummy_block_table[:num_reqs],
            slot_mapping=self._dummy_slot_mapping[:num_tokens],
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            positions=input_batch.positions,
        )

        attn_metadata: dict[str, Any] = {}
        for group in self.encoder_attn_groups:
            metadata = group.get_metadata_builder(0).build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for name in group.layer_names:
                attn_metadata[name] = metadata
        return attn_metadata
