# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    compute_mm_prefix_ranges,
    create_attn_groups,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        # Pruner is used for multimodal embedding pruning (EVS).
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
        )
        self.token_type_ids: dict[str, torch.Tensor] = {}
        self.encoder_only_attn_groups = self._init_encoder_only_attn_groups()

    def _init_encoder_only_attn_groups(self) -> list[AttentionGroup]:
        if self.model_config.runner_type != "pooling":
            return []

        layer_specs: dict[str, KVCacheSpec] = {}
        for (
            layer_name,
            layer,
        ) in self.vllm_config.compilation_config.static_forward_context.items():
            if (
                not isinstance(layer, Attention)
                or layer.attn_type != AttentionType.ENCODER_ONLY
            ):
                continue
            layer_specs[layer_name] = EncoderOnlyAttentionSpec(
                block_size=self.vllm_config.cache_config.block_size,
                num_kv_heads=layer.num_kv_heads,
                head_size=layer.head_size,
                dtype=layer.kv_cache_torch_dtype,
            )
        # Encoder-only attention does not allocate a KV cache group.
        return create_attn_groups(self.vllm_config, layer_specs, -1)

    def get_additional_attn_groups(self) -> list[AttentionGroup]:
        return self.encoder_only_attn_groups

    def _add_encoder_only_attn_metadata(
        self,
        attn_metadata: dict[str, Any],
        *,
        num_reqs: int,
        num_tokens: int,
        query_start_loc_gpu: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        max_query_len: int,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        for_capture: bool,
    ) -> None:
        if not self.encoder_only_attn_groups:
            return
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens[:num_reqs],
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=torch.empty(
                (num_reqs, 0), dtype=torch.int32, device=self.device
            ),
            slot_mapping=torch.empty(0, dtype=torch.int64, device=self.device),
            causal=False,
        )
        for group in self.encoder_only_attn_groups:
            assert group.metadata_builders, (
                "GPUModelRunner.initialize_kv_cache must initialize encoder-only "
                "metadata builders before attention metadata is built."
            )
            builder = group.get_metadata_builder()
            if for_capture:
                metadata = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                metadata = builder.build(0, common_attn_metadata)
            for layer_name in group.layer_names:
                attn_metadata[layer_name] = metadata

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        pooling_params = new_req_data.pooling_params
        if pooling_params is not None and pooling_params.extra_kwargs is not None:
            token_type_start = pooling_params.extra_kwargs.get(
                "compressed_token_type_ids"
            )
            if token_type_start is not None:
                assert new_req_data.prompt_token_ids is not None
                self.token_type_ids[new_req_data.req_id] = (
                    torch.arange(len(new_req_data.prompt_token_ids), dtype=torch.int32)
                    >= token_type_start
                ).to(torch.int32)

        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def remove_request(self, req_id: str) -> None:
        self.token_type_ids.pop(req_id, None)

    def apply_staged_writes(self) -> None:
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        return self.encoder_runner.inputs_embeds[:num_tokens]

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            # Execute the multimodal encoder.
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            # Cache the encoder outputs by mm_hash
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
        if self.mm_pruner is not None and mm_embeds:
            # EVS: recompute mrope positions for pruned media.
            mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)
            # We must flush the staged rope updates for prepare_inputs() to pick up.
            self.apply_staged_writes()

        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]
        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_ids_unpadded, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(
            input_batch, draft_lookahead
        )
        if self.mm_pruner is not None:
            # EVS: strip the appended mrope-position channels.
            mm_embeds = self.mm_pruner.strip(mm_embeds)
        return mm_embeds, is_mm_embed

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs: dict[str, torch.Tensor | None] = {}
        if self.token_type_ids:
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

        if self.rope_state is None:
            return model_inputs  # Common case (1D positions).

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        model_inputs["positions"] = self.rope_state.get_positions(
            input_batch.num_tokens_after_padding
        )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if self.supports_mm_inputs:
            inputs_embeds = self.encoder_runner.inputs_embeds[:num_tokens]
            model_inputs["inputs_embeds"] = inputs_embeds
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL or (
            cudagraph_mode == CUDAGraphMode.PIECEWISE
            and is_breakable_cudagraph_enabled()
        ):
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            req_doc_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            mm_req_doc_ranges=req_doc_ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        self._add_encoder_only_attn_metadata(
            attn_metadata,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            for_capture=for_capture,
        )
        return attn_metadata
