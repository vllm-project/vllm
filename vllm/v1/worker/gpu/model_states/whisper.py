# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class WhisperModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.model_config.max_model_len
        self.device = device

        assert encoder_cache is not None
        self.encoder_cache = encoder_cache
        self.encoder_runner = EncoderRunner(
            model=self.model,
            max_num_tokens=self.max_num_tokens,
            hidden_size=self.model_config.get_inputs_embeds_size(),
            encoder_cache=self.encoder_cache,
            dtype=self.model_config.dtype,
            device=self.device,
        )

        self.max_encoder_len = getattr(
            self.model_config.hf_config,
            "max_source_positions",
            self.max_model_len,
        )
        # CUDA-graph capture only needs a representative encoder length for
        # kernel selection. Keep it bounded by max_model_len to avoid creating
        # slot mappings that can exceed capture-time block-table capacity.
        self.max_capture_encoder_len = min(self.max_encoder_len, self.max_model_len)
        self.encoder_seq_lens_cpu = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.encoder_seq_lens_gpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )

        self.encoder_outputs: list[torch.Tensor] = []

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        return None

    def apply_staged_writes(self) -> None:
        return None

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        # Ensure encoder inputs are ordered consistently with input_batch.req_ids.
        encoder_inputs: dict[str, list[int]] = {}
        for req_id in input_batch.req_ids:
            req_encoder_inputs = scheduled_encoder_inputs.get(req_id, [])
            if req_encoder_inputs:
                encoder_inputs[req_id] = req_encoder_inputs
        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        if mm_kwargs:
            # Whisper consumes encoder outputs through `encoder_outputs`, not
            # `inputs_embeds`.
            self.encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = {"encoder_outputs": self.encoder_outputs}
        self.encoder_outputs = []
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"encoder_outputs": []}

    def prepare_attn(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> dict[str, Any]:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        encoder_seq_lens_by_kv_group = self.get_encoder_seq_lens_by_kv_group(
            attn_groups=attn_groups,
            num_reqs=input_batch.num_reqs,
            req_ids=input_batch.req_ids,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            encoder_seq_lens_by_kv_group=encoder_seq_lens_by_kv_group,
        )
        return attn_metadata

    def _get_encoder_seq_lens(
        self,
        num_reqs: int,
        req_ids: list[str] | None,
        for_cudagraph_capture: bool,
    ) -> tuple[torch.Tensor, np.ndarray]:
        encoder_seq_lens_cpu = self.encoder_seq_lens_cpu[:num_reqs]
        encoder_seq_lens_cpu.fill(0)

        if for_cudagraph_capture:
            encoder_seq_lens_cpu[:] = self.max_capture_encoder_len
        elif req_ids is not None:
            for req_index, req_id in enumerate(req_ids):
                mm_features = self.encoder_cache.mm_features.get(req_id)
                if not mm_features:
                    continue
                encoder_seq_lens_cpu[req_index] = sum(
                    feature.mm_position.length for feature in mm_features
                )

        self.encoder_seq_lens_gpu[:num_reqs].copy_(
            torch.from_numpy(encoder_seq_lens_cpu), non_blocking=True
        )
        return self.encoder_seq_lens_gpu[:num_reqs], encoder_seq_lens_cpu

    def get_encoder_seq_lens_by_kv_group(
        self,
        attn_groups: list[list[AttentionGroup]],
        num_reqs: int,
        req_ids: list[str] | None,
        for_cudagraph_capture: bool = False,
    ) -> dict[int, tuple[torch.Tensor, np.ndarray]]:
        encoder_seq_lens, encoder_seq_lens_cpu = self._get_encoder_seq_lens(
            num_reqs=num_reqs,
            req_ids=req_ids,
            for_cudagraph_capture=for_cudagraph_capture,
        )
        seq_lens_by_group: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        for kv_cache_group_idx, groups in enumerate(attn_groups):
            has_cross_attn = any(
                isinstance(attn_group.kv_cache_spec, CrossAttentionSpec)
                for attn_group in groups
            )
            if has_cross_attn:
                seq_lens_by_group[kv_cache_group_idx] = (
                    encoder_seq_lens,
                    encoder_seq_lens_cpu,
                )
        return seq_lens_by_group
