# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class MossVLModelState(ModelState):
    """Model state for MOSS-VL cross-attention encoder outputs.

    MOSS-VL consumes multimodal encoder outputs through cross-attention, so the
    outputs must be passed to model forward rather than merged into text
    embeddings.
    """

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
        self.encoder_seq_lens_gpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.encoder_outputs: list[torch.Tensor] = []
        self.encoder_seq_lens_by_req: dict[str, int] = {}

    def get_supported_generation_tasks(self):
        return ("generate",)

    def add_request(self, req_index: int, new_req_data) -> None:
        return None

    def apply_staged_writes(self) -> None:
        return None

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        encoder_inputs: dict[str, list[int]] = {}
        for req_id in input_batch.req_ids:
            req_encoder_inputs = scheduled_encoder_inputs.get(req_id, [])
            if req_encoder_inputs:
                encoder_inputs[req_id] = req_encoder_inputs

        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        if not mm_kwargs:
            self.encoder_outputs = []
            return None

        outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        self.encoder_outputs = outputs

        output_idx = 0
        for req_id, input_ids in encoder_inputs.items():
            req_len = 0
            for _ in input_ids:
                req_len += int(outputs[output_idx].shape[0])
                output_idx += 1
            self.encoder_seq_lens_by_req[req_id] = req_len
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
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        encoder_seq_lens = self._get_encoder_seq_lens(
            input_batch.req_ids, attn_groups, for_capture
        )
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = (
            self.max_model_len
            if for_capture
            else int(seq_lens_cpu_upper_bound[:num_reqs].max().item())
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            encoder_seq_lens=encoder_seq_lens,
        )

    def _get_encoder_seq_lens(
        self,
        req_ids: list[str],
        attn_groups: list[list[AttentionGroup]],
        for_capture: bool,
    ) -> dict[int, tuple[torch.Tensor, np.ndarray]]:
        num_reqs = len(req_ids)
        encoder_seq_lens_np = np.zeros(num_reqs, dtype=np.int32)
        if for_capture:
            # CUDA graph capture uses dummy inputs and only needs a small,
            # nonzero encoder length to initialize cross-attention metadata.
            # Using max_model_len here is text-context-sized (e.g. 262k for
            # MOSS-VL) and creates huge cross-attention slot mappings.
            encoder_seq_lens_np[:] = 1
        else:
            for idx, req_id in enumerate(req_ids):
                encoder_seq_lens_np[idx] = self.encoder_seq_lens_by_req.get(req_id, 0)

        self.encoder_seq_lens_gpu[:num_reqs].copy_(
            torch.from_numpy(encoder_seq_lens_np), non_blocking=True
        )
        self.encoder_seq_lens_gpu[num_reqs:].fill_(0)
        encoder_seq_lens_gpu = self.encoder_seq_lens_gpu[:num_reqs]

        seq_lens_by_group: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        for kv_cache_group_idx, groups in enumerate(attn_groups):
            if any(
                isinstance(attn_group.kv_cache_spec, CrossAttentionSpec)
                for attn_group in groups
            ):
                seq_lens_by_group[kv_cache_group_idx] = (
                    encoder_seq_lens_gpu,
                    encoder_seq_lens_np,
                )
        return seq_lens_by_group
