# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig
from vllm.v1.worker.cpu.attn_utils import build_attn_metadata
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.input_batch import InputBatch
from vllm.v1.worker.cpu.model_states.interface import (
    ModelSpecificAttnMetadata,
    ModelState,
)
from vllm.v1.worker.cpu.states import CPURequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class EncoderDecoderAttnMetadata(ModelSpecificAttnMetadata):
    encoder_seq_lens: dict[int, tuple[torch.Tensor, np.ndarray]]

    def get_extra_common_attn_kwargs(
        self, kv_cache_group_id: int, num_reqs: int
    ) -> dict[str, Any]:
        encoder_seq_lens = self.encoder_seq_lens.get(kv_cache_group_id)
        if encoder_seq_lens is None:
            return {}
        encoder_seq_lens_tensor, encoder_seq_lens_np = encoder_seq_lens
        return {
            "encoder_seq_lens": encoder_seq_lens_tensor[:num_reqs],
            "encoder_seq_lens_cpu": encoder_seq_lens_np[:num_reqs],
        }


class EncoderDecoderModelState(ModelState):
    """ModelState for cross-attention encoder-decoder models
    (Whisper, CohereASR, NemotronParse, ...)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        assert encoder_cache is not None
        if vllm_config.model_config.enable_prompt_embeds:
            raise ValueError(
                "--enable-prompt-embeds is not supported with encoder-decoder models."
            )
        super().__init__(vllm_config, model, encoder_cache, device)

        self.max_encoder_len = getattr(
            self.model_config.hf_config,
            "max_source_positions",
            self.max_model_len,
        )
        self.encoder_seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.encoder_seq_lens_np = self.encoder_seq_lens.numpy()
        self.encoder_outputs: list[torch.Tensor] = []

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: CPURequestState,
    ) -> None:
        # Ensure encoder inputs are ordered consistently with input_batch.req_ids.
        encoder_inputs: dict[str, list[int]] = {}
        for req_id in input_batch.req_ids:
            req_encoder_inputs = scheduled_encoder_inputs.get(req_id, [])
            if req_encoder_inputs:
                encoder_inputs[req_id] = req_encoder_inputs

        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        if mm_kwargs:
            # Encoder-decoder models consume encoder outputs through the
            # `encoder_outputs` forward kwarg, not `inputs_embeds`. Single modality
            # so execute_mm_encoder preserves request order; use its return value
            # directly. No need to store in encoder_cache: cross-attention K/V are
            # written to the KV cache on the first step; decode steps use the cache.
            with self.encoder_runner.timed_encoder_operation(encoder_inputs.keys()):
                self.encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        else:
            # Decode steps: encoder K/V are in cross-attention KV cache.
            self.encoder_outputs = []
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: CPURequestState
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
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        enc_dec_attn_metadata = EncoderDecoderAttnMetadata(
            self._get_encoder_seq_lens(input_batch.req_ids, attn_groups)
        )

        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = int(seq_lens_cpu_upper_bound[:num_reqs].max())
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc=input_batch.query_start_loc,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            model_specific_attn_metadata=enc_dec_attn_metadata,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        return attn_metadata

    def _get_encoder_seq_lens(
        self,
        req_ids: list[str],
        attn_groups: list[list[AttentionGroup]],
    ) -> dict[int, tuple[torch.Tensor, np.ndarray]]:
        num_reqs = len(req_ids)
        encoder_seq_lens_np = self.encoder_seq_lens_np[:num_reqs]
        encoder_seq_lens_np.fill(0)
        for i, req_id in enumerate(req_ids):
            mm_features = self.encoder_cache.mm_features.get(req_id, [])
            encoder_seq_lens_np[i] = sum(
                feature.mm_position.get_num_embeds() for feature in mm_features
            )

        seq_lens_by_group: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        encoder_seq_lens = self.encoder_seq_lens[:num_reqs]
        for kv_cache_group_idx, groups in enumerate(attn_groups):
            has_cross_attn = any(
                isinstance(attn_group.kv_cache_spec, CrossAttentionSpec)
                for attn_group in groups
            )
            if has_cross_attn:
                seq_lens_by_group[kv_cache_group_idx] = (
                    encoder_seq_lens,
                    encoder_seq_lens_np,
                )
        return seq_lens_by_group
