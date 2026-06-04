# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState


class BaseSpeculator(ABC):
    @abstractmethod
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    @abstractmethod
    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        pass

    @abstractmethod
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        pass


class DraftModelSpeculator(BaseSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Widen for HC-multiplexed residuals (e.g. DeepSeek V4 feeds the MTP
        # draft the target's pre-hc_head (T, hc_mult * hidden_size) residual).
        # Non-HC models default to hc_mult=1 and are unaffected.
        hc_mult = getattr(self.draft_model_config.hf_config, "hc_mult", 1)
        self.hidden_size = self.hidden_size * hc_mult
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel

        # DP configuration
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=torch.float32,
                device=device,
            )

    @abstractmethod
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self.load_draft_model(target_model, target_attn_layer_names)

        all_attn_layers = set[str](
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.draft_attn_layer_names = all_attn_layers - target_attn_layer_names

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups, _, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
    ) -> dict[str, Any] | None:
        query_start_loc_cpu = torch.clamp(
            self.arange[: num_reqs_padded + 1], max=num_reqs
        )
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=1,
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        return attn_metadata

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
