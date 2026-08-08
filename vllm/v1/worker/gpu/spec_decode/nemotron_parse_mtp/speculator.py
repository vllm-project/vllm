# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight MTP speculator for models with a trained auxiliary head.

This is the Model Runner V2 speculator behind ``method="mtp"`` for Nemotron
Parse, which ships a single dependent auxiliary prediction head instead of a
transformer draft model. It is selected via
``SpeculativeConfig.use_nemotron_parse_mtp()`` and, unlike the draft-model MTP
path, loads no separate model, allocates no draft KV cache, and runs no
attention -- the target model owns all proposal math and weights. Tensor
parallel execution uses the target model's vocab-parallel embedding and global
draft-token reduction while keeping the auxiliary head replicated.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)

class NemotronParseMTPSpeculator(BaseSpeculator):
    """A single-token, hidden-state-conditioned auxiliary-head speculator.

    The target model owns all proposal math and weights. This adapter only
    selects the hidden state that produced the latest sampled token and invokes
    ``target_model.propose_draft_token_ids(hidden_states, token_ids)``.
    """

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        if speculative_config.num_speculative_tokens != 1:
            raise ValueError(
                "Nemotron Parse MTP speculation supports exactly one token"
            )
        if speculative_config.draft_sample_method != "greedy":
            raise ValueError(
                "Nemotron Parse MTP speculation supports greedy drafts only"
            )
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.dtype = vllm_config.model_config.dtype
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.model: nn.Module | None = None
        self.cudagraph_manager: CudaGraphManager | None = None

        self.hidden_states = torch.zeros(
            self.max_num_reqs,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        self.d1_token_ids = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.draft_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

    def load_model(self, target_model: nn.Module) -> None:
        propose = getattr(target_model, "propose_draft_token_ids", None)
        if not callable(propose):
            raise TypeError(
                f"{type(target_model).__name__} does not implement "
                "propose_draft_token_ids(hidden_states, token_ids)"
            )
        self.model = target_model

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            # PIECEWISE graphs are not supported for this tiny head-only path.
            cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_manager = CudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=1,
        )

    def capture(self) -> None:
        if self.cudagraph_manager is None or not self.cudagraph_manager.needs_capture():
            return
        logger.info("Capturing CUDA graphs for Nemotron Parse MTP speculator...")
        self.hidden_states.zero_()
        self.d1_token_ids.zero_()
        self.draft_tokens.zero_()

        def create_forward_fn(
            desc,
            warmup: bool,
        ) -> Callable[[CUDAGraphMode], None]:
            num_reqs_padded = desc.num_reqs or desc.num_tokens
            return lambda cg_mode: self._run_mtp_head(num_reqs_padded)

        self.cudagraph_manager.capture(
            create_forward_fn,
            progress_bar_desc="Capturing Nemotron Parse MTP CUDA graphs",
        )

    def _prepare_mtp_inputs(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        req_state_indices = input_batch.idx_mapping[:num_reqs]
        sampled = num_sampled[:num_reqs] > 0

        hidden_indices = (
            input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs] - 1
        ).clamp_min(0)
        self.hidden_states[:num_reqs] = last_hidden_states[
            hidden_indices.to(torch.long)
        ]

        d1 = last_sampled[req_state_indices].reshape(-1).to(torch.long)
        self.d1_token_ids[:num_reqs] = torch.where(
            sampled, d1, torch.zeros_like(d1)
        )
        return sampled

    def _run_mtp_head(self, num_reqs_padded: int) -> None:
        if self.model is None:
            raise RuntimeError("target model has not been bound to the speculator")
        h = self.hidden_states[:num_reqs_padded]
        d1 = self.d1_token_ids[:num_reqs_padded]
        self.draft_tokens[:num_reqs_padded] = self.model.propose_draft_token_ids(h, d1)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            slot_mappings,
            aux_hidden_states,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        if self.model is None:
            raise RuntimeError("target model has not been bound to the speculator")

        num_reqs = input_batch.num_reqs
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )
        num_reqs_padded = batch_desc.num_reqs or num_reqs

        sampled = self._prepare_mtp_inputs(
            input_batch,
            last_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            num_reqs,
        )

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.cudagraph_manager is not None
            self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._run_mtp_head(num_reqs_padded)

        proposed = self.draft_tokens[:num_reqs]
        proposed = torch.where(sampled, proposed, torch.zeros_like(proposed))
        return proposed.to(torch.long).view(num_reqs, 1)
