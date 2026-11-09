# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

from .ngram_embedding import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLEDiskOffloadEmbedding,
    stage_checkpoint_rows,
)


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self.disk_offload_modules = [
            (module, module.ngram_embedding)
            for module in model.modules()
            if isinstance(module, Qwen4ExpNGramEmbedding)
            and isinstance(module.ngram_embedding, Qwen4ExpPLEDiskOffloadEmbedding)
        ]
        engram_config = vllm_config.engram_config
        self.disk_offload_overlap = bool(
            self.disk_offload_modules
            and engram_config
            and engram_config.disk_offload_overlap
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def _predict_next_chunk(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Predict each prefilling request's next prompt chunk.

        Assumes the scheduler gives a request the same number of tokens as
        this step, capped at its remaining prompt. A wrong guess only makes
        the next step read more rows itself.
        """
        num_reqs = input_batch.num_reqs
        if not input_batch.has_prefill or num_reqs == 0:
            return None
        scheduled = input_batch.num_scheduled_tokens[:num_reqs].astype(np.int64)
        start = input_batch.num_computed_prefill_tokens_np[:num_reqs] + scheduled
        remaining = input_batch.prefill_len_np[:num_reqs] - start
        selected = np.flatnonzero(
            input_batch.is_prefilling_np[:num_reqs] & (remaining > 0)
        )
        if selected.size == 0:
            return None
        lengths = np.minimum(remaining[selected], scheduled[selected])
        start = start[selected]
        requests = input_batch.idx_mapping_np[selected]
        query_start_loc = np.zeros(selected.size + 1, dtype=np.int32)
        np.cumsum(lengths, out=query_start_loc[1:])
        token_requests = np.repeat(requests, lengths)
        token_positions = np.arange(query_start_loc[-1]) + np.repeat(
            start - query_start_loc[:-1], lengths
        )
        context_positions = start[:, None] + np.arange(-self.ngram_context_len, 0)

        def to_device(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(self.device, non_blocking=True)

        all_token_ids = req_states.all_token_ids.gpu
        input_ids = all_token_ids[to_device(token_requests), to_device(token_positions)]
        context = all_token_ids[
            to_device(requests)[:, None], to_device(context_positions.clip(min=0))
        ]
        context = torch.where(
            to_device(context_positions >= 0),
            context,
            context.new_full((), self.ngram_eos_token_id),
        )
        return input_ids, to_device(query_start_loc), context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        if self.disk_offload_modules:
            next_chunk = (
                self._predict_next_chunk(input_batch, req_states)
                if self.disk_offload_overlap
                else None
            )
            stage_checkpoint_rows(
                self.disk_offload_modules,
                input_batch.input_ids,
                query_start_loc,
                ngram_context,
                next_chunk,
            )
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
