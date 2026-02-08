# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        # For now, only support prompt logprobs for the prompt tokens (not top-k).
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        request_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
        # [max_num_reqs]
        prefill_lens: np.ndarray,
        # [max_num_reqs]
        num_computed_prefill_tokens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        prompt_lens = prompt_lens[idx_mapping_np]
        # NOTE(woosuk): -1 because the last prompt token's hidden state is not
        # needed for prompt logprobs.
        computed_prefill = num_computed_prefill_tokens[idx_mapping_np]
        includes_prompt = computed_prefill < prompt_lens - 1
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < prefill_lens[idx_mapping_np]
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            request_token_ids,
        )
        # Compute the prompt logprobs.
        prompt_logprobs, prompt_ranks = compute_prompt_logprobs_with_chunking(
            prompt_logprobs_token_ids,
            hidden_states[: input_batch.num_tokens],
            logits_fn,
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_token_ids = prompt_logprobs_token_ids.unsqueeze(-1)
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not is_prompt_chunked[i]:
                end_idx -= 1
            logprobs = LogprobsTensors(
                logprob_token_ids=prompt_token_ids[start_idx:end_idx],
                logprobs=prompt_logprobs[start_idx:end_idx],
                selected_token_ranks=prompt_ranks[start_idx:end_idx],
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if is_prompt_chunked[i]:
                # Prompt is chunked. Do not return the logprobs yet.
                prompt_logprobs_list.append(logprobs)
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                prompt_logprobs_list.append(logprobs)
                logprobs = LogprobsTensors(
                    logprob_token_ids=torch.cat(
                        [x.logprob_token_ids for x in prompt_logprobs_list]
                    ),
                    logprobs=torch.cat([x.logprobs for x in prompt_logprobs_list]),
                    selected_token_ranks=torch.cat(
                        [x.selected_token_ranks for x in prompt_logprobs_list]
                    ),
                )
                prompt_logprobs_list.clear()

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    request_token_ids_ptr,
    request_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            request_token_ids_ptr
            + req_state_idx * request_token_ids_stride
            + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    request_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        request_token_ids,
        request_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    logprobs = []
    ranks = []
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        prompt_logprobs = compute_topk_logprobs(
            prompt_logits,
            0,  # num_logprobs
            prompt_token_ids[start_idx:end_idx],
        )
        logprobs.append(prompt_logprobs.logprobs)
        ranks.append(prompt_logprobs.selected_token_ranks)

    logprobs = torch.cat(logprobs, dim=0) if len(logprobs) > 1 else logprobs[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return logprobs, ranks
