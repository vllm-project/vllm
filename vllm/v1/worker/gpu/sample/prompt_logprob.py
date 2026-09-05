# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import numpy as np
import torch

from vllm.config.model import LogprobsMode
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import (
    compute_token_logprobs,
    compute_topk_scores,
)

CHUNK_SIZE = 1024


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int, logprobs_mode: LogprobsMode = "raw_logprobs"):
        self.max_num_reqs = max_num_reqs
        self.logprobs_mode = logprobs_mode

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_id -> (candidate token IDs, first prompt row to score)
        self.prompt_logprob_token_ids: dict[str, tuple[list[int], int]] = {}
        self.in_progress_prompt_token_id_logprobs: dict[str, list[torch.Tensor]] = {}
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []
        if sampling_params.prompt_logprob_token_ids is not None:
            self.prompt_logprob_token_ids[req_id] = (
                list(sampling_params.prompt_logprob_token_ids),
                sampling_params.prompt_logprob_start or 0,
            )
            self.in_progress_prompt_token_id_logprobs[req_id] = []

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)
        self.prompt_logprob_token_ids.pop(req_id, None)
        self.in_progress_prompt_token_id_logprobs.pop(req_id, None)

    def compute_prompt_token_id_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        prompt_lens: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Compute fixed-ID scores and aggregate them across prompt chunks."""
        if not self.prompt_logprob_token_ids:
            return {}
        out: dict[str, torch.Tensor] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            request_data = self.prompt_logprob_token_ids.get(req_id)
            if request_data is None:
                continue
            candidate_ids_for_req, prompt_start = request_data
            state_idx = int(input_batch.idx_mapping_np[i])
            prompt_len = int(prompt_lens[state_idx])
            # NOTE: A request resumed after preemption re-prefills its prompt
            # plus the tokens it already generated; its scores were emitted
            # before the preemption. Skip it, as compute_prompt_logprobs does.
            if prompt_len < int(input_batch.prefill_len_np[i]):
                continue
            row_start = int(input_batch.query_start_loc_np[i])
            row_end = int(input_batch.query_start_loc_np[i + 1])
            prompt_row_start = int(input_batch.num_computed_prefill_tokens_np[i])
            pending = self.in_progress_prompt_token_id_logprobs[req_id]
            row_start += max(prompt_start - prompt_row_start, 0)
            prefill_end = prompt_row_start + int(input_batch.num_scheduled_tokens[i])
            is_last_chunk = prefill_end >= prompt_len
            # The final prompt row predicts the first decode token and is not
            # a prompt-position score, matching the existing prompt-logprobs
            # output convention.
            if is_last_chunk:
                row_end -= 1
            if row_start >= row_end:
                # This chunk scores nothing; still emit whatever earlier
                # chunks accumulated once the prompt is fully prefilled.
                if is_last_chunk and pending:
                    out[req_id] = torch.cat(pending)
                    pending.clear()
                continue
            ids = torch.as_tensor(
                candidate_ids_for_req,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            part = compute_prompt_token_id_logprobs_with_chunking(
                ids,
                hidden_states[row_start:row_end],
                logits_fn,
                self.logprobs_mode,
            )
            if not is_last_chunk:
                pending.append(part)
                continue
            if pending:
                part = torch.cat([*pending, part])
                pending.clear()
            out[req_id] = part
        return out

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
                self.logprobs_mode,
            )
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            # no logprobs if start_idx >= end_idx
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)
            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                logprobs = LogprobsTensors.cat(prompt_logprobs_list)
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
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
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
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
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    token_ids = []
    scores = []
    ranks = []
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    scores = torch.cat(scores, dim=0) if len(scores) > 1 else scores[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, scores, ranks


def compute_prompt_token_id_logprobs_with_chunking(
    candidate_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> torch.Tensor:
    """Gather caller-selected IDs from prompt logits without prompt-logprob semantics.

    ``candidate_token_ids`` is the request-wise ID list, a 1-D int64 tensor,
    scored at every row of ``prompt_hidden_states``. Unlike
    ``compute_prompt_logprobs_with_chunking``, this function does not add the
    actual target token or compute its rank.

    Returns:
        ``[num_rows, len(candidate_token_ids)]`` float32 scores.
    """
    num_rows = prompt_hidden_states.shape[0]
    score_chunks: list[torch.Tensor] = []
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    # compute_token_logprobs indexes token_ids as contiguous rows, so expand the
    # request-wise IDs once at chunk width and slice the tail chunk out of it.
    ids_block = candidate_token_ids.expand(min(num_rows, CHUNK_SIZE), -1).contiguous()
    for start_idx in range(0, num_rows, CHUNK_SIZE):
        logits = logits_fn(prompt_hidden_states[start_idx : start_idx + CHUNK_SIZE])
        ids = ids_block[: logits.shape[0]]
        if logits_mode:
            scores = logits.gather(-1, ids).to(torch.float32)
        else:
            # Reuses the sampler's kernel: it accumulates the logsumexp in fp32
            # and emits only the requested columns, so no [tokens, vocab]
            # tensor is materialized.
            scores = compute_token_logprobs(logits, ids)
        score_chunks.append(scores)
    return torch.cat(score_chunks) if len(score_chunks) > 1 else score_chunks[0]
