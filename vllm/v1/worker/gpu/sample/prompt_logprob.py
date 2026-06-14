# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int):
        self.max_num_reqs = max_num_reqs

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}
        self.prompt_token_ids: dict[str, list[int]] = {}

    def add_request(
        self,
        req_id: str,
        req_idx: int,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int],
    ):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []
            self.prompt_token_ids[req_id] = list(prompt_token_ids)

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)
        self.prompt_token_ids.pop(req_id, None)

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
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

        prompt_logprobs_token_ids = self._get_prompt_logprobs_token_ids(
            input_batch,
            computed_prefill,
            needs_prompt_logprobs,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
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

            if logprobs is None:
                continue

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict

    def _get_prompt_logprobs_token_ids(
        self,
        input_batch: InputBatch,
        computed_prefill: np.ndarray,
        needs_prompt_logprobs: np.ndarray,
    ) -> torch.Tensor:
        token_ids = [0] * input_batch.num_tokens
        query_start_loc_np = input_batch.query_start_loc_np
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            prompt_token_ids = self.prompt_token_ids[req_id]
            start_idx = int(query_start_loc_np[i])
            end_idx = int(query_start_loc_np[i + 1])
            target_start = int(computed_prefill[i]) + 1
            for offset, out_idx in enumerate(range(start_idx, end_idx)):
                target_idx = target_start + offset
                if target_idx < len(prompt_token_ids):
                    token_ids[out_idx] = prompt_token_ids[target_idx]

        return torch.tensor(
            token_ids,
            dtype=torch.int64,
            device=input_batch.input_ids.device,
        )


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    token_ids = []
    logprobs = []
    ranks = []
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num_prompt_logprobs = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        prompt_logprobs = compute_topk_logprobs(
            prompt_logits,
            requested_num_prompt_logprobs,
            prompt_token_ids[start_idx:end_idx],
        )
        token_ids.append(prompt_logprobs.logprob_token_ids)
        logprobs.append(prompt_logprobs.logprobs)
        ranks.append(prompt_logprobs.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    logprobs = torch.cat(logprobs, dim=0) if len(logprobs) > 1 else logprobs[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, logprobs, ranks
