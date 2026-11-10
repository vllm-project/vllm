# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config.model import LogprobsMode
from vllm.platforms import current_platform
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.cpu.input_batch import InputBatch

CHUNK_SIZE = 1024


@dataclass
class _TokenIdScores:
    token_ids: torch.Tensor  # [num_rows, num_ids], on device
    pad: torch.Tensor  # [num_rows, num_ids] bool, True at padded columns
    scores: torch.Tensor | None = None  # [num_rows, num_ids], filled per chunk


class LogprobTokenIdsState:
    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.num_token_ids_np = self.num_token_ids.numpy()
        self.token_ids = torch.zeros(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids_np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids_np[req_idx] = n
        self.token_ids[req_idx, :n] = torch.tensor(
            token_ids, dtype=torch.int32, device=self.token_ids.device
        )

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids_np[idx_mapping_np].max(initial=0))


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
)
def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    token_ids = token_ids.to(torch.int64)
    selected = logits.gather(-1, token_ids).to(torch.float32)
    # Accumulate the normalizer in FP32. Compilation fuses this cast into the
    # reduction instead of materializing a full FP32 logits tensor.
    normalizer = torch.logsumexp(logits.to(torch.float32), dim=-1, keepdim=True)
    return selected - normalizer


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
)
def _compute_token_ranks(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(-1, token_ids.unsqueeze(-1))
    return (logits >= selected).sum(-1)


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    num_per_req_token_ids: torch.Tensor | None = None,
    per_req_token_ids: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    sampled_token_ids = sampled_token_ids.to(torch.int64)

    if max_per_req_token_ids == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_token_ids), dim=1)
        valid_mask = None
    else:
        assert num_per_req_token_ids is not None
        assert per_req_token_ids is not None
        assert expanded_idx_mapping is not None
        batch_size = logits.shape[0]
        num_cols = max(num_logprobs, max_per_req_token_ids)
        selected_token_ids = sampled_token_ids.new_zeros((batch_size, num_cols))
        valid_mask = torch.zeros_like(selected_token_ids, dtype=torch.bool)

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            selected_token_ids[:, :num_logprobs] = topk_token_ids
        valid_mask[:, :num_logprobs] = True

        req_indices = expanded_idx_mapping.to(torch.int64)
        num_custom = num_per_req_token_ids[req_indices].to(torch.int64)
        custom_token_ids = torch.zeros_like(selected_token_ids)
        custom_token_ids[:, :max_per_req_token_ids] = per_req_token_ids[
            req_indices, :max_per_req_token_ids
        ].to(torch.int64)
        custom_valid_mask = torch.arange(num_cols, device=logits.device).unsqueeze(
            0
        ) < num_custom.unsqueeze(1)
        use_custom = num_custom.unsqueeze(1) > 0
        selected_token_ids = torch.where(
            use_custom, custom_token_ids, selected_token_ids
        )
        valid_mask = torch.where(use_custom, custom_valid_mask, valid_mask)

        logprob_token_ids = torch.cat(
            (sampled_token_ids.unsqueeze(-1), selected_token_ids), dim=1
        )
        valid_mask = torch.cat(
            (
                torch.ones((batch_size, 1), dtype=torch.bool, device=logits.device),
                valid_mask,
            ),
            dim=1,
        )

    if logits_mode:
        logprobs = logits.gather(-1, logprob_token_ids).to(torch.float32)
    else:
        logprobs = compute_token_logprobs(logits, logprob_token_ids)
    if valid_mask is not None:
        logprobs.masked_fill_(~valid_mask, float("-inf"))

    sampled_ranks = _compute_token_ranks(logits, sampled_token_ids)

    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=sampled_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    return ops.get_prompt_logprobs_token_ids_cpu(
        num_tokens,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )


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


class CPUPromptLogprobsWorker:
    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ):
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.logprobs_mode = logprobs_mode

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.prompt_logprob_start = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}
        # req_id -> fixed-ID scoring state
        self.token_id_scores: dict[str, _TokenIdScores] = {}

    def add_request(
        self, req_id: str, req_idx: int, sampling_params: SamplingParams
    ) -> None:
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        self.prompt_logprob_start[req_idx] = sampling_params.prompt_logprob_start or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []
        if sampling_params.prompt_logprob_token_ids is not None:
            ids = sampling_params.prompt_logprob_token_ids
            # Decoded arrays may be read-only; torch.as_tensor warns on those.
            if not ids.flags.writeable:
                ids = ids.copy()
            ids = torch.as_tensor(ids, dtype=torch.int64, device=self.device)
            self.token_id_scores[req_id] = _TokenIdScores(ids.clamp_min(0), ids < 0)

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)
        self.token_id_scores.pop(req_id, None)

    def compute_prompt_token_id_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        prompt_lens: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Fill scores row (prompt row - start) per chunk; emit on the last."""
        if not self.token_id_scores:
            return {}
        logits_mode = self.logprobs_mode in ("raw_logits", "processed_logits")
        out: dict[str, torch.Tensor] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            req = self.token_id_scores.get(req_id)
            if req is None:
                continue
            req_idx = input_batch.idx_mapping_np[i]
            prompt_len = int(prompt_lens[req_idx])
            start = int(self.prompt_logprob_start[req_idx])
            chunk_start = int(input_batch.num_computed_prefill_tokens_np[i])
            chunk_end = chunk_start + int(input_batch.num_scheduled_tokens[i])
            # Skip decode steps and, as compute_prompt_logprobs does, requests
            # resumed after preemption: their scores were already emitted.
            if chunk_start >= prompt_len or prompt_len < input_batch.prefill_len_np[i]:
                self.token_id_scores.pop(req_id, None)
                continue
            if req.scores is None:
                if chunk_start > start:
                    # This prefill starts past the first scored row, so those
                    # rows will never be written; drop the request instead of
                    # emitting a buffer with unwritten rows.
                    self.token_id_scores.pop(req_id, None)
                    continue
                req.scores = torch.empty_like(req.token_ids, dtype=torch.float32)
            # The last prompt row predicts the first decode token; skip it.
            lo = max(start, chunk_start)
            hi = min(chunk_end, prompt_len - 1)
            base = int(input_batch.query_start_loc_np[i]) - chunk_start
            for a in range(lo, hi, CHUNK_SIZE):
                b = min(a + CHUNK_SIZE, hi)
                logits = logits_fn(hidden_states[base + a : base + b])
                ids = req.token_ids[a - start : b - start]
                req.scores[a - start : b - start] = (
                    logits.gather(-1, ids)
                    if logits_mode
                    else compute_token_logprobs(logits, ids)
                )
            if chunk_end >= prompt_len:
                req.scores.masked_fill_(req.pad, float("-inf"))
                out[req_id] = req.scores
                self.token_id_scores.pop(req_id, None)
        return out

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        all_token_ids: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_logprob_start = self.prompt_logprob_start[idx_mapping_np]
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
            # Skip rows before prompt_logprob_start.
            start_idx += max(int(prompt_logprob_start[i]) - int(computed_prefill[i]), 0)

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
