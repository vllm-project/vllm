# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm import envs
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    # Online log-sum-exp: each vocabulary element is read exactly once.
    max_val = float("-inf")
    sum_exp = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            row_ptr + block, mask=block < vocab_size, other=float("-inf")
        ).to(tl.float32)
        block_max = tl.max(logits)
        new_max = tl.maximum(max_val, block_max)
        safe_new_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        old_scale = tl.where(
            max_val == float("-inf"), 0.0, tl.exp(max_val - safe_new_max)
        )
        block_sum = tl.sum(tl.exp(logits - safe_new_max))
        sum_exp = sum_exp * old_scale + block_sum
        max_val = new_max

    logsumexp = max_val + tl.log(sum_exp)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk_ids_stride + k_offset,
            mask=k_mask,
            other=0,
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask, other=0.0)
        logits = logits.to(tl.float32)
        tl.store(
            output_ptr + req_idx * topk + k_offset,
            logits - logsumexp,
            mask=k_mask,
        )


@triton.jit
def _partial_logsumexp_kernel(
    logits_ptr,
    logits_stride,
    partial_max_ptr,
    partial_sum_ptr,
    vocab_size,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    offsets = split_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_ptr = logits_ptr + req_idx * logits_stride
    logits = tl.load(
        row_ptr + offsets, mask=offsets < vocab_size, other=float("-inf")
    ).to(tl.float32)

    local_max = tl.max(logits)
    safe_local_max = tl.where(local_max == float("-inf"), 0.0, local_max)
    local_sum = tl.sum(tl.exp(logits - safe_local_max))
    # An all -inf chunk contributes zero to the merged exponential sum.
    local_sum = tl.where(local_max == float("-inf"), 0.0, local_sum)

    output_offset = req_idx * NUM_SPLITS + split_idx
    tl.store(partial_max_ptr + output_offset, local_max)
    tl.store(partial_sum_ptr + output_offset, local_sum)


@triton.jit
def _merge_logsumexp_and_gather_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    vocab_size,
    partial_max_ptr,
    partial_sum_ptr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    split_offsets = tl.arange(0, BLOCK_SPLITS)
    partial_offset = req_idx * NUM_SPLITS + split_offsets
    partial_max = tl.load(
        partial_max_ptr + partial_offset,
        mask=split_offsets < NUM_SPLITS,
        other=float("-inf"),
    )
    partial_sum = tl.load(
        partial_sum_ptr + partial_offset,
        mask=split_offsets < NUM_SPLITS,
        other=0.0,
    )
    max_val = tl.max(partial_max)
    safe_max_val = tl.where(max_val == float("-inf"), 0.0, max_val)
    scaled_sums = partial_sum * tl.exp(partial_max - safe_max_val)
    scaled_sums = tl.where(partial_max == float("-inf"), 0.0, scaled_sums)
    sum_exp = tl.sum(scaled_sums)
    logsumexp = tl.where(
        max_val == float("-inf"), float("-inf"), max_val + tl.log(sum_exp)
    )

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk_ids_stride + k_offset,
            mask=k_mask,
            other=0,
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask, other=0.0)
        logits = logits.to(tl.float32)
        tl.store(
            output_ptr + req_idx * topk + k_offset,
            logits - logsumexp,
            mask=k_mask,
        )


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap gather width so large num_logprobs requests use bounded-size chunks.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)

    # Splitting the vocabulary creates more independent programs for small
    # decode batches. Keep the one-program-per-row path for large batches and
    # batch-invariant execution, where the reduction order must not depend on
    # how many other rows are scheduled with a request.
    if (
        batch_size <= 4
        and vocab_size > 1024
        and not envs.VLLM_BATCH_INVARIANT
    ):
        num_splits = triton.cdiv(vocab_size, 1024)
        partial_max = torch.empty(
            (batch_size, num_splits), device=logits.device, dtype=torch.float32
        )
        partial_sum = torch.empty_like(partial_max)
        _partial_logsumexp_kernel[(batch_size, num_splits)](
            logits,
            logits.stride(0),
            partial_max,
            partial_sum,
            vocab_size,
            NUM_SPLITS=num_splits,
            BLOCK_SIZE=1024,  # type: ignore
        )
        _merge_logsumexp_and_gather_kernel[(batch_size,)](
            logprobs,
            logits,
            logits.stride(0),
            token_ids,
            token_ids.stride(0),
            num_logprobs,
            partial_max,
            partial_sum,
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            TOPK_BLOCK_SIZE=topk_block_size,
        )
    else:
        _topk_log_softmax_kernel[(batch_size,)](
            logprobs,
            logits,
            logits.stride(0),
            token_ids,
            token_ids.stride(0),
            num_logprobs,
            vocab_size,
            BLOCK_SIZE=1024,  # type: ignore
            TOPK_BLOCK_SIZE=topk_block_size,
        )
    return logprobs

def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single triton kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = torch.topk(logits, num_logprobs, dim=-1).indices
            topk_token_ids = topk_token_ids.to(torch.int32)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_zeros((batch_size, 1 + num_cols))
        valid_mask = torch.zeros_like(logprob_token_ids, dtype=torch.bool)
        _fill_logprob_token_ids_kernel[(batch_size,)](
            logprob_token_ids,
            logprob_token_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu.stride(0),
            NUM_TOPK=num_logprobs,
            PADDED_COLS=triton.next_power_of_2(num_cols),
        )
        if logits_mode:
            scores = logits.gather(-1, logprob_token_ids).to(torch.float32)
        else:
            scores = compute_token_logprobs(logits, logprob_token_ids)
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.token_ids = StagedWriteTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
            dtype=torch.int32,
            device=device,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))
