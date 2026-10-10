# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

# Token logprobs = logits[token_ids] - logsumexp(logits), per row.
#
# Memory traffic: each row's vocabulary is read exactly ONCE. The only loop
# over the vocabulary is `_online_max_sumexp`, which keeps a running
# (max, sum of exp) so no separate max pass is needed. After that, only the
# `num_logprobs` requested logits are re-loaded by `_gather_token_logprobs`
# (a gather, not a vocab pass). Two kernel layouts share these helpers:
#   * fused  (`_topk_log_softmax_kernel`): one program per row.
#   * split  (`_partial_max_sumexp_kernel` + `_merge_and_gather_kernel`): each
#     row's vocab is cut into disjoint chunks reduced by separate programs, so
#     small batches can use more than `batch_size` programs. Each logit still
#     belongs to exactly one chunk and is read once; stage 2 only reads the
#     [batch_size, num_splits] partials.

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024
# Vocab tile width of the logsumexp loops.
_BLOCK_SIZE = 1024
# Split-vocab heuristic (see `_get_num_vocab_splits`).
_SPLIT_TARGET_PROGRAMS_PER_SM = 2
_SPLIT_MIN_CHUNK_SIZE = 8192
_MAX_VOCAB_SPLITS = 64


@triton.jit
def _online_max_sumexp(
    row_ptr,
    start,
    end,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass (online) max and sum(exp(x - max)) over row[start:end].

    This is the only loop over the vocabulary: every logit in [start, end) is
    loaded once, and the running sum is rescaled by exp(m_old - m_new)
    whenever the running max grows.

    Returns (m, s) in FP32 such that logsumexp(row[start:end]) = m + log(s).
    An empty or all -inf range returns (-inf, 0), which is the identity of the
    merge `M = max(m_i); S = sum(s_i * exp(m_i - M))`.
    """
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    for i in range(start, end, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < end
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        x = tl.load(row_ptr + block, mask=mask, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # While the running max is still -inf (only -inf seen so far), shift
        # by 0 instead to avoid (-inf) - (-inf) = NaN. exp(-inf - 0) = 0, so
        # masked lanes and -inf logits contribute nothing either way.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        s = s * tl.exp(m - m_safe) + tl.sum(tl.exp(x - m_safe), axis=0)
        m = m_new
    return m, s


@triton.jit
def _gather_token_logprobs(
    output_ptr,
    row_ptr,
    topk_ids_ptr,
    topk_ids_stride,
    req_idx,
    topk,
    max_val,
    lse,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk_ids_stride + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


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
    # One program per row; single (online) pass over the vocab.
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val, se = _online_max_sumexp(row_ptr, 0, vocab_size, BLOCK_SIZE)
    lse = tl.log(se)
    _gather_token_logprobs(
        output_ptr,
        row_ptr,
        topk_ids_ptr,
        topk_ids_stride,
        req_idx,
        topk,
        max_val,
        lse,
        TOPK_BLOCK_SIZE,
    )


@triton.jit
def _partial_max_sumexp_kernel(
    # [batch_size, num_splits]
    partial_max_ptr,
    # [batch_size, num_splits]
    partial_sumexp_ptr,
    num_splits,
    logits_ptr,
    logits_stride,
    vocab_size,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Split-vocab stage 1: grid = (batch_size, num_splits). Each program
    # reduces one contiguous vocab chunk of its row to a local (max, sumexp).
    req_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    row_ptr = logits_ptr + req_idx * logits_stride

    start = split_idx * chunk_size
    end = tl.minimum(start + chunk_size, vocab_size)
    m, s = _online_max_sumexp(row_ptr, start, end, BLOCK_SIZE)
    tl.store(partial_max_ptr + req_idx * num_splits + split_idx, m)
    tl.store(partial_sumexp_ptr + req_idx * num_splits + split_idx, s)


@triton.jit
def _merge_and_gather_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    topk,
    # [batch_size, num_splits]
    partial_max_ptr,
    # [batch_size, num_splits]
    partial_sumexp_ptr,
    num_splits,
    PADDED_NUM_SPLITS: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    # Split-vocab stage 2: grid = (batch_size,). Merge the per-chunk partials
    # into the row's logsumexp, then gather logprobs at `topk_ids`.
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    splits = tl.arange(0, PADDED_NUM_SPLITS)
    split_mask = splits < num_splits
    maxes = tl.load(
        partial_max_ptr + req_idx * num_splits + splits,
        mask=split_mask,
        other=float("-inf"),
    )
    sumexps = tl.load(
        partial_sumexp_ptr + req_idx * num_splits + splits,
        mask=split_mask,
        other=0.0,
    )
    max_val = tl.max(maxes, axis=0)
    # Empty / all -inf chunks carry (-inf, 0) and add 0 * exp(-inf - M) = 0.
    # If every chunk is -inf (an all -inf row), M = -inf and the result is
    # NaN, which matches torch.log_softmax and the fused path.
    se = tl.sum(sumexps * tl.exp(maxes - max_val), axis=0)
    lse = tl.log(se)
    _gather_token_logprobs(
        output_ptr,
        row_ptr,
        topk_ids_ptr,
        topk_ids_stride,
        req_idx,
        topk,
        max_val,
        lse,
        TOPK_BLOCK_SIZE,
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


def _get_num_compute_units(device: torch.device) -> int:
    try:
        return num_compute_units(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    except Exception:
        # Platform does not expose its compute-unit count; disable splitting.
        return 0


def _get_num_vocab_splits(
    batch_size: int, vocab_size: int, device: torch.device
) -> int:
    """Number of programs to split each row's vocab across.

    One program per row leaves most SMs idle for small decode batches (the
    common case), so split each row across up to `_SPLIT_TARGET_PROGRAMS_PER_SM
    * num_SMs / batch_size` programs, each handling at least
    `_SPLIT_MIN_CHUNK_SIZE` logits. Returns 1 (the fused single-kernel path)
    when splitting would not help, and always under VLLM_BATCH_INVARIANT since
    the split count depends on the batch size and changes the FP32
    accumulation order.
    """
    if envs.VLLM_BATCH_INVARIANT:
        return 1
    target_programs = _get_num_compute_units(device) * _SPLIT_TARGET_PROGRAMS_PER_SM
    num_splits = min(
        target_programs // max(batch_size, 1),
        triton.cdiv(vocab_size, _SPLIT_MIN_CHUNK_SIZE),
        _MAX_VOCAB_SPLITS,
    )
    if num_splits < 2:
        return 1
    # Round down to a power of 2 to bound the number of compiled variants of
    # the merge kernel (PADDED_NUM_SPLITS is a constexpr).
    return 1 << (num_splits.bit_length() - 1)


def compute_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    num_vocab_splits: int | None = None,
) -> torch.Tensor:
    """Compute log_softmax(logits)[i, token_ids[i, j]] for every (i, j).

    Args:
        logits: [batch_size, vocab_size] logits (fp32/bf16/fp16). The row
            stride may be arbitrary (e.g. a slice of a padded buffer).
        token_ids: [batch_size, num_logprobs] integer token ids.
        num_vocab_splits: Override for the number of programs per row (testing
            and benchmarking only). None picks it from the batch size and the
            device's SM count; 1 forces the fused single-kernel path.

    Returns:
        [batch_size, num_logprobs] FP32 logprobs.

    """
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernels compute the row
    # max + logsumexp in a single online pass and only emit logprobs at
    # `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    if batch_size == 0 or num_logprobs == 0:
        return logprobs
    # The kernels index columns with unit stride (rows may be strided).
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    if token_ids.stride(1) != 1:
        token_ids = token_ids.contiguous()
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)

    if num_vocab_splits is None:
        num_vocab_splits = _get_num_vocab_splits(batch_size, vocab_size, logits.device)
    assert num_vocab_splits >= 1

    if num_vocab_splits == 1:
        _topk_log_softmax_kernel[(batch_size,)](
            logprobs,
            logits,
            logits.stride(0),
            token_ids,
            token_ids.stride(0),
            num_logprobs,
            vocab_size,
            BLOCK_SIZE=_BLOCK_SIZE,  # type: ignore
            TOPK_BLOCK_SIZE=topk_block_size,
        )
        return logprobs

    # Split-vocab path: stage 1 reduces each (row, chunk) to a local
    # (max, sumexp); stage 2 merges them and gathers the requested logprobs.
    chunk_size = triton.cdiv(triton.cdiv(vocab_size, num_vocab_splits), _BLOCK_SIZE)
    chunk_size *= _BLOCK_SIZE
    partial_max = logits.new_empty((batch_size, num_vocab_splits), dtype=torch.float32)
    partial_sumexp = torch.empty_like(partial_max)
    _partial_max_sumexp_kernel[(batch_size, num_vocab_splits)](
        partial_max,
        partial_sumexp,
        num_vocab_splits,
        logits,
        logits.stride(0),
        vocab_size,
        chunk_size,
        BLOCK_SIZE=_BLOCK_SIZE,  # type: ignore
    )
    _merge_and_gather_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_logprobs,
        partial_max,
        partial_sumexp,
        num_vocab_splits,
        PADDED_NUM_SPLITS=triton.next_power_of_2(num_vocab_splits),
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
