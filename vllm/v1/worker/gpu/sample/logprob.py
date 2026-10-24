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

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024
_MIN_TILE_ELEMENTS = 2048
_MAX_TILES_PER_ROW = 64


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

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

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
def _vector_logsoftmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    token_ids_stride,
    num_tokens,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        logits_ptr + row * logits_stride + column,
        column < vocab_size,
        other=float("-inf"),
    ).to(tl.float32)
    maximum = tl.max(values, 0)
    exponentials = tl.where(
        (column < vocab_size) & (maximum != float("-inf")),
        tl.exp(values - maximum),
        0.0,
    )
    log_total = tl.log(tl.sum(exponentials, 0))
    for start in range(0, num_tokens, TOKEN_BLOCK):
        token_column = start + tl.arange(0, TOKEN_BLOCK)
        valid = token_column < num_tokens
        ids = tl.load(
            token_ids_ptr + row * token_ids_stride + token_column,
            valid,
            other=0,
        )
        selected = tl.load(logits_ptr + row * logits_stride + ids, valid, other=0.0).to(
            tl.float32
        )
        tl.store(
            output_ptr + row * num_tokens + token_column,
            selected - maximum - log_total,
            valid,
        )


@triton.jit
def _tile_logsumexp_kernel(
    partials_ptr,
    logits_ptr,
    logits_stride,
    vocab_size,
    num_tiles,
    num_rows,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce one contiguous vocabulary tile using a single vector load."""
    tile_id = tl.program_id(0)
    row = (tile_id // num_tiles).to(tl.int64)
    tile = tile_id % num_tiles
    lane = tl.arange(0, BLOCK_SIZE)
    column = tile * TILE_SIZE + lane
    valid = (lane < TILE_SIZE) & (column < vocab_size)
    values = tl.load(
        logits_ptr + row * logits_stride + column,
        mask=valid,
        other=float("-inf"),
    ).to(tl.float32)
    maximum = tl.max(values, 0)
    exponentials = tl.where(
        valid & (maximum != float("-inf")),
        tl.exp(values - maximum),
        0.0,
    )
    total = tl.sum(exponentials, 0)
    offset = row * num_tiles + tile
    tl.store(partials_ptr + offset, maximum)
    tl.store(partials_ptr + num_rows * num_tiles + offset, total)


@triton.jit
def _finish_tiled_logprobs_kernel(
    output_ptr,
    partials_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    token_ids_stride,
    num_tokens,
    num_tiles,
    num_rows,
    SPLIT_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.arange(0, SPLIT_BLOCK)
    valid_tile = tile < num_tiles
    partial_offset = row * num_tiles + tile
    maximums = tl.load(partials_ptr + partial_offset, valid_tile, other=float("-inf"))
    totals = tl.load(
        partials_ptr + num_rows * num_tiles + partial_offset,
        valid_tile,
        other=0.0,
    )
    maximum = tl.max(maximums, 0)
    terms = tl.where(
        valid_tile & (totals != 0.0),
        totals * tl.exp(maximums - maximum),
        0.0,
    )
    log_total = tl.log(tl.sum(terms, 0))

    for start in range(0, num_tokens, TOKEN_BLOCK):
        token_column = start + tl.arange(0, TOKEN_BLOCK)
        valid_token = token_column < num_tokens
        ids = tl.load(
            token_ids_ptr + row * token_ids_stride + token_column,
            valid_token,
            other=0,
        )
        logits = tl.load(
            logits_ptr + row * logits_stride + ids, valid_token, other=0.0
        ).to(tl.float32)
        logprobs = logits - maximum - log_total
        tl.store(
            output_ptr + row * num_tokens + token_column,
            logprobs,
            valid_token,
        )


def _get_num_vocab_splits(
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> int:
    if envs.VLLM_BATCH_INVARIANT or device.type != "cuda" or vocab_size <= 8192:
        return 1
    return _get_static_num_vocab_splits(batch_size, vocab_size, device)


def _get_static_num_vocab_splits(
    batch_size: int, vocab_size: int, device: torch.device
) -> int:
    try:
        device_index = device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        sms = num_compute_units(device_index)
    except Exception:
        return 1
    max_by_work = vocab_size // _MIN_TILE_ELEMENTS
    max_by_occupancy = 64 * sms // max(batch_size, 1)
    count = min(max_by_work, max_by_occupancy, _MAX_TILES_PER_ROW)
    return 1 << (count.bit_length() - 1) if count >= 2 else 1


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
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    num_vocab_splits: int | None = None,
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    if batch_size == 0 or num_logprobs == 0:
        return logprobs
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    if token_ids.stride(1) != 1:
        token_ids = token_ids.contiguous()
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    if num_vocab_splits is None:
        num_vocab_splits = _get_num_vocab_splits(batch_size, vocab_size, logits.device)
    if num_vocab_splits < 1:
        raise ValueError("num_vocab_splits must be positive")
    if num_vocab_splits > 1:
        tile_size = triton.cdiv(vocab_size, num_vocab_splits)
        partials = torch.empty(
            (2, batch_size, num_vocab_splits),
            device=logits.device,
            dtype=torch.float32,
        )
        _tile_logsumexp_kernel[(batch_size * num_vocab_splits,)](
            partials,
            logits,
            logits.stride(0),
            vocab_size,
            num_vocab_splits,
            batch_size,
            TILE_SIZE=tile_size,
            BLOCK_SIZE=triton.next_power_of_2(tile_size),
        )
        _finish_tiled_logprobs_kernel[(batch_size,)](
            logprobs,
            partials,
            logits,
            logits.stride(0),
            token_ids,
            token_ids.stride(0),
            num_logprobs,
            num_vocab_splits,
            batch_size,
            SPLIT_BLOCK=triton.next_power_of_2(num_vocab_splits),
            TOKEN_BLOCK=topk_block_size,
        )
        return logprobs
    if vocab_size <= 8192:
        _vector_logsoftmax_kernel[(batch_size,)](
            logprobs,
            logits,
            logits.stride(0),
            token_ids,
            token_ids.stride(0),
            num_logprobs,
            vocab_size,
            BLOCK_SIZE=triton.next_power_of_2(vocab_size),
            TOKEN_BLOCK=topk_block_size,
        )
        return logprobs
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
