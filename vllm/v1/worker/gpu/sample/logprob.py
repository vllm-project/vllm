# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
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
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
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
    logits_for_topk: torch.Tensor | None = None,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    topk_logits = logits if logits_for_topk is None else logits_for_topk
    batch_size, vocab_size = logits.shape

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = torch.topk(topk_logits, num_logprobs, dim=-1).indices
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
            topk_token_ids = torch.topk(topk_logits, num_logprobs, dim=-1).indices
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

    if logits_for_topk is not None:
        masked = logits_for_topk.gather(-1, logprob_token_ids).isneginf()
        # Padding may repeat the sampled token; retain its score in every column.
        masked &= logprob_token_ids != sampled_token_ids.unsqueeze(-1)
        scores.masked_fill_(masked, float("-inf"))

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


@triton.jit
def _mask_logprob_logits_kernel(
    output_ptr,
    output_stride,
    logits_ptr,
    logits_stride,
    masks_ptr,
    mask_stride,
    idx_mapping_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(idx_mapping_ptr + row).to(tl.int64)
    token = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = token < vocab_size
    word = tl.load(masks_ptr + req_idx * mask_stride + token // 32, mask=valid, other=0)
    allowed = ((word >> (token % 32)) & 1) != 0
    value = tl.load(logits_ptr + row * logits_stride + token, mask=valid, other=0)
    tl.store(
        output_ptr + row * output_stride + token,
        tl.where(allowed, value, float("-inf")),
        mask=valid,
    )


class LogprobTokenMaskState:
    """Per-request top-k candidate masks, independent of sampling logits."""

    def __init__(self, max_num_reqs: int, vocab_size: int, device: torch.device):
        self.enabled = np.zeros(max_num_reqs, dtype=bool)
        self.num_words = triton.cdiv(vocab_size, 32)
        self.device = device
        self.masks: StagedWriteTensor | None = None
        # Host masks can be allocated on demand. The non-UVA fallback must be
        # included in device memory profiling before the KV cache is sized.
        if not is_uva_available():
            self.masks = self._allocate_masks()

    def _allocate_masks(self) -> StagedWriteTensor:
        masks = StagedWriteTensor(
            (len(self.enabled), self.num_words),
            dtype=torch.int32,
            device=self.device,
            uva_instead_of_gpu=True,
        )
        masks.gpu.fill_(-1)
        return masks

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        mask = sampling_params._logprob_token_mask
        if mask is not None:
            if self.masks is None:
                self.masks = self._allocate_masks()
            self.masks.stage_write(req_idx, 0, mask)
        elif self.enabled[req_idx]:
            assert self.masks is not None
            self.masks.stage_write(req_idx, 0, [-1] * self.num_words)
        self.enabled[req_idx] = mask is not None

    def apply_staged_writes(self) -> None:
        if self.masks is not None:
            self.masks.apply_write()

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor | None:
        if not np.any(self.enabled[idx_mapping_np]):
            return None
        assert self.masks is not None
        masked_logits = torch.empty_like(logits)
        _mask_logprob_logits_kernel[
            (logits.shape[0], triton.cdiv(logits.shape[1], 1024))
        ](
            masked_logits,
            masked_logits.stride(0),
            logits,
            logits.stride(0),
            self.masks.gpu,
            self.num_words,
            expanded_idx_mapping,
            logits.shape[1],
            BLOCK_SIZE=1024,
        )
        return masked_logits
