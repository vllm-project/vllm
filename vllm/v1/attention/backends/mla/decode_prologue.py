# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch decode-step prologue for the DeepSeek sparse attention paths.

``DecodePrologueBatch`` brackets a step's attention-metadata build: armed
builders stage their per-token decode builds (token-to-request mapping,
compressed and ring slot mappings, native decode lens, SWA indices, the
engram lookback gather) as rows of a parameter table, and ``flush`` issues a
single pointer-indirected kernel covering every staged phase. While a CUDA
graph is recording, or when a step stages a single phase, the flush runs each
phase's own kernel instead.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.ops.metadata import (
    _token_request,
    compute_token_to_req_indices,
)
from vllm.v1.worker.gpu.buffer_utils import _load_ptr

_TRITON_BLOCK_SIZE = 1024

# Parameter-table row layout: phase kind, program count, nine buffer
# addresses, seven scalars. The kernel indexes rows with these offsets.
_ROW_WIDTH = 18
_KIND_TOKEN_MAP = 0
_KIND_COMPRESSED_SLOT = 1
_KIND_NATIVE_LENS = 2
_KIND_RING_SLOT = 3
_KIND_SWA_INDEX = 4
_KIND_LOOKBACK = 5
_KIND_FILL_I64 = 6
_KIND_ZERO_ROWS = 7
_KIND_DUMMY_BATCH = 8


@triton.jit
def _load_scalar(params, offset):
    # The table is int64 because the address columns need 64 bits; every
    # scalar column is a count or size that fits int32, the type the phase
    # bodies were written against.
    return tl.load(params + offset).to(tl.int32)


@triton.jit
def _token_map_body(
    output_ptr,
    query_start_loc_ptr,
    num_reqs,
    num_mapped,
    num_tokens,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0) * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    req = _token_request(query_start_loc_ptr, token, num_reqs)
    mapped = tl.where(token < num_mapped, req, 0)
    tl.store(output_ptr + token, mapped, token < num_tokens)


@triton.jit
def _compressed_slot_body(
    compressed_slot_mapping_ptr,
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    compress_ratio,
    dcp_world_size,
    cp_interleave,
    num_tokens,
    out_len,
    num_rows,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    if batch_idx >= num_rows:
        # Request programs cover [0, num_tokens); graph replays read the
        # buffer's full padded width, so tail programs clear the rest.
        offsets = (
            num_tokens
            + (batch_idx - num_rows) * TRITON_BLOCK_SIZE
            + tl.arange(0, TRITON_BLOCK_SIZE)
        )
        tl.store(
            compressed_slot_mapping_ptr + offsets,
            PAD_ID,
            mask=offsets < out_len,
        )
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        # A replayed token has a PAD slot (SWA bounded replay: its KV is
        # cached already); its compressed KV is cached too, so PAD here.
        slot = tl.load(slot_mapping_ptr + query_start + offset, mask=mask, other=PAD_ID)
        is_valid = ((pos + 1) % compress_ratio == 0) & (slot != PAD_ID)
        # Under DCP a PAD slot also marks tokens owned by other ranks; the
        # interleave is a multiple of compress_ratio, so a state lives
        # entirely on the rank owning its last token.
        local_pos = (
            pos // (cp_interleave * dcp_world_size) * cp_interleave
            + pos % cp_interleave
        )
        pos_after_compress = local_pos // compress_ratio

        block_ids = pos_after_compress // block_size
        # Padded or invalid table entries clamp to block 0, the null
        # block, so a bogus write lands in scratch instead of an illegal
        # slot; the load index itself can be negative for padding rows.
        block_numbers = tl.maximum(
            tl.load(
                block_table_ptr + batch_idx * block_table_stride + block_ids,
                mask=mask & is_valid,
            ),
            0,
        )
        slot_ids = block_numbers.to(tl.int64) * block_size + (
            pos_after_compress % block_size
        )
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(
            compressed_slot_mapping_ptr + query_start + offset, slot_ids, mask=mask
        )


@triton.jit
def _native_decode_lens_body(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    query_start_loc_ptr,
    decode_lens_ptr,
    per_req_decode_lens_ptr,
    num_decodes,
    max_decode_len,
    compress_ratio,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < num_decodes * max_decode_len
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len
    # Clamp at 0: padding requests have seq_len == 0, which would
    # otherwise make token 0 negative (next_n=2 gives 0-2+1+0 = -1).
    # Downstream kernels read these as uint32, turning -1 into ~4e9.
    seq_len = tl.load(seq_lens_ptr + req_id, mask=mask, other=0)
    per_token_seq_len = tl.maximum(seq_len - max_decode_len + local_idx + 1, 0)
    if compress_ratio > 1:
        per_token_seq_len = per_token_seq_len // compress_ratio
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len, mask=mask)

    # The token-0 lane of each request publishes its query length, so
    # build() needs no eager diff or per-request copy. query_start_loc is
    # the [num_decodes] view; the req_id + 1 load stays inside the same
    # [max_num_seqs + 1] allocation.
    req_mask = mask & (local_idx == 0)
    query_start = tl.load(query_start_loc_ptr + req_id, mask=req_mask, other=0)
    query_end = tl.load(query_start_loc_ptr + req_id + 1, mask=req_mask, other=0)
    decode_len = query_end - query_start
    tl.store(decode_lens_ptr + req_id, decode_len, mask=req_mask)
    tl.store(per_req_decode_lens_ptr + req_id, decode_len, mask=req_mask)


@triton.jit(
    do_not_specialize=[
        "block_table_stride",
        "num_reqs",
        "num_mapped_tokens",
        "num_actual_tokens",
        "num_tokens",
        "capacity",
    ]
)
def _ring_slot_body(
    slot_mapping_ptr,
    block_table_ptr,
    query_start_loc_ptr,
    positions_ptr,
    block_table_stride,
    num_reqs,
    num_mapped_tokens,
    num_actual_tokens,
    num_tokens,
    capacity,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_actual_tokens
    # The staged token map is written by another phase of this same launch,
    # so the request index is searched from query_start_loc directly. Tokens
    # at or past num_mapped_tokens map to 0, like the staged mapping does.
    req = _token_request(query_start_loc_ptr, offsets, num_reqs)
    req = tl.where(valid & (offsets < num_mapped_tokens), req, 0).to(tl.int64)
    block = tl.load(block_table_ptr + req * block_table_stride, mask=valid, other=0)
    pos = tl.load(positions_ptr + offsets, mask=valid, other=0)
    slot = block.to(tl.int64) * capacity + pos % capacity
    # Block 0 is the null block. A request without a ring block (dummy or padding
    # rows, which the runners fill with the null block) must not write.
    tl.store(
        slot_mapping_ptr + offsets,
        tl.where(valid & (block != 0), slot, -1),
        mask=offsets < num_tokens,
    )


@triton.jit
def _swa_index_body(
    swa_indices_ptr,
    swa_lens_ptr,
    is_valid_token_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    replay_start_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    swa_indices_stride,
    block_table_stride,
    block_size,
    window_size,
    index_width,
    num_valid_tokens,
    num_reqs,
    num_mapped_tokens,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid
    is_valid = tl.load(slot_mapping_ptr + token_idx) >= 0
    tl.store(is_valid_token_ptr + token_idx, is_valid.to(tl.int8))
    # Same tail bound as the per-group kernel: lens zero to the buffer end,
    # index row-clears only within num_valid_tokens.
    if not is_valid or pid >= num_valid_tokens:
        tl.store(swa_lens_ptr + pid, 0)
        if pid < num_valid_tokens:
            for i in range(0, index_width, TRITON_BLOCK_SIZE):
                offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
                tl.store(
                    swa_indices_ptr + pid * swa_indices_stride + offset,
                    -1,
                    mask=offset < index_width,
                )
        return

    # The staged token map is written by another phase of this same launch,
    # so the request index is searched from query_start_loc directly.
    token_vec = tl.full((1,), token_idx, tl.int32)
    req_idx = tl.sum(_token_request(query_start_loc_ptr, token_vec, num_reqs), 0)
    req_idx = tl.where(token_idx < num_mapped_tokens, req_idx, 0)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - (window_size - 1), 0)
    start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )


@triton.jit
def _lookback_body(
    lookback_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    positions_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    begin = tl.load(query_start_loc_ptr + batch_idx, mask=in_batch, other=0)
    end = tl.load(query_start_loc_ptr + batch_idx + 1, mask=in_batch, other=0)
    in_batch = in_batch & (end > begin)
    num_computed = tl.load(positions_ptr + begin, mask=in_batch, other=0)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx.to(tl.int64) * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


@triton.jit
def _fill_i64_body(output_ptr, value, num_elements, TRITON_BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)
    tl.store(output_ptr + offs, value.to(tl.int64), mask=offs < num_elements)


@triton.jit
def _zero_rows_body(block_table_ptr, stride, TRITON_BLOCK_SIZE: tl.constexpr):
    # pid is the block-table row; per-group column counts differ, hence the
    # stride loop rather than a grid over columns.
    row_ptr = block_table_ptr + tl.program_id(0).to(tl.int64) * stride
    for i in tl.range(0, stride, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        tl.store(row_ptr + offset, 0, mask=offset < stride)


@triton.jit
def _dummy_batch_body(
    idx_mapping_ptr,  # [num_reqs] int64
    expanded_local_pos_ptr,  # [num_reqs] int32
    cu_num_logits_ptr,  # [num_reqs + 1] int32
    logits_indices_ptr,  # [num_reqs] int32
    seq_lens_ptr,  # [max_num_reqs] int32
    query_start_loc_ptr,  # [max_num_reqs + 1] int32
    input_ids_ptr,  # [num_tokens] int32
    positions_ptr,  # [num_tokens] int64
    is_padding_ptr,  # [num_tokens] int8
    num_reqs,
    num_tokens,
    base_tokens,
    num_extra,
    max_num_reqs,
    is_padding,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)

    is_req = offs < num_reqs
    zeros_i32 = tl.zeros([TRITON_BLOCK_SIZE], tl.int32)
    tl.store(idx_mapping_ptr + offs, offs.to(tl.int64), mask=is_req)
    tl.store(expanded_local_pos_ptr + offs, zeros_i32, mask=is_req)

    # seq_len equals query_len; the last num_extra requests carry the remainder.
    seq_len = base_tokens + tl.where(offs >= num_reqs - num_extra, 1, 0)
    tl.store(
        seq_lens_ptr + offs,
        tl.where(is_req, seq_len, 0).to(tl.int32),
        mask=offs < max_num_reqs,
    )

    # query_start_loc[k] = cumsum(seq_lens)[k] for k <= num_reqs, and the
    # full-graph padding value num_tokens beyond that.
    qsl = offs * base_tokens + tl.maximum(offs - (num_reqs - num_extra), 0)
    qsl = tl.where(offs <= num_reqs, qsl, num_tokens)
    tl.store(query_start_loc_ptr + offs, qsl.to(tl.int32), mask=offs < max_num_reqs + 1)

    tl.store(cu_num_logits_ptr + offs, offs.to(tl.int32), mask=offs <= num_reqs)
    next_qsl = (offs + 1) * base_tokens + tl.maximum(
        offs + 1 - (num_reqs - num_extra), 0
    )
    tl.store(logits_indices_ptr + offs, (next_qsl - 1).to(tl.int32), mask=is_req)

    is_tok = offs < num_tokens
    tl.store(input_ids_ptr + offs, zeros_i32, mask=is_tok)
    tl.store(positions_ptr + offs, tl.zeros([TRITON_BLOCK_SIZE], tl.int64), mask=is_tok)
    tl.store(is_padding_ptr + offs, (is_padding != 0).to(tl.int8), mask=is_tok)


@triton.jit(
    do_not_specialize=[
        "num_reqs_padded",
        "num_mapped_tokens",
        "num_output_tokens",
        "num_actual_tokens",
        "num_decodes",
        "max_decode_len",
    ]
)
def _decode_prologue_kernel(
    phase_params_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    positions_ptr,
    num_reqs_padded,
    num_mapped_tokens,
    num_output_tokens,
    num_actual_tokens,
    num_decodes,
    max_decode_len,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Decode prologue phases of one step over an int64 [num_rows, 18] table.

    Each row names a phase kind, its program count, nine buffer addresses
    and seven scalars; program (pid, row) runs the row's phase when pid is
    below its program count. Rows whose phases read the per-token request
    mapping inline the query_start_loc search instead: another row of this
    same launch writes that mapping.
    """
    pid = tl.program_id(0)
    params = phase_params_ptr + tl.program_id(1) * 18
    if pid >= tl.load(params + 1):
        return
    kind = tl.load(params + 0)
    if kind == 0:
        _token_map_body(
            _load_ptr(params + 2, tl.int32),
            query_start_loc_ptr,
            num_reqs_padded,
            num_mapped_tokens,
            num_output_tokens,
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
    elif kind == 1:
        _compressed_slot_body(
            _load_ptr(params + 2, tl.int64),
            _load_ptr(params + 3, tl.int64),
            query_start_loc_ptr,
            seq_lens_ptr,
            _load_ptr(params + 4, tl.int32),
            _load_scalar(params, 11),
            _load_scalar(params, 12),
            _load_scalar(params, 13),
            _load_scalar(params, 14),
            _load_scalar(params, 15),
            num_actual_tokens,
            _load_scalar(params, 16),
            _load_scalar(params, 17),
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
    elif kind == 2:
        _native_decode_lens_body(
            seq_lens_ptr,
            _load_ptr(params + 2, tl.int32),
            query_start_loc_ptr,
            _load_ptr(params + 3, tl.int32),
            _load_ptr(params + 4, tl.int32),
            num_decodes,
            max_decode_len,
            _load_scalar(params, 11),
            BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
    elif kind == 3:
        _ring_slot_body(
            _load_ptr(params + 2, tl.int64),
            _load_ptr(params + 3, tl.int32),
            query_start_loc_ptr,
            positions_ptr,
            _load_scalar(params, 11),
            num_reqs_padded,
            num_mapped_tokens,
            num_actual_tokens,
            _load_scalar(params, 13),
            _load_scalar(params, 12),
            BLOCK=TRITON_BLOCK_SIZE,
        )
    elif kind == 4:
        _swa_index_body(
            _load_ptr(params + 2, tl.int32),
            _load_ptr(params + 3, tl.int32),
            _load_ptr(params + 4, tl.int8),
            _load_ptr(params + 5, tl.int64),
            _load_ptr(params + 6, tl.int32),
            _load_ptr(params + 7, tl.int32),
            query_start_loc_ptr,
            seq_lens_ptr,
            _load_scalar(params, 11),
            _load_scalar(params, 12),
            _load_scalar(params, 13),
            _load_scalar(params, 14),
            _load_scalar(params, 15),
            _load_scalar(params, 16),
            num_reqs_padded,
            num_mapped_tokens,
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
    elif kind == 5:
        _lookback_body(
            _load_ptr(params + 2, tl.int32),
            _load_ptr(params + 3, tl.int64),
            query_start_loc_ptr,
            positions_ptr,
            _load_ptr(params + 4, tl.int32),
            _load_scalar(params, 11),
            _load_scalar(params, 12),
            DEPTH=DEPTH,
            BLOCK_DEPTH=BLOCK_DEPTH,
        )
    elif kind == 6:
        _fill_i64_body(
            _load_ptr(params + 2, tl.int64),
            _load_scalar(params, 11),
            _load_scalar(params, 12),
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
    elif kind == 7:
        _zero_rows_body(
            _load_ptr(params + 2, tl.int32),
            _load_scalar(params, 11),
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )
    else:
        _dummy_batch_body(
            _load_ptr(params + 2, tl.int64),
            _load_ptr(params + 3, tl.int32),
            _load_ptr(params + 4, tl.int32),
            _load_ptr(params + 5, tl.int32),
            seq_lens_ptr,
            query_start_loc_ptr,
            _load_ptr(params + 6, tl.int32),
            positions_ptr,
            _load_ptr(params + 7, tl.int8),
            _load_scalar(params, 11),
            _load_scalar(params, 12),
            _load_scalar(params, 13),
            _load_scalar(params, 14),
            _load_scalar(params, 15),
            _load_scalar(params, 16),
            TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
        )


class DecodePrologueKernel(VllmTritonJitKernel["DecodePrologueKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        lookback_depth: int

    kernel = staticmethod(_decode_prologue_kernel)

    def dispatch(  # type: ignore[override]
        self,
        *,
        lookback_depth: int,
    ) -> CompileKey:
        return self.CompileKey(lookback_depth=lookback_depth)

    def get_warmup_keys(self, *, lookback_depth: int) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(lookback_depth=lookback_depth)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            phase_params=TritonWarmupTensor(
                torch.int64, shape=(1, _ROW_WIDTH), strides=(_ROW_WIDTH, 1)
            ),
            query_start_loc=int32_ptr,
            seq_lens=int32_ptr,
            positions=TritonWarmupTensor(torch.int64),
            max_blocks=1,
            num_rows=1,
            num_reqs_padded=1,
            num_mapped_tokens=1,
            num_output_tokens=1,
            num_actual_tokens=1,
            num_decodes=1,
            max_decode_len=1,
            lookback_depth=compile_key.lookback_depth,
        )

    @kernel_launcher
    def __call__(
        self,
        phase_params: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        positions: torch.Tensor,
        *,
        max_blocks: int,
        num_rows: int,
        num_reqs_padded: int,
        num_mapped_tokens: int,
        num_output_tokens: int,
        num_actual_tokens: int,
        num_decodes: int,
        max_decode_len: int,
        lookback_depth: int,
    ) -> LaunchSpec:
        return (max_blocks, num_rows), dict(
            DEPTH=lookback_depth,
            BLOCK_DEPTH=triton.next_power_of_2(lookback_depth),
            TRITON_BLOCK_SIZE=_TRITON_BLOCK_SIZE,
        )


_DECODE_PROLOGUE_KERNEL = DecodePrologueKernel()


@dataclass
class _StagedPhase:
    params: np.ndarray
    fallback: Callable[[], Any]


@dataclass
class _DeferredSchedule:
    context_lens: torch.Tensor
    block_size: int
    num_sms: int
    indices: torch.Tensor | None
    out: torch.Tensor


class DecodePrologueBatch:
    """Stages one step's decode prologue phases for one launch.

    The Deepseek V4.1 model state brackets the step's attention-metadata
    build with :meth:`begin` and :meth:`flush`: each armed builder's decode
    build records its buffers instead of launching, and the flush issues a
    single kernel covering every staged phase. While a CUDA graph is
    recording, the flush launches each phase's own kernel instead, keeping
    captured builds replay-stable.
    """

    def __init__(self, device: torch.device | None = None):
        self._device = device
        self._rows: list[_StagedPhase] = []
        self._deferred: list[_DeferredSchedule] = []
        self._open = False
        self._decode_only = False
        self._capturing = False
        self._tensors: dict[str, torch.Tensor] = {}
        self._ints: dict[str, int] = {}
        self._depth = 1
        # Staging for the launch's parameter table, sized to the largest
        # phase count seen. The host side must be pinned: a pageable H2D
        # copy serializes on the stream through a driver staging buffer
        # (measured as a uniform per-step stall), while a pinned copy is a
        # true async DMA. Refilling a pinned source is only safe once the
        # previous copy retired, so an event guards the refill; while it is
        # pending the launch falls back to a fresh table, whose lifetime the
        # caching allocator already guards. Unlike the SWA batch's table,
        # every row changes per step, so the copy runs per flush.
        self._params_host: torch.Tensor | None = None
        self._params_host_rows: np.ndarray | None = None
        self._params_device: torch.Tensor | None = None
        self._params_copied: torch.cuda.Event | None = None

    @property
    def open(self) -> bool:
        return self._open

    def begin(self, *, decode_only: bool) -> None:
        assert not self._open, (
            "a decode prologue batch began before the previous one flushed"
        )
        self._rows = []
        self._deferred = []
        self._tensors = {}
        self._ints = {}
        self._depth = 1
        self._decode_only = decode_only
        # Capture state cannot change inside a bracket; asking the driver
        # once here keeps every per-stage guard off it.
        self._capturing = (
            self._device is not None
            and self._device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        )
        self._open = True

    def _fuse_ok(self) -> bool:
        # The open check short-circuits before the device checks, so the
        # unarmed singleton never touches CUDA.
        return (
            self._open
            and self._decode_only
            and self._device is not None
            and self._device.type == "cuda"
            and not self._capturing
        )

    def _note_shared_tensor(self, name: str, tensor: torch.Tensor) -> None:
        was = self._tensors.get(name)
        if was is not None and (
            was.data_ptr() != tensor.data_ptr() or was.dtype != tensor.dtype
        ):
            raise ValueError(
                f"decode prologue phases disagree on {name}; batched builds "
                "require one shared buffer"
            )
        self._tensors[name] = tensor

    def _note_shared_int(self, name: str, value: int) -> None:
        was = self._ints.get(name)
        if was is not None and was != value:
            raise ValueError(
                f"decode prologue phases disagree on {name} ({was} vs {value}); "
                "batched builds require one shared value"
            )
        self._ints[name] = value

    def _add_row(
        self,
        kind: int,
        blocks: int,
        ptrs: list[torch.Tensor],
        scalars: list[int],
        fallback: Callable[[], Any],
    ) -> None:
        assert len(ptrs) <= 9 and len(scalars) <= 7
        params = np.zeros(_ROW_WIDTH, dtype=np.int64)
        params[0] = kind
        params[1] = blocks
        for i, ptr in enumerate(ptrs):
            params[2 + i] = ptr.data_ptr()
        for i, scalar in enumerate(scalars):
            params[11 + i] = scalar
        self._rows.append(_StagedPhase(params, fallback))

    def token_to_req_indices(
        self, cm: CommonAttentionMetadata, buffer: torch.Tensor
    ) -> torch.Tensor:
        """Build or reuse the per-token request mapping, staged when fusing.

        Mirrors ``CommonAttentionMetadata.token_to_req_indices`` including
        its cache threading, so only the step's first builder call builds.
        """
        num_tokens = cm.num_actual_tokens
        cache = cm._token_to_req_indices_cache
        if cache is not None:
            assert cache.device == buffer.device
            assert cache.dtype == torch.int32
            assert cache.shape[0] >= num_tokens
            return cache[:num_tokens]
        if not self._fuse_ok():
            return cm.token_to_req_indices(buffer)

        num_mapped_tokens = int(cm.query_start_loc_cpu[-1])
        num_output_tokens = max(num_mapped_tokens, num_tokens)
        assert buffer.dtype == torch.int32
        assert buffer.shape[0] >= num_output_tokens

        def fallback() -> torch.Tensor:
            return compute_token_to_req_indices(
                cm.query_start_loc, buffer, num_mapped_tokens, num_output_tokens
            )

        self._note_shared_tensor("query_start_loc", cm.query_start_loc)
        self._note_shared_int("num_reqs_padded", cm.query_start_loc.shape[0] - 1)
        self._note_shared_int("num_mapped_tokens", num_mapped_tokens)
        self._note_shared_int("num_output_tokens", num_output_tokens)
        self._add_row(
            _KIND_TOKEN_MAP,
            cdiv(num_output_tokens, _TRITON_BLOCK_SIZE),
            [buffer],
            [],
            fallback,
        )
        cm._token_to_req_indices_cache = buffer[:num_output_tokens]
        return buffer[:num_tokens]

    def stage_compressed_slot_mapping(
        self,
        *,
        fuse: bool,
        num_tokens: int,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
        out: torch.Tensor,
        dcp_world_size: int = 1,
        cp_interleave: int = 1,
    ) -> torch.Tensor:
        """Stage the compressed slot mapping build; returns ``out[:num_tokens]``."""

        def fallback() -> torch.Tensor:
            from vllm.v1.attention.backends.mla.compressor_utils import (
                get_compressed_slot_mapping,
            )

            return get_compressed_slot_mapping(
                num_tokens,
                slot_mapping,
                query_start_loc,
                seq_lens,
                block_table,
                block_size,
                compress_ratio,
                out=out,
                dcp_world_size=dcp_world_size,
                cp_interleave=cp_interleave,
            )

        if not (fuse and self._fuse_ok()):
            return fallback()

        num_rows = block_table.shape[0]
        out_len = out.numel()
        self._note_shared_tensor("query_start_loc", query_start_loc)
        self._note_shared_tensor("seq_lens", seq_lens)
        self._note_shared_int("num_actual_tokens", num_tokens)
        self._add_row(
            _KIND_COMPRESSED_SLOT,
            num_rows + cdiv(max(out_len - num_tokens, 0), _TRITON_BLOCK_SIZE),
            [out, slot_mapping, block_table],
            [
                block_table.stride(0),
                block_size,
                compress_ratio,
                dcp_world_size,
                cp_interleave,
                out_len,
                num_rows,
            ],
            fallback,
        )
        return out[:num_tokens]

    def stage_native_decode_lens(
        self,
        *,
        fuse: bool,
        seq_lens: torch.Tensor,
        decode_seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        decode_lens: torch.Tensor,
        per_req_decode_lens: torch.Tensor,
        num_decodes: int,
        max_decode_len: int,
        compress_ratio: int,
    ) -> None:
        """Stage the native per-token decode lens build."""

        def fallback() -> None:
            from vllm.v1.attention.backends.mla.indexer import (
                _PREPARE_NATIVE_DECODE_LENS_KERNEL,
            )

            _PREPARE_NATIVE_DECODE_LENS_KERNEL(
                seq_lens,
                decode_seq_lens,
                query_start_loc,
                decode_lens,
                per_req_decode_lens,
                num_decodes,
                max_decode_len,
                compress_ratio,
            )

        if not (fuse and self._fuse_ok()):
            fallback()
            return
        self._note_shared_tensor("query_start_loc", query_start_loc)
        self._note_shared_tensor("seq_lens", seq_lens)
        self._note_shared_int("num_decodes", num_decodes)
        self._note_shared_int("max_decode_len", max_decode_len)
        self._add_row(
            _KIND_NATIVE_LENS,
            cdiv(num_decodes * max_decode_len, _TRITON_BLOCK_SIZE),
            [decode_seq_lens, decode_lens, per_req_decode_lens],
            [compress_ratio],
            fallback,
        )

    def stage_ring_slot_mapping(
        self,
        *,
        fuse: bool,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        num_mapped_tokens: int,
        num_actual_tokens: int,
        num_tokens: int,
        capacity: int,
    ) -> None:
        """Stage the compressor ring slot mapping build."""

        def fallback() -> None:
            _ring_slot_body[(cdiv(num_tokens, 256),)](
                slot_mapping,
                block_table,
                query_start_loc,
                positions,
                block_table.stride(0),
                num_reqs,
                num_mapped_tokens,
                num_actual_tokens,
                num_tokens,
                capacity,
                BLOCK=256,
            )

        if not (fuse and self._fuse_ok()):
            fallback()
            return
        self._note_shared_tensor("query_start_loc", query_start_loc)
        self._note_shared_tensor("positions", positions)
        self._note_shared_int("num_reqs_padded", num_reqs)
        self._note_shared_int("num_mapped_tokens", num_mapped_tokens)
        self._note_shared_int("num_actual_tokens", num_actual_tokens)
        self._add_row(
            _KIND_RING_SLOT,
            cdiv(num_tokens, _TRITON_BLOCK_SIZE),
            [slot_mapping, block_table],
            [block_table.stride(0), capacity, num_tokens],
            fallback,
        )

    def stage_swa_indices(
        self,
        *,
        fuse: bool,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        is_valid_token: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        replay_start: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        window_size: int,
        block_size: int,
        num_tokens: int,
        num_valid_tokens: int,
        num_mapped_tokens: int,
    ) -> None:
        """Stage one group's causal decode SWA index build; shared arguments
        must be identical across the groups of a step."""

        def fallback() -> None:
            from vllm.v1.attention.backends.mla.sparse_swa import (
                _launch_causal_decode_swa_indices,
            )

            _launch_causal_decode_swa_indices(
                swa_indices,
                swa_lens,
                window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                slot_mapping,
                is_valid_token,
                block_table,
                block_size,
                replay_start,
                num_tokens,
                num_valid_tokens,
            )

        if not (fuse and self._fuse_ok()):
            fallback()
            return
        self._note_shared_tensor("query_start_loc", query_start_loc)
        self._note_shared_tensor("seq_lens", seq_lens)
        self._note_shared_int("num_reqs_padded", query_start_loc.shape[0] - 1)
        self._note_shared_int("num_mapped_tokens", num_mapped_tokens)
        self._add_row(
            _KIND_SWA_INDEX,
            num_tokens,
            [
                swa_indices,
                swa_lens,
                is_valid_token,
                slot_mapping,
                block_table,
                replay_start,
            ],
            [
                swa_indices.stride(0),
                block_table.stride(0),
                block_size,
                window_size,
                swa_indices.shape[-1],
                num_valid_tokens,
            ],
            fallback,
        )

    def stage_lookback(
        self,
        *,
        window: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        all_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Stage the engram lookback gather."""
        depth = window.shape[1]

        def fallback() -> None:
            _lookback_body[(window.shape[0],)](
                window,
                idx_mapping,
                query_start_loc,
                positions,
                all_token_ids,
                all_token_ids.stride(0),
                num_reqs,
                DEPTH=depth,
                BLOCK_DEPTH=triton.next_power_of_2(depth),
            )

        if not self._fuse_ok():
            fallback()
            return
        self._depth = depth
        self._note_shared_tensor("query_start_loc", query_start_loc)
        self._note_shared_tensor("positions", positions)
        self._add_row(
            _KIND_LOOKBACK,
            window.shape[0],
            [window, idx_mapping, all_token_ids],
            [all_token_ids.stride(0), num_reqs],
            fallback,
        )

    def stage_fill_i64(self, *, fuse: bool, tensor: torch.Tensor, value: int) -> None:
        """Stage an int64 tensor fill; the direct path fills eagerly."""

        def fallback() -> None:
            tensor.fill_(value)

        if not (fuse and self._fuse_ok()):
            fallback()
            return
        numel = tensor.numel()
        self._add_row(
            _KIND_FILL_I64,
            cdiv(numel, _TRITON_BLOCK_SIZE),
            [tensor],
            [value, numel],
            fallback,
        )

    def stage_zero_block_table_rows(
        self,
        *,
        fuse: bool,
        block_tables: Sequence[torch.Tensor],
        strides: Sequence[int],
        num_rows: int,
        fallback: Callable[[], None],
    ) -> None:
        """Stage the dummy block-table row zeroing, one row per KV group."""
        if not (fuse and self._fuse_ok()):
            fallback()
            return
        for block_table, stride in zip(block_tables, strides):
            self._add_row(
                _KIND_ZERO_ROWS,
                num_rows,
                [block_table],
                [stride],
                fallback,
            )

    def stage_dummy_batch(
        self,
        *,
        fuse: bool,
        idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        logits_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_padding: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        base_tokens: int,
        num_extra: int,
        max_num_reqs: int,
        is_padding_value: int,
    ) -> None:
        """Stage the dummy-batch buffer build.

        The staged row writes the step's shared buffers (query_start_loc,
        seq_lens, positions), so its launch must not also carry rows that
        read them: the dummy bracket flushes before the attention phases
        batch their own launch.
        """

        def fallback() -> None:
            from vllm.v1.worker.gpu.input_batch import _init_dummy_batch_kernel

            _init_dummy_batch_kernel[(cdiv(max(max_num_reqs + 1, num_tokens), 1024),)](
                idx_mapping,
                expanded_local_pos,
                cu_num_logits,
                logits_indices,
                seq_lens,
                query_start_loc,
                input_ids,
                positions,
                is_padding,
                num_reqs,
                num_tokens,
                base_tokens,
                num_extra,
                max_num_reqs,
                is_padding_value,
                BLOCK_SIZE=1024,  # type: ignore
            )

        if not (fuse and self._fuse_ok()):
            fallback()
            return
        self._note_shared_tensor("query_start_loc", query_start_loc)
        self._note_shared_tensor("seq_lens", seq_lens)
        self._note_shared_tensor("positions", positions)
        self._add_row(
            _KIND_DUMMY_BATCH,
            cdiv(max(max_num_reqs + 1, num_tokens), _TRITON_BLOCK_SIZE),
            [
                idx_mapping,
                expanded_local_pos,
                cu_num_logits,
                logits_indices,
                input_ids,
                is_padding,
            ],
            [
                num_reqs,
                num_tokens,
                base_tokens,
                num_extra,
                max_num_reqs,
                is_padding_value,
            ],
            fallback,
        )

    def defer_paged_mqa_schedule(
        self,
        *,
        fuse: bool,
        context_lens: torch.Tensor,
        block_size: int,
        num_sms: int,
        indices: torch.Tensor | None,
        out: torch.Tensor,
    ) -> bool:
        """Stage the deep_gemm paged MQA schedule for after the batch's launch.

        ``get_paged_mqa_logits_metadata`` allocates, so its result is copied
        into the caller's persistent buffer either way; deferring only moves
        that copy behind the batch's launch. Returns whether the schedule
        was deferred; the caller runs it directly otherwise.
        """
        if not (fuse and self._fuse_ok()):
            return False
        self._deferred.append(
            _DeferredSchedule(context_lens, block_size, num_sms, indices, out)
        )
        return True

    def flush(self) -> None:
        try:
            if not self._rows and not self._deferred:
                return
            if len(self._rows) <= 1 or self._device is None or self._capturing:
                # A recording must capture each phase's own launch; the
                # megakernel reads a parameter table this method rebuilds per
                # step, which a replay would not refresh. A single phase pays
                # the table upload to fuse nothing: one launch either way, so
                # it takes the direct path too.
                for row in self._rows:
                    row.fallback()
            else:
                self._launch_batched()
            self._commit_deferred_schedules()
        finally:
            self._open = False
            self._rows = []
            self._deferred = []

    def _launch_batched(self) -> None:
        num_rows = len(self._rows)
        rows_np = np.stack([row.params for row in self._rows])
        max_blocks = int(rows_np[:, 1].max())
        if (
            self._params_host is None
            or self._params_host_rows is None
            or self._params_device is None
            or self._params_host.shape[0] < num_rows
            or (self._params_copied is not None and not self._params_copied.query())
        ):
            self._params_host = torch.empty(
                num_rows, _ROW_WIDTH, dtype=torch.int64, pin_memory=True
            )
            self._params_host_rows = self._params_host.numpy()
            self._params_device = torch.empty(
                num_rows, _ROW_WIDTH, dtype=torch.int64, device=self._device
            )
            self._params_copied = None
        self._params_host_rows[:num_rows] = rows_np
        self._params_device[:num_rows].copy_(
            self._params_host[:num_rows], non_blocking=True
        )
        if self._params_copied is None:
            self._params_copied = torch.cuda.Event()
        self._params_copied.record()
        _DECODE_PROLOGUE_KERNEL(
            self._params_device[:num_rows],
            self._shared_tensor("query_start_loc", torch.int32),
            self._shared_tensor("seq_lens", torch.int32),
            self._shared_tensor("positions", torch.int64),
            max_blocks=max_blocks,
            num_rows=num_rows,
            num_reqs_padded=self._ints.get("num_reqs_padded", 0),
            num_mapped_tokens=self._ints.get("num_mapped_tokens", 0),
            num_output_tokens=self._ints.get("num_output_tokens", 0),
            num_actual_tokens=self._ints.get("num_actual_tokens", 0),
            num_decodes=self._ints.get("num_decodes", 0),
            max_decode_len=self._ints.get("max_decode_len", 0),
            lookback_depth=self._depth,
        )

    def _shared_tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        tensor = self._tensors.get(name)
        if tensor is None:
            # No staged phase reads it; the launch signature still needs one.
            tensor = torch.empty(1, dtype=dtype, device=self._device)
            self._tensors[name] = tensor
        return tensor

    def _commit_deferred_schedules(self) -> None:
        # Steps that stage phases without deferring a schedule reach flush
        # with an empty list, and _foreach_copy_ rejects empty lists.
        if not self._deferred:
            return
        sources = [
            get_paged_mqa_logits_metadata(
                schedule.context_lens,
                schedule.block_size,
                schedule.num_sms,
                indices=schedule.indices,
            )
            for schedule in self._deferred
        ]
        targets = [schedule.out for schedule in self._deferred]
        if len(targets) == 1:
            targets[0].copy_(sources[0])
        else:
            torch._foreach_copy_(targets, sources)


# Builders hold this until a model state arms them with the step's live
# batch; its stages always take the direct path.
UNARMED_PROLOGUE_BATCH = DecodePrologueBatch()
