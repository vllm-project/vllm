# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve an nvfp4_ds_mla KV cache through the FlashInfer sparse MLA kernel.

FlashInfer's SM100 sparse MLA kernel (TRTLLM-gen) reads a dense
``[num_pages, page_size, 576]`` cache of a single dtype; for fp8 it is e4m3
with one per-tensor scale. The nvfp4_ds_mla cache instead stores every token
as a self-describing 352 B record (see flashmla_sparse.py): 256 B of packed
e2m1 NoPE, 64 B of unscaled e4m3 RoPE and 32 B of permuted e4m3 block scales.

Following TensorRT-LLM's NVFP4 DSA path, the rows a batch attends to are
dequantized into an FP8 staging buffer laid out the way the FlashInfer kernel
reads it, and the stock FP8 kernel then runs on that buffer:

- decode tokens gather their own top-k rows into a private ``[topk, 576]``
  slice (an index-driven gather, so the traffic is bounded by top-k);
- prefill tokens share a per-request context gather: each prefill request's
  whole context is dequantized once into a workspace, and the tokens' top-k
  indices are remapped to workspace rows (as the FlashMLA prefill path does
  with its BF16 upconvert, at half the bytes).

The staged values are ``dequant(x) / k_scale`` saturated to e4m3, i.e. what
the fp8 cache would have stored for the same k_scale, so the fp8 bmm scales
apply unchanged.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import PIN_MEMORY, np_to_pinned_tensor
from vllm.v1.attention.backends.utils import split_prefill_chunks

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata

# nvfp4_ds_mla record, 352 B/token (keep in sync with
# csrc/libtorch_stable/nvfp4_ds_mla_cache_kernels.cu):
#   [0,   256)  512 x e2m1 NoPE, packed 2/byte (low nibble = even element)
#   [256, 320)  64  x e4m3 RoPE, unscaled
#   [320, 352)  32  x e4m3 NoPE scales; element block s at byte
#               320 + 8 * (s & 3) + (s >> 2)
NVFP4_DS_MLA_ROW_BYTES = 352
NVFP4_DS_MLA_KV_LORA_RANK = 512
NVFP4_DS_MLA_ROPE_DIM = 64
# Width of a staged FP8 row, as the FlashInfer kernel reads it.
FP8_STAGING_ROW_DIM = NVFP4_DS_MLA_KV_LORA_RANK + NVFP4_DS_MLA_ROPE_DIM
# Tokens per page of the staging views handed to the FlashInfer kernel. The
# sparse indices address flat rows, so this only has to be a supported size.
FP8_STAGING_PAGE_SIZE = 64
# Rows per program of the gather kernels.
_GATHER_BLOCK_ROWS = 4
# Decode tokens staged per pass (2048-wide top-k: ~300 MB of FP8 rows); larger
# decode batches loop over the staging buffer.
_MAX_DECODE_STAGING_TOKENS = 256


def nvfp4_fp8_prefill_workspace_rows(max_model_len: int) -> int:
    """Context rows the prefill workspace holds (FlashMLA's prefill sizing).

    At 576 B/row this is half the bytes of FlashMLA's BF16 prefill workspace.
    """
    return round_up(5 * max_model_len, FP8_STAGING_PAGE_SIZE)


def nvfp4_fp8_max_decode_tokens(vllm_config: "VllmConfig") -> int:
    """Decode tokens one staging pass holds; larger batches loop over it."""
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    tokens_per_req = 1
    if (
        speculative_config is not None
        and speculative_config.num_speculative_tokens is not None
    ):
        tokens_per_req += speculative_config.num_speculative_tokens
    return max(
        1,
        min(
            scheduler_config.max_num_batched_tokens,
            scheduler_config.max_num_seqs * tokens_per_req,
            _MAX_DECODE_STAGING_TOKENS,
        ),
    )


def nvfp4_fp8_context_search_steps(max_num_reqs: int) -> int:
    """Bisection steps resolving up to ``max_num_reqs`` requests.

    Fixed per engine so the context-gather kernel compiles once; extra steps
    are no-ops once the range has collapsed.
    """
    return max(1, (max_num_reqs - 1).bit_length())


def nvfp4_fp8_gather_unsupported_reason(vllm_config: "VllmConfig") -> str | None:
    """Why the FlashInfer sparse MLA backend cannot serve nvfp4_ds_mla here."""
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.decode_context_parallel_size > 1
        or parallel_config.prefill_context_parallel_size > 1
    ):
        return (
            "FLASHINFER_MLA_SPARSE does not support the nvfp4_ds_mla kv-cache "
            "dtype with context parallelism"
        )
    if vllm_config.attention_config.hisparse_config is not None:
        return (
            "FLASHINFER_MLA_SPARSE does not support the nvfp4_ds_mla kv-cache "
            "dtype with HiSparse"
        )
    model_config = vllm_config.model_config
    if model_config is not None:
        hf_text_config = model_config.hf_text_config
        if (
            getattr(hf_text_config, "kv_lora_rank", None) != NVFP4_DS_MLA_KV_LORA_RANK
            or getattr(hf_text_config, "qk_rope_head_dim", None)
            != NVFP4_DS_MLA_ROPE_DIM
        ):
            return (
                "The nvfp4_ds_mla kv-cache dtype requires kv_lora_rank="
                f"{NVFP4_DS_MLA_KV_LORA_RANK} and qk_rope_head_dim="
                f"{NVFP4_DS_MLA_ROPE_DIM}"
            )
    return None


@triton.jit
def _e2m1_to_f32(nib):
    # e2m1: bit 3 is the sign, bits 2..1 the exponent and bit 0 the mantissa,
    # giving magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}: 0.5 * man when the
    # exponent is 0, else (1 + man / 2) * 2^(exp - 1).
    mag = nib & 7
    exp = mag >> 1
    man = (mag & 1).to(tl.float32)
    pow2 = tl.where(exp == 1, 0.5, tl.where(exp == 2, 1.0, 2.0))
    val = tl.where(exp == 0, 0.5 * man, (2.0 + man) * pow2)
    return tl.where((nib & 8) != 0, -val, val)


@triton.jit
def _dequant_rows_to_fp8(
    src_row_ptr,  # [BLOCK_ROWS] uint8 pointers to 352 B nvfp4_ds_mla records
    dst_row_ptr,  # [BLOCK_ROWS] e4m3 pointers to 576-wide staging rows
    row_mask,  # [BLOCK_ROWS] rows to convert
    inv_scale,  # 1 / k_scale of the fp8 kernel
    UNIT_SCALE: tl.constexpr,
):
    mask = row_mask[:, None]

    # NoPE: output column c is nibble (c & 1) of byte c // 2, scaled by the
    # e4m3 factor of its 16-element block, stored at the permuted scale byte.
    cols = tl.arange(0, 512)
    packed = tl.load(src_row_ptr[:, None] + (cols // 2)[None, :], mask=mask, other=0)
    nib = (packed.to(tl.int32) >> ((cols & 1) * 4)[None, :]) & 15
    blk = cols // 16
    sf_byte = 320 + 8 * (blk & 3) + (blk >> 2)
    sf = tl.load(src_row_ptr[:, None] + sf_byte[None, :], mask=mask, other=0)
    # The e2m1 x e4m3 product is exact in fp32; the e4m3 store is the only
    # rounding step.
    val = _e2m1_to_f32(nib) * sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    if not UNIT_SCALE:
        val = val * inv_scale
    val = tl.minimum(tl.maximum(val, -448.0), 448.0)
    tl.store(dst_row_ptr[:, None] + cols[None, :], val.to(tl.float8e4nv), mask=mask)

    # RoPE is already e4m3; with a unit k_scale it is copied bit for bit.
    rope_cols = tl.arange(0, 64)
    rope = tl.load(
        src_row_ptr[:, None] + (256 + rope_cols)[None, :], mask=mask, other=0
    )
    if UNIT_SCALE:
        rope_fp8 = rope.to(tl.float8e4nv, bitcast=True)
    else:
        rope_val = rope.to(tl.float8e4nv, bitcast=True).to(tl.float32) * inv_scale
        rope_val = tl.minimum(tl.maximum(rope_val, -448.0), 448.0)
        rope_fp8 = rope_val.to(tl.float8e4nv)
    tl.store(dst_row_ptr[:, None] + (512 + rope_cols)[None, :], rope_fp8, mask=mask)


@triton.jit(do_not_specialize=["topk_stride", "out_idx_stride"])
def _gather_topk_rows_kernel(
    src_ptr,  # uint8 flat rows of the nvfp4_ds_mla cache
    topk_ptr,  # int32 [num_tokens, TOPK] physical rows, -1 = invalid
    topk_stride,
    dst_ptr,  # e4m3 [num_tokens * TOPK, 576]
    out_idx_ptr,  # int32 [num_tokens, TOPK] staging rows, -1 = invalid
    out_idx_stride,
    inv_scale,
    TOPK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    UNIT_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    in_range = cols < TOPK
    idx = tl.load(topk_ptr + tok * topk_stride + cols, mask=in_range, other=-1)
    valid = in_range & (idx >= 0)
    # Each token owns staging rows [tok * TOPK, (tok + 1) * TOPK); column j
    # keeps its position, so the valid-first order and counts carry over.
    dst_row = tok * TOPK + cols
    tl.store(
        out_idx_ptr + tok * out_idx_stride + cols,
        tl.where(valid, dst_row, -1).to(tl.int32),
        mask=in_range,
    )
    _dequant_rows_to_fp8(
        src_ptr + idx.to(tl.int64) * 352,
        dst_ptr + dst_row * 576,
        valid,
        inv_scale,
        UNIT_SCALE,
    )


@triton.jit(
    do_not_specialize=[
        "block_table_stride",
        "num_reqs",
        "total_rows",
        "cache_block_stride",
    ]
)
def _gather_context_rows_kernel(
    src_ptr,  # uint8 nvfp4_ds_mla cache [num_blocks, BLOCK_SIZE, 352]
    block_table_ptr,  # int32 [num_reqs, max_blocks]
    block_table_stride,
    workspace_starts_ptr,  # int32 [num_reqs], non-decreasing, first == 0
    num_reqs,
    total_rows,
    cache_block_stride,  # bytes between consecutive cache blocks
    dst_ptr,  # e4m3 [>= total_rows, 576]
    inv_scale,
    BLOCK_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    UNIT_SCALE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row_mask = rows < total_rows

    # The request owning a row is the last one starting at or before it, as in
    # cp_gather_and_upconvert_nvfp4_kv_cache. SEARCH_STEPS bisections resolve
    # num_reqs candidates.
    lo = tl.zeros([BLOCK_ROWS], dtype=tl.int32)
    hi = tl.zeros([BLOCK_ROWS], dtype=tl.int32) + (num_reqs - 1)
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi + 1) >> 1
        mid_start = tl.load(workspace_starts_ptr + mid, mask=row_mask, other=0)
        go_right = mid_start <= rows
        lo = tl.where(go_right, mid, lo)
        hi = tl.where(go_right, hi, mid - 1)

    offset = rows - tl.load(workspace_starts_ptr + lo, mask=row_mask, other=0)
    block = tl.load(
        block_table_ptr + lo.to(tl.int64) * block_table_stride + offset // BLOCK_SIZE,
        mask=row_mask,
        other=0,
    )
    src_row_ptr = (
        src_ptr
        + block.to(tl.int64) * cache_block_stride
        + (offset % BLOCK_SIZE).to(tl.int64) * 352
    )
    _dequant_rows_to_fp8(
        src_row_ptr,
        dst_ptr + rows.to(tl.int64) * 576,
        row_mask,
        inv_scale,
        UNIT_SCALE,
    )


def _check_nvfp4_cache(kv_cache: torch.Tensor) -> None:
    assert kv_cache.dtype == torch.uint8, kv_cache.dtype
    assert kv_cache.dim() == 3 and kv_cache.shape[-1] == NVFP4_DS_MLA_ROW_BYTES, (
        f"expected a [num_blocks, block_size, {NVFP4_DS_MLA_ROW_BYTES}] "
        f"nvfp4_ds_mla cache, got {tuple(kv_cache.shape)}"
    )
    assert kv_cache.stride(2) == 1 and kv_cache.stride(1) == NVFP4_DS_MLA_ROW_BYTES, (
        f"nvfp4_ds_mla records must be packed within a block: {kv_cache.stride()}"
    )


def _check_staging_rows(rows: torch.Tensor, min_rows: int) -> None:
    assert rows.dtype == current_platform.fp8_dtype(), rows.dtype
    assert rows.is_contiguous() and rows.shape[-1] == FP8_STAGING_ROW_DIM
    assert rows.shape[0] >= min_rows, (rows.shape, min_rows)


def gather_nvfp4_ds_mla_topk_to_fp8(
    kv_cache: torch.Tensor,
    physical_topk: torch.Tensor,
    out_rows: torch.Tensor,
    out_indices: torch.Tensor,
    inv_scale: float,
) -> None:
    """Stage each token's top-k rows as FP8 for the FlashInfer kernel.

    Args:
        kv_cache: uint8 nvfp4_ds_mla cache ``[num_blocks, block_size, 352]``.
        physical_topk: int32 ``[num_tokens, topk]`` rows of
            ``flat_kv_row_view(kv_cache, block_size)``; -1 marks invalid slots.
        out_rows: e4m3 ``[>= num_tokens * topk, 576]`` staging rows. Token
            ``t`` owns rows ``[t * topk, (t + 1) * topk)``.
        out_indices: int32 ``[num_tokens, topk]`` (contiguous); receives the
            staging row of every valid slot and -1 elsewhere.
        inv_scale: ``1 / k_scale`` of the fp8 kernel.

    """
    _check_nvfp4_cache(kv_cache)
    num_tokens, topk = physical_topk.shape
    if num_tokens == 0:
        return
    assert physical_topk.dtype == torch.int32 and physical_topk.stride(1) == 1
    assert out_indices.shape == (num_tokens, topk) and out_indices.is_contiguous()
    _check_staging_rows(out_rows, num_tokens * topk)
    use_pdl = current_platform.is_arch_support_pdl()
    grid = (num_tokens, cdiv(topk, _GATHER_BLOCK_ROWS))
    _gather_topk_rows_kernel[grid](
        kv_cache,
        physical_topk,
        physical_topk.stride(0),
        out_rows,
        out_indices,
        out_indices.stride(0),
        inv_scale,
        TOPK=topk,
        BLOCK_ROWS=_GATHER_BLOCK_ROWS,
        UNIT_SCALE=inv_scale == 1.0,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )


def gather_nvfp4_ds_mla_context_to_fp8(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    num_rows: int,
    search_steps: int,
    out_rows: torch.Tensor,
    inv_scale: float,
) -> None:
    """Stage whole request contexts as FP8 rows of a prefill workspace.

    Request ``r`` (row ``r`` of ``block_table``) occupies workspace rows
    ``[workspace_starts[r], workspace_starts[r + 1])``; the last request runs to
    ``num_rows``.
    """
    _check_nvfp4_cache(kv_cache)
    num_reqs = block_table.shape[0]
    if num_rows == 0 or num_reqs == 0:
        return
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert workspace_starts.dtype == torch.int32 and workspace_starts.is_contiguous()
    assert workspace_starts.shape[0] == num_reqs
    assert (num_reqs - 1).bit_length() <= search_steps
    _check_staging_rows(out_rows, num_rows)
    use_pdl = current_platform.is_arch_support_pdl()
    grid = (cdiv(num_rows, _GATHER_BLOCK_ROWS),)
    _gather_context_rows_kernel[grid](
        kv_cache,
        block_table,
        block_table.stride(0),
        workspace_starts,
        num_reqs,
        num_rows,
        kv_cache.stride(0),
        out_rows,
        inv_scale,
        BLOCK_SIZE=kv_cache.shape[1],
        SEARCH_STEPS=search_steps,
        BLOCK_ROWS=_GATHER_BLOCK_ROWS,
        UNIT_SCALE=inv_scale == 1.0,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )


@dataclass
class NVFP4GatherPrefillChunk:
    """Prefill requests whose contexts fit the workspace together."""

    # Query-token rows of the chunk, relative to the first prefill token.
    tokens_slice: slice
    block_table: torch.Tensor  # int32 [num_chunk_reqs, max_blocks]
    workspace_starts: torch.Tensor  # int32 [num_chunk_reqs], chunk-local
    num_rows: int  # context rows staged for the chunk
    search_steps: int


@dataclass
class NVFP4GatherPrefillMetadata:
    # Prefill request index of every prefill token (maps into workspace_starts).
    request_ids: torch.Tensor  # int32 [num_prefill_tokens]
    # Chunk-local workspace start of every prefill request.
    workspace_starts: torch.Tensor  # int32 [num_prefills]
    chunks: list[NVFP4GatherPrefillChunk]


def build_nvfp4_gather_prefill_metadata(
    common_attn_metadata: "CommonAttentionMetadata",
    num_decodes: int,
    num_prefills: int,
    workspace_rows: int,
    request_ids_buffer: torch.Tensor,
    workspace_starts_buffer: torch.Tensor,
) -> NVFP4GatherPrefillMetadata:
    """Plan the per-request context gather for the prefill tokens of a batch.

    Mirrors the FlashMLA prefill chunking: requests are packed into chunks
    whose contexts fit ``workspace_rows``, and starts restart at 0 per chunk.
    ``workspace_starts_buffer`` holds one entry per schedulable request; its
    size fixes the context-gather bisection depth.
    """
    search_steps = nvfp4_fp8_context_search_steps(workspace_starts_buffer.shape[0])
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    assert seq_lens_cpu is not None
    # The upper bound is exact for prefill rows, so no D2H sync is needed.
    prefill_seq_lens = seq_lens_cpu[num_decodes : num_decodes + num_prefills]
    qsl_cpu = common_attn_metadata.query_start_loc_cpu
    prefill_qsl = (
        qsl_cpu[num_decodes : num_decodes + num_prefills + 1] - qsl_cpu[num_decodes]
    ).numpy()
    num_prefill_tokens = int(prefill_qsl[-1])

    request_ids = request_ids_buffer[:num_prefill_tokens]
    request_ids.copy_(
        np_to_pinned_tensor(
            np.repeat(np.arange(num_prefills, dtype=np.int32), np.diff(prefill_qsl))
        ),
        non_blocking=True,
    )

    block_table = common_attn_metadata.block_table_tensor[
        num_decodes : num_decodes + num_prefills
    ]
    starts_cpu = torch.zeros(num_prefills, dtype=torch.int32, pin_memory=PIN_MEMORY)
    chunks = []
    for start, end in split_prefill_chunks(prefill_seq_lens, workspace_rows):
        chunk_lens = prefill_seq_lens[start:end]
        if end - start > 1:
            starts_cpu[start + 1 : end] = torch.cumsum(chunk_lens[:-1], dim=0)
        chunks.append(
            NVFP4GatherPrefillChunk(
                tokens_slice=slice(int(prefill_qsl[start]), int(prefill_qsl[end])),
                block_table=block_table[start:end],
                workspace_starts=workspace_starts_buffer[start:end],
                num_rows=int(chunk_lens.sum()),
                search_steps=search_steps,
            )
        )
    workspace_starts = workspace_starts_buffer[:num_prefills]
    workspace_starts.copy_(starts_cpu, non_blocking=True)
    return NVFP4GatherPrefillMetadata(
        request_ids=request_ids,
        workspace_starts=workspace_starts,
        chunks=chunks,
    )
