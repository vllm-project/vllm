# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up sparse-MLA Triton metadata kernels."""

import itertools
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "ROCM_FLASHMLA_SPARSE_DSV4",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_GENERIC_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE",
        "FLASHINFER_MLA_SPARSE",
        "FLASHINFER_MLA_SPARSE_SM120",
    }
)
_INDEXER_PREFILL_CHUNK_METADATA_BACKENDS = frozenset({"DEEPSEEK_V32_INDEXER"})

_SPARSE_PREFILL_METADATA_NUM_PREFILLS = (1, 2, 4, 8)
_SPARSE_PREFILL_METADATA_NUM_DECODES = (0, 1, 2)
_DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS = (4, 128)
_PREFILL_CHUNK_METADATA_SEQ_LEN_MULTIPLIERS = (2, 3)
_PREFILL_CHUNK_METADATA_QUERY_SLICE_OFFSETS = (
    # query_slice_start offset, query_slice_stop offset
    (0, 0),
    (0, -1),
    (1, 0),
    (1, -1),
)
_COMBINE_TOPK_SWA_INPUT_VARIANTS = (
    # offset_topk, offset_query_and_seq, offset_gather
    (False, False, False),
    (False, True, False),
    (True, True, True),
)
_C128A_TOPK_ALIGNMENT = 128


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _hf_config_int(runner: "GPUModelRunner", name: str, default: int) -> int:
    model_config = getattr(runner.vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    return int(getattr(hf_config, name, default) or default)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_attention_backend(
    runner: "GPUModelRunner",
    backend_names: frozenset[str],
) -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in backend_names:
                return True
    return False


def _warm_sparse_swa_prefill_metadata_kernel(
    device: torch.device,
    window_size: int,
    prefill_tokens: int,
) -> None:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_prefill_metadata_kernel,
    )

    for num_prefills in _SPARSE_PREFILL_METADATA_NUM_PREFILLS:
        for num_decodes in _SPARSE_PREFILL_METADATA_NUM_DECODES:
            query_lens = [1] * num_decodes
            query_lens += [prefill_tokens] * num_prefills
            query_start_locs = [0]
            for query_len in query_lens:
                query_start_locs.append(query_start_locs[-1] + query_len)
            query_start_loc = torch.tensor(
                query_start_locs,
                dtype=torch.int32,
                device=device,
            )
            seq_lens = torch.tensor(
                [1] * num_decodes + [window_size + q for q in query_lens[num_decodes:]],
                dtype=torch.int32,
                device=device,
            )
            prefill_gather_lens = torch.empty(
                num_prefills, dtype=torch.int32, device=device
            )
            _compute_prefill_metadata_kernel[(1,)](
                prefill_gather_lens,
                seq_lens,
                query_start_loc,
                num_prefills,
                num_decodes,
                window_size,
                BLOCK_SIZE=_next_power_of_2(num_prefills),
            )


def _warm_prefill_chunk_metadata_kernel(
    device: torch.device,
    compress_ratio: int,
    query_len: int,
) -> None:
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata

    num_reqs = 2
    query_start_loc_cpu = torch.arange(
        0, (num_reqs + 1) * query_len, query_len, dtype=torch.int32
    )
    query_start_loc = query_start_loc_cpu.to(device=device)

    uncompressed_seq_lens_cpu = torch.tensor(
        [
            compress_ratio * multiplier + query_len
            for multiplier in _PREFILL_CHUNK_METADATA_SEQ_LEN_MULTIPLIERS
        ],
        dtype=torch.int32,
    )
    compressed_seq_lens_cpu = uncompressed_seq_lens_cpu // compress_ratio
    uncompressed_seq_lens = uncompressed_seq_lens_cpu.to(device=device)
    compressed_seq_lens = compressed_seq_lens_cpu.to(device=device)
    block_table = torch.zeros(
        (num_reqs, int(compressed_seq_lens_cpu.max().item())),
        dtype=torch.int32,
        device=device,
    )

    offset_uncompressed_seq_lens = torch.empty(
        num_reqs + 1, dtype=torch.int32, device=device
    )[1:]
    offset_uncompressed_seq_lens.copy_(uncompressed_seq_lens)
    query_slices = tuple(
        slice(start, num_reqs * query_len + stop)
        for start, stop in _PREFILL_CHUNK_METADATA_QUERY_SLICE_OFFSETS
    )
    for warmup_uncompressed_seq_lens in (
        uncompressed_seq_lens,
        offset_uncompressed_seq_lens,
    ):
        for query_slice in query_slices:
            build_prefill_chunk_metadata(
                0,
                num_reqs,
                query_start_loc,
                query_start_loc_cpu,
                warmup_uncompressed_seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                block_table,
                compress_ratio,
                query_slice=query_slice,
            )


def _derive_combine_topk_swa_modes(
    runner: "GPUModelRunner",
) -> list[tuple[int, int, int, int]]:
    """Derive (compress_ratio, top_k, topk_width, n_prod) from model config.

    Each tuple produces a distinct set of Triton constexpr values:

    * ``COMPRESS_RATIO = compress_ratio``
    * ``TOP_K = top_k``
    * ``PADDED_TOP_K = next_power_of_2(topk_width)``
    * ``WINDOW_SIZE`` is passed separately from ``sliding_window``.

    Three modes based on compress_ratio:

    * SWA-only (cr ≤ 1): ``TOP_K=0``, ``topk_width=index_topk``
    * C4A (cr ≤ 4):      ``TOP_K=index_topk``, ``topk_width=index_topk``
    * C128A (cr ≥ 128):  ``TOP_K=c128a_max_compressed``,
      ``topk_width=c128a_max_compressed``

    ``n_prod`` is the production N value used only for ``n_non_one``
    selection inside the M/N 2×2 enumeration.
    """
    index_topk = _hf_config_int(runner, "index_topk", 512)
    max_model_len = int(runner.vllm_config.model_config.max_model_len)

    compress_ratios: set[int] = set()
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            spec = getattr(group, "kv_cache_spec", None)
            if spec is not None:
                compress_ratios.add(getattr(spec, "compress_ratio", 1))

    modes: list[tuple[int, int, int, int]] = []
    for cr in sorted(compress_ratios):
        if cr <= 1:
            # SWA-only: TOP_K=0, topk_width=index_topk, N=0
            modes.append((1, 0, index_topk, 0))
        elif cr <= 4:
            # C4A: TOP_K=index_topk, topk_width=index_topk
            n_prod = cdiv(max_model_len, cr)
            modes.append((cr, index_topk, index_topk, n_prod))
        else:
            # C128A: TOP_K=c128a_max_compressed, topk_width=c128a_max_compressed
            c128a_max = cdiv(max_model_len, cr)
            c128a_max = cdiv(c128a_max, _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT
            n_prod = cdiv(max_model_len, cr)
            modes.append((cr, c128a_max, c128a_max, n_prod))
    return modes


def _warm_combine_topk_swa_indices_kernel(
    device: torch.device,
    num_tokens: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    topk_width: int,
    n: int,
) -> None:
    """Exhaustively warm up ``_combine_topk_swa_indices_kernel``.

    Full specialization space (per constexpr combo):
      - 4 constexprs: TOP_K, COMPRESS_RATIO, WINDOW_SIZE, PADDED_TOP_K
        (fixed per call, derived from model config)
      - 2 runtime ints specialized when ==1: M, N → 2^2 = 4 combos
      - 6 pointer divisibility (16B alignment): combined_indices_ptr,
        combined_lens_ptr, topk_indices_ptr, query_start_loc_ptr,
        seq_lens_ptr, gather_lens_ptr → 2^6 = 64 combos

    Total cache entries per constexpr combo: 4 × 64 = 256.

    ``combined_indices_stride`` (≥128 after alignment) and
    ``topk_indices_stride`` (= topk_width ≥ 512) are never 1 in
    production, so they always produce a single 'D' cubin — no
    enumeration needed.
    """
    import triton

    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        _SPARSE_PREFILL_TOPK_ALIGNMENT,
        _combine_topk_swa_indices_kernel,
    )

    if num_tokens <= 0:
        return

    NUM_WORKERS = 128
    num_reqs = 1
    padded_top_k = triton.next_power_of_2(topk_width)
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )

    # Triton specializes int args on three distinct cases:
    #   ==1 → promoted to constexpr
    #   !=1 and %16==0 → 'D' (divisible) in key
    #   !=1 and %16!=0 → '' (not divisible) in key
    # M and N are buffer offsets whose values are arbitrary at runtime,
    # so all three cases must be enumerated.
    # Use 16 for the divisible case and 3 for the non-divisible case.
    # combined_indices_stride (≥128) and topk_indices_stride (≥512) are
    # always multiples of 16 in production, so they always produce 'D'
    # and don't need enumeration.
    m_values = (1, 16, 3)
    n_values = (1, 16, 3)

    NUM_PTRS = 6
    total = len(m_values) * len(n_values) * (1 << NUM_PTRS)

    def _alloc(
        size: int,
        unaligned: bool,
        dtype: torch.dtype = torch.int32,
        fill_value: int | None = None,
    ) -> torch.Tensor:
        alloc_size = size + 1 if unaligned else size
        t = torch.empty(alloc_size, dtype=dtype, device=device)
        # Always zero-fill: Triton mask= does not prevent the GPU from
        # issuing the memory request, only from using the result.
        t.fill_(0 if fill_value is None else fill_value)
        return t[1:] if unaligned else t

    def _alloc_with_values(
        values: list[int],
        unaligned: bool,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        src = torch.tensor(values, dtype=dtype, device=device)
        if not unaligned:
            return src
        storage = torch.empty(len(values) + 1, dtype=dtype, device=device)
        storage[1:] = src
        return storage[1:]

    logger.info(
        "Warming up _combine_topk_swa_indices_kernel: "
        "compress_ratio=%d, topk=%d, topk_width=%d, window_size=%d, n=%d "
        "→ %d M/N combos × %d div combos = %d entries",
        compress_ratio,
        topk,
        topk_width,
        window_size,
        n,
        len(m_values) * len(n_values),
        1 << NUM_PTRS,
        total,
    )
    ok_count = 0
    fail_count = 0
    for m_value in m_values:
        for n_value in n_values:
            for div_mask in range(1 << NUM_PTRS):
                ci_off = bool(div_mask & 0x01)
                cl_off = bool(div_mask & 0x02)
                ti_off = bool(div_mask & 0x04)
                qs_off = bool(div_mask & 0x08)
                sl_off = bool(div_mask & 0x10)
                gl_off = bool(div_mask & 0x20)

                combined_indices = _alloc(
                    num_tokens * combined_topk,
                    ci_off,
                ).view(num_tokens, combined_topk)
                combined_lens = _alloc(num_tokens, cl_off)
                topk_indices = _alloc(
                    num_tokens * topk_width,
                    ti_off,
                    fill_value=0,
                ).view(num_tokens, topk_width)
                query_start_loc = _alloc_with_values(
                    [0, num_tokens],
                    qs_off,
                )
                seq_lens = _alloc_with_values(
                    [window_size + num_tokens],
                    sl_off,
                )
                gather_lens = _alloc_with_values(
                    [min(window_size + num_tokens, window_size + num_tokens - 1)],
                    gl_off,
                )

                try:
                    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
                        combined_indices,
                        combined_indices.stride(0),
                        combined_lens,
                        topk_indices,
                        topk_indices.stride(0),
                        query_start_loc,
                        seq_lens,
                        gather_lens,
                        m_value,
                        n_value,
                        TOP_K=topk,
                        COMPRESS_RATIO=compress_ratio,
                        WINDOW_SIZE=window_size,
                        PADDED_TOP_K=padded_top_k,
                    )
                    ok_count += 1
                except Exception:
                    fail_count += 1

    torch.accelerator.synchronize()
    logger.info(
        "Warmed up _combine_topk_swa_indices_kernel: cr=%d, topk=%d, "
        "topk_width=%d, n=%d → %d ok, %d failed (total %d)",
        compress_ratio,
        topk,
        topk_width,
        n,
        ok_count,
        fail_count,
        ok_count + fail_count,
    )


def _find_c128a_kv_cache_spec(runner: "GPUModelRunner"):
    """Find the MLAAttentionSpec with compress_ratio==128, if any."""
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            spec = getattr(group, "kv_cache_spec", None)
            if spec is None:
                continue
            if getattr(spec, "compress_ratio", 1) == 128:
                return spec
    return None


def _warmup_build_c128a_topk_metadata_kernel(
    device: torch.device,
    max_model_len: int,
    storage_block_size: int,
    block_table_stride: int,
) -> None:
    """Exhaustively warm up ``_build_c128a_topk_metadata_kernel``.

    Full specialization space:
      - 1 constexpr: BLOCK_SIZE=1024 (fixed)
      - 6 runtime ints specialized when ==1: global_decode_stride,
        prefill_local_stride, compress_ratio, max_compressed_tokens,
        block_table_stride, block_size → 2^6 = 64 combos
      - 1 runtime int with 3 divisibility cases: num_decode_tokens
        (1→'S', 16→'D', 3→'') → 3 combos
      - 7 pointer divisibility (16B alignment): global_decode_ptr,
        decode_lens_ptr, prefill_local_ptr, positions_ptr,
        token_to_req_indices_ptr, block_table_ptr, slot_mapping_ptr
        → 2^7 = 128 combos

    Total cache entries: 64 × 3 × 128 = 24576.
    """
    from vllm.models.deepseek_v4.sparse_mla import (
        _build_c128a_topk_metadata_kernel,
    )

    COMPRESS_RATIO = 128
    max_compressed = cdiv(max_model_len, COMPRESS_RATIO)
    max_compressed = cdiv(max_compressed, _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT

    # Triton specializes int args on three distinct cases:
    #   ==1 → promoted to constexpr
    #   !=1 and %16==0 → 'D' (divisible) in key
    #   !=1 and %16!=0 → '' (not divisible) in key
    # Most params are strides/sizes that are always multiples of 16 in
    # production (max_compressed=8192, compress_ratio=128, block_size=64,
    # etc.), so (1, non1) suffices.  However, num_decode_tokens is the
    # actual decode batch size — an arbitrary runtime value that can be
    # any positive integer (1, 2, 3, ..., batch_size).  It must use all
    # three divisibility cases: (1, 16, 3).
    int_param_choices = [
        (1, max_compressed),  # global_decode_stride (always %16==0)
        (1, max_compressed),  # prefill_local_stride (always %16==0)
        (1, COMPRESS_RATIO),  # compress_ratio (always 128, %16==0)
        (1, max_compressed),  # max_compressed_tokens (always %16==0)
        (1, 16, 3),  # num_decode_tokens — arbitrary runtime value!
        (1, block_table_stride),  # block_table_stride (always %16==0)
        (1, storage_block_size),  # block_size (always %16==0)
    ]
    int_combos = list(itertools.product(*int_param_choices))

    NUM_PTRS = 7
    # Must be >= max num_decode_tokens choice (16) so _is_safe doesn't skip it.
    num_tokens = 16
    buf_cols = max(max_compressed, 1)

    def _alloc(
        size: int,
        unaligned: bool,
        dtype: torch.dtype = torch.int32,
        fill_value: int | None = None,
    ) -> torch.Tensor:
        alloc_size = size + 1 if unaligned else size
        t = torch.empty(alloc_size, dtype=dtype, device=device)
        # Always zero-fill: Triton mask= does not prevent the GPU from
        # issuing the memory request, only from using the result.  An
        # uninitialized token_to_req buffer gives a garbage req_idx that
        # makes block_table_ptr + req_idx*stride point to unmapped memory,
        # triggering cudaErrorIllegalAddress even with mask=False.
        t.fill_(0 if fill_value is None else fill_value)
        return t[1:] if unaligned else t

    def _is_safe(
        v_g_decode_stride: int,
        v_p_stride: int,
        v_compress_ratio: int,
        v_max_compressed: int,
        v_num_decode: int,
        v_bt_stride: int,
        v_block_size: int,
    ) -> bool:
        """Pre-check whether this int combo can launch without OOB.

        Kernel reads/writes:
          - global_decode[token * g_decode_stride + offset],
            offset in [0, max_compressed_tokens)
          - prefill_local[token * p_stride + offset],
            offset in [0, max_compressed_tokens)
          - block_table[req_idx * block_table_stride + (offset // block_size)]
        Buffers have buf_cols columns.  num_decode_tokens <= num_tokens.
        """
        bs = max(v_block_size, 1)
        if v_g_decode_stride < 1 or v_g_decode_stride > buf_cols:
            return False
        if v_p_stride < 1 or v_p_stride > buf_cols:
            return False
        if v_max_compressed < 1 or v_max_compressed > buf_cols:
            return False
        # block_table index = req_idx * block_table_stride + offset // block_size
        # req_idx in [0, num_tokens), offset in [0, v_max_compressed)
        # Worst-case index = (num_tokens-1)*bt_stride + (v_max_compressed-1)//bs
        if v_bt_stride < 1:
            return False
        worst_bt_idx = (num_tokens - 1) * v_bt_stride + (v_max_compressed - 1) // bs
        if worst_bt_idx >= num_tokens * buf_cols:
            return False
        if v_num_decode < 0 or v_num_decode > num_tokens:
            return False
        return v_compress_ratio >= 1

    total = len(int_combos) * (1 << NUM_PTRS)
    logger.info(
        "Warming up _build_c128a_topk_metadata_kernel: "
        "%d int combos × %d div combos = %d entries",
        len(int_combos),
        1 << NUM_PTRS,
        total,
    )

    import time

    start_ts = time.monotonic()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    progress_interval = 1024
    next_log = progress_interval
    done = 0
    # Sync after each launch so that any illegal memory access is reported
    # on this call (and caught by try/except) instead of accumulating and
    # corrupting the CUDA context for later combos.
    sync_each = True
    for int_idx, (
        v_g_decode_stride,
        v_p_stride,
        v_compress_ratio,
        v_max_compressed,
        v_num_decode,
        v_bt_stride,
        v_block_size,
    ) in enumerate(int_combos):
        safe_block_size = max(v_block_size, 1)
        if not _is_safe(
            v_g_decode_stride,
            v_p_stride,
            v_compress_ratio,
            v_max_compressed,
            v_num_decode,
            v_bt_stride,
            v_block_size,
        ):
            # Skip entire int combo for all div masks: mark 128 entries as skipped.
            skip_count += 1 << NUM_PTRS
            done += 1 << NUM_PTRS
            if done >= next_log:
                elapsed = time.monotonic() - start_ts
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(
                    "Warming up _build_c128a_topk_metadata_kernel: "
                    "progress %d/%d (%.1f%%), int=%d/%d div=0x-- (skipped), "
                    "ok=%d fail=%d skip=%d, %.1fs elapsed, ETA %.1fs (%.1f/s)",
                    done,
                    total,
                    100.0 * done / total,
                    int_idx + 1,
                    len(int_combos),
                    ok_count,
                    fail_count,
                    skip_count,
                    elapsed,
                    eta,
                    rate,
                )
                next_log = done + progress_interval
                torch.accelerator.empty_cache()
            continue
        for div_mask in range(1 << NUM_PTRS):
            gd_off = bool(div_mask & 0x01)
            dl_off = bool(div_mask & 0x02)
            pl_off = bool(div_mask & 0x04)
            pos_off = bool(div_mask & 0x08)
            ttr_off = bool(div_mask & 0x10)
            bt_off = bool(div_mask & 0x20)
            sm_off = bool(div_mask & 0x40)

            global_decode_buffer = _alloc(
                num_tokens * buf_cols,
                gd_off,
            ).view(num_tokens, buf_cols)
            decode_lens_buffer = _alloc(num_tokens, dl_off)
            prefill_buffer = _alloc(
                num_tokens * buf_cols,
                pl_off,
            ).view(num_tokens, buf_cols)
            positions = _alloc(num_tokens, pos_off, dtype=torch.int64)
            token_to_req = _alloc(num_tokens, ttr_off)
            block_table = _alloc(
                num_tokens * buf_cols,
                bt_off,
            ).view(num_tokens, buf_cols)
            slot_mapping = _alloc(num_tokens, sm_off, dtype=torch.int64)

            try:
                _build_c128a_topk_metadata_kernel[(num_tokens,)](
                    global_decode_buffer,
                    v_g_decode_stride,
                    decode_lens_buffer,
                    prefill_buffer,
                    v_p_stride,
                    positions,
                    v_compress_ratio,
                    v_max_compressed,
                    v_num_decode,
                    token_to_req,
                    block_table,
                    v_bt_stride,
                    safe_block_size,
                    slot_mapping,
                    BLOCK_SIZE=1024,
                )
                if sync_each:
                    torch.accelerator.synchronize()
                ok_count += 1
            except Exception:
                fail_count += 1

            done += 1
            if done >= next_log:
                elapsed = time.monotonic() - start_ts
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                logger.info(
                    "Warming up _build_c128a_topk_metadata_kernel: "
                    "progress %d/%d (%.1f%%), int=%d/%d div=0x%02x, "
                    "ok=%d fail=%d skip=%d, %.1fs elapsed, ETA %.1fs (%.1f/s)",
                    done,
                    total,
                    100.0 * done / total,
                    int_idx + 1,
                    len(int_combos),
                    div_mask,
                    ok_count,
                    fail_count,
                    skip_count,
                    elapsed,
                    eta,
                    rate,
                )
                next_log = done + progress_interval
                # Periodic allocator cleanup to keep peak memory bounded.
                # Triton's internal JIT cache (CUmodule per specialization)
                # is the real memory consumer and cannot be freed here, but
                # empty_cache at least releases PyTorch's caching allocator
                # slack back to the driver so the fragmentation doesn't grow.
                torch.accelerator.empty_cache()

    torch.accelerator.synchronize()
    elapsed = time.monotonic() - start_ts
    logger.info(
        "Warmed up _build_c128a_topk_metadata_kernel: "
        "%d ok, %d failed, %d skipped (total %d) in %.1fs (%.1f/s)",
        ok_count,
        fail_count,
        skip_count,
        ok_count + fail_count + skip_count,
        elapsed,
        total / elapsed if elapsed > 0 else 0,
    )


@torch.inference_mode()
def sparse_mla_triton_warmup(
    runner: "GPUModelRunner",
    num_tokens: int,
    *,
    compress_ratios: tuple[int, ...],
    combine_topk_swa_cases: tuple[tuple[int, int, int, int], ...]
    | list[tuple[int, int, int, int]] = (),
) -> None:
    device = getattr(runner, "device", torch.device("cuda"))
    window_size = _hf_config_int(runner, "sliding_window", 128)

    _warm_sparse_swa_prefill_metadata_kernel(device, window_size, num_tokens)
    for compress_ratio in compress_ratios:
        _warm_prefill_chunk_metadata_kernel(device, compress_ratio, num_tokens)
    for compress_ratio, topk, topk_width, n in combine_topk_swa_cases:
        _warm_combine_topk_swa_indices_kernel(
            device,
            num_tokens,
            window_size,
            compress_ratio,
            topk,
            topk_width,
            n,
        )


def deepseek_v4_sparse_triton_warmup(
    runner: "GPUModelRunner",
    num_tokens: int,
) -> None:
    combine_topk_swa_cases = _derive_combine_topk_swa_modes(runner)
    sparse_mla_triton_warmup(
        runner,
        num_tokens,
        compress_ratios=_DSV4_PREFILL_CHUNK_METADATA_COMPRESS_RATIOS,
        combine_topk_swa_cases=combine_topk_swa_cases,
    )

    # Diagnose: check if warmup kernels are actually in the Triton cache.
    import os

    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
    combine_entries = 0
    c128a_entries = 0
    if cache_dir and os.path.isdir(cache_dir):
        for entry in os.listdir(cache_dir):
            full = os.path.join(cache_dir, entry)
            if not os.path.isdir(full):
                continue
            try:
                files = os.listdir(full)
            except OSError:
                continue
            if any("combine_topk" in f for f in files):
                combine_entries += 1
            if any("c128a_topk" in f for f in files):
                c128a_entries += 1
    logger.info(
        "sparse MLA warmup done. TRITON_CACHE_DIR=%s, combine_topk cache "
        "entries=%d, c128a cache entries=%d (total dirs=%d)",
        cache_dir,
        combine_entries,
        c128a_entries,
        len(os.listdir(cache_dir)) if cache_dir and os.path.isdir(cache_dir) else 0,
    )
    c128a_spec = _find_c128a_kv_cache_spec(runner)
    if c128a_spec is not None:
        max_model_len = int(runner.vllm_config.model_config.max_model_len)
        storage_block_size = c128a_spec.storage_block_size
        block_table_stride = cdiv(max_model_len, c128a_spec.block_size)
        _warmup_build_c128a_topk_metadata_kernel(
            device=getattr(runner, "device", torch.device("cuda")),
            max_model_len=max_model_len,
            storage_block_size=storage_block_size,
            block_table_stride=block_table_stride,
        )


def sparse_mla_triton_warmup_if_needed(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    num_tokens = _clamp_warmup_tokens(8, max_tokens)
    if num_tokens <= 0:
        return

    try:
        if _has_attention_backend(runner, _DEEPSEEK_V4_SPARSE_MLA_BACKENDS):
            deepseek_v4_sparse_triton_warmup(runner, num_tokens)
        elif _has_attention_backend(runner, _GENERIC_SPARSE_MLA_BACKENDS):
            sparse_mla_triton_warmup(
                runner,
                num_tokens,
                compress_ratios=(1,),
            )
        elif _has_attention_backend(runner, _INDEXER_PREFILL_CHUNK_METADATA_BACKENDS):
            _warm_prefill_chunk_metadata_kernel(
                getattr(runner, "device", torch.device("cuda")),
                compress_ratio=1,
                query_len=num_tokens,
            )
    except Exception:
        logger.warning("Skipping sparse MLA Triton warmup.", exc_info=True)
