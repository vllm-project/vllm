# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008, N806 -- FlyDSL launch signatures and local tile constants
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tile32, 16-wave D128/D256 grouped-query SplitKV stage for RDNA4."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from .rdna4_splitkv_common import (
    LOG2E,
    WAVE_SIZE,
    _decode_fp8,
    _dequant_fp8x8,
    _flat_view,
    _wave_reduce,
)

HEAD_DIM = 128
QUERY_ROWS = 16
TILE_TOKENS = 32
NUM_WAVES = 16
QK_WAVES = 2
PV_WAVES = 8
BLOCK_THREADS = NUM_WAVES * WAVE_SIZE
K_FRAGMENTS = HEAD_DIM // 16
TOKEN_FRAGMENTS = TILE_TOKENS // 16


@functools.lru_cache(maxsize=128)
def compile_gqa_stage(
    *,
    dtype: str,
    kv_dtype: str,
    splits: int,
    num_kv_heads: int,
    page_size: int,
    softmax_scale: float,
    head_dim: int = 128,
    query_group_size: int = 16,
):
    """Compile the Tile32 register-PV specialization."""

    if head_dim not in (128, 256):
        raise ValueError(f"head_dim must be 128 or 256, got {head_dim}")
    if not 1 <= query_group_size <= 16:
        raise ValueError(f"query_group_size must be in [1, 16], got {query_group_size}")
    if dtype not in ("bf16", "fp16"):
        raise ValueError(f"dtype must be 'bf16' or 'fp16', got {dtype!r}")
    if kv_dtype not in ("bf16", "fp16", "fp8", "fp8fnuz"):
        raise ValueError(f"unsupported KV dtype {kv_dtype!r}")
    if splits < 1:
        raise ValueError(f"splits must be positive, got {splits}")
    if num_kv_heads < 1:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if page_size < 16 or page_size % 16:
        raise ValueError(f"page_size must be a multiple of 16, got {page_size}")
    HEAD_DIM = head_dim
    PV_WAVES = head_dim // 16
    K_FRAGMENTS = head_dim // 16

    query_type = fx.BFloat16 if dtype == "bf16" else fx.Float16
    pv_type = fx.BFloat16 if kv_dtype == "bf16" else fx.Float16
    cache_type = {
        "bf16": fx.BFloat16,
        "fp16": fx.Float16,
        "fp8": fx.Uint8,
        "fp8fnuz": fx.Uint8,
    }[kv_dtype]
    is_fp8 = kv_dtype in ("fp8", "fp8fnuz")
    is_fp8fnuz = kv_dtype == "fp8fnuz"
    qk_type = query_type
    qk_bytes = 2
    cache_pack = 16 if is_fp8 else 8
    q_words = QUERY_ROWS * HEAD_DIM * qk_bytes // 4
    k_words = TILE_TOKENS * HEAD_DIM * qk_bytes // 4
    v_words = TILE_TOKENS * HEAD_DIM // 2

    @fx.struct
    class SharedStorage:
        query_or_scores: fx.Array[fx.Int32, q_words, 16]
        key: fx.Array[fx.Int32, k_words, 16]
        value: fx.Array[fx.Int32, v_words, 16]
        row_scale: fx.Array[fx.Float32, QUERY_ROWS, 16]

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def stage(
        query_ptr: fx.Tensor,
        key_ptr: fx.Tensor,
        value_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,
        seq_lens_ptr: fx.Tensor,
        query_start_loc_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        v_scale_ptr: fx.Tensor,
        mid_out_ptr: fx.Tensor,
        mid_lse_ptr: fx.Tensor,
        batch: fx.Int32,
        table_stride: fx.Int32,
        q_stride0: fx.Int32,
        q_stride1: fx.Int32,
        k_stride0: fx.Int32,
        k_stride1: fx.Int32,
        k_stride2: fx.Int32,
        k_stride3: fx.Int32,
        k_stride4: fx.Int32,
        v_stride0: fx.Int32,
        v_stride1: fx.Int32,
        v_stride2: fx.Int32,
        v_stride3: fx.Int32,
        mo_stride0: fx.Int32,
        mo_stride1: fx.Int32,
        mo_stride2: fx.Int32,
        ml_stride0: fx.Int32,
        ml_stride1: fx.Int32,
        ml_stride2: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        block = fx.Int32(gpu.block_idx.x)
        split = block % splits
        item = block // splits
        seq = item // num_kv_heads
        kv_head = item % num_kv_heads

        query = _flat_view(query_ptr)
        value = _flat_view(value_ptr)
        raw_value = fx.make_view(
            fx.recast_iter(fx.Uint8, fx.get_iter(value_ptr)),
            fx.make_layout(1 << 30, 1),
        )
        block_tables = _flat_view(block_tables_ptr)
        seq_lens = _flat_view(seq_lens_ptr)
        query_start_loc = _flat_view(query_start_loc_ptr)
        k_scales = _flat_view(k_scale_ptr)
        v_scales = _flat_view(v_scale_ptr)
        mid_out = _flat_view(mid_out_ptr)
        mid_lse = _flat_view(mid_lse_ptr)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        value_tile = fx.make_view(
            fx.recast_iter(pv_type, storage.value.ptr),
            fx.make_layout((HEAD_DIM, TILE_TOKENS), (TILE_TOKENS, 1)),
        )
        scores = fx.make_view(
            fx.recast_iter(fx.Float32, storage.query_or_scores.ptr),
            fx.make_layout((QUERY_ROWS, TILE_TOKENS), (TILE_TOKENS, 1)),
        )
        weights = fx.make_view(
            fx.recast_iter(pv_type, storage.query_or_scores.ptr),
            fx.make_layout((QUERY_ROWS, TILE_TOKENS), (TILE_TOKENS, 1)),
        )
        row_scale = fx.make_view(storage.row_scale.ptr, fx.make_layout(QUERY_ROWS, 1))

        def lds_barrier():
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        k_scale = fx.Float32(1.0)
        v_scale = fx.Float32(1.0)
        if const_expr(is_fp8):
            k_scale = fx.Float32(k_scales[0])
            v_scale = fx.Float32(v_scales[0])

        query_row = fx.Int32(query_start_loc[seq])
        is_decode = fx.Int32(query_start_loc[seq + 1]) - query_row == 1
        length = is_decode.select(fx.Int32(seq_lens[seq]), fx.Int32(0))
        tokens_per_split = (length + splits - 1) // splits
        begin = fx.min(split * tokens_per_split, length)
        end = fx.min(begin + tokens_per_split, length)

        for query_iteration in range_constexpr(
            (QUERY_ROWS * (HEAD_DIM // 4)) // BLOCK_THREADS
        ):
            query_word = tid + query_iteration * BLOCK_THREADS
            query_local_row = query_word // (HEAD_DIM // 4)
            query_live = query_local_row < query_group_size
            safe_query_row = query_live.select(query_local_row, fx.Int32(0))
            query_head = kv_head * query_group_size + safe_query_row
            query_d = (query_word % (HEAD_DIM // 4)) * 4
            query_values = []
            for element in range_constexpr(4):
                query_index = (
                    query_row * q_stride0 + query_head * q_stride1 + query_d + element
                )
                query_value = (query_live & is_decode).select(
                    fx.Float32(query[query_index]), fx.Float32(0.0)
                )
                query_values.append(query_value)
            query_packed = (
                fx.Vector.from_elements(query_values, dtype=fx.Float32)
                .to(query_type)
                .bitcast(fx.Int32)
            )
            for word in range_constexpr(2):
                storage.query_or_scores[query_word * 2 + word] = query_packed[word]
        lds_barrier()

        q_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), qk_type)
            for _ in range_constexpr(K_FRAGMENTS)
        ]
        fragment_row = lane % QUERY_ROWS
        fragment_half = lane // QUERY_ROWS
        repair_row_one = ((wave >= QK_WAVES) & (wave < 2 * QK_WAVES)).select(
            fx.Int32(1), fragment_row
        )
        for k_fragment in range_constexpr(K_FRAGMENTS):
            query_fragment_word = (
                repair_row_one * (HEAD_DIM * qk_bytes // 4)
                + (k_fragment * 16 + fragment_half * 8) * qk_bytes // 4
            )
            query_fragment_words = fx.Vector.from_elements(
                [
                    storage.query_or_scores[query_fragment_word + word]
                    for word in range_constexpr(2 * qk_bytes)
                ],
                dtype=fx.Int32,
            )
            q_fragments[k_fragment].store(query_fragment_words.bitcast(qk_type))
        if const_expr(HEAD_DIM == 256):
            fx.rocdl.s_waitcnt(lgkmcnt=0)
        lds_barrier()

        qk_mma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, qk_type, fx.Float32))
        pv_mma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, pv_type, fx.Float32))
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        init_state = [neg_inf, zero] + [zero for _ in range_constexpr(8)]

        key_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), cache_type)
            for _ in range_constexpr(HEAD_DIM // 128)
        ]
        key_source = fx.make_view(fx.get_iter(key_ptr), fx.make_layout(1 << 30, 1))
        key_divided = fx.logical_divide(key_source, fx.make_layout(1, 1))
        key_copy = fx.make_copy_atom(
            fx.UniversalCopy64b() if is_fp8 else fx.UniversalCopy128b(),
            cache_type,
        )
        score_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
            for _ in range_constexpr(HEAD_DIM // 128)
        ]
        probability_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), pv_type)
            for _ in range_constexpr(TOKEN_FRAGMENTS)
        ]
        value_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), pv_type)
            for _ in range_constexpr(TOKEN_FRAGMENTS)
        ]
        output_fragment = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)

        for tile, state in range(  # type: ignore[call-overload]
            begin, end, fx.Int32(TILE_TOKENS), init=init_state
        ):
            tile_i32 = fx.Int32(tile)
            valid_tokens = fx.min(fx.Int32(TILE_TOKENS), end - tile_i32)

            key_token_local = wave * 2 + lane // 16
            key_token_valid = key_token_local < valid_tokens
            key_token = tile_i32 + fx.min(key_token_local, valid_tokens - 1)
            key_page = key_token // page_size
            key_in_page = key_token - key_page * page_size
            key_physical = fx.Int32(block_tables[seq * table_stride + key_page])
            for key_dimension_block in range_constexpr(HEAD_DIM // 128):
                key_d = key_dimension_block * 128 + (lane % 16) * 8
                key_index = (
                    key_physical * k_stride0
                    + kv_head * k_stride1
                    + (key_d // cache_pack) * k_stride2
                    + key_in_page * k_stride3
                    + (key_d % cache_pack) * k_stride4
                )
                fx.copy(
                    key_copy,
                    fx.slice(key_divided, (None, key_index)),
                    key_fragments[key_dimension_block],
                )
            fx.rocdl.s_wait_loadcnt(0)
            for key_dimension_block in range_constexpr(HEAD_DIM // 128):
                key_d = key_dimension_block * 128 + (lane % 16) * 8
                if const_expr(is_fp8):
                    key_values = _dequant_fp8x8(
                        fx.Vector(key_fragments[key_dimension_block].load()),
                        fx.Float32(1.0),
                        query_type,
                        is_fp8fnuz=is_fp8fnuz,
                    )
                else:
                    key_values = fx.Vector(key_fragments[key_dimension_block].load())
                key_words_value = key_values.bitcast(fx.Int32)
                for word in range_constexpr(4):
                    storage.key[
                        key_token_local * (HEAD_DIM // 2) + key_d // 2 + word
                    ] = key_token_valid.select(key_words_value[word], fx.Int32(0))

            value_token_local = lane
            value_token_valid = value_token_local < valid_tokens
            value_token = tile_i32 + fx.min(value_token_local, valid_tokens - 1)
            value_page = value_token // page_size
            value_in_page = value_token - value_page * page_size
            value_physical = fx.Int32(block_tables[seq * table_stride + value_page])
            for dimension_iteration in range_constexpr(HEAD_DIM // NUM_WAVES):
                d = wave + dimension_iteration * NUM_WAVES
                value_index = (
                    value_physical * v_stride0
                    + kv_head * v_stride1
                    + d * v_stride2
                    + value_in_page * v_stride3
                )
                if const_expr(is_fp8):
                    value_element = fx.Float32(
                        _decode_fp8(
                            fx.Int32(raw_value[value_index]), 0, is_fp8fnuz=is_fp8fnuz
                        )
                    )
                else:
                    value_element = fx.Float32(value[value_index])
                value_tile[d, value_token_local] = value_token_valid.select(
                    value_element.to(pv_type), pv_type(0.0)
                )
            lds_barrier()

            if wave < QK_WAVES:
                token_block = wave
                for score_part in range_constexpr(HEAD_DIM // 128):
                    score_fragments[score_part].fill(0.0)
                    for part_fragment in range_constexpr(8):
                        k_fragment_index = score_part * 8 + part_fragment
                        key_mma_fragment = fx.make_rmem_tensor(
                            fx.make_layout(8, 1), qk_type
                        )
                        key_row = token_block * 16 + lane % 16
                        key_fragment_word = (
                            key_row * (HEAD_DIM * qk_bytes // 4)
                            + (k_fragment_index * 16 + fragment_half * 8)
                            * qk_bytes
                            // 4
                        )
                        key_fragment_words = fx.Vector.from_elements(
                            [
                                storage.key[key_fragment_word + word]
                                for word in range_constexpr(2 * qk_bytes)
                            ],
                            dtype=fx.Int32,
                        )
                        key_mma_fragment.store(key_fragment_words.bitcast(qk_type))
                        fx.mma_atom_call(
                            qk_mma,
                            score_fragments[score_part],
                            q_fragments[k_fragment_index],
                            key_mma_fragment,
                            score_fragments[score_part],
                        )
                score_values = fx.Vector(score_fragments[0].load())
                if const_expr(HEAD_DIM == 256):
                    score_values = score_values + fx.Vector(score_fragments[1].load())
                score_row_base = fragment_half * 8
                score_column = token_block * 16 + lane % 16
                for score_row in range_constexpr(8):
                    if const_expr(
                        not (
                            HEAD_DIM == 256 and query_group_size == 4 and score_row == 1
                        )
                    ):
                        scores[score_row_base + score_row, score_column] = (
                            score_values[score_row] * softmax_scale * k_scale
                        )
            if const_expr(HEAD_DIM == 256 and query_group_size == 4):  # noqa: SIM102
                if (wave >= QK_WAVES) & (wave < 2 * QK_WAVES):
                    token_block = wave - QK_WAVES
                    for score_part in range_constexpr(HEAD_DIM // 128):
                        score_fragments[score_part].fill(0.0)
                        for part_fragment in range_constexpr(8):
                            k_fragment_index = score_part * 8 + part_fragment
                            key_mma_fragment = fx.make_rmem_tensor(
                                fx.make_layout(8, 1), qk_type
                            )
                            key_row = token_block * 16 + lane % 16
                            key_fragment_word = (
                                key_row * (HEAD_DIM * qk_bytes // 4)
                                + (k_fragment_index * 16 + fragment_half * 8)
                                * qk_bytes
                                // 4
                            )
                            key_fragment_words = fx.Vector.from_elements(
                                [
                                    storage.key[key_fragment_word + word]
                                    for word in range_constexpr(2 * qk_bytes)
                                ],
                                dtype=fx.Int32,
                            )
                            key_mma_fragment.store(key_fragment_words.bitcast(qk_type))
                            fx.mma_atom_call(
                                qk_mma,
                                score_fragments[score_part],
                                q_fragments[k_fragment_index],
                                key_mma_fragment,
                                score_fragments[score_part],
                            )
                    repair_score = fx.Vector(score_fragments[0].load()) + fx.Vector(
                        score_fragments[1].load()
                    )
                    if lane < 16:
                        scores[1, token_block * 16 + lane] = (
                            repair_score[0] * softmax_scale * k_scale
                        )
            lds_barrier()

            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            score = (lane < valid_tokens).select(
                fx.Float32(scores[wave, lane]), neg_inf
            )
            tile_max = fx.Float32(
                gpu.shuffle_idx(_wave_reduce(score, "max"), 0, WAVE_SIZE)
            )
            next_max = fx.max(running_max, tile_max)
            old_scale = fmath.exp2((running_max - next_max) * LOG2E)
            probability = (lane < valid_tokens).select(
                fmath.exp2((score - next_max) * LOG2E), zero
            )
            tile_sum = fx.Float32(
                gpu.shuffle_idx(_wave_reduce(probability, "sum"), 0, WAVE_SIZE)
            )
            next_sum = running_sum * old_scale + tile_sum
            # Weights reuse the FP32 score storage across different waves.
            lds_barrier()
            weights[wave, lane] = probability.to(pv_type)
            row_scale[wave] = old_scale
            lds_barrier()

            next_accumulators = [
                fx.Float32(state[2 + element]) for element in range_constexpr(8)
            ]
            if wave < PV_WAVES:
                for output_row in range_constexpr(8):
                    accumulator_row = fragment_half * 8 + output_row
                    output_fragment[output_row] = next_accumulators[
                        output_row
                    ] * fx.Float32(row_scale[accumulator_row])
                d = wave * 16 + lane % 16
                for token_fragment in range_constexpr(TOKEN_FRAGMENTS):
                    token_base = token_fragment * 16 + fragment_half * 8
                    for element in range_constexpr(8):
                        probability_fragments[token_fragment][element] = weights[
                            lane % 16, token_base + element
                        ]
                        value_fragments[token_fragment][element] = value_tile[
                            d, token_base + element
                        ]
                    fx.mma_atom_call(
                        pv_mma,
                        output_fragment,
                        probability_fragments[token_fragment],
                        value_fragments[token_fragment],
                        output_fragment,
                    )
                output_values = fx.Vector(output_fragment.load())
                for output_row in range_constexpr(8):
                    next_accumulators[output_row] = output_values[output_row]
            lds_barrier()
            results = yield [next_max, next_sum] + next_accumulators

        final_sum = fx.Float32(results[1])
        row_scale[wave] = final_sum
        lds_barrier()

        has_tokens = end > begin
        if wave < PV_WAVES:
            d = wave * 16 + lane % 16
            for output_row in range_constexpr(8):
                row = fragment_half * 8 + output_row
                if row < query_group_size:
                    denominator = fx.Float32(row_scale[row])
                    partial = fx.Float32(results[2 + output_row]) / (
                        denominator + 1.0e-10
                    )
                    out_index = (
                        seq * mo_stride0
                        + (kv_head * query_group_size + row) * mo_stride1
                        + split * mo_stride2
                        + d
                    )
                    mid_out[out_index] = has_tokens.select(partial * v_scale, zero)
        if (lane == 0) & (wave < query_group_size):
            lse = fx.Float32(results[0]) + fmath.log2(final_sum) / LOG2E
            lse_index = (
                seq * ml_stride0
                + (kv_head * query_group_size + wave) * ml_stride1
                + split * ml_stride2
            )
            mid_lse[lse_index] = has_tokens.select(lse, neg_inf)

    @flyc.jit
    def launch(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        block_tables: fx.Tensor,
        seq_lens: fx.Tensor,
        query_start_loc: fx.Tensor,
        k_scale: fx.Tensor,
        v_scale: fx.Tensor,
        mid_out: fx.Tensor,
        mid_lse: fx.Tensor,
        batch: fx.Int32,
        table_stride: fx.Int32,
        q_stride0: fx.Int32,
        q_stride1: fx.Int32,
        k_stride0: fx.Int32,
        k_stride1: fx.Int32,
        k_stride2: fx.Int32,
        k_stride3: fx.Int32,
        k_stride4: fx.Int32,
        v_stride0: fx.Int32,
        v_stride1: fx.Int32,
        v_stride2: fx.Int32,
        v_stride3: fx.Int32,
        mo_stride0: fx.Int32,
        mo_stride1: fx.Int32,
        mo_stride2: fx.Int32,
        ml_stride0: fx.Int32,
        ml_stride1: fx.Int32,
        ml_stride2: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        stage(
            query,
            key,
            value,
            block_tables,
            seq_lens,
            query_start_loc,
            k_scale,
            v_scale,
            mid_out,
            mid_lse,
            batch,
            table_stride,
            q_stride0,
            q_stride1,
            k_stride0,
            k_stride1,
            k_stride2,
            k_stride3,
            k_stride4,
            v_stride0,
            v_stride1,
            v_stride2,
            v_stride3,
            mo_stride0,
            mo_stride1,
            mo_stride2,
            ml_stride0,
            ml_stride1,
            ml_stride2,
        ).launch(
            grid=(batch * num_kv_heads * splits, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["compile_gqa_stage"]
