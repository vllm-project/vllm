# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008, SIM102 -- FlyDSL kernels preserve staged control flow
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Four-wave native-cache D128/GQA16 SplitKV stage for RDNA4."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from .rdna4_splitkv_common import LOG2E, WAVE_SIZE, _dequant_fp8x8, _flat_view

HEAD_DIM = 128
QUERY_ROWS = 16
TILE_TOKENS = 16
NUM_WAVES = 4
BLOCK_THREADS = NUM_WAVES * WAVE_SIZE
K_FRAGMENTS = HEAD_DIM // 16
OUTPUT_FRAGMENTS = HEAD_DIM // 64


@functools.lru_cache(maxsize=64)
def compile_native_d128_gqa16_stage(
    *,
    dtype: str,
    kv_dtype: str | None = None,
    splits: int,
    num_kv_heads: int,
    page_size: int,
    softmax_scale: float,
    schedule: str = "direct",
):
    """Compile a four-wave direct-register or LDS D128/GQA16 stage."""

    if dtype not in ("bf16", "fp16"):
        raise ValueError(f"dtype must be 'bf16' or 'fp16', got {dtype!r}")
    if kv_dtype is None:
        kv_dtype = dtype
    if kv_dtype not in ("bf16", "fp16", "fp8", "fp8fnuz"):
        raise ValueError(f"unsupported KV dtype {kv_dtype!r}")
    if splits < 1:
        raise ValueError(f"splits must be positive, got {splits}")
    if num_kv_heads < 1:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if page_size < TILE_TOKENS or page_size % TILE_TOKENS:
        raise ValueError(
            f"page_size must be a multiple of {TILE_TOKENS}, got {page_size}"
        )
    if schedule not in ("direct", "lds"):
        raise ValueError(f"schedule must be 'direct' or 'lds', got {schedule!r}")
    elem_type = fx.BFloat16 if dtype == "bf16" else fx.Float16
    is_fp8 = kv_dtype in ("fp8", "fp8fnuz")
    is_fp8fnuz = kv_dtype == "fp8fnuz"
    cache_pack = 16 if is_fp8 else 8
    shared_elements = (
        QUERY_ROWS * HEAD_DIM if schedule == "direct" else 2 * TILE_TOKENS * HEAD_DIM
    )

    @fx.struct
    class SharedStorage:
        words: fx.Array[fx.Int32, shared_elements // 2, 16]

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
        key = _flat_view(key_ptr)
        value = _flat_view(value_ptr)
        block_tables = _flat_view(block_tables_ptr)
        seq_lens = _flat_view(seq_lens_ptr)
        query_start_loc = _flat_view(query_start_loc_ptr)
        k_scales = _flat_view(k_scale_ptr)
        v_scales = _flat_view(v_scale_ptr)
        mid_out = _flat_view(mid_out_ptr)
        mid_lse = _flat_view(mid_lse_ptr)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        shared = fx.make_view(
            fx.recast_iter(elem_type, storage.words.ptr),
            fx.make_layout(shared_elements, 1),
        )
        scores = fx.make_view(
            fx.recast_iter(fx.Float32, storage.words.ptr),
            fx.make_layout((QUERY_ROWS, TILE_TOKENS), (TILE_TOKENS, 1)),
        )

        def make_global_fragments(tensor_ptr, count, fragment_type, load_atom):
            tensor = fx.make_view(fx.get_iter(tensor_ptr), fx.make_layout(1 << 30, 1))
            divided = fx.logical_divide(tensor, fx.make_layout(1, 1))
            fragments = [
                fx.make_rmem_tensor(fx.make_layout(8, 1), fragment_type)
                for _ in range_constexpr(count)
            ]

            def prefetch(slot, offset):
                fx.copy(
                    load_atom,
                    fx.slice(divided, (None, offset)),
                    fragments[slot],
                )

            return fragments, prefetch

        k_scale = fx.Float32(1.0)
        v_scale = fx.Float32(1.0)
        if const_expr(is_fp8):
            k_scale = fx.Float32(k_scales[0])
            v_scale = fx.Float32(v_scales[0])

        query_row = fx.Int32(query_start_loc[seq])
        is_decode = fx.Int32(query_start_loc[seq + 1]) - query_row == 1
        length = is_decode.select(fx.Int32(seq_lens[seq]), fx.Int32(0))
        if const_expr(schedule == "direct"):
            pages = (length + page_size - 1) // page_size
            pages_per_split = (pages + splits - 1) // splits
            begin = split * pages_per_split * page_size
            end = fx.min(begin + pages_per_split * page_size, length)
        else:
            tokens_per_split = (length + splits - 1) // splits
            begin = fx.min(split * tokens_per_split, length)
            end = fx.min(begin + tokens_per_split, length)

        for load_iter in range_constexpr((QUERY_ROWS * HEAD_DIM) // BLOCK_THREADS):
            linear = tid + load_iter * BLOCK_THREADS
            row = linear // HEAD_DIM
            d = linear % HEAD_DIM
            q_head = kv_head * QUERY_ROWS + row
            shared[linear] = query[query_row * q_stride0 + q_head * q_stride1 + d].to(
                elem_type
            )
        fx.rocdl.s_wait_dscnt(0)
        gpu.barrier()

        q_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), elem_type)
            for _ in range_constexpr(K_FRAGMENTS)
        ]
        q_row = lane % QUERY_ROWS
        q_half = lane // QUERY_ROWS
        for k_fragment in range_constexpr(K_FRAGMENTS):
            for element in range_constexpr(8):
                d = k_fragment * 16 + q_half * 8 + element
                q_fragments[k_fragment][element] = shared[q_row * HEAD_DIM + d]
        fx.rocdl.s_wait_dscnt(0)
        gpu.barrier()

        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, elem_type, fx.Float32))
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        init_state = [neg_inf, zero] + [
            zero for _ in range_constexpr(8 * OUTPUT_FRAGMENTS)
        ]

        key_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), elem_type)
            for _ in range_constexpr(K_FRAGMENTS)
        ]
        value_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), elem_type)
            for _ in range_constexpr(OUTPUT_FRAGMENTS)
        ]
        if const_expr(is_fp8):
            raw_key_fragments, prefetch_key = make_global_fragments(
                key_ptr,
                K_FRAGMENTS,
                fx.Uint8,
                fx.make_copy_atom(fx.UniversalCopy64b(), fx.Uint8),
            )
            raw_value_fragments, prefetch_value = make_global_fragments(
                value_ptr,
                OUTPUT_FRAGMENTS,
                fx.Uint8,
                fx.make_copy_atom(fx.UniversalCopy64b(), fx.Uint8),
            )
        else:
            key_fragments, prefetch_key = make_global_fragments(
                key_ptr,
                K_FRAGMENTS,
                elem_type,
                fx.make_copy_atom(fx.UniversalCopy128b(), elem_type),
            )
            value_fragments, prefetch_value = make_global_fragments(
                value_ptr,
                OUTPUT_FRAGMENTS,
                elem_type,
                fx.make_copy_atom(fx.UniversalCopy128b(), elem_type),
            )
        score_fragment = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
        probability_fragment = fx.make_rmem_tensor(fx.make_layout(8, 1), elem_type)
        output_fragments = [
            fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float32)
            for _ in range_constexpr(OUTPUT_FRAGMENTS)
        ]

        for tile, state in range(  # type: ignore[call-overload]
            begin, end, fx.Int32(TILE_TOKENS), init=init_state
        ):
            tile_i32 = fx.Int32(tile)
            token = tile_i32 + (lane // 16) * 8
            if const_expr(schedule == "direct"):  # noqa: SIM102
                logical_page = token // page_size
                in_page = token - logical_page * page_size
                physical_page = fx.Int32(
                    block_tables[seq * table_stride + logical_page]
                )

                for output_fragment in range_constexpr(OUTPUT_FRAGMENTS):
                    d = wave * 16 + output_fragment * 64 + lane % 16
                    value_index = (
                        physical_page * v_stride0
                        + kv_head * v_stride1
                        + d * v_stride2
                        + in_page * v_stride3
                    )
                    prefetch_value(output_fragment, value_index)
            else:
                for token_iteration in range_constexpr(TILE_TOKENS // NUM_WAVES):
                    token_local = wave + token_iteration * NUM_WAVES
                    load_token = tile_i32 + token_local
                    valid_token = load_token < end
                    logical_page = load_token // page_size
                    in_page = load_token - logical_page * page_size
                    physical_page = fx.Int32(
                        block_tables[seq * table_stride + logical_page]
                    )
                    for dimension_iteration in range_constexpr(HEAD_DIM // WAVE_SIZE):
                        d = lane + dimension_iteration * WAVE_SIZE
                        key_index = (
                            physical_page * k_stride0
                            + kv_head * k_stride1
                            + (d // cache_pack) * k_stride2
                            + in_page * k_stride3
                            + (d % cache_pack) * k_stride4
                        )
                        value_index = (
                            physical_page * v_stride0
                            + kv_head * v_stride1
                            + d * v_stride2
                            + in_page * v_stride3
                        )
                        shared[token_local * HEAD_DIM + d] = valid_token.select(
                            key[key_index].to(elem_type), elem_type(0.0)
                        )
                        shared[
                            TILE_TOKENS * HEAD_DIM + d * TILE_TOKENS + token_local
                        ] = valid_token.select(
                            value[value_index].to(elem_type), elem_type(0.0)
                        )
                fx.rocdl.s_wait_dscnt(0)
                fx.rocdl.s_barrier()

            if wave == 0:
                if const_expr(schedule == "direct"):
                    key_token = tile_i32 + lane % 16
                    key_page = key_token // page_size
                    key_in_page = key_token - key_page * page_size
                    key_physical = fx.Int32(block_tables[seq * table_stride + key_page])
                    for k_fragment in range_constexpr(K_FRAGMENTS):
                        d = k_fragment * 16 + (lane // 16) * 8
                        key_index = (
                            key_physical * k_stride0
                            + kv_head * k_stride1
                            + (d // cache_pack) * k_stride2
                            + key_in_page * k_stride3
                            + (d % cache_pack) * k_stride4
                        )
                        prefetch_key(k_fragment, key_index)
                    if const_expr(is_fp8):
                        for k_fragment in range_constexpr(K_FRAGMENTS):
                            key_fragments[k_fragment].store(
                                _dequant_fp8x8(
                                    fx.Vector(raw_key_fragments[k_fragment].load()),
                                    k_scale,
                                    elem_type,
                                    is_fp8fnuz=is_fp8fnuz,
                                )
                            )
                else:
                    key_token_local = lane % 16
                    for k_fragment in range_constexpr(K_FRAGMENTS):
                        for element in range_constexpr(8):
                            d = k_fragment * 16 + (lane // 16) * 8 + element
                            key_fragments[k_fragment][element] = shared[
                                key_token_local * HEAD_DIM + d
                            ]

                score_fragment.fill(0.0)
                for k_fragment in range_constexpr(K_FRAGMENTS):
                    fx.mma_atom_call(
                        mma_atom,
                        score_fragment,
                        q_fragments[k_fragment],
                        key_fragments[k_fragment],
                        score_fragment,
                    )
                score_values = fx.Vector(score_fragment.load())
                score_row_base = (lane // 16) * 8
                score_column = lane % 16
                for score_row in range_constexpr(8):
                    scores[score_row_base + score_row, score_column] = (
                        score_values[score_row] * softmax_scale
                    )
            fx.rocdl.s_wait_dscnt(0)
            fx.rocdl.s_barrier()

            if const_expr(schedule == "lds"):
                for output_fragment in range_constexpr(OUTPUT_FRAGMENTS):
                    d = wave * 16 + output_fragment * 64 + lane % 16
                    for element in range_constexpr(8):
                        value_fragments[output_fragment][element] = shared[
                            TILE_TOKENS * HEAD_DIM
                            + d * TILE_TOKENS
                            + (lane // 16) * 8
                            + element
                        ]
            elif const_expr(is_fp8):
                for output_fragment in range_constexpr(OUTPUT_FRAGMENTS):
                    value_fragments[output_fragment].store(
                        _dequant_fp8x8(
                            fx.Vector(raw_value_fragments[output_fragment].load()),
                            v_scale,
                            elem_type,
                            is_fp8fnuz=is_fp8fnuz,
                        )
                    )

            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            local_max = neg_inf
            row = lane % QUERY_ROWS
            token_base = (lane // QUERY_ROWS) * 8
            score_values = []
            for token_inner in range_constexpr(8):
                token_local = token_base + token_inner
                valid = tile_i32 + token_local < end
                score = valid.select(fx.Float32(scores[row, token_local]), neg_inf)
                score_values.append(score)
                local_max = fx.max(local_max, score)
            tile_max = fx.max(
                local_max, fx.Float32(gpu.shuffle_xor(local_max, 16, WAVE_SIZE))
            )
            next_max = fx.max(running_max, tile_max)
            alpha = fmath.exp2((running_max - next_max) * LOG2E)
            local_sum = zero
            for token_inner in range_constexpr(8):
                probability = fmath.exp2((score_values[token_inner] - next_max) * LOG2E)
                probability_fragment[token_inner] = probability.to(elem_type)
                local_sum = local_sum + probability
            tile_sum = local_sum + fx.Float32(gpu.shuffle_xor(local_sum, 16, WAVE_SIZE))
            next_sum = running_sum * alpha + tile_sum

            if const_expr(schedule == "direct"):
                if tile_i32 + TILE_TOKENS > end:
                    for output_fragment in range_constexpr(OUTPUT_FRAGMENTS):
                        for element in range_constexpr(8):
                            value_valid = token + element < end
                            value_fragments[output_fragment][element] = (
                                value_valid.select(
                                    value_fragments[output_fragment][element],
                                    elem_type(0.0),
                                )
                            )

            next_accumulators = []
            for output_fragment in range_constexpr(OUTPUT_FRAGMENTS):
                for output_row in range_constexpr(8):
                    accumulator_row = (lane // 16) * 8 + output_row
                    alpha_row = fx.Float32(
                        gpu.shuffle_idx(alpha, accumulator_row, WAVE_SIZE)
                    )
                    output_fragments[output_fragment][output_row] = (
                        fx.Float32(state[2 + output_fragment * 8 + output_row])
                        * alpha_row
                    )
                fx.mma_atom_call(
                    mma_atom,
                    output_fragments[output_fragment],
                    probability_fragment,
                    value_fragments[output_fragment],
                    output_fragments[output_fragment],
                )
                output_values = fx.Vector(output_fragments[output_fragment].load())
                for output_row in range_constexpr(8):
                    next_accumulators.append(output_values[output_row])
            if const_expr(schedule == "direct"):
                gpu.barrier()
            else:
                fx.rocdl.s_wait_dscnt(0)
                fx.rocdl.s_barrier()
            results = yield [next_max, next_sum] + next_accumulators

        has_tokens = end > begin
        final_sum = fx.Float32(results[1])
        output_row_base = (lane // 16) * 8
        for output_fragment in range_constexpr(OUTPUT_FRAGMENTS):
            d = wave * 16 + output_fragment * 64 + lane % 16
            for output_row in range_constexpr(8):
                row = output_row_base + output_row
                q_head = kv_head * QUERY_ROWS + row
                denominator = fx.Float32(gpu.shuffle_idx(final_sum, row, WAVE_SIZE))
                partial = fx.Float32(results[2 + output_fragment * 8 + output_row]) / (
                    denominator + 1.0e-10
                )
                out_index = (
                    seq * mo_stride0 + q_head * mo_stride1 + split * mo_stride2 + d
                )
                mid_out[out_index] = has_tokens.select(partial, zero)
        if (wave == 0) & (lane < QUERY_ROWS):
            q_head = kv_head * QUERY_ROWS + lane
            lse = fx.Float32(results[0]) + fmath.log2(final_sum) / LOG2E
            lse_index = seq * ml_stride0 + q_head * ml_stride1 + split * ml_stride2
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


__all__ = ["compile_native_d128_gqa16_stage"]
