# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Eight-wave per-query-head SplitKV stage for RDNA4."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from .rdna4_splitkv_common import LOG2E, WAVE_SIZE, _decode_fp8, _wave_reduce

NUM_WAVES = 8
BLOCK_THREADS = NUM_WAVES * WAVE_SIZE


@functools.lru_cache(maxsize=128)
def compile_wave8_stage(
    *,
    query_dtype: str,
    kv_dtype: str,
    head_dim: int,
    splits: int,
    num_query_heads: int,
    num_kv_heads: int,
    query_group_size: int,
    page_size: int,
    softmax_scale: float,
    layout_key: tuple = (),
):
    """Compile a scalar-FP32, eight-wave token-partition stage."""

    if query_dtype not in ("bf16", "fp16"):
        raise ValueError(f"unsupported query dtype {query_dtype!r}")
    if kv_dtype not in ("bf16", "fp16", "fp8", "fp8fnuz"):
        raise ValueError(f"unsupported KV dtype {kv_dtype!r}")
    if head_dim not in (128, 256):
        raise ValueError(f"head_dim must be 128 or 256, got {head_dim}")
    if splits < 1:
        raise ValueError(f"splits must be positive, got {splits}")
    if num_query_heads < 1 or num_kv_heads < 1:
        raise ValueError("query and KV head counts must be positive")
    if not 1 <= query_group_size <= 16:
        raise ValueError(f"query_group_size must be in [1, 16], got {query_group_size}")
    if page_size < 8 or page_size % 8:
        raise ValueError(f"page_size must be a positive multiple of 8, got {page_size}")
    if not isinstance(layout_key, tuple):
        raise TypeError("layout_key must be a tuple")

    is_fp8 = kv_dtype in ("fp8", "fp8fnuz")
    is_fp8fnuz = kv_dtype == "fp8fnuz"
    values_per_lane = head_dim // WAVE_SIZE
    lds_values = NUM_WAVES * head_dim + 3 * NUM_WAVES

    @fx.struct
    class SharedStorage:
        values: fx.Array[fx.Float32, lds_values, 16]

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
        output_ptr: fx.Tensor,
        split_counters_ptr: fx.Tensor,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        query_head = fx.Int32(gpu.block_idx.x)
        split = fx.Int32(gpu.block_idx.y)
        seq = fx.Int32(gpu.block_idx.z)
        kv_head = query_head // query_group_size

        raw_key = fx.make_view(
            fx.recast_iter(fx.Uint8, fx.get_iter(key_ptr)),
            key_ptr.layout,
        )
        raw_value = fx.make_view(
            fx.recast_iter(fx.Uint8, fx.get_iter(value_ptr)),
            value_ptr.layout,
        )

        k_scale = fx.Float32(1.0)
        v_scale = fx.Float32(1.0)
        if const_expr(is_fp8):
            k_scale = fx.Float32(k_scale_ptr[0])
            v_scale = fx.Float32(v_scale_ptr[0])

        query_row = fx.Int32(query_start_loc_ptr[seq])
        query_end = fx.Int32(query_start_loc_ptr[seq + 1])
        is_decode = query_end - query_row == 1
        length = is_decode.select(fx.Int32(seq_lens_ptr[seq]), fx.Int32(0))
        tokens_per_split = (length + splits - 1) // splits
        begin = fx.min(split * tokens_per_split, length)
        end = fx.min(begin + tokens_per_split, length)

        query_loads = []
        for element in range_constexpr(values_per_lane):
            d = lane + element * WAVE_SIZE
            query_loads.append(query_ptr[query_row, query_head, d])
        query_values = []
        for element in range_constexpr(values_per_lane):
            query_value = is_decode.select(
                fx.Float32(query_loads[element]), fx.Float32(0.0)
            )
            query_values.append(query_value * k_scale)

        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)

        def issue_token_loads(token):
            token_i32 = fx.Int32(token)
            logical_page = token_i32 // page_size
            in_page = token_i32 - logical_page * page_size
            physical_page = fx.Int32(0)
            if lane == 0:
                physical_page = fx.Int32(block_tables_ptr[seq, logical_page])
            physical_page = fx.Int32(gpu.shuffle_idx(physical_page, 0, WAVE_SIZE))
            loads = []
            for element in range_constexpr(values_per_lane):
                d = lane + element * WAVE_SIZE
                key_index = (
                    physical_page,
                    kv_head,
                    d // (16 if is_fp8 else 8),
                    in_page,
                    d % (16 if is_fp8 else 8),
                )
                if const_expr(is_fp8):
                    loads.append(raw_key[key_index])
                else:
                    loads.append(key_ptr[key_index])
            for element in range_constexpr(values_per_lane):
                d = lane + element * WAVE_SIZE
                value_index = (physical_page, kv_head, d, in_page)
                if const_expr(is_fp8):
                    loads.append(raw_value[value_index])
                else:
                    loads.append(value_ptr[value_index])
            return loads

        def update_online_state(loads, state):
            dot = zero
            for element in range_constexpr(values_per_lane):
                if const_expr(is_fp8):
                    key_value = fx.Float32(
                        _decode_fp8(fx.Int32(loads[element]), 0, is_fp8fnuz=is_fp8fnuz)
                    )
                else:
                    key_value = fx.Float32(loads[element])
                dot = dot + query_values[element] * key_value
            score = (
                fx.Float32(gpu.shuffle_idx(_wave_reduce(dot, "sum"), 0, WAVE_SIZE))
                * softmax_scale
            )
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            next_max = fx.max(running_max, score)
            old_scale = fmath.isfinite(running_max).select(
                fmath.exp2((running_max - next_max) * LOG2E), zero
            )
            weight = fmath.exp2((score - next_max) * LOG2E)
            next_accumulators = []
            for element in range_constexpr(values_per_lane):
                if const_expr(is_fp8):
                    value_element = fx.Float32(
                        _decode_fp8(
                            fx.Int32(loads[values_per_lane + element]),
                            0,
                            is_fp8fnuz=is_fp8fnuz,
                        )
                    )
                else:
                    value_element = fx.Float32(loads[values_per_lane + element])
                next_accumulators.append(
                    fx.Float32(state[2 + element]) * old_scale + weight * value_element
                )
            return [
                next_max,
                running_sum * old_scale + weight,
            ] + next_accumulators

        init_state = [neg_inf, zero] + [zero for _ in range_constexpr(values_per_lane)]
        for token, state in range(  # type: ignore[call-overload]
            begin + wave, end, fx.Int32(NUM_WAVES), init=init_state
        ):
            results = yield update_online_state(issue_token_loads(token), state)

        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        partial = fx.make_view(
            storage.values.ptr,
            fx.make_layout((NUM_WAVES, head_dim), (head_dim, 1)),
        )
        state_max = fx.make_view(
            fx.add_offset(storage.values.ptr, NUM_WAVES * head_dim),
            fx.make_layout(NUM_WAVES, 1),
        )
        state_sum = fx.make_view(
            fx.add_offset(storage.values.ptr, NUM_WAVES * head_dim + NUM_WAVES),
            fx.make_layout(NUM_WAVES, 1),
        )
        state_weight = fx.make_view(
            fx.add_offset(storage.values.ptr, NUM_WAVES * head_dim + 2 * NUM_WAVES),
            fx.make_layout(NUM_WAVES, 1),
        )
        for element in range_constexpr(values_per_lane):
            partial[wave, lane + element * WAVE_SIZE] = fx.Float32(results[2 + element])
        if lane == 0:
            state_max[wave] = fx.Float32(results[0])
            state_sum[wave] = fx.Float32(results[1])
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()

        if tid == 0:
            merged_max = neg_inf
            for source_wave in range_constexpr(NUM_WAVES):
                merged_max = fx.max(merged_max, fx.Float32(state_max[source_wave]))
            merged_sum = zero
            for source_wave in range_constexpr(NUM_WAVES):
                source_max = fx.Float32(state_max[source_wave])
                merge_weight = fmath.isfinite(source_max).select(
                    fmath.exp2((source_max - merged_max) * LOG2E), zero
                )
                state_weight[source_wave] = merge_weight
                merged_sum = (
                    merged_sum + fx.Float32(state_sum[source_wave]) * merge_weight
                )
            state_max[0] = merged_max
            state_sum[0] = merged_sum
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()

        for output_element in range_constexpr(
            (head_dim + BLOCK_THREADS - 1) // BLOCK_THREADS
        ):
            d = tid + output_element * BLOCK_THREADS
            if d < head_dim:
                merged_value = zero
                for source_wave in range_constexpr(NUM_WAVES):
                    merged_value = merged_value + fx.Float32(
                        state_weight[source_wave]
                    ) * fx.Float32(partial[source_wave, d])
                merged_sum = fx.Float32(state_sum[0])
                mid_out_ptr[seq, query_head, split, d] = (merged_sum > 0.0).select(
                    merged_value / merged_sum * v_scale, zero
                )
        if tid == 0:
            merged_sum = fx.Float32(state_sum[0])
            lse = fx.Float32(state_max[0]) + fmath.log2(merged_sum) / LOG2E
            mid_lse_ptr[seq, query_head, split] = (merged_sum > 0.0).select(
                lse, neg_inf
            )

        gpu.barrier()
        old = fx.Int32(-1)
        if tid == 0:
            fx.llvm.memory_fence(
                ordering=fx.AtomicOrdering.Release,
                syncscope=fx.rocdl.SyncScope.Agent,
            )
            counter_index = (seq * num_query_heads + query_head) * 16
            counter_ptr = fx.add_offset(fx.get_iter(split_counters_ptr), counter_index)
            old = fx.llvm.atomic_add(
                counter_ptr,
                fx.Int32(1),
                ordering=fx.AtomicOrdering.Monotonic,
                syncscope=fx.rocdl.SyncScope.Agent,
            )
            state_weight[0] = fx.Float32(old)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        old = fx.Int32(state_weight[0])

        if old == splits - 1:
            if tid == 0:
                fx.llvm.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
                counter_index = (seq * num_query_heads + query_head) * 16
                counter_ptr = fx.add_offset(
                    fx.get_iter(split_counters_ptr), counter_index
                )
                fx.llvm.generic_store(
                    counter_ptr,
                    fx.Int32(0),
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
            gpu.barrier()

            if (wave == 0) & is_decode:
                max_lse = neg_inf
                for source_split in range_constexpr(splits):
                    max_lse = fx.max(
                        max_lse,
                        fx.Float32(mid_lse_ptr[seq, query_head, source_split]),
                    )
                denominator = zero
                result = [zero for _ in range_constexpr(values_per_lane)]
                for source_split in range_constexpr(splits):
                    partial_lse = fx.Float32(mid_lse_ptr[seq, query_head, source_split])
                    weight = fmath.isfinite(partial_lse).select(
                        fmath.exp2((partial_lse - max_lse) * LOG2E),
                        zero,
                    )
                    denominator = denominator + weight
                    for element in range_constexpr(values_per_lane):
                        d = lane + element * WAVE_SIZE
                        result[element] = result[element] + weight * fx.Float32(
                            mid_out_ptr[seq, query_head, source_split, d]
                        )
                for element in range_constexpr(values_per_lane):
                    d = lane + element * WAVE_SIZE
                    normalized = (denominator > 0.0).select(
                        result[element] / denominator, zero
                    )
                    if const_expr(query_dtype == "fp16"):
                        output_ptr[query_row, query_head, d] = normalized.to(fx.Float16)
                    else:
                        output_ptr[query_row, query_head, d] = normalized.to(
                            fx.BFloat16
                        )

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
        output: fx.Tensor,
        split_counters: fx.Tensor,
        batch: fx.Int32,
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
            output,
            split_counters,
        ).launch(
            grid=(num_query_heads, splits, batch),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["compile_wave8_stage"]
