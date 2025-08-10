# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm.utils import cdiv


def _kv_cache_update_kernel(
    # Prefetch
    slices_ref,  # [3, padded_num_slices], list of (kv_cache_start,
    # new_kv_start, slice_len)
    num_slices_ref,  # [1]
    # Input
    new_kv_hbm_ref,  # [num_tokens, num_combined_kv_heads, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    # Output
    _,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    # Scratch
    scratch,  # [num_slices_per_block, page_size, num_combined_kv_heads,
    # head_dim]
    sem,
):
    async_copies = []
    block_idx = pl.program_id(0)
    num_slices_per_block = scratch.shape[0]

    # Copy from new_kv_hbm_ref to scratch
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        new_kv_start = jax.lax.select(offset_i < num_slices_ref[0],
                                      slices_ref[1, offset_i], 0)
        length = jax.lax.select(offset_i < num_slices_ref[0],
                                slices_ref[2, offset_i], 0)
        async_copy = pltpu.make_async_copy(
            new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
            scratch.at[i, pl.ds(0, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)

    for async_copy in async_copies:
        async_copy.wait()

    # Copy from scratch to kv_cache_hbm_ref
    async_copies.clear()
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = jax.lax.select(offset_i < num_slices_ref[0],
                                        slices_ref[0, offset_i], 0)
        length = jax.lax.select(offset_i < num_slices_ref[0],
                                slices_ref[2, offset_i], 0)
        async_copy = pltpu.make_async_copy(
            scratch.at[i, pl.ds(0, length), ...],
            kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)
    for async_copy in async_copies:
        async_copy.wait()


@functools.partial(
    jax.jit,
    static_argnames=["page_size", "num_slices_per_block"],
)
def kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_combined_kv_heads, head_dim]
    slices: jax.
    Array,  # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    kv_cache: jax.
    Array,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    num_kv_update_slices: jax.Array,  # [1]
    *,
    page_size: int = 32,
    num_slices_per_block: int = 8,
):
    _, num_combined_kv_heads, head_dim = new_kv.shape
    assert kv_cache.shape[1] == num_combined_kv_heads
    assert kv_cache.shape[2] == head_dim
    assert head_dim % 128 == 0
    # TODO: Add dynamic check to make sure that the all the slice lengths are
    # smaller or equal to page_size

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
    ]

    out_specs = [pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)]
    out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

    scalar_prefetches = [slices, num_kv_update_slices]
    scratch = pltpu.VMEM(
        (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
        new_kv.dtype,
    )

    scratch_shapes = [
        scratch,
        pltpu.SemaphoreType.DMA,
    ]

    kernel = pl.pallas_call(
        _kv_cache_update_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(cdiv(num_kv_update_slices[0], num_slices_per_block), ),
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        input_output_aliases={len(scalar_prefetches) + 1: 0},
    )

    return kernel(*scalar_prefetches, new_kv, kv_cache)[0]
