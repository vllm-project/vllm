# SPDX-License-Identifier: Apache-2.0
import functools
import math
from typing import List

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch.library import impl
from torch_xla.experimental.custom_kernel import (XLA_LIB, jax_import_guard,
                                                  make_kernel_from_pallas)

XLA_LIB.define(
    "bgmv_shrink(Tensor inputs, Tensor loras, Tensor idxs) -> Tensor")

# bgmv_expand needs a flag to enable LoRA laning since it expects its inputs to
# be the outputs of a LoRA laned bgmv_shrink. This is not always the case when
# we use bgmv_expand
XLA_LIB.define(
    "bgmv_expand(Tensor inputs, Tensor loras, Tensor idxs, bool enable_laning) -> Tensor"
)
"""
LoRA Laning Optimization for TPU Matrix Multiplication

When we run with the TPU we need to keep its MXU (matrix multiplication unit)
well fed to achieve maximum utilisation.
The MXU can perform an (8x128) by (128x128) matmul once every 8 cycles.

LoRA computations typically take a series of T (1xD) vectors and matmul them
with a (DxL) matrix (shrinking) followed by another matmul with a (LxD) matrix
(expanding). Grouping the vectors we get a (TxD) matrix, so our computations
become matmul((TxD), (DxL)) and matmul((TxL), (LxD)).

The number of tokens (T) and the hidden dimension (D) are usually greater than
8 and 128 respectively, however the LoRA rank (L) is usually a smaller value,
around 8-64, which means we need to pad L to allow it to fit in a TPU register.

                              +------------------+
                              | Shrink Operation |
                              +------------------+

                                       L
                              +------------------+
            D                 | 1111000000000000 |              L
   +------------------+       | 1111000000000000 |     +------------------+
   | 1111111111111111 |       | 1111000000000000 |     | 1111000000000000 |
 T | 2222222222222222 |  x  D | 1111000000000000 | = T | 1111000000000000 |
   +------------------+       | 1111000000000000 |     +------------------+
                              | 1111000000000000 |
                              | 1111000000000000 |
                              | 1111000000000000 |
                              +------------------+

Here we have 4 tokens each needing a different LoRA adapter, and 1 LoRA adapter
loaded into the MXU. After the matmul we end up with the result of applying
LoRA 1 to all T tokens, but since only one token needs LoRA 1, we mask out
everything we don't need to get:

                                       D
                              +------------------+
                              | 1111000000000000 |
                              | 0000000000000000 |
                              +------------------+

However, we need:

                                       L
                              +------------------+
                              | 1111000000000000 |
                              | 2222000000000000 |
                              +------------------+

So we'll have to perform another matmul.
Overall this shrink wastes time and memory padding the LoRA adapters and running
extra matmuls.

We can get both reduce the number of matmuls used and the amount of applied
padding by grouping the LoRA adapters into multiple "lanes".

                                       L
                              +------------------+
            D                 | 1111222200000000 |              L
   +------------------+       | 1111222200000000 |     +------------------+
   | 1111111111111111 |       | 1111222200000000 |     | 1111222200000000 |
 T | 2222222222222222 |  x  D | 1111222200000000 | = T | 1111222200000000 |
   +------------------+       | 1111222200000000 |     +------------------+
                              | 1111222200000000 |
                              | 1111222200000000 |
                              | 1111222200000000 |
                              +------------------+


Now we're able to compute the outputs of 4 different LoRA adapters in the same
8 cycles. However we don't need all these results so we'll again mask out
everything we don't need to get:

                                       L
                              +------------------+
                              | 1111000000000000 |
                              | 0000222200000000 |
                              +------------------+

But now our outputs aren't aligned properly, so we would need to apply an extra
shuffle operation.

                              +------------------+
                              | Expand Operation |
                              +------------------+

When expanding we end up wasting space in both matrix registers.

                                       D
                              +------------------+
            L                 | 1111111111111111 |              D
   +------------------+       | 1111111111111111 |     +------------------+
   | 1111000000000000 |       | 1111111111111111 |     | 1111111111111111 |
 T | 2222000000000000 |  x  L | 1111111111111111 | = T | 1111111111111111 |
   +------------------+       | 0000000000000000 |     +------------------+
                              | 0000000000000000 |
                              | 0000000000000000 |
                              | 0000000000000000 |
                              +------------------+

But, if we use LoRA Laning like before, we can waste less space. We would also 
have to shuffle the input so it applies to the right adapter.

                                       D
                              +------------------+
            L                 | 1111111111111111 |              D
   +------------------+       | 1111111111111111 |     +------------------+
   | 1111000000000000 |       | 1111111111111111 |     | 1111111111111111 |
 T | 0000222200000000 |  x  L | 1111111111111111 | = T | 2222222222222222 |
   +------------------+       | 2222222222222222 |     +------------------+
                              | 2222222222222222 |
                              | 2222222222222222 |
                              | 2222222222222222 |
                              +------------------+

Since this shuffling is the exact opposite of the operation we do at the end of
the Shrink operation, we can skip both shuffles.

"""


def _bgmv_shrink_kernel(bT: int, bL: int, n_lora_lanes: int, lane_size: int,
                        max_num_loras: int, idx_ref, inp_ref, lora_ref,
                        out_ref, acc_ref, mask_ref):

    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    if max_num_loras == 1 and n_lora_lanes == 1:
        acc_ref[...] += jax.lax.dot_general(inp_ref[...],
                                            lora_ref[0, ...],
                                            (((1, ), (1, )), ((), ())),
                                            preferred_element_type=jnp.float32)
    else:
        t = pl.program_id(0)

        ones = jnp.ones((lane_size, ), dtype=jnp.float32)

        base_lora_idx = 0
        for lane_idx in range(max_num_loras):
            mask_ref[...] = jnp.zeros_like(mask_ref[...], dtype=jnp.float32)
            valid = False
            for j in range(bT):
                idx = idx_ref[j + bT * t]
                for k in range(n_lora_lanes):
                    lora_idx = base_lora_idx + k
                    set_mask = idx == lora_idx
                    valid |= set_mask

                    @pl.when(set_mask)
                    def _():
                        lane_start = k * lane_size
                        lane_end = lane_start + lane_size

                        mask_ref.at[j, lane_start:lane_end].set(ones)

            base_lora_idx += n_lora_lanes

            @pl.when(valid)
            def _():
                acc_ref[...] += jax.lax.dot_general(
                    inp_ref[...],
                    lora_ref[lane_idx, ...], (((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32) * mask_ref[...]

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit,
                   static_argnames=[
                       "TOKEN_BLOCK", "LORA_BLOCK", "DIM_BLOCK",
                       "N_LORA_LANES", "LANE_SIZE"
                   ])
def _bgmv_shrink(
        idxs: jax.Array,  # (T, ) int32
        inputs: jax.Array,  # (T, D) model dtype
        loras: jax.Array,  # (N, L, D) model dtype
        *,
        TOKEN_BLOCK: int,
        LORA_BLOCK: int,
        DIM_BLOCK: int,
        N_LORA_LANES: int,
        LANE_SIZE: int) -> jax.Array:  # (T, L) model dtype
    T, D = inputs.shape
    N, L, _ = loras.shape

    return pl.pallas_call(
        kernel=functools.partial(_bgmv_shrink_kernel, TOKEN_BLOCK, LORA_BLOCK,
                                 N_LORA_LANES, LANE_SIZE, N),
        out_shape=jax.ShapeDtypeStruct((T, L), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T // TOKEN_BLOCK, L // LORA_BLOCK, D // DIM_BLOCK),
            in_specs=[
                pl.BlockSpec((TOKEN_BLOCK, DIM_BLOCK),
                             lambda i, j, k, block_idx: (i, k)),
                pl.BlockSpec((N, LORA_BLOCK, DIM_BLOCK),
                             lambda i, j, k, block_idx: (0, j, k)),
            ],
            out_specs=pl.BlockSpec((TOKEN_BLOCK, LORA_BLOCK),
                                   lambda i, j, k, block_idx: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32),
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32)
            ]),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="bgmv")(idxs, inputs, loras)


def bgmv_shrink_shape_function(idxs, inputs, loras):
    T, _ = inputs.shape
    _, L, _ = loras.shape

    return [((T, L), inputs.dtype)]


@impl(XLA_LIB, "bgmv_shrink", "XLA")
def bgmv_shrink_xla(inputs: torch.Tensor, loras: torch.Tensor,
                    idxs: torch.IntTensor):
    inputs = inputs.to(dtype=loras.dtype)

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    T, _ = inputs.shape
    N, L, D = loras.shape

    TOKEN_BLOCK = get_bounded_value(16, next_multiple_of(T, 16), 128)
    LORA_BLOCK = 256
    DIM_BLOCK = largest_divisor(D, [256, 512, 1024])

    # See if we can fit multiple LoRAs in a register. This would activate LoRA
    # laning
    N_LORA_LANES = math.ceil(LORA_BLOCK / L)
    LANE_SIZE = min(L, LORA_BLOCK)
    if N_LORA_LANES > 1 and N > 1:
        pad_N = next_multiple_of(N, N_LORA_LANES) - N
        new_N = N + pad_N

        loras = torch.nn.functional.pad(loras, (0, 0, 0, 0, 0, pad_N))
        loras = loras.reshape((new_N // N_LORA_LANES, LORA_BLOCK, D))
        N, L, D = loras.shape

    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU
    # register. This has to happen in pytorch, doing it in Jax will lead to NaNs
    pad_L = 0
    if LORA_BLOCK > L or L % LORA_BLOCK != 0:
        pad_L = next_multiple_of(L, LORA_BLOCK) - L
    pad_D = 0
    if DIM_BLOCK > D or D % DIM_BLOCK != 0:
        pad_D = next_multiple_of(D, DIM_BLOCK) - D

    pad_T = 0
    if TOKEN_BLOCK > T or T % TOKEN_BLOCK != 0:
        pad_T = next_multiple_of(T, TOKEN_BLOCK) - T

    if pad_D != 0 or pad_L != 0:
        loras = torch.nn.functional.pad(loras, (0, pad_D, 0, pad_L, 0, 0))
    if pad_D != 0 or pad_T != 0:
        inputs = torch.nn.functional.pad(inputs, (0, pad_D, 0, pad_T))
        if pad_T != T:
            idxs = torch.nn.functional.pad(idxs, ((0, pad_T)))

    jax_import_guard()
    kernel = make_kernel_from_pallas(
        functools.partial(_bgmv_shrink,
                          TOKEN_BLOCK=TOKEN_BLOCK,
                          LORA_BLOCK=LORA_BLOCK,
                          DIM_BLOCK=DIM_BLOCK,
                          N_LORA_LANES=N_LORA_LANES,
                          LANE_SIZE=LANE_SIZE), bgmv_shrink_shape_function)

    return kernel(idxs, inputs, loras)[:T, :L]


@impl(XLA_LIB, "bgmv_shrink", "CompositeExplicitAutograd")
def bgmv_shrink_non_xla(inputs: torch.Tensor, loras: torch.Tensor,
                        idxs: torch.IntTensor):
    T, _ = inputs.shape

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    N, L, _ = loras.shape

    LORA_BLOCK = 256
    N_LORA_LANES = math.ceil(LORA_BLOCK / L)
    if N_LORA_LANES > 1 and N > 1:
        L = LORA_BLOCK

    return torch.empty((T, L), device=inputs.device)


# This kernel is similar to the one above but it assumes that the LoRA adapters
# have been pre-transposed. This lets us skip the data copies involved in
# transposing.
# We only need this for the expand op since the LoRA dimensions in the shrink op
# are small enough that the TPU can gather them without a data copy.
def _bgmv_expand_kernel(bT: int, bL: int, max_num_loras: int, idx_ref, inp_ref,
                        lora_ref, out_ref, acc_ref, mask_ref):

    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    if max_num_loras == 1:
        acc_ref[...] += jax.lax.dot(inp_ref[...],
                                    lora_ref[0, ...],
                                    preferred_element_type=jnp.float32)
    else:
        t = pl.program_id(0)

        ones = jnp.ones((bL, ), dtype=jnp.float32)

        for i in range(max_num_loras):
            mask_ref[...] = jnp.zeros_like(mask_ref[...], dtype=jnp.float32)
            valid = False
            for j in range(bT):
                valid |= idx_ref[j + bT * t] == i

                @pl.when(idx_ref[j + bT * t] == i)
                def _():
                    mask_ref.at[j, :].set(ones)

            @pl.when(valid)
            def _():
                acc_ref[...] += jax.lax.dot(
                    inp_ref[...],
                    lora_ref[i, ...],
                    preferred_element_type=jnp.float32) * mask_ref[...]

        @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
        def _():
            out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit,
                   static_argnames=["TOKEN_BLOCK", "LORA_BLOCK", "DIM_BLOCK"])
def _bgmv_expand(
        idxs: jax.Array,  # (T, ) int32
        inputs: jax.Array,  # (T, D) model dtype
        loras: jax.Array,  # (N, L, D) model dtype
        *,
        TOKEN_BLOCK: int,
        LORA_BLOCK: int,
        DIM_BLOCK: int) -> jax.Array:  # (T, L) model dtype
    T, D = inputs.shape
    N, _, L = loras.shape

    return pl.pallas_call(
        kernel=functools.partial(_bgmv_expand_kernel, TOKEN_BLOCK, LORA_BLOCK,
                                 N),
        out_shape=jax.ShapeDtypeStruct((T, L), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T // TOKEN_BLOCK, L // LORA_BLOCK, D // DIM_BLOCK),
            in_specs=[
                pl.BlockSpec((TOKEN_BLOCK, DIM_BLOCK),
                             lambda i, j, k, block_idx: (i, k)),
                pl.BlockSpec((N, DIM_BLOCK, LORA_BLOCK),
                             lambda i, j, k, block_idx: (0, k, j)),
            ],
            out_specs=pl.BlockSpec((TOKEN_BLOCK, LORA_BLOCK),
                                   lambda i, j, k, block_idx: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32),
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32)
            ]),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="bgmv_pre_transpose")(idxs, inputs, loras)


def bgmv_expand_shape_function(idxs, inputs, loras):
    T, _ = inputs.shape
    _, _, L = loras.shape

    return [((T, L), inputs.dtype)]


@impl(XLA_LIB, "bgmv_expand", "XLA")
def bgmv_expand_xla(inputs: torch.Tensor, loras: torch.Tensor,
                    idxs: torch.IntTensor, enable_laning: bool):
    inputs = inputs.to(dtype=loras.dtype)

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    T, _ = inputs.shape
    N, D, L = loras.shape

    TOKEN_BLOCK = get_bounded_value(16, next_multiple_of(T, 16), 128)
    LORA_BLOCK = largest_divisor(L, [256, 512, 1024])
    DIM_BLOCK = 256

    # See if we can fit multiple LoRAs in a register. This would activate LoRA
    # laning
    N_LORA_LANES = math.ceil(DIM_BLOCK / D)
    if enable_laning and N_LORA_LANES > 1 and N > 1:
        pad_N = next_multiple_of(N, N_LORA_LANES) - N
        new_N = N + pad_N

        loras = torch.nn.functional.pad(loras, (0, 0, 0, 0, 0, pad_N))
        loras = loras.reshape((new_N // N_LORA_LANES, DIM_BLOCK, L))
        idxs = idxs // N_LORA_LANES
        N, D, L = loras.shape

    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU
    # register. This has to happen in pytorch, doing it in Jax will lead to NaNs
    pad_L = 0
    if LORA_BLOCK > L or L % LORA_BLOCK != 0:
        pad_L = next_multiple_of(L, LORA_BLOCK) - L

    pad_D = 0
    if DIM_BLOCK > D or D % DIM_BLOCK != 0:
        pad_D = next_multiple_of(D, DIM_BLOCK) - D

    pad_T = 0
    if TOKEN_BLOCK > T or T % TOKEN_BLOCK != 0:
        pad_T = next_multiple_of(T, TOKEN_BLOCK) - T

    if pad_D != 0 or pad_L != 0:
        loras = torch.nn.functional.pad(loras, (0, pad_L, 0, pad_D, 0, 0))
    if pad_D != 0 or pad_T != 0:
        inputs = torch.nn.functional.pad(inputs, (0, pad_D, 0, pad_T))
        if pad_T != T:
            idxs = torch.nn.functional.pad(idxs, ((0, pad_T)))

    jax_import_guard()

    kernel = make_kernel_from_pallas(
        functools.partial(_bgmv_expand,
                          TOKEN_BLOCK=TOKEN_BLOCK,
                          LORA_BLOCK=LORA_BLOCK,
                          DIM_BLOCK=DIM_BLOCK), bgmv_expand_shape_function)

    return kernel(idxs, inputs, loras)[:T, :L]


@impl(XLA_LIB, "bgmv_expand", "CompositeExplicitAutograd")
def bgmv_expand_non_xla(inputs: torch.Tensor, loras: torch.Tensor,
                        idxs: torch.IntTensor, enable_laning: bool):
    T, _ = inputs.shape

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    _, _, L = loras.shape

    return torch.empty((T, L), device=inputs.device)


def largest_divisor(n: int, divs: List[int]) -> int:
    for div in sorted(divs, reverse=True):
        if n % div == 0:
            return div
    return max(divs)


def next_multiple_of(n: int, mult: int) -> int:
    return math.ceil(n / mult) * mult


def get_bounded_value(_min: int, val: int, _max: int) -> int:
    return min(max(_min, val), _max)
