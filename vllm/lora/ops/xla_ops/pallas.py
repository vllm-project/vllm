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
XLA_LIB.define(
    "bgmv_expand(Tensor inputs, Tensor loras, Tensor idxs) -> Tensor")


def _bgmv_shrink_kernel(bT: int, bL: int, max_num_loras: int, idx_ref, inp_ref,
                        lora_ref, out_ref, acc_ref, mask_ref):

    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

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
            acc_ref[...] += jax.lax.dot_general(
                inp_ref[...],
                lora_ref[i, ...], (((1, ), (1, )), ((), ())),
                preferred_element_type=jnp.float32) * mask_ref[...]

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit,
                   static_argnames=["TOKEN_BLOCK", "LORA_BLOCK", "DIM_BLOCK"])
def _bgmv_shrink(
        idxs: jax.Array,  # (T, ) int32
        inputs: jax.Array,  # (T, D) model dtype
        loras: jax.Array,  # (N, L, D) model dtype
        *,
        TOKEN_BLOCK: int,
        LORA_BLOCK: int,
        DIM_BLOCK: int) -> jax.Array:  # (T, L) model dtype
    T, D = inputs.shape
    N, L, _ = loras.shape

    return pl.pallas_call(
        kernel=functools.partial(_bgmv_shrink_kernel, TOKEN_BLOCK, LORA_BLOCK,
                                 N),
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
    _, L, D = loras.shape
    is_expand = L > D

    jax_import_guard()

    TOKEN_BLOCK = get_bounded_value(16, next_multiple_of(T, 16), 128)
    LORA_BLOCK = 256
    DIM_BLOCK = min(1024, next_multiple_of(D, 256))

    kernel = make_kernel_from_pallas(
        functools.partial(_bgmv_shrink,
                          TOKEN_BLOCK=TOKEN_BLOCK,
                          LORA_BLOCK=LORA_BLOCK,
                          DIM_BLOCK=DIM_BLOCK), bgmv_shrink_shape_function)

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

    return kernel(idxs, inputs, loras)[:T, :L]


@impl(XLA_LIB, "bgmv_shrink", "CompositeExplicitAutograd")
def bgmv_shrink_non_xla(inputs: torch.Tensor, loras: torch.Tensor,
                        idxs: torch.IntTensor):
    T, _ = inputs.shape

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    _, L, _ = loras.shape

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
                    idxs: torch.IntTensor):
    inputs = inputs.to(dtype=loras.dtype)

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    T, _ = inputs.shape
    _, D, L = loras.shape

    jax_import_guard()

    TOKEN_BLOCK = get_bounded_value(16, next_multiple_of(T, 16), 128)
    LORA_BLOCK = min(1024, next_multiple_of(L, 256))
    DIM_BLOCK = 256

    kernel = make_kernel_from_pallas(
        functools.partial(_bgmv_expand,
                          TOKEN_BLOCK=TOKEN_BLOCK,
                          LORA_BLOCK=LORA_BLOCK,
                          DIM_BLOCK=DIM_BLOCK), bgmv_expand_shape_function)

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

    return kernel(idxs, inputs, loras)[:T, :L]


@impl(XLA_LIB, "bgmv_expand", "CompositeExplicitAutograd")
def bgmv_expand_non_xla(inputs: torch.Tensor, loras: torch.Tensor,
                        idxs: torch.IntTensor):
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
