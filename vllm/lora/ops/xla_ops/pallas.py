# SPDX-License-Identifier: Apache-2.0
import functools

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch.library import impl
from torch_xla.experimental.custom_kernel import (XLA_LIB, jax_import_guard,
                                                  make_kernel_from_pallas)

def _bgmv_kernel(bT: int, bL: int, max_num_loras: int, idx_ref, inp_ref, lora_ref, out_ref,
                 acc_ref, mask_ref):
    t = pl.program_id(0)
    d = pl.program_id(2)
    ds = pl.num_programs(2)

    @pl.when(d == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    for i in range(max_num_loras):
        mask_ref[...] = jnp.zeros_like(mask_ref[...], dtype=jnp.float32)
        valid = False
        for j in range(bT):
            valid |= idx_ref[j + bT * t] == i

            @pl.when(idx_ref[j + bT * t] == i)
            def _():
                mask_ref[j, :] = jnp.ones((bL, ), dtype=jnp.float32)

        @pl.when(valid)
        def _():
            acc_ref[...] += jax.lax.dot_general(
                inp_ref[...],
                lora_ref[i, ...], (((1, ), (1, )), ((), ())),
                preferred_element_type=jnp.float32) * mask_ref[...]

    @pl.when(d == ds - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=["TOKEN_BLOCK", "LORA_BLOCK", "DIM_BLOCK"])
def _bgmv(
    idxs: jax.Array,  # (T, ) int32
    inputs: jax.Array,  # (T, D) model dtype
    loras: jax.Array,  # (N, L, D) model dtype
    *,
    TOKEN_BLOCK: int,
    LORA_BLOCK: int,
    DIM_BLOCK: int
) -> jax.Array:  # (T, L) model dtype
    T, D = inputs.shape
    N, L, _ = loras.shape

    return pl.pallas_call(
        kernel=functools.partial(_bgmv_kernel, TOKEN_BLOCK, LORA_BLOCK, N),
        out_shape=jax.ShapeDtypeStruct((T, L), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T // TOKEN_BLOCK, L // LORA_BLOCK,
                  D // DIM_BLOCK),
            in_specs=[
                pl.BlockSpec((TOKEN_BLOCK, DIM_BLOCK),
                             lambda t, l, d, block_idx: (t, d)),
                pl.BlockSpec((N, LORA_BLOCK, DIM_BLOCK),
                             lambda t, l, d, block_idx: (0, l, d)),
            ],
            out_specs=pl.BlockSpec((TOKEN_BLOCK, LORA_BLOCK),
                                   lambda t, l, d, block_idx: (t, l)),
            scratch_shapes=[
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32),
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32)
            ]),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="bgmv")(idxs, inputs, loras)


def _fused_bgmv_shrink_expand_kernel(
    bT: int, bL: int, bD: int, max_num_loras: int,
    idx_ref, inp_ref, lora_a_ref, lora_b_ref, out_ref, cache_ref, acc_ref, mask_a_ref, mask_b_ref):

    t = pl.program_id(0)
    d1 = pl.program_id(1)
    l = pl.program_id(2)
    d2 = pl.program_id(3)

    ls = pl.num_programs(2)
    ds = pl.num_programs(3)

    should_compute_cache = d1 == 0
    @pl.when(should_compute_cache)
    def _():
        @pl.when(d2 == 0)
        def _():
            cache_ref[l, ...] = jnp.zeros_like(cache_ref[l, ...], dtype=jnp.float32)

        for i in range(max_num_loras):
            mask_a_ref[...] = jnp.zeros_like(mask_a_ref[...], dtype=jnp.float32)
            valid = False
            for j in range(bT):
                valid |= idx_ref[j + bT * t] == i

                @pl.when(idx_ref[j + bT * t] == i)
                def _():
                    mask_a_ref[j, :] = jnp.ones((bL, ), dtype=jnp.float32)

            @pl.when(valid)
            def _():
                cache_ref[l, ...] += jax.lax.dot_general(
                    inp_ref[...],
                    lora_a_ref[i, ...], (((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32) * mask_a_ref[...]

    cache_valid = d2 == (ds - 1)
    @pl.when(cache_valid)
    def _():
        @pl.when(l == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

        for i in range(max_num_loras):
            mask_b_ref[...] = jnp.zeros_like(mask_b_ref[...], dtype=jnp.float32)
            valid = False
            for j in range(bT):
                valid |= idx_ref[j + bT * t] == i

                @pl.when(idx_ref[j + bT * t] == i)
                def _():
                    mask_b_ref[j, :] = jnp.ones((bD, ), dtype=jnp.float32)

            @pl.when(valid)
            def _():
                acc_ref[...] += jax.lax.dot_general(
                    cache_ref[l, ...],
                    lora_b_ref[i, ...], (((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32) * mask_b_ref[...]

        @pl.when(l == ls - 1)
        def _():
            out_ref[...] = acc_ref[...].astype(out_ref.dtype)

@functools.partial(jax.jit, static_argnames=["TOKEN_BLOCK", "LORA_BLOCK", "DIM_BLOCK"])
def _fused_bgmv_shrink_expand(
    idxs: jax.Array,  # (T, ) int32
    inputs: jax.Array,  # (T, D) model dtype
    loras_a: jax.Array,  # (N, L, D) model dtype
    loras_b: jax.Array,  # (N, D, L) model dtype
    *,
    TOKEN_BLOCK: int,
    LORA_BLOCK: int,
    DIM_BLOCK: int
) -> jax.Array:  # (T, D) model dtype
    T, D = inputs.shape
    N, L, _ = loras_a.shape

    return pl.pallas_call(
        kernel=functools.partial(_fused_bgmv_shrink_expand_kernel, TOKEN_BLOCK, LORA_BLOCK, DIM_BLOCK, N),
        out_shape=jax.ShapeDtypeStruct((T, D), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T // TOKEN_BLOCK, D // DIM_BLOCK, L // LORA_BLOCK, D // DIM_BLOCK),
            in_specs=[
                pl.BlockSpec((TOKEN_BLOCK, DIM_BLOCK), lambda t, d1, l, d2, block_idx: (t, d2)), # Inputs
                pl.BlockSpec((N, LORA_BLOCK, DIM_BLOCK), lambda t, d1, l, d2, block_idx: (0, l, d2)), # LoRA A
                pl.BlockSpec((N, DIM_BLOCK, LORA_BLOCK), lambda t, d1, l, d2, block_idx: (0, d1, l)), # LoRA B
            ],
            out_specs=pl.BlockSpec((TOKEN_BLOCK, DIM_BLOCK), lambda t, d1, l, d2, block_idx: (t, d1)),
            scratch_shapes=[
                pltpu.VMEM((L // LORA_BLOCK, TOKEN_BLOCK, LORA_BLOCK), jnp.float32), # Intermediates cache
                pltpu.VMEM((TOKEN_BLOCK, DIM_BLOCK), jnp.float32), # Final accumulator
                pltpu.VMEM((TOKEN_BLOCK, LORA_BLOCK), jnp.float32), # LoRA A mask
                pltpu.VMEM((TOKEN_BLOCK, DIM_BLOCK), jnp.float32) # LoRA B mask
            ]
        ),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary", "arbitrary")
        ),
        name="fused_bgmv_shrink_expand"
    )(idxs, inputs, loras_a, loras_b)


XLA_LIB.define("bgmv(Tensor inputs, Tensor loras, Tensor idxs) -> Tensor")

def bgmv_shape_function(idxs, inputs, loras):
    T, _ = inputs.shape
    _, L, _ = loras.shape

    return [((T, L), inputs.dtype)]

@impl(XLA_LIB, "bgmv", "XLA")
def bgmv_xla(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    inputs = inputs.to(dtype=loras.dtype)

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    T, _ = inputs.shape
    _, L, D = loras.shape

    jax_import_guard()

    TOKEN_BLOCK=16
    if L > D: # Expand
        LORA_BLOCK=1024
        DIM_BLOCK=256
    else: # Shrink
        LORA_BLOCK=256
        DIM_BLOCK=1024


    kernel = make_kernel_from_pallas(
        functools.partial(
            _bgmv,
            TOKEN_BLOCK=TOKEN_BLOCK,
            LORA_BLOCK=LORA_BLOCK,
            DIM_BLOCK=DIM_BLOCK
        ),
        bgmv_shape_function
    )

    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU
    # register. This has to happen in pytorch, doing it in Jax will lead to NaNs
    L1 = L
    if LORA_BLOCK > L or L % LORA_BLOCK != 0:
        L1 = (L // LORA_BLOCK + 1) * LORA_BLOCK

    D1 = D
    if DIM_BLOCK > D or D % DIM_BLOCK != 0:
        D1 = (D // DIM_BLOCK + 1) * DIM_BLOCK

    T1 = T
    if TOKEN_BLOCK > T or T % TOKEN_BLOCK != 0:
        T1 = (T // TOKEN_BLOCK + 1) * TOKEN_BLOCK

    if D1 != D or L1 != L:
        loras = torch.nn.functional.pad(loras, (0, D1 - D, 0, L1 - L, 0, 0))
    if D1 != D or T1 != T:
        inputs = torch.nn.functional.pad(inputs, (0, D1 - D, 0, T1 - T))
        if T1 != T:
            idxs = torch.nn.functional.pad(idxs, ((0, T1 - T)))

    return kernel(idxs, inputs, loras)[:T, :L]


@impl(XLA_LIB, "bgmv", "CompositeExplicitAutograd")
def bgmv_non_xla(inputs: torch.Tensor, loras: torch.Tensor,
                 idxs: torch.IntTensor):
    T, _ = inputs.shape

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    _, L, _ = loras.shape

    return torch.empty((T, L), device=inputs.device)

def fused_bgmv_shrink_expand_shape_function(idxs, inputs, loras_a, loras_b):
    return [(inputs.shape, inputs.dtype)]

XLA_LIB.define("fused_bgmv_shrink_expand(Tensor inputs, Tensor loras_a, Tensor loras_b, Tensor idxs) -> Tensor")

@impl(XLA_LIB, "fused_bgmv_shrink_expand", "XLA")
def fused_bgmv_shrink_expand_xla(inputs: torch.Tensor, loras_a: torch.Tensor, loras_b: torch.Tensor, idxs: torch.IntTensor):
    inputs = inputs.to(dtype=loras_a.dtype)

    if len(loras_a.shape) == 4:
        loras_a = loras_a.squeeze(axis=1)
    if len(loras_b.shape) == 4:
        loras_b = loras_b.squeeze(axis=1)

    T, _ = inputs.shape
    _, L, D = loras_a.shape

    jax_import_guard()

    TOKEN_BLOCK=16
    LORA_BLOCK=256
    DIM_BLOCK=1024


    kernel = make_kernel_from_pallas(
        functools.partial(
            _fused_bgmv_shrink_expand,
            TOKEN_BLOCK=TOKEN_BLOCK,
            LORA_BLOCK=LORA_BLOCK,
            DIM_BLOCK=DIM_BLOCK
        ),
        fused_bgmv_shrink_expand_shape_function
    )

    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU
    # register. This has to happen in pytorch, doing it in Jax will lead to NaNs
    L1 = L
    if LORA_BLOCK > L or L % LORA_BLOCK != 0:
        L1 = (L // LORA_BLOCK + 1) * LORA_BLOCK

    D1 = D
    if DIM_BLOCK > D or D % DIM_BLOCK != 0:
        D1 = (D // DIM_BLOCK + 1) * DIM_BLOCK

    T1 = T
    if TOKEN_BLOCK > T or T % TOKEN_BLOCK != 0:
        T1 = (T // TOKEN_BLOCK + 1) * TOKEN_BLOCK

    if D1 != D or L1 != L:
        loras_a = torch.nn.functional.pad(loras_a, (0, D1 - D, 0, L1 - L, 0, 0))
        loras_b = torch.nn.functional.pad(loras_b, (0, L1 - L, 0, D1 - D, 0, 0))
    if D1 != D or T1 != T:
        inputs = torch.nn.functional.pad(inputs, (0, D1 - D, 0, T1 - T))
        if T1 != T:
            idxs = torch.nn.functional.pad(idxs, ((0, T1 - T)))

    return kernel(idxs, inputs, loras_a, loras_b)[:T, :D]


@impl(XLA_LIB, "fused_bgmv_shrink_expand", "CompositeExplicitAutograd")
def fused_bgmv_shrink_expand_non_xla(inputs: torch.Tensor, loras_a: torch.Tensor, loras_b: torch.Tensor,
                 idxs: torch.IntTensor):
    return torch.empty_like(inputs)