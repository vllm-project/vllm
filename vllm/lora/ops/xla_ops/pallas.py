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
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    t = pl.program_id(0)
    
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

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=["TOKEN_BLOCK_SIZE", "LORA_RANK_BLOCK_SIZE", "DIM_BLOCK_SIZE"])
def _bgmv(
    idxs: jax.Array,  # (T, ) int32
    inputs: jax.Array,  # (T, D) model dtype
    loras: jax.Array,  # (N, L, D) model dtype
    *,
    TOKEN_BLOCK_SIZE: int,
    LORA_RANK_BLOCK_SIZE: int,
    DIM_BLOCK_SIZE: int
) -> jax.Array:  # (T, L) model dtype
    T, D = inputs.shape
    N, L, _ = loras.shape

    return pl.pallas_call(
        kernel=functools.partial(_bgmv_kernel, TOKEN_BLOCK_SIZE, LORA_RANK_BLOCK_SIZE, N),
        out_shape=jax.ShapeDtypeStruct((T, L), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T // TOKEN_BLOCK_SIZE, L // LORA_RANK_BLOCK_SIZE,
                  D // DIM_BLOCK_SIZE),
            in_specs=[
                pl.BlockSpec((TOKEN_BLOCK_SIZE, DIM_BLOCK_SIZE),
                             lambda i, j, k, block_idx: (i, k)),
                pl.BlockSpec((N, LORA_RANK_BLOCK_SIZE, DIM_BLOCK_SIZE),
                             lambda i, j, k, block_idx: (0, j, k)),
            ],
            out_specs=pl.BlockSpec((TOKEN_BLOCK_SIZE, LORA_RANK_BLOCK_SIZE),
                                   lambda i, j, k, block_idx: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((TOKEN_BLOCK_SIZE, LORA_RANK_BLOCK_SIZE), jnp.float32),
                pltpu.VMEM((TOKEN_BLOCK_SIZE, LORA_RANK_BLOCK_SIZE), jnp.float32)
            ]),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        name="bgmv")(idxs, inputs, loras)


def bgmv_shape_function(idxs, inputs, loras):
    T, _ = inputs.shape
    _, L, _ = loras.shape

    return [((T, L), inputs.dtype)]


XLA_LIB.define("bgmv(Tensor inputs, Tensor loras, Tensor idxs) -> Tensor", )


@impl(XLA_LIB, "bgmv", "XLA")
def bgmv_xla(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    inputs = inputs.to(dtype=loras.dtype)

    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    T, _ = inputs.shape
    _, L, D = loras.shape

    jax_import_guard()

    TOKEN_BLOCK_SIZE=16
    if L > D: # Expand
        LORA_RANK_BLOCK_SIZE=1024
        DIM_BLOCK_SIZE=256
    else: # Shrink
        LORA_RANK_BLOCK_SIZE=256
        DIM_BLOCK_SIZE=1024


    kernel = make_kernel_from_pallas(
        functools.partial(
            _bgmv,
            TOKEN_BLOCK_SIZE=TOKEN_BLOCK_SIZE,
            LORA_RANK_BLOCK_SIZE=LORA_RANK_BLOCK_SIZE,
            DIM_BLOCK_SIZE=DIM_BLOCK_SIZE
        ),
        bgmv_shape_function
    )

    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU
    # register. This has to happen in pytorch, doing it in Jax will lead to NaNs
    L1 = L
    if LORA_RANK_BLOCK_SIZE > L or L % LORA_RANK_BLOCK_SIZE != 0:
        L1 = (L // LORA_RANK_BLOCK_SIZE + 1) * LORA_RANK_BLOCK_SIZE

    D1 = D
    if DIM_BLOCK_SIZE > D or D % DIM_BLOCK_SIZE != 0:
        D1 = (D // DIM_BLOCK_SIZE + 1) * DIM_BLOCK_SIZE

    T1 = T
    if TOKEN_BLOCK_SIZE > T or T % TOKEN_BLOCK_SIZE != 0:
        T1 = (T // TOKEN_BLOCK_SIZE + 1) * TOKEN_BLOCK_SIZE

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
