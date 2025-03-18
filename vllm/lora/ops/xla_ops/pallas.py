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

    TOKEN_BLOCK = 16
    if L > D: # Expand
        LORA_BLOCK = 1024
        DIM_BLOCK = 256
    else: # Shrink
        LORA_BLOCK = 256
        DIM_BLOCK = 1024

    TOKEN_BLOCK = min(max(TOKEN_BLOCK, pl.next_power_of_2(T)), 128)
    LORA_BLOCK = max(LORA_BLOCK, pl.next_power_of_2(L))
    DIM_BLOCK = max(DIM_BLOCK, pl.next_power_of_2(D))

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
