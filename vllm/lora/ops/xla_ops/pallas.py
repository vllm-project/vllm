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

# TODO: Tune these
TOKENS_BLOCK = 16
LORA_RANK_BLOCK = 128
DIM_BLOCK_SIZE = 128


def _bgmv_kernel(bT: int, bL: int, idx_ref, inp_ref, lora_ref, out_ref,
                 acc_ref, mask_ref):

    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    t = pl.program_id(0)

    for i in range(bT):
        idx = idx_ref[i + bT * t]
        mask_ref[...] = jnp.zeros_like(mask_ref[...], dtype=jnp.float32)
        mask_ref[i, :] = jnp.ones((bL, ), dtype=jnp.float32)

        acc_ref[...] += jax.lax.dot_general(
            inp_ref[...],
            lora_ref[idx, ...], (((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32) * mask_ref[...]

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@jax.jit
def _bgmv(
    idxs: jax.Array,  # (T, ) int32
    inputs: jax.Array,  # (T, D) model dtype
    loras: jax.Array  # (N, L, D) model dtype
) -> jax.Array:  # (T, L) model dtype
    T, D = inputs.shape
    N, L, _ = loras.shape

    return pl.pallas_call(
        kernel=functools.partial(_bgmv_kernel, TOKENS_BLOCK, LORA_RANK_BLOCK),
        out_shape=jax.ShapeDtypeStruct((T, L), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T // TOKENS_BLOCK, L // LORA_RANK_BLOCK,
                  D // DIM_BLOCK_SIZE),
            in_specs=[
                pl.BlockSpec((TOKENS_BLOCK, DIM_BLOCK_SIZE),
                             lambda i, j, k, block_idx: (i, k)),
                pl.BlockSpec((N, LORA_RANK_BLOCK, DIM_BLOCK_SIZE),
                             lambda i, j, k, block_idx: (0, j, k)),
            ],
            out_specs=pl.BlockSpec((TOKENS_BLOCK, LORA_RANK_BLOCK),
                                   lambda i, j, k, block_idx: (i, j)),
            scratch_shapes=[
                pltpu.VMEM((TOKENS_BLOCK, LORA_RANK_BLOCK), jnp.float32),
                pltpu.VMEM((TOKENS_BLOCK, LORA_RANK_BLOCK), jnp.float32)
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

    jax_import_guard()
    kernel = make_kernel_from_pallas(_bgmv, bgmv_shape_function)

    T, _ = inputs.shape
    _, L, D = loras.shape

    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU
    # register. This has to happen in pytorch, doing it in Jax will lead to NaNs
    L1 = L
    if LORA_RANK_BLOCK > L or L % LORA_RANK_BLOCK != 0:
        L1 = (L // LORA_RANK_BLOCK + 1) * LORA_RANK_BLOCK

    D1 = D
    if DIM_BLOCK_SIZE > D or D % DIM_BLOCK_SIZE != 0:
        D1 = (D // DIM_BLOCK_SIZE + 1) * DIM_BLOCK_SIZE

    T1 = T
    if TOKENS_BLOCK > T or T % TOKENS_BLOCK != 0:
        T1 = (T // TOKENS_BLOCK + 1) * TOKENS_BLOCK

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
