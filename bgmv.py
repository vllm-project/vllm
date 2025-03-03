# SPDX-License-Identifier: Apache-2.0

import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def create_tensors(T, D, L, N):
    """
    Inputs: (All integers)
        T: Total number of tokens
        D: Input dim
        L: LoRA Dim
        N: N LoRAs
    
    Outputs:
        inputs:     jax.Array - shape (T, D)
        loras:      jax.Array - shape (N, 1, L, D)
        idxs:       jax.Array - shape (T, ) - all values must be in [0, N)
        
        ref_output: jax.Array - shape (T, L) - inputs @ loras[idxs].T
    """
    inputs = jax.random.normal(jax.random.PRNGKey(0), (T, D))
    loras = jax.random.normal(jax.random.PRNGKey(0), (N, 1, L, D))
    idxs = jax.random.randint(jax.random.PRNGKey(0),
                              shape=(T, ),
                              minval=0,
                              maxval=N)

    ref_output = jnp.einsum("td,__ld->tl", inputs, loras[idxs])

    return inputs, loras, idxs, ref_output


def create_debug_tensors(T, D, L, N):
    """
    Inputs: (All integers)
        T: Total number of tokens
        D: Input dim
        L: LoRA Dim
        N: N LoRAs
    
    Outputs:
        inputs:     jax.Array - shape (T, D)
        loras:      jax.Array - shape (N, 1, L, D)
        idxs:       jax.Array - shape (T, ) - all values must be in [0, N)
        
        ref_output: jax.Array - shape (T, L) - inputs @ loras[idxs].T
    """
    inputs = jnp.ones((T, D))
    loras = jnp.ones((N, 1, L, D)) * jnp.arange(0, N)[:, None, None, None]
    idxs = jax.random.randint(jax.random.PRNGKey(0),
                              shape=(T, ),
                              minval=0,
                              maxval=N)

    ref_output = jnp.einsum("td,t_ld->tl", inputs, loras[idxs])

    return inputs, loras, idxs, ref_output


def bgmv_kernel(bT: int, bL: int, idx_ref, inp_ref, lora_ref, out_ref, acc_ref,
                mask_ref):

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
            lora_ref[idx, 0, ...], (((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32) * mask_ref[...]

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@jax.jit
def bgmv(inputs: jax.Array, lora: jax.Array, idxs: jax.Array):
    T, D = inputs.shape
    N, _, L, _ = lora.shape
    
    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU register
    L1 = L
    if L < 128 or L % 128 != 0:
        L1 = (L // 128 + 1) * 128
        lora = jnp.pad(lora, ((0,0), (0,0), (0,L1-L), (0,0)))

    # TODO: Tune these
    bT = 8
    bL = 128
    bD = 128

    return pl.pallas_call(kernel=functools.partial(bgmv_kernel, bT, bL),
                          out_shape=jax.ShapeDtypeStruct((T, L1),
                                                         dtype=inputs.dtype),
                          grid_spec=pltpu.PrefetchScalarGridSpec(
                              num_scalar_prefetch=1,
                              grid=(T // bT, L1 // bL, D // bD),
                              in_specs=[
                                  pl.BlockSpec((bT, bD),
                                               lambda i, j, k, block_idx:
                                               (i, k)),
                                  pl.BlockSpec((N, 1, bL, bD),
                                               lambda i, j, k, block_idx:
                                               (0, 0, j, k)),
                              ],
                              out_specs=pl.BlockSpec(
                                  (bT, bL), lambda i, j, k, block_idx: (i, j)),
                              scratch_shapes=[
                                  pltpu.VMEM((bT, bL), jnp.float32),
                                  pltpu.VMEM((bT, bL), jnp.float32)
                              ]),
                          compiler_params=pltpu.TPUCompilerParams(
                              dimension_semantics=("parallel", "parallel",
                                                   "arbitrary")),
                          interpret=True)(idxs, inputs, lora)[:, :L]


if __name__ == "__main__":
    T, D, L, N = 16, 3072, 8, 8
    inputs, lora, idxs, ref_output = create_debug_tensors(T, D, L, N)
    print(idxs)
    # breakpoint()

    print(lora.shape, inputs.shape, ref_output.shape)

    output = bgmv(inputs, lora, idxs)

    print(jnp.isnan(output).sum(), "NaN values")

    print("Err", jnp.max(jnp.abs(ref_output - output)))

    output_idxs = (output / D)[:, 0]
    print(output_idxs)
    print(output_idxs == idxs)

    # breakpoint()
    # np.testing.assert_allclose(ref_output, output1, rtol=1e-2)
