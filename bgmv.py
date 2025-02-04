# SPDX-License-Identifier: Apache-2.0

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
        lora:       jax.Array - shape (L, D)
        
        ref_output: jax.Array - shape (T, L) - inputs @ loras[idxs].T
        
    Ignored:
        idxs:       jax.Array - shape (T, ) - all values must be in [0, N)
        loras:      jax.Array - shape (N, 1, L, D)
    """
    inputs = jax.random.normal(jax.random.PRNGKey(0), (T, D))
    lora = jax.random.normal(jax.random.PRNGKey(1), (L, D))
    ref_output = inputs @ lora.T

    return inputs, lora, ref_output


def bgmv_kernel(inp_ref, lora_ref, out_ref, acc_ref):

    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    acc_ref[...] += jax.lax.dot_general(inp_ref[...],
                                        lora_ref[...],
                                        (((1, ), (1, )), ((), ())),
                                        preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@jax.jit
def bgmv(inputs: jax.Array, lora: jax.Array):
    T, D = inputs.shape
    L, _ = lora.shape

    # TODO: Tune
    # Also figure out how to make bT % 128 instead of bL,
    # or pick block sizes based off dims
    bT = 8
    bL = 128
    bD = 128

    return pl.pallas_call(
        kernel=bgmv_kernel,
        out_shape=jax.ShapeDtypeStruct((T, L), dtype=inputs.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(T // bT, L // bL, D // bD),
            in_specs=[
                pl.BlockSpec((bT, bD), lambda i, j, k: (i, k)),
                pl.BlockSpec((bL, bD), lambda i, j, k: (j, k)),
            ],
            out_specs=pl.BlockSpec((bT, bL), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bT, bL), jnp.float32)]),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=True)(inputs, lora)


if __name__ == "__main__":
    T, D, L, N = 128, 3072, 128, 8
    inputs, lora, ref_output = create_tensors(T, D, L, N)

    print(lora.shape, inputs.shape, ref_output.shape)

    output1 = bgmv(inputs, lora)

    print(jnp.isnan(output1).sum(), "NaN values")

    # np.testing.assert_allclose(ref_output, output1)
    # print("Success")
