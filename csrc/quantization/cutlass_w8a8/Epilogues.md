# CUTLASS Epilogues

## Introduction
This document describes the various CUTLASS epilogues implemented for fusing de-quantization operations onto GEMMs. 

Currently, we only support symmetric quantization for weights,
and symmetric and asymmetric quantization for activations.
Both can be quantized per-tensor or per-channel (weights) / per-token (activations).

There are 4 epilogues:
1. ScaledEpilogue: symmetric quantization for activations, no bias.
1. ScaledEpilogueBias: symmetric quantization for activations, supports bias.
1. ScaledEpilogueAzp: asymmetric per-tensor quantization for activations, supports bias.
1. ScaledEpilogueAzpPerToken: asymmetric per-token quantization for activations, supports bias.

We do not have epilogues for asymmetric quantization of activations without bias in order to reduce final binary size.
Instead, if no bias is passed, the epilogue will use 0 as the bias.
That induces a redundant addition operation (and runtime check), but the performance impact is minor.

## Underlying Linear Algebra

More details available in the [Activation Quantization RFC](https://github.com/vllm-project/vllm/issues/3975).

If $` \widehat X `$ is the quantized $` X `$, our matrices become the following

```math
A = s_a (\widehat A - J_a z_a)
```
```math
B = s_b \widehat B
```
```math
D = A B + C
```
```math
D = s_a s_b \widehat D + C
```

Here, D is the output of the GEMM, and C is the bias.
A is the activations and supports asymmetric quantization,
and B is the weights and only supports symmetric quantization.
$ s_a $ and $s_b$ are the scales for activations and weights, respectively.
$ z_a $ is the zero-point for activations, and $ J_a $ is the matrix of all ones with dimensions of A.
Additional epilogues would be required to support asymmetric quantization for weights.

Expanding further, we can calculate $` \widehat D `$ as follows:

```math
A B = s_a ( \widehat A - J_a z_a ) s_b \widehat B
```
```math
A B = s_a s_b \left( \widehat A \widehat B - J_a z_a \widehat B \right)
```
```math
\widehat D = \widehat A \widehat B - z_a J_a \widehat B
```

Note that $` \widehat A \widehat B `$ is the raw output of the GEMM,
and $` J_a \widehat B `$ is known ahead of time.
Each row of it is equal to $` \mathbf 1 \widehat B `$, which is a row-vector of column sums of $` \widehat B `$.

## Epilogues

### ScaledEpilogue
This epilogue computes the symmetric quantization for activations without bias, meaning $` C = 0 `$ and $` z_a = 0 `$.
The output of the GEMM is:

```math
\widehat D = \widehat A \widehat B
```
```math
D = s_a s_b \widehat D
```
```math
D = s_a s_b \widehat A \widehat B
```

Epilogue parameters:
- `scale_a` is the scale for activations, can be per-tensor (scalar) or per-token (column-vector).
- `scale_b` is the scale for weights, can be per-tensor (scalar) or per-channel (row-vector).

### ScaledEpilogueBias
This epilogue computes the symmetric quantization for activations with bias, meaning $` z_a = 0 `$.
The output of the GEMM is:

```math
\widehat D = \widehat A \widehat B
```
```math
D = s_a s_b \widehat D + C 
```
```math
D = s_a s_b \widehat A \widehat B + C
```


Epilogue parameters:
- `scale_a` is the scale for activations, can be per-tensor (scalar) or per-token (column-vector).
- `scale_b` is the scale for weights, can be per-tensor (scalar) or per-channel (row-vector).
- `bias` is the bias, is always per-channel (row-vector).

### ScaledEpilogueAzp
This epilogue computes the asymmetric per-tensor quantization for activations with bias.
The output of the GEMM is:

```math
\widehat D = \widehat A \widehat B - z_a J_a \widehat B
```
```math
D = s_a s_b \widehat D + C 
```
```math
D = s_a s_b \left( \widehat A \widehat B - z_a J_a \widehat B \right) + C
```

Because $` z_a `$ is a scalar, the zero-point term $` z_a J_a \widehat B `$ has every row equal to $` z_a \mathbf 1 B `$. 
That is precomputed and stored in `azp_with_adj` as a row-vector.

Epilogue parameters:
- `scale_a` is the scale for activations, can be per-tensor (scalar) or per-token (column-vector).
  - Generally this will be per-tensor as the zero-points are per-tensor.
- `scale_b` is the scale for weights, can be per-tensor (scalar) or per-channel (row-vector).
- `azp_with_adj` is the precomputed zero-point term ($` z_a J_a \widehat B `$), is per-channel (row-vector).
- `bias` is the bias, is always per-channel (row-vector).

To use these kernels efficiently, users must precompute the `azp_with_adj` term offline and pass it to the kernel.

### ScaledEpilogueAzpPerToken
This epilogue computes the asymmetric per-token quantization for activations with bias.

The output of the GEMM is the same as above, but the $` z_a `$ is a column-vector.
That means the zero-point term $` z_a J_a \widehat B `$ becomes an outer product of $` z_a `$ and $` \mathbf 1 \widehat B `$.

Epilogue parameters:
- `scale_a` is the scale for activations, can be per-tensor (scalar) or per-token (column-vector).
  - Generally this will be per-token as the zero-points are per-token.
- `scale_b` is the scale for weights, can be per-tensor (scalar) or per-channel (row-vector).
- `azp_adj` is the precomputed zero-point adjustment term ($` \mathbf 1 \widehat B `$), is per-channel (row-vector).
- `azp` is the zero-point (`z_a`), is per-token (column-vector).
- `bias` is the bias, is always per-channel (row-vector).

To use these kernels efficiently, users must precompute the `azp_adj` term offline and pass it to the kernel.

The epilogue performs the following computation (where `Dq` is the raw quantized output of the GEMM):
```
out = scale_a * scale_b * (Dq - azp_adj * azp) + bias
```
