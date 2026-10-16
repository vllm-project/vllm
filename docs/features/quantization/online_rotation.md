# Checkpoint-backed rotations before RMSNorm

`rotation_config` attaches a checkpoint-backed Composite Hadamard transform to
the input of selected `RMSNorm` modules. This is useful when an offline-rotated
residual stream must return to the original coordinates before the final norm.
For MTP models, inverse rotation can be applied before both the main-model final
norm and MTP's own norm, restoring their inputs for the original norm and head.
The main-model final norm also supplies the hidden state consumed by MTP.

Use a checkpoint whose weights already incorporate the matching forward
rotation and whose tensors include the rotation parameters described below.

## Configuration

The following GLM-5.3 example uses hidden size `6144 = 12 * 512` and 78 main
decoder layers:

```json
{
  "rotation_config": {
    "rotations": {
      "0": {
        "type": "composite_hadamard",
        "hidden_size": 6144,
        "hadamard_k": 12
      }
    },
    "apply": [
      {
        "tag": 0,
        "targets": ["model.norm", "model.layers.78.shared_head.norm"],
        "mode": "pre",
        "inverse": true
      }
    ]
  }
}
```

| Field | Meaning |
| --- | --- |
| `rotations` | Rotation definitions indexed by tag. Tags are normalized to strings. |
| `type` | Currently `composite_hadamard`. |
| `hidden_size` | Last dimension of the input and the target RMSNorm. |
| `hadamard_k` | Order of the checkpoint-provided small mixing matrix. |
| `hadamard_p` | Optional power-of-two width; defaults to `hidden_size / hadamard_k`. |
| `tag` | A key in `rotations`; consumers using the same tag share parameters. |
| `targets` | Runtime module paths, resolved across the target and loaded draft model. |
| `mode` | Currently `pre`, applying the transform before RMS normalization. |
| `inverse` | `true` applies the transpose of the checkpoint rotation. |

Target paths depend on the model's runtime module structure; use the paths for
the model being exported.

## Checkpoint tensors and execution

For tag `0`, the target model owns these additional checkpoint tensors:

| Name | Shape | Dtype and convention |
| --- | --- | --- |
| `rotations.0.sign` | `[hidden_size]` | INT8, each element is either -1 or +1. |
| `rotations.0.hk` | `[K, K]` | FP32, unnormalized `H_K.T`, where `H_K` is the producer's Hadamard base; `hk.T @ hk = K * I`. |

### Matrix orientation

Use row-vector hidden states `X[..., N]`, with `N = K * P` and `P = 2^m`.
Let `D = diag(sign)`, `H_P` be the unnormalized Sylvester Hadamard matrix, and
`H_K` be the unnormalized K-by-K base used by the checkpoint producer.
**The checkpoint stores `hk = H_K.T`, already transposed and unnormalized.**
Using the same signs and base, this matches the matrix convention of official
[LLM Compressor's SpinQuantModifier](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/transform/spinquant/base.py)
with `transform_type="random-hadamard"`, implemented in
[Compressed Tensors](https://github.com/vllm-project/compressed-tensors/blob/37e416a370560e3bedddf37ff5b14750da08ad53/src/compressed_tensors/transform/utils/hadamard.py):

$$
Q = D\frac{\mathrm{hk} \otimes H_P}{\sqrt{N}} = D\frac{H_K^T \otimes H_P}{\sqrt{N}}
$$

The official `random_hadamard_matrix` returns the unnormalized matrix
`R = D (H_K.T ⊗ H_P)`; its `_matmul_hadU` mixes the K axis by left-multiplying
with the base fetched by `_fetch_hadamard_divisor`.
[HadamardTransform](https://github.com/vllm-project/compressed-tensors/blob/37e416a370560e3bedddf37ff5b14750da08ad53/src/compressed_tensors/transform/factory/hadamard.py)
then divides by `sqrt(N)`, giving the normalized rotation `Q = R / sqrt(N)`.

The forward transform rotates row-vector hidden states:

$$
X_{\mathrm{rot}} = XQ
$$

The inverse transform restores the original coordinates:

$$
X = X_{\mathrm{rot}} Q^T, \qquad Q^T = \frac{H_K \otimes H_P}{\sqrt{N}}D
$$

For `K > 1`, prepare the checkpoint tensor from the producer's base at export
time:

```python
import torch

hk_checkpoint = H_K.T.to(dtype=torch.float32).contiguous()
```

For `K = 1`, store `hk = [[1.0]]`. Load the checkpoint tensor as stored, with
no additional transpose or normalization. The forward kernel uses `hk` for its
K-axis right multiplication; the inverse kernel uses `hk.T` and applies `sign`
last. Both directions apply `1 / sqrt(N)` inside the operator. `H_K` may be
asymmetric, so its orientation matters. Export a base satisfying
`H_K.T @ H_K = K * I`; loading checks finiteness and valid signs.

Parameters are registered before checkpoint loading. Binding runs after target
and optional draft loading, so both models can reference the target registry
without duplicating the checkpoint namespace. Missing or invalid sign/mixing
tensors fail validation before inference.

If the norm receives a residual, execution is:

```text
hidden + residual
       ↓
inverse Composite Hadamard
       ↓
original RMSNorm dispatch
       ↓
normalized hidden; restored residual
```

The adapter preserves the original norm instance, weight namespace, and selected
CustomOp dispatch. The shared fused AllReduce/RMSNorm helper falls back to
AllReduce followed by the adapter, because reading norm weights directly would
bypass its pre-normalization transform.

Without a draft model, unmatched targets are logged as a warning and skipped,
including when all targets belong to MTP. This allows the same checkpoint
configuration to run with MTP enabled or disabled. With a draft model loaded,
every target must resolve in the main or draft model; otherwise binding raises
an error.

## Execution requirements

The adapter supports plain `RMSNorm` with `mode: "pre"`; custom RMSNorm subclasses
are rejected to preserve their own forward behavior. CUDA execution requires
TileLang and uses a kernel for NVIDIA warp32 GPUs. For the power-of-two width
`P`, the kernel uses an H32 GEMM primitive when `P >= 32` and butterfly stages
when `P < 32`. A PyTorch reference implementation is available for CPU execution.
