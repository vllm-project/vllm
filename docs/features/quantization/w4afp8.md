# W4AFP8

`w4afp8` is a mixed-precision format for Mixture-of-Experts models: routed
expert weights are INT4 (group size 128) while activations stay in FP8, and
dense `Linear` layers are FP8 block-128 (W8A8).

vLLM supports the variant published by PhalaCloud, where
`config.json` declares `"quant_method": "w4afp8"`. `PhalaCloud/GLM-5.3-W4AFP8`
is the reference checkpoint.

## Usage

```bash
vllm serve PhalaCloud/GLM-5.3-W4AFP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --trust-remote-code \
  --moe-backend humming
```

`--moe-backend` accepts `humming` or `cutlass`. On SM90 with aligned shapes
(hidden size and per-partition intermediate size divisible by 256) `cutlass` is
selected automatically.

## How it works

No dedicated kernel is involved. The loader translates the on-disk layout at
load time and then reuses the compressed-tensors W4A8 FP8 MoE path
(the Humming / CUTLASS W4A8 oracle):

| On disk | vLLM / kernel expects |
| --- | --- |
| signed INT4, two nibbles per byte | `uint4b8`, offset-by-8 unsigned INT4 |
| `int8` pack=2 | `int32` pack=8 |
| `*_proj.weight_scale_inv` | `*_weight_scale` |
| `w{1,2,3}.input_scale` (static) | dropped; per-token dynamic FP8 scales |

Dense layers already match the native FP8 block-128 serialization
(`weight_scale_inv`), so they are loaded by the regular `fp8` path.

## Limitations

- Group size must be 128 and weights must be symmetric INT4.
- Hidden size and the per-partition intermediate size must be multiples of 256.
- Static per-tensor activation scales in the checkpoint are ignored: the W4A8
  FP8 kernels recompute activation scales per token.
