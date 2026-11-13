# Qwen3.8-2.4T specialized decode

`--monokernel` on MI355X/gfx950 runs each pure decode step of 1 to 64 rows
of Qwen3.8-2.4T (`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`) at TP8 on persistent
FlyDSL launches (`runtime/decode.py`). Steps of 17 to 64 rows run one launch
per decoder layer (`wide/layer.py`):

| layer | launches | phases |
| --- | --- | --- |
| GDN (69) | `gdn` | [embedding →] input norm, GDN projections + conv, gated delta rule + gated norm, out_proj + all-reduce + post-attention norm, router, routing + MXFP4 quantization, experts, MoE combine + all-reduce + the next layer's input norm [→ the next layer's qkv_proj] |
| attention (23) | vLLM's attention, `attn` | q / k norms, RoPE and attention stay vLLM's; then o_proj (gated) + all-reduce + post-attention norm and the MoE as above |

Up to 16 rows (`wide.layer.SPLIT_MAX`) each layer's MLP half is
`mono.blocks.layer_block` from the Qwen3.8 mono decode (#60833, vendored
unchanged under `mono/`) after the layer's split launch: a GDN layer's
residual add, input norm and GDN to the core output, an attention layer's
input norm and qkv_proj. Up to 2 rows (`runtime.layers.MONO_GDN_MAX`) a GDN
layer whose in_proj_qkvz is plain MXFP4 (out_proj not Hadamard-rotated) runs
`mono.blocks.gdn_block` instead of its split launch.

The embedding, the head and sampling stay vLLM's. Prefill and mixed steps,
and decode steps wider than 64 rows, run vLLM's forward: from 65 rows up
stock's MoE streams the experts near the HBM floor, and the wide kernels
measure no faster there. The kernels are built for each of the deployment's
CUDA graph capture sizes in 1-64; a decode step runs the smallest captured
width that holds it (vLLM's graph padding).

## Supported

- TP8 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU), FULL CUDA
  graphs for decode (PIECEWISE steps run vLLM's forward);
- pure decode of 1-64 rows (graph padding rows are inert: their GDN state
  slot is the null block);
- the checkpoint's shapes (`wide/layout.py`, `SHAPES`); GDN projections
  (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`) bf16, online MXFP4
  or online MXFP4 with a 32-block Hadamard rotation (`mxfp4_hadamard32`),
  on either AITER FP4 GEMM layout (ASM: shuffled; Triton: row-major,
  transposed scales); `AITER_MXFP4_MXFP4` experts with
  `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` (the shared expert is routed
  expert 512); a BF16 conv state and an FP32 recurrent state with
  `--no-enable-prefix-caching`.

## Refusals

Default-deny, in three stages, each reporting every reason in one error
(`runtime/__init__.py`, `runtime/guards.py`): the config (TP other than 8,
PP, DP, EP, EPLB, MoE sequence parallelism, LoRA, KV connectors,
routed-expert capture, speculative decoding, sleep mode, CPU weight offload,
weight updates, GDN prefix caching, other GDN cache dtypes, an unfused shared
expert, `STOCK_TORCH_COMPILE`, other model shapes, no capture size in 1-64, a
platform other than unpartitioned gfx950); the loaded model (every layer's
modules, quantization methods, weight layouts and shapes); the GDN caches at
every bind. The TP ranks agree on every stage.

## Numerics

The stock path's rounding points are kept: bf16 residual and norm outputs,
bf16 router logits, bf16 GDN projection outputs and core output. MXFP4
activations follow the stock kernel each replaces: AITER's
`per_1x32_f4_quant` rule (`ceil_pow2(amax / 6)`) for the ASM FP4 GEMM and
the MoE (input and intermediate), the Triton GEMM's floor rule for a
row-major MXFP4 projection (`wide/stages/mx.py`). A Hadamard projection
rotates its input per 32-block in the fast transform's order before the
quantization. The TP all-reduces are exact bf16 sums in rank order, as
stock's custom all-reduce at these sizes.

## Failure

Every in-kernel wait (grid barrier, TP exchange line, mailbox) gives up after
a bounded time: the CTA records it in the rank's health words and sets
FAILED, so the launch runs to its end. `runtime/guards.FailStop` reads
FAILED after every step and the ranks raise together.

## Code

- `runtime/`: config refusals and widths (`__init__`), step routing
  (`routing`), model and cache refusals and `FailStop` (`guards`), the per-rank
  launch state and the layer launches (`layers`), the executor (`decode`), peer
  memory (`peer`).
- `wide/`: the build key, ABI and scratch layout (`layer`), the shapes
  (`layout`), the health words (`abi`); `common/`: device primitives (`ops`),
  the grid barrier and fold counters (`grid`), build keys (`plan`);
  `stages/`: the batch-tiled GEMM (`tiled`), tagged hand-offs (`tagged`),
  norms and combines (`norm`), the gated delta net (`gdn`), the MoE (`moe`),
  MXFP4 activations (`mx`), the TP exchange (`exchange`).
- `mono/`: #60833's `gdn_block` and `layer_block`.

For profiling, `runtime.layers.TRACE = True` (set in the worker before the
model is built) builds every wide launch with per-phase clock stamps
(`wide.layer.phase_names`).

## Tests

`tests/models/qwen3_5/amd_specialized_decode/` checks the launches at TP8
against a float64 reference with the stock rounding points (`test_layer.py`:
the wide widths; `test_split.py`: the split widths, and the GDN launch's
MXFP4 rows bit-exact against AITER's quantization on one GPU).
