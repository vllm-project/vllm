# Qwen3.8-Flash-Next specialized decode (wide)

`--monokernel` on MI355X/gfx950 runs each pure decode step of 1 to 128 rows
of Qwen3.8-Flash-Next (`amd/Qwen3.8-Flash-Next-Quark-MXFP4`) at TP2 as one
persistent FlyDSL launch per decoder layer (`runtime/decode.py`,
`wide/layer.py`):

| layer | launches | phases |
| --- | --- | --- |
| GDN (36) | `gdn` | [stats] attn mix, GDN projection + conv, gated delta rule + gated norm, out_proj + all-reduce + combine, MLP mix, router, routing + MXFP4 quantization, experts, MoE combine + all-reduce + combine |
| QSA (12) | `qsa_front`, indexer, attention, `qsa_back` | attn mix → x, qkv_proj + index_qk_proj; vLLM's fused prepare, the indexer scores and the top-k select (`wide/indexer.py`, `wide/select.py`); sparse attention on the paged KV (`wide/attention.py`); gated o_proj + all-reduce + combine and the MLP half |

The embedding, the PLE layer, the final mixer, the head and sampling stay
vLLM's. Steps of a width without a build, prefill and mixed steps run vLLM's
forward. The kernels are built for each of the deployment's CUDA graph capture
sizes in 1-128; a decode step runs the smallest captured width that holds it
(vLLM's graph padding), so capture sizes 1, 2, 4, 8, 16 serve 3 rows at width
4 and 12 at 16.

The kernels are built from an expert-major fused W4A4 MoE, batch-tiled
MFMA GEMMs with split-K folded by the last arriver, a line-based in-kernel TP
exchange and a striped grid barrier, in vLLM's execution model: one launch
per layer on vLLM's buffers, weights and caches in place, no multi-step
device loop. Up to 16 rows (`tiled.RESIDENT_MAX`) a GEMM unit stages its whole
K slice once, and a mix, and the GDN projection through the attention combine,
each run as one phase whose CTAs hand data over as tagged pairs
(`wide/stages/tagged.py`) rather than through grid barriers. Up to 12 rows
(`moe.SMALL_MAX`) the MoE takes a small-batch path: every CTA computes whole
router logits for its gate rows, each routed or shared expert's gate/up runs
as pieces of 32 intermediate rows spread over the grid, and 160 CTAs own the
down projection's output rows and push the exchange lines.

## MoE schedule

A unit is one expert and up to 16 of its rows; its time is one CU's stream
of the expert's weights (~1.3 MB a rank), whatever its row count. Real
routing is far from uniform (128 rows of natural text: ~300 distinct experts
a layer, up to ~80 rows on one; uniform: ~470), so the schedule
(`wide/stages/moe.py`, `experts`) adapts to the step's distribution:

- units of one expert are consecutive and run on one XCD (runs of 8 rotate
  over the XCDs), so a hot expert's weights come through one L2;
- a last round at most half full (for example 300 units on 256 CTAs) runs
  as two intermediate halves a unit (half the gate/up rows and the down
  projection's K each, the second half's partial in its own buffer, marked
  for the combine), so the round ends after about half a unit instead of a
  whole one.

## Supported

- TP2 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU), Model
  Runner V2, FULL CUDA graphs for decode (PIECEWISE steps run vLLM's forward);
- pure decode of 1-128 rows (graph padding rows are inert: their GDN state
  slot is the null block);
- the checkpoint's shapes (`wide/layout.py`, `SHAPES`) and formats: online
  MXFP4 GDN projections (`--quantization-config.targets` for
  `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`) on either AITER FP4
  GEMM layout (ASM: shuffled; Triton: row-major, transposed scales),
  `AITER_MXFP4_MXFP4` experts with `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`
  (the shared expert is routed expert 512; gate and up padded to any multiple
  of 128 rows), a BF16 conv state and an FP32 recurrent state with
  `--no-enable-prefix-caching`.

## Refusals

Default-deny, in three stages, each reporting every reason in one error
(`runtime/__init__.py`, `runtime/guards.py`): the config (TP other than 2,
PP, DP, EP, EPLB, DCP, PCP, DBO, HC/MoE sequence parallelism, LoRA, KV
connectors, routed-expert capture, speculative decoding, sleep mode, CPU
weight offload, weight updates, GDN prefix caching, other GDN cache dtypes,
an unfused shared expert, the V1 runner, `STOCK_TORCH_COMPILE`, other model
shapes, no capture size in 1-128, a platform other than unpartitioned
gfx950); the loaded model (every layer's modules, quantization methods,
weight layouts and shapes); the GDN caches at every bind (dtypes, shapes,
dense slots). The TP ranks agree on every stage.

## Numerics

The stock path's rounding points are kept: bf16 multi-stream state, bf16
gated-residual GEMM outputs, bf16 router logits, MXFP4 MoE activations and
intermediate with AITER's scale rule, bf16 GDN projection outputs and core
output. The online-MXFP4 dense projections run W4A16 (the weights decoded to
bf16 in registers; stock quantizes the activations to MXFP4 too).

The TP all-reduces follow the deployment's QuickReduce setting
(`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`, which already quantizes the prefill
all-reduces): with `INT4` (or `INT3`) they send QuickReduce's INT4 codec
(16-column groups, the group's signed extremum mapped to -8, here with a bf16
scale so small partials do not flush to zero), 3.2x fewer bytes on the link;
otherwise they are exact bf16 sums in rank order, as stock's custom
all-reduce at decode sizes. Every rank decodes every rank's copy, its own
included, so the ranks agree. The lines are one-shot: each rank quantizes
its partial once, where QuickReduce's two-shot all-reduce quantizes twice
(the scattered partials, then the reduced shard); at TP2 two-shot would move
the same bytes and add a round trip.

## Failure

Every in-kernel wait (grid barrier, TP exchange line) gives up after 10 s:
the CTA records it in the rank's health words, raises the rank's flag on the
output rank and sets FAILED, so the launch runs to its end.
`runtime/guards.FailStop` reads FAILED after every step and the ranks raise
together every 16 steps.

## Code

- `runtime/`: config refusals and widths (`__init__`), step routing
  (`routing`), model and cache refusals and `FailStop` (`guards`), the per-rank
  launch state and the layer launches (`layers`), the executor (`decode`), peer
  memory (`peer`, as the Kimi-K3 decode's).
- `wide/`: the build key, ABI and scratch layout (`layer`), the shapes
  (`layout`), the health words (`abi`); `common/`: device primitives (`ops`),
  the grid barrier and fold counters (`grid`), build keys (`plan`);
  `stages/`: the batch-tiled GEMM (`tiled`), the gated residual (`hc`), the
  gated delta net (`gdn`), the MoE (`moe`), MXFP4 activations (`mx`), the TP
  exchange (`exchange`).

For profiling, `runtime.layers.TRACE = True` (set in the worker before the
model is built, for example by a `--worker-extension-cls` module) builds
every launch with per-phase clock stamps (`Launcher.trace`,
`wide.layer.phase_names`).

## Toolchain

FlyDSL comes in through AITER; measured with FlyDSL 0.4.0.
`tests/models/qwen4_exp/amd_specialized_decode/` checks each stage against a
float64 reference with the stock rounding points.
