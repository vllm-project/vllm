# fp8 dense projections for Kimi-K3 on ROCm

Kimi-K3 ships MXFP4 routed experts and leaves every dense projection in bf16 --
`self_attn`, `shared_experts` and the dense MLP are all in the checkpoint's
`ignore` list. This directory turns a stock checkpoint into one that serves the
shared-expert projections as fp8, without copying the weights.

Measured on MI355X, TP8, conc=1, MTP depth 6, 1200s:

| | interactivity p90 | GSM8k (full 1319) |
|---|---|---|
| stock bf16 | 110.28 | 0.9659 |
| **fp8 `shared_experts`** | **111.99 (+1.55%)** | **0.9659** |

Accuracy is unchanged. The noise floor for a 1200s conc=1 arm is ~1%, so treat
+1.55% as real but not precise; a 3600s run tightens it to ~0.6%.

## Quantize only `shared_experts`

This is the whole recommendation, and it is deliberately narrow. Three other
targets were built, gated and measured, and **none of them helped**:

| target | Linears | p90 | GSM8k | verdict |
|---|---|---|---|---|
| `shared_experts` | 184 | +1.55% | 0.9659 | **use this** |
| `o_proj` | 93 | +1.32% | 0.9651 | adds nothing on top |
| `shared_experts` + `o_proj` | 277 | +1.56% | 0.9659 | same as shared_experts alone |
| `mla_proj` | 96 | -0.29% | 0.9689 | no gain |

The combined arm is the informative one: 277 quantized Linears scored 112.00
against 184 Linears at 111.99, with an identical tpot p90 floor of 0.00893.
**The levers do not compose.** At conc=1 this step is launch-bound, not
weight-bandwidth-bound, so once the first target relieves the fixed cost there
is nothing left for the second to take. Quantizing more surface buys accuracy
risk and build time for no measured speedup.

For the same reason, MXFP4 on these projections is worse on both axes:
**-0.48%** p90 and **0.9568** GSM8k, despite halving the bytes again. It adds a
per-Linear activation-quant kernel, and kernel launches are the binding
constraint here.

## Build

One command, no chaining:

```bash
python build_overlay.py \
    --src /path/to/Kimi-K3 \
    --dst /path/to/Kimi-K3-fp8se \
    --target shared_experts
```

~12 GB of new bytes, CPU only, well under an hour. The original shards are
symlinked, not copied, so nothing is duplicated and nothing is modified.
Deleting the output directory reverts everything.

### Why an overlay rather than a requantized copy

The 276 tensors are spread across 92 of 96 shards, so a conventional rewrite
touches ~1553 GB of a 1561 GB checkpoint. A sharded checkpoint is loaded
through its index, and the index is authoritative: a tensor present in a shard
but absent from the weight map is never read. So the builder symlinks every
original shard, writes fp8 weights and scales into a few small new shards, and
emits an index repointing those tensors at the new shards. The bf16 originals
remain on disk, simply unreferenced.

## Verify before trusting a number

```bash
python verify_overlay.py --overlay /path/to/Kimi-K3-fp8se --target shared_experts
```

Expect `none ignored, none in an unexpected group`.

This check exists because the failure mode is silent. `should_ignore_layer` is
consulted **before** `config_groups`, so a layer still covered by an `ignore`
regex is served as bf16 even though byte-perfect fp8 tensors sit on disk for
it. The server logs a clean load, accuracy is unchanged (it is the original
model), and the perf result looks like a null result rather than a bug. This
cost an hour of GSM8k before the check existed.

## Serve

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve /path/to/Kimi-K3-fp8se --load-format safetensors -tp 8 ...
```

Two flags are load-bearing:

* **`--load-format safetensors` is required.** vLLM narrows the *file* list by
  the index but then yields *every* tensor in each file, so without the
  tensor-level filter the stale bf16 originals are emitted under the same names
  and filename sort order decides which wins. `fastsafetensors` uses a separate
  `ParallelLoader` path with no equivalent hook and is **not** covered -- it
  will silently serve bf16.
* **`VLLM_ROCM_USE_AITER=1`** routes the fp8 linear to aiter. Without it the
  scheme resolves elsewhere and these numbers do not apply.

Nothing here changes behaviour for a stock checkpoint: the `mxfp4.py` change is
inert unless the checkpoint declares more than one config group, and the
`weight_utils.py` change only acts on a contradiction between the index and the
file being read, which an ordinary checkpoint never has.

## Accuracy gating

Gate on the **full 1319** GSM8k questions with real joint verification
(`EVAL_ONLY`), not a 250-question subset -- a subset scores ~0.99 and makes a
real regression look like noise.

Quantization error here is uniform and mantissa-limited: per-tensor 0.02653,
per-channel 0.02646, per-block-32 0.02398. A finer scheme buys ~9%, because
e4m3 carries its own exponent bits. If accuracy fails, quantize *less*, not
*better*. For comparison MXFP4 sits at 0.116 on the same weights -- 4.4x
noisier -- which is why it costs ~0.9 GSM8k points.
