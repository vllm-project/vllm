# fp8 dense projections for Kimi-K3 on ROCm

Kimi-K3 ships MXFP4 routed experts and leaves every dense projection in bf16 --
`self_attn`, `shared_experts` and the dense MLP are all in the checkpoint's
`ignore` list. This directory builds an overlay checkpoint that serves the
shared-expert projections as fp8, without copying the weights, and documents
how to reproduce the measured result end to end.

Measured on MI355X, TP8, conc=1, DSpark MTP depth 6, 1200 s:

| | interactivity p90 | GSM8k (full 1319) |
|---|---|---|
| stock bf16 | 110.28 | 0.9659 |
| **fp8 `shared_experts` overlay** | **111.99 (+1.55%)** | **0.9659** |

Accuracy is unchanged. The noise floor of a 1200 s conc=1 arm is roughly 1%,
so +1.55% is real but not precise; a 3600 s run tightens it to about 0.6%.

## Quick start

You need 8x MI355X (or another ROCm GPU with aiter) for TP8, the
`moonshotai/Kimi-K3` base checkpoint, the `Inferact/Kimi-K3-DSpark` draft model
for MTP, and ~12 GB of free disk.

**This directory ships no quantized weights.** The PR makes a mixed mxfp4+fp8
checkpoint servable; you build the checkpoint yourself in step 1.

```bash
cd examples/features/kimi_k3_fp8_dense

# 1. build the overlay (CPU only, ~12 GB, under an hour)
python build_overlay.py --src /path/to/Kimi-K3 \
                        --dst /path/to/Kimi-K3-fp8se \
                        --target shared_experts

# 2. verify it will actually be served as fp8 -- do not skip this
python verify_overlay.py --overlay /path/to/Kimi-K3-fp8se \
                         --target shared_experts

# 3. serve (full flag list under "Run the conc=1 agentic workload")
export VLLM_ROCM_USE_AITER=1
vllm serve /path/to/Kimi-K3-fp8se --load-format safetensors \
    --tensor-parallel-size 8 ...

# 4. confirm the layers were really delegated -- expect 184
grep -c 'delegated to CompressedTensorsLinearMethod' server.log
```

Then drive an agentic workload at **concurrency 1** for at least 1200 s and
read interactivity p90. Gate accuracy separately on the full 1319 GSM8k
questions with `"rejection_sample_method":"block"`.

Three things decide whether this works at all:

* **`--load-format safetensors` is mandatory.** `fastsafetensors` has no
  tensor-level index hook and will silently serve the original bf16 weights.
* **`VLLM_ROCM_USE_AITER=1`** routes the fp8 linear to aiter.
* **Step 4 returning 0** means you are measuring the baseline, not the change.
  184 rather than 276 is correct: vLLM fuses `gate_proj`+`up_proj` into a
  single `gate_up_proj`.

Each step is expanded below.

## Scope: quantize only `shared_experts`

This is the entire recommendation, and it is deliberately narrow. Three other
targets were built, accuracy-gated and measured. **None of them helped.**

| target | Linears | p90 | vs control | GSM8k | verdict |
|---|---|---|---|---|---|
| `shared_experts` | 184 | 111.99 | +1.55% | 0.9659 | **use this** |
| `o_proj` | 93 | 111.74 | +1.32% | 0.9651 | adds nothing on top |
| `shared_experts` + `o_proj` | 277 | 112.00 | +1.56% | 0.9659 | same as the first alone |
| `mla_proj` | 96 | 109.96 | -0.29% | 0.9689 | no gain |

The combined arm is the informative one: **277 quantized Linears scored 112.00
against 184 Linears at 111.99**, with an identical tpot p90 floor of 0.00893.
The levers do not compose. At conc=1 this step is not weight-bandwidth-bound,
so once the first target relieves the cost there is nothing left for the second
to take. Quantizing more surface buys accuracy risk and build time for no
measured speedup.

For the same reason MXFP4 on these projections is worse on **both** axes:
**-0.48%** p90 and **0.9568** GSM8k. It halves the bytes again but adds a
per-Linear activation-quant kernel, and the GPU is already ~98% busy.

## Overlay vs online quantization

vLLM can also quantize these layers online from the stock checkpoint, with no
overlay at all:

```bash
vllm serve /path/to/Kimi-K3 \
    --quantization-config '{"targets": {"re:.*shared_experts.*": "fp8_per_channel"}}'
```

That is far easier to adopt, so it was measured on the same 184 Linears:

| approach | p90 | vs control | tpot p90 | build cost |
|---|---|---|---|---|
| online `fp8_per_channel` | 111.14 | +0.78% | 0.00900 | none |
| **overlay (this directory)** | **111.99** | **+1.55%** | **0.00893** | ~12 GB, one command |

The overlay measured roughly twice the gain. The likely reason is the
activation scheme: the online shorthands are **static**, while the overlay is
**w8a8 with dynamic per-token activations**.

Read that gap carefully. The 0.85-point difference is **inside the ~1% noise
floor** of a 1200 s arm, so it is not statistically resolved on its own. What
supports the ordering is that tpot p90 is monotone across all three
measurements (0.00907 control, 0.00900 online, 0.00893 overlay). If you need
the difference established rather than suggested, run both at 3600 s.

Either way the online path needs **its own accuracy gate** before use: static
activations are a different scheme from the one validated here.

## Build the overlay

One command, no chaining. Run from the repository root:

```bash
python examples/features/kimi_k3_fp8_dense/build_overlay.py \
    --src /path/to/Kimi-K3 \
    --dst /path/to/Kimi-K3-fp8se \
    --target shared_experts
```

About 12 GB of new bytes, CPU only, well under an hour. The original shards are
symlinked rather than copied, so nothing is duplicated and the source
checkpoint is never written to. Deleting the output directory reverts
everything.

### Why an overlay rather than a requantized copy

The 276 tensors are spread across 92 of 96 shards, so a conventional rewrite
touches ~1553 GB of a 1561 GB checkpoint. A sharded checkpoint is loaded
through its index, and the index is authoritative: a tensor present in a shard
but absent from the weight map is never read. So the builder symlinks every
original shard, writes fp8 weights and scales into a few small new shards, and
emits an index repointing those tensors at the new shards. The bf16 originals
stay on disk, simply unreferenced.

## Verify before trusting a number

```bash
python examples/features/kimi_k3_fp8_dense/verify_overlay.py \
    --overlay /path/to/Kimi-K3-fp8se --target shared_experts
```

Expect:

```
OK:  276 layers -> group_fp8_shared_experts (w8a8-dynamic, channel)
OK: 276 fp8 layers total; none ignored, none in an unexpected group
```

This check is not optional scaffolding. `should_ignore_layer` is consulted
**before** `config_groups`, so a layer still matched by an `ignore` regex is
served as bf16 even though byte-perfect fp8 tensors exist for it on disk. The
server logs a clean load, accuracy is unchanged because it is the original
model, and the speedup simply fails to appear. That looks like a null result,
not a bug. This cost an hour of GSM8k before the check existed.

## Run the conc=1 agentic workload

### 1. Environment

```bash
export VLLM_ROCM_USE_AITER=1          # routes the fp8 linear to aiter
export DRAFT_ATTN_BACKEND=ROCM_AITER_MLA
export AITER_MLA_NON_CAUSAL=1
```

`VLLM_ROCM_USE_AITER=1` is load-bearing: without it the fp8 scheme resolves to
a different implementation and these numbers do not apply.

### 2. Serve

```bash
vllm serve /path/to/Kimi-K3-fp8se \
    --served-model-name moonshotai/Kimi-K3 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --load-format safetensors \
    --moe-backend auto \
    --gpu-memory-utilization 0.9 \
    --language-model-only \
    --max-num-seqs 2 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3 \
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}' \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":14,"custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[2,3,4,5,6,7,8,9,10,11,12,13,14]}' \
    --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":6,"method":"dspark","attention_backend":"ROCM_AITER_MLA","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"synthetic"}'
```

**`--load-format safetensors` is required and is the one real footgun.** vLLM
narrows the *file* list by the index but then yields *every* tensor in each
file, so without the tensor-level filter the stale bf16 originals are emitted
under the same names and filename sort order decides which wins.
`fastsafetensors` uses a separate `ParallelLoader` path with no equivalent hook
and is **not** covered -- it will silently serve bf16 and you will measure the
control.

Confirm from the server log that the layers were actually delegated:

```bash
grep -c 'delegated to CompressedTensorsLinearMethod' server.log   # expect 184
```

184, not 276, because vLLM fuses `gate_proj` + `up_proj` into a single
`gate_up_proj` Linear (92 fused + 92 `down_proj`).

### 3. Measure interactivity p90 at conc=1

Drive it with an agentic client at **concurrency 1** and read interactivity
p90 (output tokens per second per user). Two settings decide what you are
measuring:

* `"rejection_sample_method":"synthetic"` with a pinned acceptance length is
  the **throughput** configuration. Acceptance is a configured constant, not a
  measurement, so this number says nothing about accuracy.
* `"rejection_sample_method":"block"` performs real joint verification. Use it
  for accuracy runs, and expect it to be slower.

Run for at least **1200 s**; the conc=1 noise floor is ~1% at that length and
~0.6% at 3600 s. Differences below the floor are not resolvable -- several
optimizations on this stack have landed inside a 1.1-point band.

### 4. Gate accuracy

Score GSM8k on the **full 1319** questions, 5-shot, with
`rejection_sample_method` set to `block`:

```
expect 0.9659 on both strict-match and flexible-extract
```

Do not gate on a 250-question subset: it scores ~0.99 and makes a real
regression look like noise.

Quantization error here is uniform and mantissa-limited -- per-tensor 0.02653,
per-channel 0.02646, per-block-32 0.02398. A finer scheme buys ~9%, because
e4m3 carries its own exponent bits. If accuracy fails, quantize *less*, not
*better*. MXFP4 on the same weights sits at 0.116, which is 4.4x noisier, and
costs ~0.9 GSM8k points.

## What this changes for a stock checkpoint

Nothing. The `mxfp4.py` change is inert unless the checkpoint declares more
than one config group, and the `weight_utils.py` change only acts on a
contradiction between the index and the file being read, which an ordinary
checkpoint never has.
