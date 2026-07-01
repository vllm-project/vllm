# XQA Spec-Decode Status for vLLM / FlashInfer

Date: 2026-07-01

Workspace: `/home/scratch/scratch.dblanaru/bench_serving`

vLLM worktree: `vllm-xqa_decode_kernels-squash-rebase`

## Executive Summary

The SM90 XQA speculative-decode path is functionally correct and can be faster
than FAv3 in the right FP8-KV, high-active-batch regime. However, end-to-end
benchmarking is difficult because the serving stack often measures other
bottlenecks before it measures the bare XQA decode kernel:

- `ngram` speculation is model-free but produces ragged verification lengths.
  FlashInfer's public XQA API only exposes uniform `q_len_per_req`, so ragged
  rows fall back to FlashInfer prefill instead of XQA decode.
- EAGLE3 speculation gives a uniform model-based path that can exercise XQA,
  but vLLM's padded EAGLE bookkeeping introduces tiny D2H synchronization
  points for accepted-token counts. These can dominate some e2e cases.
- Long-context SPEED-Bench also includes substantial prefill work, which dilutes
  decode-kernel wins.

The current state is: correctness is good, the kernel microbench is positive in
the expected FP8/high-batch region, and a few e2e cases show XQA wins. The
remaining work is to separate kernel performance from vLLM spec-decode
bookkeeping and to expose ragged XQA support through FlashInfer for model-free
ngram testing.

## Correctness And Enablement

The vLLM branch under test made these behavior changes:

- Added a cached XQA spec-decode causal mask helper.
- Removed the `NotImplementedError` for XQA with `q_len_per_req > 1`.
- Passes `mask=decode_mask` to FlashInfer XQA decode.
- Re-enabled spec-as-decode batching for XQA.
- Updated CUDA graph support so SM90 XQA can advertise `UNIFORM_BATCH`.

Focused correctness tests passed:

```text
tests/v1/attention/test_attention_backends.py::test_flashinfer_xqa_spec_decode_causal_mask
tests/v1/attention/test_attention_backends.py::test_flashinfer_sm90_xqa_decode_correctness
tests/v1/attention/test_attention_backends.py::test_flashinfer_sm90_xqa_spec_decode_correctness

3 passed, 21 warnings in 8.32s
```

Full MMLU ngram correctness was within noise:

| Run | Spec decode | Attention path | MMLU acc | Stderr |
| --- | --- | --- | ---: | ---: |
| FAv3 | ngram, 5 draft tokens | `FLASH_ATTN` | 0.7768 | 0.0033 |
| FI+XQA | ngram, 5 draft tokens | `FLASHINFER`, `decode_backend=xqa` | 0.7750 | 0.0033 |

CUDA graph smoke confirmed full graph support:

```text
FlashInfer resolved query dtypes: prefill=torch.float8_e4m3fn,
decode=torch.bfloat16, decode_backend=xqa,
kv_cache_dtype=torch.float8_e4m3fn, arch=sm90

Profiling CUDA graph memory: PIECEWISE=25 (largest=192), FULL=13 (largest=96)
```

## Kernel Microbenchmark

Artifact:

`vllm-xqa_decode_kernels-squash-rebase/artifacts/xqa_specdecode_pr_grid_cg/speedup_tables.md`

Speedup is `FLASH_ATTN mean_time / FLASHINFER mean_time`; values above `1.00x`
mean FI/XQA is faster.

Model dimensions:

`num_layers=1`, `num_q_heads=32`, `num_kv_heads=8`, `head_dim=128`,
`block_size=16`.

Main reading:

- BF16 KV is mostly slower for FI/XQA.
- FP8 KV is favorable at larger active batches.
- The favorable region starts around active batch `32+`, especially for
  `q=4` and `q=8`.

Representative FP8 tables:

### q=4, FP8 KV

| batch \ ctx | 1k | 2k | 4k | 8k | 16k | 32k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.08x | 0.86x | 0.78x | 0.70x | 0.66x | 0.64x |
| 16 | 0.85x | 0.79x | 0.70x | 0.66x | 0.64x | 0.64x |
| 32 | 1.55x | 1.39x | 1.28x | 1.22x | 1.20x | 1.20x |
| 64 | 1.36x | 1.23x | 1.15x | 1.11x | 1.08x | 1.08x |
| 128 | 1.59x | 1.50x | 1.43x | 1.39x | 1.35x | 1.34x |

### q=8, FP8 KV

| batch \ ctx | 1k | 2k | 4k | 8k | 16k | 32k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.05x | 0.87x | 0.80x | 0.71x | 0.66x | 0.64x |
| 16 | 0.86x | 0.79x | 0.70x | 0.66x | 0.64x | 0.63x |
| 32 | 1.53x | 1.42x | 1.32x | 1.27x | 1.24x | 1.23x |
| 64 | 1.43x | 1.36x | 1.28x | 1.25x | 1.23x | 1.22x |
| 128 | 1.37x | 1.33x | 1.27x | 1.24x | 1.22x | 1.22x |

## End-To-End Results

### EAGLE3, Natural Chat, Fast Case

Artifact:

`artifacts/qwen3-30b-a3b-instruct-fp8-eagle3-sharegpt-pair_20260630_1805/summary.md`

Config:

- Verifier: `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- Speculator: `RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3`
- Dataset: `Aeala/ShareGPT_Vicuna_unfiltered`
- OSL: `500`
- Prompts: `16`
- `max-concurrency=8`
- Endpoint: `/v1/chat/completions`
- Spec config: EAGLE3, 3 draft tokens

Result:

| Case | Output tok/s | Mean TPOT ms | Mean ITL ms | Draft accept | Mean accept len |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAv3 | 1317.10 | 4.56 | 10.00 | 0.4017 | 2.2052 |
| FI+XQA | 1411.46 | 4.59 | 9.95 | 0.3934 | 2.1803 |

`XQA / FAv3 output throughput: 1.072x`

Interpretation: this is the cleanest proof that uniform model-based spec decode
can exercise XQA and show a measurable Hopper GQA win.

### EAGLE3, Random 32k/500, Slow Case

Artifact:

`artifacts/qwen3-30b-a3b-instruct-fp8-eagle3-specdec-pair_20260630_1755/summary.md`

Config:

- Dataset: random-token prompts
- ISL/OSL: `32000/500`
- Prompts: `16`
- `max-concurrency=8`
- Endpoint: `/v1/chat/completions`
- Spec config: EAGLE3, 3 draft tokens

Result:

| Metric | FAv3 | FI+XQA | XQA / FAv3 |
| --- | ---: | ---: | ---: |
| Output throughput (tok/s) | 279.24 | 272.24 | 0.975x |
| Mean TPOT (ms) | 20.18 | 22.55 | 1.117x |
| Mean ITL (ms) | 24.89 | 27.22 | 1.094x |
| Draft acceptance rate | 0.0793 | 0.0705 | 0.889x |
| Mean acceptance length | 1.2380 | 1.2116 | 0.979x |

Interpretation: random tokens are hostile to EAGLE3. Acceptance collapses to
about `7-8%`, so this is not a useful proof of spec-decode acceleration.

### EAGLE3, SPEED-Bench Concurrency Sweep

Artifact:

`artifacts/qwen3-30b-a3b-instruct-fp8-eagle3-speedbench-concurrency_20260630_1815/summary.md`

Config:

- Dataset: SPEED-Bench `low_entropy`
- Subsets: `throughput_8k`, `throughput_16k`
- Prompts per case: `64`
- OSL: `500`
- Concurrencies: `8`, `16`, `32`, `64`
- Endpoint: `/v1/completions` with client-side chat-template rendering
- Spec config: EAGLE3, 3 draft tokens
- KV cache: FP8

Result:

| ISL | max concurrency | FAv3 out tok/s | FI+XQA out tok/s | XQA/FAv3 tok/s | FAv3 TPOT ms | FI+XQA TPOT ms | FAv3 accept % | FI+XQA accept % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8k | 8 | 771.27 | 755.15 | 0.979x | 9.34 | 9.65 | 16.76 | 16.98 |
| 8k | 16 | 1579.61 | 1542.24 | 0.976x | 9.04 | 9.35 | 16.76 | 16.55 |
| 8k | 32 | 2281.17 | 2104.23 | 0.922x | 11.62 | 12.66 | 16.49 | 16.35 |
| 8k | 64 | 3081.76 | 3042.63 | 0.987x | 15.57 | 15.30 | 17.08 | 16.54 |
| 16k | 8 | 495.53 | 486.86 | 0.982x | 14.35 | 15.12 | 10.81 | 10.85 |
| 16k | 16 | 1178.19 | 1104.89 | 0.938x | 11.86 | 12.64 | 10.98 | 10.86 |
| 16k | 32 | 1674.26 | 1438.50 | 0.859x | 16.26 | 18.23 | 11.02 | 10.98 |
| 16k | 64 | 1932.43 | 2054.99 | 1.063x | 24.53 | 23.08 | 10.68 | 10.86 |

Interpretation:

- The kernel win surfaces at the highest tested active-batch/context point:
  `16k`, concurrency `64`, `1.063x`.
- Lower concurrency remains slower or near parity.
- Acceptance is matched, so the slow points are not explained by worse EAGLE3
  quality on XQA.

### Ngram, Model-Free, Slow Case

Artifacts:

- `vllm-xqa_decode_kernels-squash-rebase/artifacts/e2e_specbench_ngram_fullcg/summary.md`
- `vllm-xqa_decode_kernels-squash-rebase/artifacts/e2e_speedbench_ngram_fullcg_32p_500osl/summary.md`
- `vllm-xqa_decode_kernels-squash-rebase/artifacts/e2e_speedbench_ngram_concurrency_sweep/summary.md`

SpecBench, OSL 256, concurrency 16:

| Metric | FAv3 | FI+XQA | XQA / FAv3 |
| --- | ---: | ---: | ---: |
| Output throughput (tok/s) | 1608.11 | 1315.28 | 0.818x |
| Mean TPOT (ms) | 9.72 | 11.91 | 1.225x |
| Spec acceptance rate (%) | 20.30 | 20.28 | 0.999x |

SPEED-Bench ngram, concurrency sweep:

| ISL | max concurrency | FAv3 out tok/s | FI+XQA out tok/s | XQA/FAv3 tok/s |
| --- | ---: | ---: | ---: | ---: |
| 8k | 8 | 714.45 | 446.71 | 0.625x |
| 8k | 32 | 2128.79 | 782.76 | 0.368x |
| 16k | 8 | 511.35 | 295.10 | 0.577x |
| 16k | 32 | 1614.05 | 463.25 | 0.287x |
| 16k | 64 | 2081.97 | 467.52 | 0.225x |

Interpretation:

Ngram is not testing the same thing as uniform XQA spec decode. Ngram produces
variable verification lengths. The FAv3 path can use varlen attention, but the
FlashInfer XQA path currently requires uniform `q_len_per_req`, so non-uniform
rows fall back through FlashInfer prefill.

## Profile Findings

### Ngram Profile

Artifact:

`vllm-xqa_decode_kernels-squash-rebase/artifacts/e2e_speedbench_ngram_profile/summary.md`

Case: SPEED-Bench `throughput_16k`, `low_entropy`, `num_prompts=32`,
`OSL=200`, `max-concurrency=32`.

| Metric | FAv3 | FI+XQA | XQA / FAv3 |
| --- | ---: | ---: | ---: |
| Output throughput (tok/s) | 401.00 | 254.54 | 0.635x |
| Mean TPOT (ms) | 46.20 | 88.15 | 1.908x |
| Mean ITL (ms) | 54.15 | 106.98 | 1.976x |

Key XQA profile finding:

```text
65.1%  15621.5487 ms  flashinfer::FP8PrefillWithKVCacheKernel...
1.0%     250.2044 ms  kernel_mha(...)
0.7%     167.3734 ms  kernel_mha(...)
```

Interpretation: ngram slowdown is dominated by FlashInfer prefill fallback, not
by XQA `kernel_mha`.

### EAGLE3 Profile

Artifacts:

- `artifacts/qwen3-30b-a3b-instruct-fp8-eagle3-speedbench-profile_16k_c32_64p_500osl_20260630_1855/summary.md`
- `artifacts/qwen3-30b-a3b-instruct-fp8-eagle3-speedbench-profile_xqa_sqlite_16k_c32_64p_500osl_20260630_1905/summary.md`
- `artifacts/qwen3-30b-a3b-instruct-fp8-eagle3-speedbench-profile_xqa_noasync_16k_c32_64p_500osl_20260701_1200/summary.md`

Exact-shape profile case: SPEED-Bench `throughput_16k`, `num_prompts=64`,
`OSL=500`, `max-concurrency=32`.

Under profiler:

| Metric | FAv3 | FI+XQA | XQA / FAv3 |
| --- | ---: | ---: | ---: |
| Output throughput (tok/s) | 694.24 | 683.87 | 0.985x |
| Mean TPOT (ms) | 36.08 | 36.36 | 1.008x |
| Mean ITL (ms) | 47.18 | 47.69 | 1.011x |
| Acceptance rate (%) | 11.11 | 11.08 | 0.997x |

Profile observations:

- Profiling changes the c32 miss magnitude: unprofiled sweep was `0.859x`,
  exact-shape profiled run was near parity. Treat profiles as diagnostic.
- XQA `kernel_mha` itself took about `9.08s`.
- XQA also spent about `8.43s` in
  `flashinfer::FP8PrefillWithKVCacheKernel`. This is normal long-context
  prefill work, not ngram ragged fallback.
- FAv3 attention kernels totaled about `18.06s`; XQA `kernel_mha + FP8Prefill`
  totaled about `17.51s`. At this e2e shape, the attention-side win is only
  about `3%`, far smaller than the isolated decode microbenchmark.

Direct SQLite drilldown:

- Slow `cudaMemcpyAsync >= 50 ms`: `87` calls, `10894.9 ms` total API time.
- Correlated GPU copies are tiny and fast: mostly `16-128` bytes,
  Device-to-Host, about `0.002 ms` each.
- These slow calls occur inside `gpu_model_runner: preprocess`.
- Attribution over all preprocess ranges:
    - `gpu_model_runner: preprocess`: `830` ranges, `12783.2 ms` total.
    - `cudaMemcpyAsync` API inside preprocess: `11298.1 ms`.
    - Slow copies inside preprocess: `10894.9 ms`.

The byte sizes match `int32[num_reqs]` accepted-token-count buffers. Code
inspection points to the EAGLE padded drafting path:

```python
# vllm/v1/worker/gpu_model_runner.py
counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
```

Disabling async scheduling did not fix it:

| Metric | FI+XQA async | FI+XQA no async |
| --- | ---: | ---: |
| Output throughput (tok/s) | 683.87 | 647.23 |
| Mean TPOT (ms) | 36.36 | 38.53 |
| Mean ITL (ms) | 47.69 | 50.76 |
| Acceptance rate (%) | 11.08 | 10.97 |

With `--no-async-scheduling`, the wait moved from `preprocess` to `bookkeep`
and throughput got worse:

- `gpu_model_runner: preprocess`: about `1.2s`.
- `gpu_model_runner: bookkeep`: about `26.4s`.
- `cudaMemcpyAsync` API time: about `26.4s`.
- Slow copies remained tiny D2H copies, mostly `320-512` bytes.

Interpretation:

The c32 EAGLE3 miss is mostly vLLM EAGLE/spec-decode bookkeeping
synchronization, not XQA `kernel_mha`. Async scheduling hides some of the cost
by moving it into the next step's preprocess; disabling async moves the wait to
bookkeeping and makes throughput worse.

## MTP Feasibility

MTP may be worth testing, but it should not be treated as guaranteed to avoid
the observed bookkeeping bottleneck.

Relevant vLLM facts:

- MTP is for models with native multi-token prediction capability.
- It does not require a separate EAGLE-style draft model in the generic case.
- vLLM recommends starting with `num_speculative_tokens=1`.
- Supported families include MTP-capable Qwen3Next/Qwen3.6-style models,
  DeepSeek V3/R1-style models, GLM-4.x, Gemma 4 assistant checkpoints, and
  Xiaomi MiMo examples.

Candidate model directions under the Qwen3-235B cap:

| Candidate | Why it is relevant | Caveats |
| --- | --- | --- |
| `Qwen/Qwen3.6-27B` or compatible MTP variants | Closest Qwen-family size to the current 30B target; model cards show MTP usage with vLLM. | Different architecture from Qwen3-30B-A3B; some public variants are AWQ/INT4 grafts and may add quantization confounds. |
| `shawnw3i/Qwen3.6-27B-AWQ-MTP` | Single-GPU-ish quantized MTP checkpoint with explicit vLLM MTP example. | Not a clean FP8 Qwen checkpoint; third-party packaging. |
| `hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16` | MTP tensors grafted as BF16 for vLLM MTP. | Quantized main model; useful for path testing, less clean for kernel claims. |
| GLM-4.x with MTP | Documented MTP in vLLM recipes; FP8 variants exist. | Not Qwen/GQA shape; may require TP and model-specific parser flags. |
| DeepSeek V3/R1 MTP | Native MTP support; no separate draft model. | Very large MoE, MLA/sparse stack, not the same XQA GQA path. DeepSeek MTP often supports only small speculative depths cleanly. |
| Gemma 4 assistant MTP | vLLM MTP path with assistant checkpoints. | Different architecture and not a Qwen Hopper GQA proxy. |
| Xiaomi MiMo 7B | vLLM MTP docs example. | Small model; useful smoke test only. |

Important code-path caution:

The vLLM path for padded model-based spec decode still needs to know how many
tokens were accepted. The current EAGLE path copies a small
`valid_sampled_tokens_count` buffer to CPU every step. Some MTP paths may share
similar bookkeeping. Therefore MTP is a useful experiment, but the question is:
does a given MTP implementation avoid the EAGLE padded accepted-count sync, or
does it hit the same vLLM synchronization pattern?

Suggested MTP next test:

1. Start with a Qwen-family MTP candidate, ideally `Qwen3.6-27B` or a known
   vLLM-compatible MTP variant under the Qwen3-235B cap.
2. Use `num_speculative_tokens=1` first.
3. Run a small FAv3 vs FI+XQA pair on ShareGPT or SPEED-Bench 16k/c32.
4. Profile only if e2e behavior is surprising.

Example shape:

```bash
vllm serve <mtp-capable-model> \
  --tensor-parallel-size <tp> \
  --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER \
  --attention-config.use_trtllm_attention true \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

For some Qwen3Next/Qwen3.6 variants, model cards show:

```bash
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

The exact method string should be verified against the local vLLM version and
model config before benchmarking.

## FlashInfer Ragged-XQA Scope

Short answer: a Python-only editable FlashInfer change is not enough to test
ragged ngram speed on XQA.

Why:

- FlashInfer Python `trtllm_batch_decode_with_kv_cache` exposes
  `cum_seq_lens_q` and `max_q_len`, but rejects them for `backend="xqa"`:

```python
if backend == "xqa":
    if max_q_len is not None or cum_seq_lens_q is not None:
        raise ValueError("xqa backend does not support cum_seq_lens_q")
```

- FlashInfer `xqa.py` only forwards scalar `q_seq_len` and `mask`.
- FlashInfer C++ binding `flashinfer_xqa_binding.cu` does not include a
  `qCuSeqLens` / `q_cu_seq_lens` argument.
- FlashInfer C++ wrapper `csrc/xqa/xqa_wrapper.cu` passes `nullptr` where
  TRT-LLM native XQA expects `qCuSeqLens`:

```cpp
#if SPEC_DEC
    qSeqLen, nullptr, maskPtr,
#endif
```

- Native TRT-LLM XQA has the needed field:

```cpp
struct SpecDecParams {
    uint32_t qSeqLen;
    uint32_t const* qCuSeqLens; // [nbReq + 1]
    MaskType const* mask;
};
```

Likely FlashInfer changes:

1. Python API:
   - Add optional `q_cu_seq_lens` / `cum_seq_lens_q` to `flashinfer.xqa.xqa`.
   - Allow `cum_seq_lens_q` for `backend="xqa"` in `decode.py`.
   - Define the expected query layout for ragged XQA: flattened query rows with
     `max_q_len` as the compile-time/runtime max sequence length.

2. Python registered custom op:
   - Add the optional tensor to the custom-op signature and fake op.
   - Include ragged-vs-uniform in the JIT/module cache key if the generated
     code or op schema differs.

3. C++ TVM-FFI binding:
   - Add `Optional<TensorView> qCuSeqLens` to
     `flashinfer_xqa_binding.cu`.

4. C++ XQA wrapper:
   - Convert the optional tensor to `uint32_t const*`.
   - Pass it instead of `nullptr` to `launchHopperF8MHAFlashInfer` /
     `launchMHAFlashInfer` under `SPEC_DEC`.
   - Continue passing `nullptr` for uniform spec decode.

5. Mask support:
   - Current uniform mask shape is `[batch, q_seq_len, packed_cols]`.
   - TRT-LLM comments allow ragged flattened masks:
     `[qCuSeqLens[nbReq], packed_cols]`.
   - vLLM needs a ragged packed causal mask helper or an adapted mask layout
     that the XQA wrapper/kernel accepts.

6. vLLM plumbing after FlashInfer supports it:
   - Stop requiring uniform split for XQA spec decode.
   - Pass `cum_seq_lens_q`, `max_q_len`, and a ragged packed mask to FlashInfer.
   - Avoid falling back to FlashInfer prefill for variable-length ngram
     verification rows.

Can this be tested without a full FlashInfer reinstall?

Probably yes, if using FlashInfer's editable/JIT development flow, but it is not
Python-only. FlashInfer's JIT system can pick up source changes in `csrc/` and
generated wrappers after cache invalidation, so an experiment may not require a
full wheel rebuild. But it still requires changing C++/CUDA binding/wrapper
source, not just Python files.

Practical experiment path:

```bash
cd /home/scratch/scratch.dblanaru/bench_serving/flashinfer
# after C++/Python edits
rm -rf ~/.cache/flashinfer
# run a small XQA ragged unit test / repro so JIT rebuilds the modified op
```

## Recommended Path Forward

1. For the current XQA PR/status:
   - Present correctness, kernel speedups, and limited e2e wins.
   - Be explicit that e2e speedup is workload-dependent.
   - Do not claim a blanket EAGLE3 speedup.

2. For model-based uniform testing:
   - Try MTP with a compatible model, preferably Qwen-family if available.
   - Start at `num_speculative_tokens=1`.
   - Check whether MTP avoids the EAGLE accepted-count D2H synchronization.

3. For model-free ngram testing:
   - Treat ragged XQA as a FlashInfer C++/binding/vLLM integration project.
   - Python-only FlashInfer edits will not be sufficient.

4. For performance optimization:
   - Avoid optimizing `kernel_mha` first; it is not the main observed bottleneck.
   - Investigate how vLLM can avoid or defer CPU-visible accepted-token-count
     synchronization in padded model-based spec decode.
   - Investigate whether FlashInfer/XQA metadata/preprocess can be reduced or
     better overlapped.
