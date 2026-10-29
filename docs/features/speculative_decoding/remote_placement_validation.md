# Remote draft placement: validation snapshot

These measurements describe the cumulative implementation at
[`65cf5b57164efd84b882e197cd10231f27cc03a5`](https://github.com/seo-rii/vllm/commit/65cf5b57164efd84b882e197cd10231f27cc03a5)
on 2026-10-21. Results apply to that source, the pinned models and the
execution settings below.

## Environment and workload

- Two A100-SXM4-40GB GPUs, NVLink (`NV12`), bidirectional P2P;
  target TP1 and draft TP1 on separate GPUs.
- BF16, eager execution, FlashAttention, prefix caching and async scheduling
  disabled in all compared modes. Target GPU memory utilization 0.90,
  max sequences 32, max batched tokens 2048, remote draft KV budget 4 GiB,
  request timeout **5000 ms**. This is a matched configuration comparison,
  not an optimized native-serving baseline.
- PyTorch 2.14.0+cu130, CUDA toolkit 13.0.88; native artifact
  `0.26.1rc1.dev3933+g66e799b65.precompiled` matched to the then-current base.
- MiMo-7B-RL revision `6299b5a2c45daf0c429285c92b8e61a5bd011c0d`, MTP K=1.
- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, EAGLE3 K=3;
  RedHatAI/Qwen3-8B-speculator.eagle3 revision
  `08610ffa01dd9f16731fe8f627b85905b6aa51c4`.

## Correctness and actual speculation

The controlled batch-invariant experiment exercised 14 ragged prompts per
model in target-only, native local, remote inline and remote IPC modes:
112 request executions, 84 comparisons against target-only, and
**1344/1344 matching output token IDs** (16 output tokens per request).
Each speculative mode recorded accepted drafts: 70 for MiMo and 88 for Qwen3.
Same-GPU and cross-GPU CUDA IPC primitive tests both passed (2/2).
These results cover these pinned models and execution settings only.

## Performance

The fixed continuation benchmark ran 32 cells (96 runs), 2688 request executions and
three repetitions per cell. Its repetitive continuation prompts produced
unusually high acceptance (approximately 100% MiMo and 93% Qwen3); they are
not representative chat workloads. No fallback or preemption was recorded in
completed benchmark cells.

Qwen3 median output tokens/s:

| Input tokens / concurrency | Target only | Native local | Remote inline | Remote IPC |
| --- | ---: | ---: | ---: | ---: |
| 512 / 1 | 29.99 | 103.36 | 85.73 | 93.21 |
| 512 / 8 | 221.48 | 659.47 | 467.02 | 593.26 |
| 512 / 32 | 748.85 | 1719.62 | 922.32 | 1517.62 |
| 4096 / 8 | 155.79 | 286.20 | 105.57 | 260.58 |

Remote IPC was 8.95–11.75% below native local for Qwen3 and
9.71–12.98% below it for MiMo. Some MiMo cells ran on a replacement VM in a
different zone; those comparisons include a host change. Qwen3 inline was
17.05–63.11% below native; the long-prompt inline case was slower than
target-only. IPC is an explicit transport option, not zero-copy or automatic
compute overlap.

A supplemental eight-question chat experiment used real chat templates,
354–457 input tokens, 128 forced output tokens, Qwen3 thinking disabled and
one repetition per cell. Remote IPC throughput was 9.59–14.10% below native
local and 1.55–1.90 times target-only at concurrency 1 and 8. Acceptance was
80.50–82.94% for MiMo and 44.50–45.55% for Qwen3. This small supplemental run
is not a statistically established serving speedup.

Batch invariance was disabled for the chat performance runs. Some concurrency-8
outputs differed between modes (including target-only versus native local).
The first differing logits/state require controlled replay; these runs are
not evidence of general token equivalence.

## Memory and context

| Target GPU accounting, native to remote IPC | MiMo | Qwen3 |
| --- | ---: | ---: |
| Non-KV memory relieved | 417.83 MiB | 1966.06 MiB |
| Physical target KV increase | 434.94 MiB | 1936.88 MiB |
| Logical target KV token capacity | 145552 → 152688 | 126816 → 144112 |
| Capacity increase | 4.90% | 13.64% |
| Combined target + draft NVML memory | 36.04 → 46.45 GiB | 36.08 → 45.36 GiB |

Freed target memory was reassigned to KV, so target NVML usage remained around
36 GiB. Remote placement consumes an additional GPU and increases total memory.

All four modes reached the configured model caps (MiMo 32768, Qwen3 40960,
including 64 output tokens). The experiment did **not** establish a larger
maximum context length or eight simultaneously resident full-length contexts.

## Limits and reproduction

The default **100 ms** timeout was tested separately and caused terminal
fallback in Qwen3 inline and IPC cases. A 5000 ms success does not validate the
default deadline for long prefills. Output equality alone is insufficient:
check remote calls, drafted/accepted token counters and fallback reasons.

These measurements cover target TP1 and draft TP1 only.

The rendered [remote placement guide](remote_draft.md) describes configuration
and the server CLI. The committed MiMo smoke test can be run with:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -m pytest -v -s \
  tests/v1/e2e/spec_decode/remote/
```

That test checks real chunked prefill, verification, output token IDs, positive
acceptance and request reuse on one NVIDIA GPU with both processes colocated.
It is separate from the two-GPU benchmark above. The benchmark harness and
raw timing artifacts are not included here.
