# Qwen3.5-4B: vLLM / SGLang, DFlash on / off

This reproduction investigates [vLLM issue #49730: lower-than-expected DFlash performance with Qwen3.5-4B on H100](https://github.com/vllm-project/vllm/issues/49730).
Our follow-up comparison found similar baseline performance across vLLM and SGLang, but lower throughput and longer intervals between streamed chunks with vLLM DFlash, despite similar acceptance lengths.

The original measurements used one NVIDIA H100 80 GB per run, TP=1, FP8 target weights, BF16 compute, context length 32768, concurrency 1, 20 warmups followed by 200 measured requests, and 256 requested output tokens.
No profiler is enabled.

## Setup

Use a Linux NVIDIA GPU machine with a CUDA 13 compatible driver, `uv`, and `canhazgpu` installed and configured for GPU reservations.
The setup script creates separate engine environments because their PyTorch requirements differ.

```bash
git clone --branch repro/qwen35-4b-dflash https://github.com/tomasruizt/vllm.git
cd vllm
bash benchmarks/dflash_4b/setup.sh
```

If using existing environments, set `ENGINE_PYTHON` to the appropriate environment's Python when invoking `run.sh`.

## Run all four configurations

```bash
bash benchmarks/dflash_4b/run.sh vllm baseline
bash benchmarks/dflash_4b/run.sh vllm dflash
bash benchmarks/dflash_4b/run.sh sglang baseline
bash benchmarks/dflash_4b/run.sh sglang dflash
```

Results go to `benchmarks/dflash_4b/results/ENGINE_MODE/`; an existing run directory is never overwritten.
Use `--output /path/to/new-results` for repeat runs.
For parallel runs, use different `--port` values and reserve a separate GPU for each through the wrapper.

## Workload and engine settings

All four runs use AIPerf’s `spec_al_gsm8k` dataset.
`max_completion_tokens=256`; `ignore_eos` is not enabled, so EOS can end generation earlier.
Sampling uses the engines’ model defaults.

vLLM explicitly uses MRV2, max-num-seqs 128, prefix caching, the qwen3 reasoning parser, qwen3_coder tool parser, and graph capture sizes 1, 2, 4, 8, 16, 32, 64, 128.
SGLang retains its original server defaults for graph sizes and maximum running requests, with memory fraction 0.8 and metrics enabled.
DFlash uses 15 speculative tokens in vLLM and a 16-token block in SGLang (including the bonus slot).

## Metrics and logs

Each run saves `server.log`, `benchmark.log`, `run_config.json`, before/after Prometheus snapshots, all AIPerf artifacts, and `summary.json`.
SGLang also saves server-info snapshots.

- **ITL:** median `inter_chunk_latency`, the interval between streamed chunks, in milliseconds.
- **TPOT:** AIPerf's `inter_token_latency`; this differs from ITL when DFlash emits multiple tokens in a chunk.
- **Throughput:** AIPerf's `output_token_throughput.avg`, in output tokens/second.
- **AL:** includes the bonus token. vLLM uses `1 + accepted draft tokens / draft iterations` from measured-phase counter deltas. SGLang uses the measured-phase sampled `spec_accept_length` gauge average; the weighting differs, so these are approximate comparisons.

The before/after snapshots include warmups; summary AL uses AIPerf's measured-phase metrics instead.

Original H100 reference measurements used `Question: {question}\nAnswer:`; the current AIPerf loader sends the raw question.

| Engine | Baseline ITL | DFlash ITL | Baseline tokens/s | DFlash tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 3.29 ms | 8.80 ms | 294.3 | 569.9 |
| SGLang | 3.26 ms | 5.99 ms | 297.9 | 829.2 |

## Appendix: environment recorded for the original runs

These pins record the environment used on September 18, 2026, for reproducibility; they were not selected through performance tuning or verified to all be the latest releases.
The engines require different PyTorch versions.
SGLang overrides its older FlashInfer and CUTLASS pins to match the measured environment, so `uv pip check` reports those two metadata mismatches.

### Engine and dependency versions

| Component | Recorded version |
| --- | --- |
| vLLM | `b6e7c1f1f0430b5d4784aea391c42067581b7f76` |
| SGLang | `0.5.17` |
| Python | `3.12` |
| PyTorch, vLLM | `2.13.0+cu130` |
| PyTorch, SGLang | `2.11.0+cu130` |
| sglang-kernel | `0.4.5` |
| FlashInfer | `0.6.18.post1` |
| CUTLASS DSL | `4.7.1` |
| FlashAttention | `4.0.0b19` |
| AIPerf | `0.12.0` |

Setup pins the main packages and the precompiled vLLM wheel commit; it is not a full transitive dependency lock.

### Model revisions

These revisions were pinned as routine bookkeeping when packaging the benchmark, with no issue-specific reason for choosing them.
We have no evidence that these particular revisions explain the observed performance gap or are required to observe it.
The hashes below simply record what was used.

| Checkpoint | Revision |
| --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `z-lab/Qwen3.5-4B-DFlash` | `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |
