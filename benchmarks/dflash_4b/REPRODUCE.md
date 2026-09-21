# Qwen3.5-4B: vLLM / SGLang, DFlash on / off

This branch adds a portable reproduction to vLLM commit `b6e7c1f1f0430b5d4784aea391c42067581b7f76`; there are no engine changes.
The original measurements used one NVIDIA H100 80 GB per run, TP=1, FP8 target weights, BF16 compute, context length 32768, concurrency 1, 20 warmups followed by 200 measured requests, and 256 requested output tokens.
No profiler is enabled.

## Setup

Use a Linux NVIDIA GPU machine with a CUDA 13 compatible driver, `uv`, and `canhazgpu` installed and configured for GPU reservations.
The original environment used Python 3.12, vLLM at the commit above with PyTorch 2.13.0+cu130, SGLang 0.5.17 with PyTorch 2.11.0+cu130 and sglang-kernel 0.4.5, FlashInfer 0.6.18.post1, and AIPerf 0.12.0.
The setup script creates separate engine environments because their PyTorch requirements differ.
SGLang overrides its older FlashInfer and CUTLASS dependency pins with FlashInfer 0.6.18.post1 and CUTLASS DSL 4.7.1, matching the measured environment; it also pins FlashAttention 4.0.0b19.
Consequently, plain `uv pip check` in the SGLang environment reports the two intentional metadata mismatches.
Model downloads require internet access and enough disk space for both checkpoints and environments.

```bash
git clone --branch repro/qwen35-4b-dflash https://github.com/tomasruizt/vllm.git
cd vllm
bash benchmarks/dflash_4b/setup.sh
```

Setup pins the main packages and the precompiled vLLM wheel commit; it is not a full transitive dependency lock.
The pinned nightly wheel must still be available from the vLLM wheel service.
If using existing environments, set `ENGINE_PYTHON` to the appropriate environment's Python when invoking `run.sh`.

## Run all four configurations

```bash
bash benchmarks/dflash_4b/run.sh vllm baseline
bash benchmarks/dflash_4b/run.sh vllm dflash
bash benchmarks/dflash_4b/run.sh sglang baseline
bash benchmarks/dflash_4b/run.sh sglang dflash
```

Each command reserves one GPU using `canhazgpu`, starts a server bound to localhost, waits for readiness, runs AIPerf, saves results, and stops the server.
Results go to `benchmarks/dflash_4b/results/ENGINE_MODE/`; an existing run directory is never overwritten.
Use `--output /path/to/new-results` for repeat runs.
For parallel runs, use different `--port` values and reserve a separate GPU for each through the wrapper.
GPU memory, CPU speed, driver, and dependency differences can affect results.

## Workload and engine settings

`inputs.json` contains all 1319 public GSM8K test questions, in their original order, formatted as `Question: {question}\nAnswer:`.
The payloads are identical to those used in the original comparison; only random session UUIDs have been replaced by deterministic IDs.
The dataset comes from OpenAI's `grade-school-math` repository; its MIT license is included in `GSM8K_LICENSE`.
AIPerf selects the workload using the same inputs-json configuration for every run, with 20 warmups and 200 measured requests.
Requests set `max_completion_tokens=256` and do not set `ignore_eos=true` in either engine.
Therefore, 256 is a maximum, not a guaranteed output length: generation can stop earlier on an EOS (end-of-sequence) token.
Setting `ignore_eos=true` in both engines would enforce generation up to the token limit, but would change the workload from the original comparison.
Sampling parameters are omitted, matching the original requests; the engines use their model defaults, so this is not a deterministic generation or accuracy test.

Both checkpoints are pinned:

| Checkpoint | Revision |
| --- | --- |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `z-lab/Qwen3.5-4B-DFlash` | `9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` |

vLLM explicitly uses MRV2, max-num-seqs 128, prefix caching, the qwen3 reasoning parser, qwen3_coder tool parser, and graph capture sizes 1, 2, 4, 8, 16, 32, 64, 128.
SGLang retains its original server defaults for graph sizes and maximum running requests, with memory fraction 0.8 and metrics enabled.
Client concurrency is 1 for both engines.
DFlash uses 15 speculative tokens in vLLM and a 16-token block in SGLang (including the bonus slot).
The launcher preserves each engine's original parser and scheduler configuration; those settings were not identical between engines.

## Metrics and logs

Each run saves `server.log`, `benchmark.log`, `run_config.json`, before/after Prometheus snapshots, all AIPerf artifacts, and `summary.json`.
SGLang also saves server-info snapshots.

- **ITL:** median `inter_chunk_latency`, the interval between streamed chunks, in milliseconds.
- **TPOT:** AIPerf's `inter_token_latency`; this differs from ITL when DFlash emits multiple tokens in a chunk.
- **Throughput:** AIPerf's `output_token_throughput.avg`, in output tokens/second.
- **AL:** includes the bonus token. vLLM uses `1 + accepted draft tokens / draft iterations` from measured-phase counter deltas. SGLang uses the measured-phase sampled `spec_accept_length` gauge average; the weighting differs, so these are approximate comparisons.

The before/after snapshots include warmups; summary AL uses AIPerf's measured-phase metrics instead.
Missing required metrics fail summary generation, leaving the raw artifacts for inspection.

Original H100 reference measurements (not guaranteed on a different machine):

| Engine | Baseline ITL | DFlash ITL | Baseline tokens/s | DFlash tokens/s |
| --- | ---: | ---: | ---: | ---: |
| vLLM | 3.29 ms | 8.80 ms | 294.3 | 569.9 |
| SGLang | 3.26 ms | 5.99 ms | 297.9 | 829.2 |

## Publication scope

Only benchmark code, instructions, and verified public GSM8K prompts are included.
Environments, original logs, profiles, archives, and generated results are ignored.
New runtime logs can contain local paths, host details, and model output; review them separately before sharing.
