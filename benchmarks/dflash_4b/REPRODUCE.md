# Qwen3.5-4B: vLLM / SGLang, DFlash on / off

This experiment investigates [vLLM issue #49730: lower-than-expected DFlash performance on H100](https://github.com/vllm-project/vllm/issues/49730).
The observed gap: both engines have similar baseline performance and DFlash acceptance lengths, but vLLM DFlash has lower throughput and longer intervals between streamed chunks.

## Experiment

Compare vLLM and SGLang, each with DFlash enabled and disabled:

- **Models:** `Qwen/Qwen3.5-4B` with `z-lab/Qwen3.5-4B-DFlash`.
- **Hardware and precision:** one H100 80 GB per run, TP=1, FP8 target weights, BF16 compute.
- **Workload:** AIPerf `spec_al_gsm8k`, concurrency 1, 20 warmups and 200 measured requests.
- **Output:** at most 256 tokens; `ignore_eos` is not enabled. Sampling uses model defaults.
- **DFlash:** 15 speculative tokens in vLLM, equivalent to SGLang's 16-token block including the bonus slot. vLLM uses MRV2.
- **Timing:** no profiler enabled.

## Run

Requires Linux, a CUDA 13 compatible NVIDIA driver, `uv`, and `canhazgpu` configured for GPU reservations.

```bash
git clone --branch repro/qwen35-4b-dflash https://github.com/tomasruizt/vllm.git
cd vllm
bash benchmarks/dflash_4b/setup.sh

bash benchmarks/dflash_4b/run.sh vllm baseline
bash benchmarks/dflash_4b/run.sh vllm dflash
bash benchmarks/dflash_4b/run.sh sglang baseline
bash benchmarks/dflash_4b/run.sh sglang dflash
```

Results go to `benchmarks/dflash_4b/results/ENGINE_MODE/`.
Use `--output /path/to/new-results` for repeat runs; existing run directories are not overwritten.
For parallel runs, supply different `--port` values; each command reserves one GPU.

Dependency versions are in [setup.sh](setup.sh); checkpoint pins and full server arguments are in [run.py](run.py).
The checkpoint pins are bookkeeping, with no evidence linking those specific revisions to the observed gap.

## Results and interpretation

Read `summary.json` for ITL, TPOT, throughput, and acceptance length.
Each run also saves server and benchmark logs, its configuration, and raw AIPerf and server metrics.

- **ITL:** median interval between streamed chunks (`inter_chunk_latency`), not TPOT.
- **TPOT:** AIPerf's `inter_token_latency`; DFlash can emit multiple tokens per chunk.
- **Throughput:** output tokens per second (`output_token_throughput.avg`).
- **AL:** accepted draft tokens plus the bonus token per iteration. vLLM uses measured-phase counter deltas; SGLang uses a sampled gauge average, so their weighting differs.

Original H100 measurements:

| Engine | Baseline ITL | DFlash ITL | Baseline tokens/s | DFlash tokens/s | DFlash AL |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | 3.29 ms | 8.80 ms | 294.3 | 569.9 | 5.68 |
| SGLang | 3.26 ms | 5.99 ms | 297.9 | 829.2 | 5.70 |

These measurements used `Question: {question}\nAnswer:`; the current AIPerf loader sends the raw question, and has not been rebenchmarked here.
