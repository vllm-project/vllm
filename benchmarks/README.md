# Benchmarks

This directory used to contain vLLM's benchmark scripts and utilities for performance testing and evaluation.

## Contents

- **Serving benchmarks**: Scripts for testing online inference performance (latency, throughput)
- **Throughput benchmarks**: Scripts for testing offline batch inference performance
- **Specialized benchmarks**: Tools for testing specific features like structured output, prefix caching, long document QA, request prioritization, and multi-modal inference
- **Dataset utilities**: Framework for loading and sampling from various benchmark datasets (ShareGPT, HuggingFace datasets, synthetic data, etc.)

## Usage

For detailed usage instructions, examples, and dataset information, see the [Benchmark CLI documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/#benchmark-cli).

For full CLI reference see:

- <https://docs.vllm.ai/en/latest/cli/bench/latency.html>
- <https://docs.vllm.ai/en/latest/cli/bench/serve.html>
- <https://docs.vllm.ai/en/latest/cli/bench/throughput.html>

## Reference-replay oracle benchmark (`benchmark_spec_decode_oracle.py`)

`benchmark_spec_decode_oracle.py` compares baseline decoding, a reference-replay oracle, `ngram`, and `eagle3`. The oracle uses saved target-model outputs as draft tokens through the existing `custom_class` interface. Replay has nonzero CPU overhead, and proposals may be rejected. Results provide a performance reference for the tested setup, not a theoretical upper bound.

Each configuration runs in its own process:

```bash
# 1. baseline, also writes the reference outputs the oracle replays
python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B --mode baseline \
    --prompts prompts.json --max-num-seqs 32 --no-async-scheduling --write-ref ref.json --out base.json
# 2. reference-replay oracle
python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B --mode oracle --k 4 \
    --prompts prompts.json --max-num-seqs 32 --ref ref.json --out oracle_k4.json
# 3. real drafters for comparison
python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B --mode eagle3 --k 3 \
    --draft-model RedHatAI/Qwen3-8B-speculator.eagle3 --prompts prompts.json --max-num-seqs 32 \
    --no-async-scheduling --ref ref.json --out eagle3_k3.json
```

`prompts.json` is a JSON list of user-turn strings. Add `--serial --max-num-seqs 1` to run one request at a time. Otherwise, `--max-num-seqs` sets the concurrency cap, not a fixed batch size.

Each run writes throughput and raw vLLM speculative-decoding metrics to JSON. Serial mode also reports request latency and generation wall time per output token (`tpot_ms`), including prefill. Where drafts are recorded, `mean_acceptance_len` is 1 + accepted draft tokens / draft events, including the bonus token; `draft_acceptance_rate` is accepted / proposed draft tokens.

When reading the results:

- **Match scheduling settings.** Use `--no-async-scheduling` on baseline and EAGLE3 runs to match the scheduling mode used by CPU-side `custom_class` and `ngram` drafters.
- **Outputs may differ from the saved reference.** The oracle attempts to realign using the last 3 generated tokens; `--strict` disables realignment. With `--ref`, `outputs_identical_frac` measures complete token-sequence equality, not task accuracy. The benchmark does not identify the cause of mismatches.
- **Oracle statistics are diagnostic snapshots.** They are cumulative, include warm-up, and are written every 100 calls, so final calls may be missing and short runs may have no snapshot. `ms_per_call` times the Python proposer body, not a complete decoding step.

See the [Qwen3-8B / H100 experiment report](https://gist.github.com/jiangLLM/e95756b974f385846e4277b7ee68d6c8) for measured results, settings, prompts, and limitations.
