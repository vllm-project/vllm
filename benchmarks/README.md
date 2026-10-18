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

## Speculative decoding speedup ceiling (`benchmark_spec_decode_oracle.py`)

`benchmark_spec_decode_oracle.py` measures how much faster speculative decoding *could* be if the drafter were never wrong. It plugs an **oracle drafter** into the existing `custom_class` speculative method: the oracle replays the target model's own greedy output as the draft, so nearly every draft token is accepted and drafting costs nothing. The speedup it reaches is the ceiling for the current verification path; running a real drafter (`ngram`, `eagle3`) through the same script shows how much of that ceiling it captures.

Each configuration runs in its own process:

```bash
# 1. baseline, also writes the reference outputs the oracle replays
python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B --mode baseline \
    --prompts prompts.json --max-num-seqs 32 --no-async-scheduling --write-ref ref.json --out base.json
# 2. oracle ceiling
python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B --mode oracle --k 4 \
    --prompts prompts.json --max-num-seqs 32 --ref ref.json --out oracle_k4.json
# 3. real drafters for comparison
python benchmarks/benchmark_spec_decode_oracle.py --model Qwen/Qwen3-8B --mode eagle3 --k 3 \
    --draft-model RedHatAI/Qwen3-8B-speculator.eagle3 --prompts prompts.json --max-num-seqs 32 --ref ref.json --out eagle3_k3.json
```

`prompts.json` is a JSON list of user-turn strings. Add `--serial` for a batch-size-1 latency measurement. Each run writes a JSON with tokens/s (or time per output token with `--serial`), mean accepted length, draft acceptance rate, per-position acceptance, the fraction of outputs identical to the reference, and, for the oracle, per-step cost and path counters.

Two things to keep in mind when reading the numbers:

- **Compare the oracle against a baseline run with `--no-async-scheduling`.** vLLM disables async scheduling for CPU-side drafters (`custom_class`, `ngram`) but keeps it on for the default baseline and for GPU drafters such as EAGLE3. That alone is worth 15-25% on an H100, so the default baseline would understate the ceiling. Use the flag on EAGLE3 too for a like-for-like real-drafter number.
- **Greedy outputs drift under bf16.** Once a request's batch neighbours change, its greedy output can differ from the reference at some token. By default the oracle re-aligns on the last 3 generated tokens and keeps drafting; `--strict` makes it stop drafting a drifted request instead, which gives a more conservative ceiling. The gap between the two modes is itself a measure of how much that drift costs. The `outputs_identical_frac` field reports how many outputs matched the reference exactly.

Known observations on Qwen3-8B / 1xH100 / vLLM 0.31.0 that are not yet explained: at batch size 1, `k=16` was 2-4x slower per step than `k=12`, and the first speculative process in a session showed elevated per-step time at `k=1`.
