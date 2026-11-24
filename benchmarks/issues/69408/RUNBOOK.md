# #69408: Parallel sampling performance regression – measurement runbook

**Scope: benchmark reproduction and analysis; no serving fix in this PR.** The upstream issue reports a 10–14% throughput loss at `n=32` between v0.28.0 and v0.34.0 on an L40S, with `n=1` approximately unchanged.

## Exact controlled benchmark

Use separate containers with the official images pinned by digest as reported in the issue; serve the same `Qwen/Qwen3-14B` checkpoint, same vLLM flags, same hardware and current driver. **Do not run the two servers concurrently** on the GPU. Record `nvidia-smi`, driver, image digest, model revision, version, seed, effective KV block size, and prefix-cache settings for every run. Alternate which build runs first across matched rounds to limit warm-cache and thermal biases.

Issue's starting benchmark:

```bash
vllm bench serve --backend vllm --base-url http://localhost:8000 \
  --endpoint /v1/completions --model base-instruct-model \
  --tokenizer Qwen/Qwen3-14B --dataset-name random \
  --random-prefix-len 80 --random-input-len 3700 \
  --random-output-len 32 \
  --random-range-ratio '{"input": 0.3, "output": 0.5}' \
  --ignore-eos --extra-body '{"n": 32, "temperature": 0.6, "top_p": 0.9}' \
  --max-concurrency 4 --num-prompts 24 --seed 42 \
  --save-result --result-dir results --result-filename run_001.json
```

Also run the n=1 control with all other flags identical, n explicitly set to 1. For the multi-run analysis, gather JSON filenames into 4 lists by version and n, then run:

```bash
python benchmarks/issues/69408/analyze.py \
  --baseline 'v028_n32_*.json' --candidate 'v034_n32_*.json' \
  --control-baseline 'v028_n1_*.json' \
  --control-candidate 'v034_n1_*.json' --n 32
```

Every input filename or glob must match at least one file. If an input has no
matches, the analyzer exits with an error before emitting a comparison.

## Separate the release comparison from the block-size control

The [reporter's L40S bisect](https://github.com/vllm-project/vllm/issues/69408#issuecomment-7139637553)
isolated [#63231](https://github.com/vllm-project/vllm/pull/63231), which changed
the effective default KV block size from 16 to 64 tokens on that setup.
Model Runner V2 was already active in the reporter's v0.28.0 runs. The
reporter found that parallel choices recalculate the uncached prompt tail:
larger blocks can leave more tokens to prefill for each choice. At concurrency
4 on v0.34.0, the reporter measured about 20.1 samples/s at default size 64
and 22.22 samples/s with size 16.

- To reproduce the user-visible change, compare the two releases with their
  own defaults and record the effective block size in each run.
- To isolate block size, repeat on one build with explicit
  `--block-size 16` and `--block-size 64`. To check for remaining
  release differences, pin the same supported size in both releases.
  Run `n=1` and `n=32` for each comparison. Keep the prompt lengths and
  prefix-cache settings matched and do not mix default and explicit-size
  results in the same analyzer input.

Use distinct output filenames such as `v028_n32_seed42.json` and
`v034_n1_seed42.json`. Keep results with failed requests out of the
comparison. The CLI's TPOT reflects the request's streamed response and
aggregate completion-token count. For per-choice latency, record the first
and last token timestamps by choice index separately. Report CLI TPOT and
per-choice TPOT with their measurement definitions.
