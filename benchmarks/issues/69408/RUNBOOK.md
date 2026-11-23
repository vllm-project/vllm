# #69408: Parallel sampling performance regression – measurement runbook

**Status: experiment only, NO throughput fix yet.** Upstream issue reports 10–14% regression at n=32 between v0.28.0 and v0.34.0, with n=1 unchanged on an L40S.

## Exact controlled benchmark

Use separate containers with the official images pinned by digest as reported in the issue; serve the same `Qwen/Qwen3-14B` checkpoint, same vLLM flags, same hardware and current driver. **Do not run the two servers concurrently** on the GPU. Record `nvidia-smi`, driver, image digest, version and seed for every run. Alternate the ordering of versions (ABBA over at least four rounds) to limit warm-cache and thermal biases.

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

Inspect scheduling time, sampled step CPU time, per-step logits/sampling kernel work, CUDA Graph dispatch and GPU trace before editing production code. In particular test the hypothesis that Model Runner V2's parallel child handling changed, instead of assuming that `ParentRequest` itself causes the slowdown. Profiling and bisecting v0.29–v0.33 are required before proposing a fix.

Use distinct output filenames such as `v028_n32_seed42.json` and
`v034_n1_seed42.json`. Keep results with failed requests out of the
comparison. The CLI's TPOT reflects the request's streamed response and
aggregate completion-token count. For per-choice latency, record the first
and last token timestamps by choice index separately. Report CLI TPOT and
per-choice TPOT with their measurement definitions.
