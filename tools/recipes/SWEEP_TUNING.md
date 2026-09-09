# Sweep Tuning

Sweep tuning is optional. The converter always creates one initial `config.yml`
first. Users can deploy it immediately or benchmark nearby scheduler settings
and generate one measured recommendation.

## Generate

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --detect-hardware \
  --input-tokens 128 \
  --output-tokens 128 \
  --concurrency 32 \
  --ttft-sla-ms 3000 \
  --tpot-sla-ms 100 \
  --generate-sweep
```

The optional package is:

```text
sweep/
├── sweep_config.yml
├── serve_params.json
├── bench_params.json
├── run_sweep.sh
├── recommend.py
└── SWEEP.md
```

The sweep varies only `max-num-seqs` and `max-num-batched-tokens`. It keeps the
eight-point directed tuned design: a batch-budget curve at the initial sequence
count plus lower/higher batch interactions at three-quarters and one-half of
that count. It also adds three default-reference candidates:

| Reference candidate | `max-num-seqs` | `max-num-batched-tokens` |
| --- | --- | --- |
| `vllm_default_max_num_seqs` | vLLM default | initial generated value |
| `vllm_default_max_num_batched_tokens` | initial generated value | vLLM default |
| `vllm_defaults` | vLLM default | vLLM default |

The default references are intentionally **not** hard-coded to values such as
128 or 2048. The generated `sweep_config.yml` copies the initial `config.yml`
but removes `max-num-seqs` and `max-num-batched-tokens`. Tuned candidates add
explicit CLI values, while default-reference candidates omit one or both CLI
arguments so `vllm serve` resolves its normal runtime defaults.

This keeps the comparison platform-, parallel-world-size-, model-, and
usage-context-aware, and avoids passing the literal string `None` through the
generic sweep CLI. If a default-reference candidate wins,
`recommended-config.yml` omits that scheduler key so the deployed server
continues to use the vLLM runtime default.

The serving benchmark scales request count with workload concurrency, following
vLLM's `PROMPTS_PER_CONCURRENCY` model:

```text
num_prompts = min(1000, max(100, concurrency * 10))
```

This targets about ten concurrency turnovers when neither bound applies. The
100-prompt floor avoids very small samples for P99 latency and 99% compliance
measurements, while the 1000-prompt cap bounds sweep runtime. Workload
concurrency is fixed during the scheduler sweep, so every scheduler candidate
uses the same number of prompts.

## Run and Recommend

```bash
sweep/run_sweep.sh --dry-run
sweep/run_sweep.sh
sweep/recommend.py
```

Resume an interrupted sweep with:

```bash
sweep/run_sweep.sh --resume
```

`recommend.py` writes:

```text
sweep/recommended-config.yml
sweep/recommendation.json
```

With TTFT/TPOT objectives, the benchmark uses vLLM `--goodput`. The recommender
calculates duration-weighted combined compliance across repeated runs and
requires both median P99 latency objectives and the minimum compliance ratio
(default `0.99`). Among eligible configurations it selects the highest mean
output-token throughput. Use `recommend.py --minimum-compliance VALUE` to
change the compliance requirement.

If no configuration is eligible, `recommend.py` records the highest-goodput
configuration as `best_effort` in `recommendation.json`, does not write a
deployable `recommended-config.yml`, and exits with status 2. Without latency
objectives it selects highest mean output-token throughput. Failed benchmark
configurations are excluded in either mode.

When an eligible configuration exists, `recommended-config.yml` changes only
the two swept scheduler parameters. `recommendation.json` records mean, median,
and worst-run P99 values, combined compliance, and the measured evidence for
every candidate.

Deploy:

```bash
source env.sh
vllm serve --config sweep/recommended-config.yml
```

## vLLM CPU Docker Shell

Use the common CPU container setup in
[RUNTIME_TUNING.md](RUNTIME_TUNING.md#vllm-cpu-docker-shell). It ensures
hardware detection runs against the same effective CPU, NUMA, memory, and
cgroup limits used by the deployment.

Inside that container, generate the initial suggestion plus sweep package:

```bash
python3 /recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --detect-hardware \
  --input-tokens 128 \
  --output-tokens 128 \
  --concurrency 32 \
  --ttft-sla-ms 3000 \
  --tpot-sla-ms 100 \
  --config-out /output/config.yml \
  --env-out /output/env.sh \
  --generate-sweep \
  --sweep-out-dir /output/sweep
```

The initial configuration can be deployed directly:

```bash
source /output/env.sh
vllm serve --config /output/config.yml
```

Stop that manually started server before running the sweep, because
`run_sweep.sh` starts and stops its own vLLM servers.

```bash
/output/sweep/run_sweep.sh --dry-run
/output/sweep/run_sweep.sh
/output/sweep/recommend.py
```

Inspect:

```bash
cat /output/sweep/recommendation.json
```

Deploy the measured result:

```bash
source /output/env.sh
vllm serve --config /output/sweep/recommended-config.yml
```
