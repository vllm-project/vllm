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
  --input-tokens 2048 \
  --output-tokens 128 \
  --concurrency 32 \
  --ttft-sla-ms 3000 \
  --tpot-sla-ms 100 \
  --generate-sweep
```

The optional package is:

```text
sweep/
├── serve_params.json
├── bench_params.json
├── run_sweep.sh
├── recommend.py
└── SWEEP.md
```

The sweep currently varies only `max-num-seqs` and
`max-num-batched-tokens`.

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

With TTFT/TPOT objectives, the benchmark uses vLLM `--goodput`, and the
recommender selects highest mean request goodput across repeated runs. Without
latency objectives it selects highest mean output-token throughput. Failed
benchmark configurations are excluded.

`recommended-config.yml` changes only the two swept scheduler parameters.
`recommendation.json` records the measured evidence.

Deploy:

```bash
source env.sh
vllm serve --config sweep/recommended-config.yml
```

## vLLM CPU Docker Shell

From the vLLM source tree:

```bash
mkdir -p recipe-output

docker run --rm -it \
  --entrypoint bash \
  --security-opt seccomp=unconfined \
  --cap-add SYS_NICE \
  --shm-size=4g \
  -p 8000:8000 \
  -v "$PWD/tools/recipes:/recipes:ro" \
  -v "$PWD/recipe-output:/output" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /output \
  vllm/vllm-openai-cpu:latest-x86_64
```

`/output` is writable; `/recipes` remains read-only.

Recipe-only conversion inside the container:

```bash
python3 /recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --config-out /output/config.yml \
  --env-out /output/env.sh
```

Initial suggestion plus sweep:

```bash
python3 /recipes/recipe_json_to_vllm_config.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --hardware xeon6 \
  --detect-hardware \
  --input-tokens 2048 \
  --output-tokens 128 \
  --concurrency 32 \
  --ttft-sla-ms 3000 \
  --tpot-sla-ms 100 \
  --config-out /output/config.yml \
  --env-out /output/env.sh \
  --generate-sweep \
  --sweep-out-dir /output/sweep
```

Use the initial suggestion immediately if desired:

```bash
source /output/env.sh
vllm serve --config /output/config.yml
```

Stop that manually started server before running the sweep.

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
