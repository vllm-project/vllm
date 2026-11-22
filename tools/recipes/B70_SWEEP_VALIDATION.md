# B70 Recipes sweep hardware-validation procedure

These tools target the `xpu_vllm_recipes_tool` branch. Execute inside the same
vLLM environment/container in which `vllm serve` and `vllm bench` work.

## 1. Generate the existing B70 sweep

```bash
python3 tools/recipes/recipe_json_to_vllm_config.py \
  --model Qwen/Qwen3.5-4B --hardware arc_pro_b70 --detect-hardware \
  --input-tokens 128 --output-tokens 128 --concurrency 32 \
  --ttft-sla-ms 3000 --tpot-sla-ms 100 \
  --config-out config.yml --env-out env.sh \
  --generate-full-sweep --sweep-out-dir sweep_b70
```

Choose a model actually supported by the recipe and accessible to the test
machine; set a tested `max-model-len` at generation time when necessary.

## 2. Inspect and run the existing stages (priority 1)

```bash
bash tools/recipes/validate_b70_sweep.sh sweep_b70 dry-run
bash tools/recipes/validate_b70_sweep.sh sweep_b70 run
```

The second command performs actual TP/DP -> concurrency -> scheduler sweeps.
Inspect `sweep_b70/results`, `sweep_b70/*recommendation.json`, and
`sweep_b70/sweep-report.html`. The script does not suppress individual sweep
failures; use server logs and benchmark artifacts to investigate any failures.

## 3. Optional max-model-len capacity stage (priority 2)

To run context-capacity selection **between** the layout and concurrency stages,
run the stages individually instead of using the `run` option above:

```bash
./sweep_b70/run_full_sweep.sh --stages parallel-layout

python3 tools/recipes/sweep/context_length_sweep.py \
  --config sweep_b70/parallel-layout-config.yml \
  --env env.sh --output-dir sweep_b70/context-length \
  --candidates 4096,8192,16384,32768 \
  --input-tokens 128 --output-tokens 128 \
  --concurrency 32 --num-prompts 100 \
  --ttft-sla-ms 3000 --tpot-sla-ms 100 \
  --port 8001 --apply

./sweep_b70/run_full_sweep.sh --stages concurrency,scheduler
```

**Important:** Keep the same input/output workload shape throughout all stages.
The example context candidates are illustrative. Adjust candidates and the
workload to match your intended serving requirements, then regenerate the
layout sweep with the same workload shape.

The context-length runner writes per-candidate YAML, server/benchmark logs,
`context-length-results.json`, and `context-length-config.yml`. `--apply`
updates `parallel-layout-config.yml` and creates a one-time `.before-context-sweep`
backup. Without `--apply`, the selected layout config is left untouched.

The selection objective is the **largest successfully measured context length**
that satisfies the provided P99 TTFT/TPOT thresholds; it does not assume smaller
contexts necessarily improve throughput. No benchmark results are fabricated for
failed starts or missing P99 metrics. Ensure the selected context length does not
exceed the model's supported context. This is opt-in and does not change the
existing default full-sweep behavior.
