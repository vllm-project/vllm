---
name: generate-perf-charts
description: Generate performance benchmark charts for a model and add them to docs/cohere/tests/performance.md. Use when the user asks to add or update performance charts, benchmark graphs, or throughput/latency visualizations for a model.
---

# Generate Performance Charts

Generate matplotlib PNG charts from CI benchmark data on the `gh-pages` branch and update the performance documentation page.

## When to Use

- User asks to add performance charts for a new model
- User asks to refresh or update existing benchmark charts
- User asks to visualize throughput or latency benchmarks for a model

## Workflow

### Step 1: Identify the Model Name

Get the model name from the user. Model names follow the pattern used in benchmark test names (e.g. `c4-25a218t_fp8`, `c5-3a30t_fp8`, `command-r7b_fp8`).

If uncertain, check available models in the benchmark data from the repo root:
```bash
git show gh-pages:data/summary.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = set()
for entry in data:
    d = entry.get('data', entry)
    s = d.get('serving', {})
    for v in s.get('Test name', {}).values():
        import re
        m = re.match(r'serving_(.+?)_tp\d+', v)
        if m:
            models.add(m.group(1))
for m in sorted(models):
    print(m)
"
```

### Step 2: Ensure Dependencies

```bash
pip install matplotlib adjustText 2>/dev/null
```

### Step 3: Run the Chart Generation Script

Run the script from the repo root. Use `--model` for each model to include. If updating all models in the existing `performance.md`, pass all of them:

```bash
python tests/cohere/scripts/generate_perf_charts.py --model <MODEL_NAME>
```

To generate charts for multiple models at once (recommended when updating the full page):
```bash
python tests/cohere/scripts/generate_perf_charts.py --model c4-25a218t_fp8 --model c5-3a30t_fp8
```

The script will:
- Read benchmark data from `gh-pages` branch (`data/summary.json`, `data/summary_b200.json`, `data/summary_mi300x.json`)
- Filter for serving tests matching the model with output length 1000
- Compute medians of up to the last 5 CI runs for each setting
- Generate two PNG charts per model in `docs/cohere/tests/images/`:
  - `{model}_throughput_vs_batchsize.png` — Output Tput (tok/s) vs max concurrency, across H100/B200/MI300X
  - `{model}_throughput_vs_latency.png` — Output Tput (tok/s) vs Mean TTFT (ms), tradeoff curves
- Write/overwrite `docs/cohere/tests/performance.md` with sections for all specified models

### Step 4: Verify Output

1. Confirm the PNG images exist:
```bash
ls -la docs/cohere/tests/images/{MODEL_NAME}_*.png
```

2. Read `docs/cohere/tests/performance.md` to confirm the model section was added.

3. Optionally open the PNG files to visually verify the charts look correct.

### Step 5: Report Results

Tell the user:
- Which models were processed
- Which devices had data (H100, B200, MI300X) and which shapes (1K/1K, 10K/1K, 100K/1K)
- Any models or device/shape combinations that had no data

## Notes

- The script reads data directly from the local `gh-pages` branch. Run `git fetch origin gh-pages:gh-pages` first if the data may be stale.
- Not all models have data on all devices or shapes. The script silently skips missing combinations.
- Zero-throughput entries (failed benchmark runs) are automatically excluded.
- The `performance.md` file is fully rewritten each time. To add a new model without losing existing ones, pass all models in one invocation.
