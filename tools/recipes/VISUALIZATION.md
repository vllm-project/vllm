# Post-Benchmark Visualization

Visualization is an independent analysis step. It reads completed sweep
results and does not start a server, run a benchmark, or change a tuning
recommendation.

Install the plotting dependencies in the same environment as vLLM:

```bash
python3 -m pip install -r requirements.txt
```

Preview the figures that would be generated:

```bash
./visualize.py --dry-run
```

Generate figures for every completed stage:

```bash
./visualize.py
```

Select one or more stages when needed:

```bash
./visualize.py --stage concurrency-tuning
./visualize.py --stage parallel-layout --stage runtime-tuning
```

Figures are written beside the source results:

```text
results/parallel-layout/figures/
results/concurrency-tuning/figures/
results/runtime-tuning/figures/
```

The analysis produces output-throughput, P99 TTFT, and P99 TPOT figures. The
concurrency stage also produces a throughput/TTFT tradeoff figure and a goodput
figure when the benchmark results contain that metric.

The native vLLM Pareto command is intentionally not used because its efficiency
axis is expressed as tokens per second per GPU, which is not an accurate label
for Xeon CPU layouts.
