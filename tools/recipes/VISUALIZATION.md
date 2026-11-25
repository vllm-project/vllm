# Post-Benchmark Reporting

## HTML summary

`run_full_sweep.sh` automatically generates a self-contained HTML report after
the final recommendation:

```text
sweep-report.html
```

The report uses Python's standard library and does not require plotting
dependencies. It summarizes selected settings, stage results, SLA metrics, and
all measured candidates. Report-generation failures are warnings and do not
change sweep success.

The generated `report.py` uses the same TTFT and TPOT objectives supplied when
the sweep package was created. It displays both objectives and validates them
against every available recommendation JSON before writing the report.

Regenerate it manually or select a different output path:

```bash
./report.py
./report.py --output results/my-sweep-report.html
```

## Optional figures

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
