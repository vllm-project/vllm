# plotting_tools

Plotting and classification for vLLM Nsight traces (Chrome JSON exported from `.nsys-rep`).

## Quick start (job 7651157)

### 1. Export on ARC (requires `nsys`)

```bash
cd /data/engs-glass/catz0932/inference-traces/vllm
bash plotting_tools/export_nsys.sh results/7651157
```

This writes `results/7651157/exported/*.json` from:

- `ray_worker_nsight/htc-g060/worker_process_2184846.nsys-rep` (34 MB, PP rank 1 — use this)
- `nsight/vllm_api_server_htc-g059.nsys-rep` (API server / driver)
- Skips the 0-byte `htc-g059` worker trace

### 2. Plot locally

```bash
cd /path/to/vllm
uv pip install -r plotting_tools/requirements.txt
.venv/bin/python plotting_tools/analyze_job.py --job-dir results/7651157
```

Outputs:

```
results/<job-id>/plots/
├── htc-g059/              # per-node (hostname from trace path)
├── htc-g060/
└── summary_plots/         # cross-node combined metrics
```

| Plot | Per-node dir | Summary dir |
|------|--------------|-------------|
| Decomposed timeline | `decomposed_timeline.png` | — |
| Category time share (%) | `traffic_volume_pct.png` | `duty_by_node.png`, `traffic_volume_pct_mean.png` |
| Classic compute / comm / control | `compute_comm_control_timeline.png` | — |
| Expert traffic heuristic (GB) | `expert_traffic_gb.png` | `expert_traffic_gb_by_node.png` |
| Rank-to-rank comm heatmap | `rank_traffic_heatmap.png` | `rank_traffic_heatmap.png` (merged) |
| GPU-to-GPU comm heatmap (on-node TP) | `gpu_traffic_heatmap.png` | — |
| All-to-all heatmap (alias) | `all2all_traffic_heatmap.png` | — |
| Prefill / decode / all comm counts & avg size | `message_stats_prefill_decode.png` | — |
| NCCL ops (timestamp, name, bytes, shape) | `collective_ops.json` | — |
| No-comm window CDF | `nocomm_windows_cdf.png` | `nocomm_windows_cdf.png` (per node + pooled) |
| Comm start delta CDF | `comm_start_delta_cdf.png` | `comm_start_delta_cdf.png` (per node + pooled) |
| GPU idle window CDF | `gpu_idle_windows_cdf.png` | `gpu_idle_windows_cdf.png` (per node + pooled) |
| Idle before/after activity | `idle_transitions_by_time.png`, `idle_transition_heatmap.png` | `idle_transition_heatmap.png` (merged) |
| Per-gap idle context (JSON) | `idle_gaps.json` | `idle_gaps.json` (merged) |
| Collective op breakdown (bar) | `collective_ops_breakdown.png` | `collective_ops_breakdown.png` (summed) |

Nsight GUI **JSONL** export is supported directly (no `nsys export` step):

```bash
.venv/bin/python plotting_tools/analyze_job.py --job-dir results/7692897
```

Auto-discovers `ray_worker_nsight/**/*.jsonl`, plots each node under `plots/htc-g059/` etc., and writes combined charts to `plots/summary_plots/`.

## Classification

Patterns live in `classify.py` (compute / comm / control, plus MoE subcategories). Extend `ATTENTION_PATTERNS`, `GATE_PATTERNS`, etc. when new kernels show up as “Other Compute”.

Use `--strict-classify` to fail on unknown event strings.

## Limitations (7651157)

- **One good worker trace** (g060); PP rank 0 on g059 is empty.
- **Multi-architecture comparison** and **expert→GPU mapping** need multiple jobs or `enable_return_routed_experts` — not available from a single `.nsys-rep` alone.
- **Heatmap bytes** use NCCL event args when present; otherwise duration is used as a proxy.

## Options

```bash
.venv/bin/python plotting_tools/analyze_job.py \
  --job-dir results/7651157 \
  --trace results/7651157/exported/htc-g060_worker_process_2184846.json \
  --max-plot-ms 5000 \
  --n-ranks 2 \
  --num-experts 128
```
