# Cosmos3-Edge EVS Nebius results

A/B of baseline vs `--video-pruning-rate 0.5` on `nvidia/Cosmos3-Edge`
(Nebius H200, vLLM serve).

## Protocol

- Clip: 10s 720p VideoMME hurdles @ **3 FPS**
- Output: **256** fixed tokens (`min_tokens=max_tokens`, ignore EOS)
- Concurrency: one clip × N in-flight (`C = 1…32`)
- Accuracy: separate VideoMME spot-check (**120** MCQs)

## Reproduce plots

```bash
python3 benchmarks/cosmos3_edge_evs/plot_results.py
```

## Headline results

| Metric | Baseline | EVS q=0.5 |
|--------|----------|-----------|
| Prompt tokens | 24209 | 12239 (~1.98×) |
| Accuracy (120 MCQ) | 50.0% | 53.3% (+3.3 pp) |
| C=1 mean latency | ~913 ms | ~831 ms |
| C=32 throughput | ~6.4 rps | ~3.0 rps |

EVS correctly prunes tokens and holds accuracy; high-concurrency
throughput did **not** improve in this fixed-decode sweep.
