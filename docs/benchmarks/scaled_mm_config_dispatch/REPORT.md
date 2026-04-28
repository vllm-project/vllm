# Does Per-Shape Config Dispatch Actually Help? Benchmarking Helion's `scaled_mm` on B200

## TL;DR

- Per-shape config dispatch matters: using a single config per `(K, N)` pair costs **+14% on average** and up to **+525% on the worst shape**
- The sensitivity comes from fundamentally different config requirements across batch sizes — tile volumes vary by 1000x between M=1 and M=8192

---

## 1. Problem Statement

`scaled_mm` is the workhorse behind every linear layer in FP8 inference. It computes `C = diag(scale_a) @ (A @ B) @ diag(scale_b) + bias`, where A is `[M, K]` and B is `[K, N]`, both FP8. M is the batch dimension (num_tokens), while K and N are fixed by the model architecture. In practice M ranges from 1 (single-token decode) to 8192+ (prefill batches).

Helion compiles a kernel from a high-level description, but the compiled code looks very different depending on its config: tile sizes, pipeline depth, warp count, memory indexing strategy. A config tuned for M=1 uses tiny tiles (e.g., 1×16) and few warps — it's optimized to minimize launch overhead when there's barely any work. A config tuned for M=8192 uses large tiles (128×256) with deep pipelines and 8 warps — it's optimized for throughput. Force the small-batch config onto a large batch and the kernel loops thousands of extra times; force the large-batch config onto a small batch and most of the GPU sits idle.

The current implementation ships 224 configs for B200 (about 14 per `(K, N)` pair), and a dispatcher picks the one that was autotuned for the closest M. We want to verify whether this fine-grained dispatch actually helps — how much performance is lost if we use fewer configs, and what structural properties of the configs drive the sensitivity.

---

## 2. Experiment Methodology

We tested all 16 `(K, N)` pairs from the kernel's shape list, covering Qwen3-1.7B, Qwen3-8B, and Llama-3.3-70B linear layer dimensions. For each pair, we swept M (num_tokens) across 14 values from 1 to 8192, and benchmarked every M against every available config in that pair — a full 14×14 cross-config matrix. That's 3,136 individual benchmarks. Each data point is the median of 20 runs after 5 warmups via `triton.testing.do_bench`, with sub-0.1% run-to-run variation.

All penalties in this report are relative to an **oracle baseline**: for each shape, the fastest latency achieved by any config in the cross-config matrix. We use this rather than the autotuned diagonal because the autotuner occasionally picks suboptimal configs, and the oracle gives a clean, unambiguous reference point. A penalty of +X% means X% slower than the oracle — higher is worse.

---

## 3. Results and Analysis

### 3.1 Sensitivity to num_tokens (M)

For a fixed `(K, N)` pair, how much does performance depend on having the right config for the current batch size? Figure 1 shows latency-ratio heatmaps for four pairs. Each cell shows the penalty of running batch size M (y-axis) with a config tuned for a different M (x-axis). Green = near-optimal, red = bad.

![Figure 1: Config Sensitivity Heatmaps](https://raw.githubusercontent.com/gmagogsfm/vllm/scaled_mm/docs/benchmarks/scaled_mm_config_dispatch/fig1_heatmaps.png)

The pattern is consistent across all 16 pairs:

- **The diagonal is green** — the autotuned config works. No surprise there.
- **There's a green band around the diagonal** — nearby M values can share configs without much cost. The band is wider at the top (small M, where all configs are launch-overhead-bound) and narrower at the bottom (large M, where the wrong tile size is fatal).
- **The corners are deep red** — forcing a small-M config onto a large batch (lower-left) is catastrophic. The reverse (upper-right) is bad but less extreme.

The asymmetry makes sense: a small-M config has tiny tiles, so a large M requires far more iterations. A large-M config at small M just wastes warps — it's a constant-factor overhead rather than a scaling disaster.

If we had to pick just one config per `(K, N)` pair for all batch sizes, how bad would it be? Table 1 shows the best possible single config and its penalty.

**Table 1: Best single config vs oracle**

| (K, N) | Best Config | Aggregate Penalty | Worst-Shape Penalty |
|---|---|---|---|
| (512, 2048) | M=8192 | +21% | +56% |
| (2048, 1024) | M=2048 | +36% | +134% |
| (2048, 2048) | M=1024 | +31% | +133% |
| (2048, 4096) | M=8192 | +22% | +128% |
| (2048, 9216) | M=1024 | +13% | +145% |
| (2048, 12288) | M=1024 | +8% | +109% |
| (4096, 2048) | M=1024 | +40% | +251% |
| (4096, 4096) | M=8192 | +27% | +262% |
| (4096, 6144) | M=8192 | +18% | +260% |
| (4096, 24576) | M=2048 | +4% | +110% |
| (6144, 2048) | M=1024 | +47% | +323% |
| (8192, 8192) | M=1024 | +19% | +321% |
| (8192, 10240) | M=2048 | +14% | +253% |
| (8192, 57344) | M=1024 | +3% | +31% |
| (12288, 4096) | M=8192 | +39% | +525% |
| (28672, 8192) | M=1024 | +22% | +464% |
| **Weighted average** | — | **+14%** | — |

The aggregate penalty is latency-weighted, so larger shapes (which take longer) count more. The +14% average might sound tolerable, but the per-shape numbers are what matter for serving: M=1 decode latency directly determines time-to-first-token. A +525% penalty at (12288, 4096) M=1 means a 6x slowdown on what is often the latency-critical path.

![Figure 3: Single-config penalty detail](https://raw.githubusercontent.com/gmagogsfm/vllm/scaled_mm/docs/benchmarks/scaled_mm_config_dispatch/fig3_single_config.png)

To understand why, look at what actually changes in the configs. Table 2 shows the parameters for (4096, 4096) across M.

**Table 2: Config parameters for (K=4096, N=4096) across batch sizes**

| M | Block M | Block N | Block K | Warps | Stages |
|---|---|---|---|---|---|
| 1 | 2 | 16 | 1024 | 2 | 3 |
| 4 | 4 | 16 | 1024 | 2 | 3 |
| 8 | 64 | 32 | 512 | 8 | 4 |
| 32 | 16 | 16 | 128 | 1 | 5 |
| 64 | 64 | 32 | 512 | 8 | 4 |
| 128 | 64 | 64 | 512 | 8 | 3 |
| 256 | 64 | 128 | 128 | 4 | 6 |
| 512 | 128 | 128 | 128 | 8 | 7 |
| 1024 | 128 | 256 | 128 | 8 | 4 |
| 4096 | 128 | 256 | 128 | 8 | 4 |
| 8192 | 128 | 256 | 128 | 8 | 4 |

At M=1, the config uses block_M=2 with only 2 warps — just enough to cover the single row. By M=128, it's 64×64 blocks with 8 warps. At M≥1024, it stabilizes at 128×256 blocks. The tile volume varies by 1000x across the M range. This is why using the M=1 config at M=8192 is orders of magnitude slower: the kernel would need ~4000 extra iterations to tile over the M dimension with block_M=2 instead of 128.

The other pairs show the same trend. For (28672, 8192), the M=1 config uses block_M=1, while M≥512 all converge to block_M=128 with 128×256 tiles.

### 3.2 Sensitivity to N (feature_size)

The previous section shows configs are sensitive to M. But are they also sensitive to N? If pairs sharing the same K (e.g., K=4096 with N=2048, 4096, 6144, 24576) could share configs, we'd need far fewer total entries. To test this, we took each config autotuned for one N and ran it on every other N at the same K, at three representative M values: M=1 (decode), M=128 (transition), and M=4096 (prefill). Figure 2 shows the full cross-N penalty heatmaps.

![Figure 2: Cross-N Config Portability Heatmaps](https://raw.githubusercontent.com/gmagogsfm/vllm/scaled_mm/docs/benchmarks/scaled_mm_config_dispatch/fig5_cross_n_heatmaps.png)

Configs are sensitive to N as well, especially at mid-M. At M=128, using a config from a different N regularly costs 20–200%. The penalty is worst when N differs by a large factor (e.g., N=1024 config on N=12288 target: +214%).

At large M (4096), configs are more portable across N — penalties are usually under 10%. This makes sense: at large M the kernel is compute-bound with large tiles, and the tiling over N is coarse enough that modest differences in N don't change the optimal config much.

At small M (1), portability varies. Configs for similarly-sized N values are often interchangeable (0–8% penalty), but large N mismatches can cost 24–95%.

---

## 4. Conclusion

Fine-grained config dispatch is worth it. Across the M dimension, a single config per `(K, N)` pair leaves 14% on the table on average, with individual shapes losing up to 525%. Across the N dimension, sharing configs between different N values at the same K costs 20–200% at decode and transition batch sizes. Per-shape dispatch along both M and (K,N) is necessary for good performance.
