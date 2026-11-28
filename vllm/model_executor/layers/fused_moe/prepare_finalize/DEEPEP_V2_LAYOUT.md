# DeepEP v2 Layout — Summary

## Purpose

Separates **layout** (expanded vs non-expanded dispatch) from **CPU count sync** (wait for real expert counts). Leaves production default off (`deepep_v2_do_expand=false`, `deepep_v2_do_cpu_sync=false`). No adaptive threshold.

## Configuration

- `KernelConfig` fields: `deepep_v2_do_expand`, `deepep_v2_do_cpu_sync` (both default false, must match across EP ranks)
- CPU-sync requires `--enforce-eager`; polled counts before capture is rejected
- FP8 direct path: `--kernel-config '{"moe_backend":"deep_gemm","deepep_v2_do_expand":true}'`
- FP4 path includes upstream padding schedulers (#59044, #59128); fuses IDs/scales/endpoints in one kernel

## Four Cases

| Case | Expanded | CPU wait | Allocation |
| --- | --- | --- | --- |
| nonexpanded-nosync | No | No | dedup-token capacity |
| nonexpanded-sync | No | Yes | actual dedup tokens |
| expanded-nosync | Yes | No | expanded + alignment padding |
| expanded-sync | Yes | Yes | expert rows + padding |

## Receiver & Expert Logic

- Reconstructs routing IDs and real counts from GPU prefixes (gaps → ID -1, CPU counts rounded to alignment)
- Clears padding before FP8 quantization; sanitizes scales to 1 for invalid rows
- FP8/FP4 skip permute/unpermute; use grouped layout directly; GEMM + activation quant skip padded rows via `expert_ends`/`expert_ids`

## Benchmark Harness (`benchmark_deepep_v2_layout.py`)

- Fresh dispatch handle per case, 2-rank NVLink (`EP_DISABLE_GIN=1`, `NCCL_GIN_ENABLE=0`), verifies against scalar reference
- Supports `--cuda-graph` for nosync cases (warmup excluded, replay with shifted routing)
- Metrics: median max-rank wall time per step (includes dispatch→combine + sync), PyTorch peak memory; not serving latency

## Serving Validation (DeepSeek-V4.1-Flash, 8×B200)

- **Pirate2 2026-10-06:** 1271/1277/1278 correct of 1319 (TPOT 10.744/13.425/12.613 ms, TTFT 524/576/588 ms)
- **Rerun nm-ohio-b200-01 2026-11-13 (fe59004, 8×B200):** 1265 (95.91%) / 1268 (96.13%) / 1272 (96.44%) (TTFT 464/534/406 ms, TPOT 9.85/11.71/159 ms, tok/s 764.6/650.5/50.2) — parity within 0.45 pp
- Finding: fused prep ↓ TPOT ~6% vs unfused, but **non-expanded remains fastest**; no TTFT/memory win; expanded-sync is eager and ~15× slower

## Status

Both flags off by default; integrated in `DeepEPV2PrepareAndFinalize` with tests in `tests/kernels/moe/test_deepep_v2_moe.py`. Coordinate with #51589.

Ref: <https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/elastic/buffer.hpp>
