# HiSparse / NIXL P/D benchmarks

Replicates, at cluster-representative scale, the benchmarks behind the
HiSparse P/D PR series so the rework branch (`feat/hisparse-pd-rework`) can be
compared against its baseline:

| PR | Benchmark | What these scripts reproduce |
| --- | --- | --- |
| #53781 | GLM e2e, ISL/OSL 20k/10k, 32k/8k, 60k/10k, concurrency sweep | `pd_bench.sh` default grid |
| #46326 | 16-node GLM-5.2-FP8 stack, HiSparse vs plain GPU-KV decode A/B | `hisparse_vs_gpu_kv.sh` two-arm run |
| samsja's #53781 report | 8xH200 nodes, 2P+2D engines tp=8, NIXL/UCX, `host_pool_gib` on D | same scripts, `NUM_PREFILL=2 NUM_DECODE=2 P_TP=8 D_TP=8` under slurm |

## Scripts

- `pd_bench.sh` — workhorse. Launches prefill engine(s), decode engine(s) and
  the toy PD proxy (`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`),
  smoke-checks one request through P -> NIXL -> D, sweeps `vllm bench serve`
  over the (ISL:OSL) x concurrency grid, writes one result JSON per point plus
  engine logs, tears everything down. **One arm per invocation.**
- `hisparse_vs_gpu_kv.sh` — A/B driver. Runs `pd_bench.sh` twice on identical
  topology/traffic: arm `gpu-kv` (decode keeps imported KV resident in the
  normal device pools) vs arm `hisparse` (decode lands imported KV in the
  pinned host pool via `attention_config.hisparse_config`). Engines restart
  between arms because the landing policy is fixed at engine startup.
- `summarize_ab.py` — joins the two arms' result JSONs on
  (concurrency, ISL, OSL) and prints throughput / TTFT p99 / completion
  columns plus the hisparse/gpu-kv throughput ratio.

## Quickstart

Single node, 8 GPUs, P tp4 + D tp4:

```bash
MODEL=zai-org/GLM-5.2-FP8 P_TP=4 D_TP=4 HOST_POOL_GIB=64 \
  ./benchmarks/hisparse_pd/hisparse_vs_gpu_kv.sh
```

Just one arm (e.g. only the reworked hisparse path):

```bash
MODEL=zai-org/GLM-5.2-FP8 P_TP=4 D_TP=4 HOST_POOL_GIB=64 \
  ./benchmarks/hisparse_pd/pd_bench.sh
```

A shakedown on small hardware before burning the big model:

```bash
MODEL=Qwen/Qwen3-0.6B BLOCK_SIZE=64 MAX_MODEL_LEN=2048 MAX_NUM_SEQS_D=16 \
  ISL_OSL_PAIRS="512:128" CONCURRENCIES="4 8" NUM_PROMPTS=16 \
  HOST_POOL_GIB=1 GPU_MEM_UTIL_P=0.4 GPU_MEM_UTIL_D=0.4 ENFORCE_EAGER=1 \
  ./benchmarks/hisparse_pd/pd_bench.sh
```

(For Qwen-class non-DSA models the hisparse arm is inert — HiSparseConfig is
only honored on sparse-MLA models — so that shakedown only exercises the
plain P/D path. Use a small DSA model, e.g. a shrunken DeepSeek-V3.2 with
`HF_OVERRIDES`, to exercise hisparse itself cheaply.)

## Arm selection

`pd_bench.sh` resolves the decode-side landing policy in this order:

1. `DECODE_ATTENTION_CONFIG` (verbatim `--attention-config` JSON), else
2. `HOST_POOL_GIB` (optionally `DEVICE_BUFFER_SIZE`) builds
   `{"hisparse_config": {...}}`, else
3. nothing — plain GPU-resident decode.

`PREFILL_ATTENTION_CONFIG` / `PREFILL_HISPARSE=1` exist only for baselining
against the pre-rework branch (`feat/hisparse-mla-decode-main@9a21608f90`),
where the NIXL handshake required `hisparse_config` on both roles. On the
rework branch P stays stock.

## Environment reference

Topology: `MODEL` (required), `NUM_PREFILL`, `NUM_DECODE`, `P_TP`, `P_PP`
(>1 needs `KV_CONNECTOR=NixlPushConnector`), `D_TP`, `BLOCK_SIZE` (128),
`GPU_MEM_UTIL_P/D` (0.90), `MAX_MODEL_LEN` (81920 — must cover ISL+OSL),
`MAX_NUM_SEQS_D` (96 — hot buffers scale with it on the hisparse arm; see
issue # 46326), `ENFORCE_EAGER` (0), `KV_CONNECTOR`.

GPUs are taken from `CUDA_VISIBLE_DEVICES` if set (slurm does this), else all
visible devices, and split contiguously: prefill instances first, then decode.

Traffic: `ISL_OSL_PAIRS` (`"20000:10000 32000:8000 60000:10000"`),
`CONCURRENCIES` (`"16 32 64 128"`), `NUM_PROMPTS` (100), `BENCH_EXTRA_ARGS`,
`BENCH_TIMEOUT` (7200).

Serve: `P_EXTRA_ARGS`, `D_EXTRA_ARGS` (space-separated extra `vllm serve`
flags), plus the arm variables above.

Ports: `PROXY_PORT` (8192), `P_PORT_BASE` (8100), `D_PORT_BASE` (8200),
internal-port bases 20000/30000, NIXL side channels 5559+i / 5659+i*D_TP.

Outputs: `OUTPUT_DIR` (default `bench_results/hisparse_pd/<arm>_<ts>`) holds
`manifest.txt` (config + git rev — the branch comparison relies on this),
`conc<N>_isl<I>_osl<O>.json` per sweep point (with `generated_texts`; inspect
these for the long-prefix garbling class of bugs — the smoke check only
asserts non-empty output), and `logs/` per engine + proxy. Completed points
are skipped on re-run, so a crashed sweep can simply be relaunched.

## Known gaps

- No hit-rate / spill counters: PR #53782 (HiSparse metrics) is not on this
  branch. Preemption counts are approximated by `grep -ci preempt` over the
  decode logs and printed at teardown.
- No NIXL transfer-volume counter (the #46326 table reports 26.5 vs 7.0 TiB);
  would need connector-side instrumentation.
- `#53263` same-node D2H read staging is not on this branch, so single-node
  P+D runs go through UCX TCP loopback for host-landed reads — expect that
  to dominate first-token latency when P and D share a node. Multi-node runs
  (the intended shape) are unaffected.

## Cluster use

The scripts assume one shared filesystem and the repo venv
(`.venv/bin/vllm`), so on a slurm cluster a job that sources nothing and runs
`hisparse_vs_gpu_kv.sh` inside an allocation works as-is for single-node
topologies. A multi-node wrapper (P and D engines on separate nodes via
srun) is intentionally not written yet.
