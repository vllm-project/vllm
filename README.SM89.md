# DeepSeek-V4-Flash on 8× RTX 4090 (SM89 / Ada)

Runs **DeepSeek-V4-Flash** on consumer Ada GPUs. Upstream vLLM refuses to select an
attention backend for DeepSeek-V4 on compute capability 8.9, and several DSv4 code
paths call kernels that only exist for SM90+.

| | |
|---|---|
| Baseline | upstream vLLM `017e9f4448` (`0.27.2rc1.dev163`) |
| Branch | `sm89` — one commit on top of the baseline; the diff **is** the adaptation |
| Diff | 21 edits across 9 files + 3 new operator files (+2018 / −41) |
| Requires | FlashInfer **`0.6.14+sm89`** (see [FlashInfer](#flashinfer)) |
| License | Apache-2.0, same as vLLM |

---

## ⚠️ Hard constraint: tensor-parallel size must be ≤ 4

The FlashInfer sparse-MLA prefill dispatch
(`flashinfer/data/csrc/sparse_mla_sm120_prefill.cu`, `dispatch_dsv4_single`)
instantiates only these head counts:

```c
switch (num_heads) {
  case 16: ... case 32: ... case 64: ... case 128: ...
  default: return false;
}
```

DeepSeek-V4-Flash has `num_attention_heads = 64`, so:

| TP | heads per rank | works |
|---|---|---|
| 1 / 2 / 4 | 64 / 32 / 16 | ✅ |
| **8** | **8** | ❌ `Unsupported sparse-MLA prefill configuration` |

This is upstream FlashInfer behaviour — the same table is present in a clean
`0.6.16` wheel — **not** something introduced by the SM89 port.

For 8 GPUs use **TP4 × PP2**. That is also the better configuration on this
model: DSv4 uses MLA with a single KV head, so under TP8 every rank must hold
the full KV cache. Splitting by layer with PP avoids that replication and gives
roughly **6× the KV cache pool**.

---

## Hardware

| | |
|---|---|
| GPUs | 8 × RTX 4090 24 GB (SM89 / Ada), PCIe, no NVLink |
| Host RAM | ≥ 200 GB (checkpoint is ~149 GB) |
| Weights | MXFP4 MoE, served through the MARLIN backend (Ada has no hardware block-scaled MMA) |
| KV cache | `fp8_ds_mla`, 584 B/token |

Smaller counts work as long as the weights fit and TP ≤ 4.

---

## Install

Build the branch, or install the upstream wheel for the baseline commit and
apply the patch set on top of it.

```bash
# 1. upstream wheel for the baseline commit (vLLM publishes one per commit)
curl -sSL -o vllm-0.27.2.whl \
  "https://wheels.vllm.ai/017e9f4448b700e85ee16023287b025693c72b9e/vllm/vllm-0.27.2rc1.dev163%2Bg017e9f444-cp38-abi3-manylinux_2_28_x86_64.whl"

uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install ./vllm-0.27.2.whl --torch-backend=cu130
```

### FlashInfer

The sparse-MLA kernels for Ada come from a community build published as a binary
only — no source branch was ever pushed. A copy is vendored in this repository so
the configuration stays rebuildable; see [`assets/`](assets/) for provenance,
checksum and license.

```bash
uv pip install ./assets/flashinfer_python-0.6.14-py3-none-any.whl
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

Verify before installing:

```
sha256  d124369346a3d48eac67e31c42f7a3c813bcc0abc10e2e36db413b7b3dfd97df
```

Once installed it reports version `0.6.14+sm89`, which is what
`has_flashinfer_sparse_mla_sm89()` probes for.

Then overlay this branch's changed files onto the installed package, or install
from source.

---

## Serving

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --enable-prefix-caching --max-model-len 131072 \
  --gpu-memory-utilization 0.95 --max-num-seqs 16 \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4
```

Under Docker, add `--security-opt seccomp=unconfined` — otherwise OpenBLAS
fails with `pthread_create: Operation not permitted` on hosts whose
libseccomp does not know `clone3`.

### Startup is healthy when the log contains all three

```
Using 'MARLIN' Mxfp4 MoE backend.
Using FP8 indexer cache for Lightning Indexer.
Using fp8_ds_mla data type to store kv cache.
```

### Flags that do not work here

| Flag | Why |
|---|---|
| `--enable-return-routed-experts` | Requires a full-attention KV cache group; DSv4's groups are all compressed. Also incompatible with `PP > 1`, and costs a fixed 512 MiB per GPU. |
| `--tensor-parallel-size 8` | See the hard constraint above. |
| `--gpu-memory-utilization 0.97` with PP | The NCCL buffers for the PP channel live outside vLLM's budget. 0.95 is the working value. |

---

## What the patches do

| Group | Count | What |
|---|---|---|
| `gate` | 10 | Backend selection and capability probing. The highest-leverage change is `has_cutedsl()` returning `False` on SM89 — one short-circuit disables every CuTe-DSL path. Those kernels are SM90+ and on Ada they do not raise; they **silently compute wrong results**. Also: sparse-MLA allowlist accepts `(8, 9)` and `fp8_ds_mla`; the default attention backend routes SM89 to the SM120 FlashInfer sparse-MLA path; E8M0 block scales are upcast to fp32 unconditionally because Triton cannot bind E8M0 on any platform. |
| `kernel` | 9 | Operators with no upstream path on this arch. `o_proj` uses a Triton FP8 einsum instead of DeepGEMM (which builds for arch 9/10/12 only); indexer MQA logits fall back to Triton; `mhc_pre_broadcast_tilelang` is missing a `use_deep_gemm` guard upstream — added, together with `n_splits=1` off the DeepGEMM path, because the TileLang kernel asserts on it. |
| `perf` | 2 | Upstream defaults that are clearly suboptimal here. See below. |

### The M-aware GEMM default

Upstream hardcodes the fallback config for `w8a8_triton_block_scaled_mm`:

```python
config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2}
```

The launch grid is `cdiv(M, BLOCK_SIZE_M) × cdiv(N, BLOCK_SIZE_N)`. Decode steps
have `M ≤ 32`, so `cdiv(M, 64)` is always 1 — with the real DeepSeek shapes
(`N = 7168` → 56 N-blocks) that leaves well over half of an Ada part's 128 SMs
idle. Measured with CUDA-graph replay, `BLOCK_SIZE_M=16, num_stages=3` is
**2–3× faster for every M in [1, 32]** across `K ∈ {512, 1024, 2048, 4096}`,
while `BLOCK_SIZE_M=64` stays best for prefill (`M = 2048`).

Single-variable A/B on the full server, same container, same launch script:

| | TPOT P50 | throughput |
|---|---|---|
| upstream default (`BLOCK_SIZE_M=64` always) | 15.76 ms | 59.4 tok/s |
| M-aware (`M ≤ 32 → 16, stages 3`) | **12.13 ms** | **75.6 tok/s** |
| | **−23.0%** | **+27.2%** |

Triton itself is not implicated: 3.6.0 and 3.7.1 time identically at a fixed
config. Set `SM89_BLOCK_M_LOW_LIMIT=0` to restore upstream behaviour.

---

## Measured

8 × RTX 4090, TP4 × PP2, `--max-model-len 131072`, `--gpu-memory-utilization 0.95`,
prefix caching on, CUDA graphs on. Random-input benchmark, 1024 in / 256 out,
`temperature=0`, median of repeated runs.

| | |
|---|---|
| Single stream TPOT P50 | **12.16 ms** (±0.0% across runs) |
| Single stream TTFT P50 | 282–297 ms |
| 10-way aggregate | **273.9 tok/s** (per-stream TPOT 32.1 ms) |
| KV cache pool | **157,511 tokens**; 1.20× concurrency at full 131,072 context |
| Long input | 100,000-token prompt served |

Quality spot-check (chat mode, `thinking=False`, 3 runs each): code, arithmetic,
Chinese and Japanese prompts all answer correctly.

---

## Known limitations

- **TP ≤ 4.** See above.
- **FlashInfer `0.6.14+sm89` is a binary-only dependency.** No source branch was
  ever published, so it cannot be rebuilt from source. A copy is vendored under
  [`assets/`](assets/) to keep this configuration reproducible.
- **`has_cutedsl()` still only checks whether the package is installed, not the
  architecture.** On Ada, CuTe-DSL kernels produce wrong output without raising.
  The short-circuit here works around it; the upstream probe is the real defect.
- The port depends on the private FlashInfer symbol
  `mla._core._resolve_dsv4_sparse_mla_backend`. If it moves, the probe returns
  `False` and the backend is refused — a loud failure, not a silent one.

---

## Credits

The three added operator files are ports from
[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89),
Apache-2.0, the same license as vLLM. That project is also the source of the
FlashInfer `+sm89` build.

Upstream vLLM: <https://github.com/vllm-project/vllm>
