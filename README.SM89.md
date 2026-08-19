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


---

## Credits

The three added operator files are ports from
[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89),
Apache-2.0, the same license as vLLM. That project is also the source of the
FlashInfer `+sm89` build.

Upstream vLLM: <https://github.com/vllm-project/vllm>
