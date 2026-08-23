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

Smaller counts work as long as the weights fit.

**Tensor-parallel size must be ≤ 4.** The FlashInfer sparse-MLA prefill dispatch
instantiates `num_heads` in `{16, 32, 64, 128}`; DeepSeek-V4-Flash has 64 attention
heads, so TP8 asks for 8 heads per rank and the dispatch refuses. Use TP4 × PP2 on
eight GPUs — which also avoids replicating the single-KV-head MLA cache on every
rank. This is upstream FlashInfer behaviour, not specific to this port.

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
uv pip install ./assets/flashinfer_python-0.6.14+sm89-py3-none-any.whl
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Required. Without it the server starts, answers, and emits garbage from the
# first token — no crash, no warning. See assets/README.md for the mechanism.
python assets/patch-flashinfer-sm89-scale-clamp.py
```

Verify before installing:

```
sha256  95ea827b9a6303fc974f7b2872befb23efed9a3eb85074b262261e3c3944730b
```

Once installed it reports version `0.6.14+sm89`, which is what
`has_flashinfer_sparse_mla_sm89()` probes for. **Check that string, not
`pip list`** — stock FlashInfer `0.6.14` is 6 KB smaller and one filename suffix
away, installs and imports cleanly, and then rejects SM89 at backend selection.
[`assets/README.md`](assets/README.md) has three probes that tell the two apart.

After patching, point `FLASHINFER_CACHE_DIR` at a fresh directory so stale
cached kernels are not reused.

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
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4
```

Under Docker, add `--security-opt seccomp=unconfined` — otherwise OpenBLAS
fails with `pthread_create: Operation not permitted` on hosts whose
libseccomp does not know `clone3`.

### Reasoning output

DeepSeek-V4-Flash reasons by default. With `--reasoning-parser deepseek_v4` the
chain of thought is split out into **`message.reasoning`** (not
`reasoning_content`), counted separately under
`usage.completion_tokens_details.reasoning_tokens`:

```json
{"message": {"content": "17 times 23 is 391.",
             "reasoning": "Thinking. 1. **Analyze the Request:** ..."}}
```

Pass `chat_template_kwargs: {"thinking": false}` (or `{"reasoning_effort": "none"}`)
to turn it off. Note that this does not save tokens — the model writes the same
working into `content` instead. Measured on one prompt: 82 completion tokens with
reasoning on (14-character answer), 114 with `thinking=false`, 196 with
`reasoning_effort=none`.

Startup is healthy when the log contains all three:

```
Using 'MARLIN' Mxfp4 MoE backend.
Using FP8 indexer cache for Lightning Indexer.
Using fp8_ds_mla data type to store kv cache.
```

### Flags that do not work here

| Flag | Why |
|---|---|
| `--tensor-parallel-size 8` | Sparse-MLA prefill has no `num_heads=8` instantiation — see [Hardware](#hardware). Fails on the first real prefill, not at startup. |
| `--enable-return-routed-experts` | Needs a full-attention KV cache group; DSv4's are all compressed → `ValueError` during engine init. Also incompatible with `PP > 1`, and costs a fixed 512 MiB per GPU. |
| `--gpu-memory-utilization 0.97` with PP | The NCCL buffers for the PP channel live outside vLLM's budget; 0.95 is the working value. |
| `--reasoning-parser deepseek_r1` | It splits on a literal `</think>` in the text, which the DSv4 tokenizer has already consumed. With the end token absent the base parser puts the whole output in `reasoning` and leaves **`content` null**. Use `deepseek_v4`. |

### The M-aware GEMM default

Upstream hardcodes `BLOCK_SIZE_M=64` for the block-scaled Triton GEMM, so every
decode step (`M ≤ 32`) launches a single M-block and under-fills the 128 SMs.
Single-variable A/B, same container and launch script:

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
