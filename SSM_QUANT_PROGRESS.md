# SSM-State Cache Quantization — Progress Notes

Working branch: `int8-fp8-ssm-cache` (tracking `fork/int8-fp8-ssm-cache`).

## 1. Goal

Add support for **quantized SSM temporal state cache** in Mamba2-style layers so
the temporal state can live in `int8`, `int16`, or `fp8_e4m3fn` instead of
fp32/bf16/fp16. This reduces the SSM-state memory footprint (per request,
scales with `num_heads × head_dim × dstate`) and should leave output quality
close to the fp16 + stochastic-rounding baseline that already exists in tree.

Test model: **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`** (NemotronH
hybrid — Mamba2 + attention — with NVFP4 weights). Single NVIDIA GB200 per
server (~189 GB HBM), 4 GPUs available on the box.

## 2. Branch Changes (2 WIP commits on top of main)

- `b86b04065 wip` — main implementation
- `d14d2c5f2 wip: derive quant_max from dtype instead of hardcoded dict`

Files touched (+160 / −17):

| File | Summary |
|---|---|
| `vllm/config/cache.py` | Extended `MambaDType` with `"int16"`, `"int8"`, `"fp8_e4m3fn"`. |
| `vllm/utils/torch_utils.py` | Added `"int16" → torch.int16` and `"fp8_e4m3fn" → torch.float8_e4m3fn` in `STR_DTYPE_TO_TORCH_DTYPE`. |
| `vllm/model_executor/layers/mamba/mamba_utils.py` | New `QUANTIZED_SSM_STATE_DTYPES` constant, new `quantize_scaled()` helper, tuples extended with a third element (`ssm_state_scales`, fp32). |
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | `kv_cache` tuple now `(conv_state, ssm_state, ssm_state_scales)`. On prefill write-back: quantize via `quantize_scaled()` and store scale. On prefill read of initial states: dequantize before the scan. Decode path receives `state_scale` through the SSU dispatch. |
| `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` | `_selective_scan_update_kernel` gained `state_scales_ptr` + strides and a `QUANT_MAX: tl.constexpr`. On load: dequant via per-`(head, dim)` scale. On store (non-spec path): compute per-dim amax over dstate, encode/decode scale, round, clamp, cast. Guards padded CUDA-graph slots (`null_block_id`). New `_quant_max()` helper. |
| `vllm/model_executor/layers/mamba/ops/ssu_dispatch.py` | Threaded `state_scale` through `MambaSSUBackend`, Triton, FlashInfer, and the unified dispatch. |

### Key design decisions

- **Block-scale quantization**: one fp32 decode scale per `(head, dim)` channel
  (shape `(num_heads/tp, head_dim, 1)`), covering the `dstate` axis.
  Dynamic — recomputed from `amax` on every write.
- **`quant_max` is derived** from `torch.finfo(dtype).max` /
  `torch.iinfo(dtype).max` rather than a hardcoded dict (commit 2).
- **Kernel quant happens entirely inside Triton** on the decode path. On the
  prefill path the quantization is done in Python via `quantize_scaled()` and
  the fp32 scan output is then cast.
- **CUDA-graph safety**: for padded slots (`state_batch_idx == null_block_id`)
  both scale loads and stores are masked out so OOB pointer arithmetic is
  harmless.

## 3. Serving Experiment (in-progress)

### Configurations

Three servers in parallel, one per GPU, port-per-model, same Nemotron-H NVFP4
checkpoint. Venv: `/my_home/venvs/vllm/bin/python`.

| Variant | GPU | Port | `--mamba-ssm-cache-dtype` | Extra flags | Served name |
|---|---|---|---|---|---|
| FP8 SSM | 0 | 8000 | `fp8_e4m3fn` | — | `nemotron-fp8ssm` |
| INT8 SSM | 1 | 8001 | `int8` | — | `nemotron-int8ssm` |
| FP16 + SR (baseline) | 2 | 8002 | `float16` | `--enable-mamba-cache-stochastic-rounding` | `nemotron-fp16sr` |

Common flags on every server:
`--trust-remote-code --tensor-parallel-size 1 --max-model-len 8192 --gpu-memory-utilization 0.85`.

Logs: `/tmp/vllm-ssm-test/{fp8,int8,fp16sr}.log`.

### Launch commands (for reference)

```bash
# FP8 SSM (GPU 0)
CUDA_VISIBLE_DEVICES=0 /my_home/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --port 8000 --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype fp8_e4m3fn \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --served-model-name nemotron-fp8ssm \
  > /tmp/vllm-ssm-test/fp8.log 2>&1 &

# INT8 SSM (GPU 1)
CUDA_VISIBLE_DEVICES=1 /my_home/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --port 8001 --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype int8 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --served-model-name nemotron-int8ssm \
  > /tmp/vllm-ssm-test/int8.log 2>&1 &

# FP16 + Stochastic Rounding baseline (GPU 2)
CUDA_VISIBLE_DEVICES=2 /my_home/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --port 8002 --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --served-model-name nemotron-fp16sr \
  > /tmp/vllm-ssm-test/fp16sr.log 2>&1 &
```

### Startup timeline (observed)

First attempt (three engines on shared tmpfs cache, all in parallel):

| Step | FP8 | INT8 | FP16+SR |
|---|---|---|---|
| First launch | failed — needed `--trust-remote-code` | failed — same | — |
| Second launch | failed — `ModuleNotFoundError: pandas` in venv | failed — same | — |
| After `pip install pandas` | loading… | loading… | not yet launched |
| Weight load | 50.85 s | 51.29 s | 55.50 s |
| Model weights resident | 69.39 GiB | 69.39 GiB | 69.39 GiB |
| `torch.compile` | 29.39 s | 29.37 s | 30.35 s |
| Post-compile | flashinfer cubin JIT + nvcc of trtllm MoE routing kernels | same | same |
| **Result** | **deadlocked at `03:04:04`** — three engines fought for the same flashinfer JIT file locks, logs frozen 7+ min, processes alive but making no progress | same | same |

Second attempt (one engine alone, then the others — scratch exports in place):

| Step | FP8 (GPU 0) | INT8 (GPU 1) | FP16+SR (GPU 2) |
|---|---|---|---|
| Filesystem for weights | **LUSTRE** (via `HF_HOME`) | LUSTRE | LUSTRE |
| Weight load | 54.42 s | ~54 s | ~55 s |
| `torch.compile` | 32.33 s | 17.08 s (graph) + 30.78 s total | 17.18 s + 30.41 s total |
| FlashInfer JIT | **~12 min** (cold compile, see §3a) | **skipped (cache hit)** | **skipped (cache hit)** |
| Profiling/warmup | ok | 1.76 s | 1.65 s |
| **HTTP `/v1/models`** | **200 OK** at `03:31:58` | **200 OK** ~`03:37` | **200 OK** ~`03:37` |

The key win: once FP8 warmed the flashinfer cache on scratch, INT8 and
FP16+SR started in parallel and skipped the flashinfer JIT phase entirely —
they went from launch to serving in under 3 minutes instead of ~12.

### 3a. FlashInfer JIT compile (one-time cost)

Under `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>/cached_ops/`
(here: `/my_home/vllm-scratch/fi/.cache/flashinfer/0.6.7/100a/cached_ops/`),
5 modules were compiled by ninja + nvcc:

| Module | build stanzas | Notes |
|---|---|---|
| `fp4_gemm_cutlass` | 10 | |
| `fp4_quantization_100` | 8 | |
| `fp8_gemm_cutlass` | 14 | |
| `gemm` | 4 | |
| `fused_moe_trtllm_sm100` | **25** | by far the largest; ~4 per worker with `--threads=1` |

Total: **61 build stanzas** for this model / arch combination. Final cache
size: **~195 MB**. Expected to grow only if vLLM invokes additional
flashinfer entry points it hasn't compiled before.

### 3b. First end-to-end curl against the FP8 server

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nemotron-fp8ssm","prompt":"Q: What is 17 * 23?\nA:","max_tokens":60,"temperature":0.0,"top_p":0.95,"seed":1}'
```

Response (`fp8_e4m3fn` SSM cache, block-scaled per-`(head, dim)`):

```
 391
Q: What is 17 * 24?
A: 408
Q: What is 17 * 25?
A: 425
Q: What is 17 * 26?
A: 44…   (truncated at max_tokens=60)
```

All arithmetic correct. End-to-end quantized SSM path works: encode on
prefill write-back (`quantize_scaled()`), decode-path Triton kernel
dequantize-on-load + requantize-on-store with `QUANT_MAX` constexpr,
block-scale tensor sitting in `kv_cache[2]`.

### Troubleshooting notes

- `VLLM_USE_V1=1` is now an **unknown env var** in this branch; dropped it.
- User venv lacked `pandas` (imported transitively via `vllm._aiter_ops`);
  installed `pandas==3.0.2` (pulled in `numpy==1.26.4`, which conflicts with
  the installed `opencv-python-headless==4.13.0.92` — not expected to matter
  for our tests, noted for later).
- Checkpoint lives on TMPFS (74.80 GiB). No network fetch beyond initial
  metadata after the first launch.
- Warning from loader: "Checkpoint does not provide a q scaling factor. Setting
  it to k_scale." — inherent to the NVFP4 checkpoint, not related to the SSM
  cache change.

## 4. Evaluation with `lm-eval`

Evaluation is done from a **separate venv** so the eval harness does not share
site-packages with the serving venv:

- Serving venv: `/my_home/venvs/vllm/bin/python`
- **Eval venv: `/my_home/venvs/eval_venv/bin/python`** — verified to contain
  `lm_eval==0.4.11` on Python 3.12.3.

The harness drives each server through its OpenAI-compatible
`/v1/completions` endpoint via `local-completions`.

### Gotchas learned the hard way

- `lm_eval` instantiates a **tokenizer locally** via `transformers.AutoTokenizer.from_pretrained(model)`.
  If the value of `model=` is just the `--served-model-name` (e.g.
  `nemotron-fp8ssm`), that is **not** a valid HF repo and the tokenizer load
  fails. Fix: pass `tokenizer=<real_hf_id>` **and** `tokenizer_backend=huggingface`
  and `trust_remote_code=True` in `--model_args`.
- `transformers` is not in `/my_home/venvs/eval_venv/` by default — install
  it before the first run: `/my_home/venvs/eval_venv/bin/python -m pip install transformers`.

### Base command template (working version)

```bash
/my_home/venvs/eval_venv/bin/python -m lm_eval \
  --model local-completions \
  --model_args "model=$SERVED_NAME,tokenizer=$HF_ID,tokenizer_backend=huggingface,trust_remote_code=True,base_url=http://localhost:$PORT/v1/completions,api_key=EMPTY,num_concurrent=50,timeout=45000" \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs temperature=0.0,top_p=0.95,do_sample=true,seed=1 \
  --output_path ./results/gsm8k --log_samples
```

- `$SERVED_NAME` = the vLLM `--served-model-name` (what the server answers to
  on the OpenAI API).
- `$HF_ID` = the real HuggingFace model id (used only for tokenizer loading);
  here always `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`.
- `$PORT` = server port.

### Per-variant commands (copy-paste)

```bash
mkdir -p /my_home/vllm/results /tmp/vllm-ssm-test
HF_ID=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4

# FP8 SSM — GPU 0, port 8000, served as nemotron-fp8ssm
/my_home/venvs/eval_venv/bin/python -m lm_eval \
  --model local-completions \
  --model_args "model=nemotron-fp8ssm,tokenizer=${HF_ID},tokenizer_backend=huggingface,trust_remote_code=True,base_url=http://localhost:8000/v1/completions,api_key=EMPTY,num_concurrent=50,timeout=45000" \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs temperature=0.0,top_p=0.95,do_sample=true,seed=1 \
  --output_path /my_home/vllm/results/gsm8k-fp8ssm --log_samples \
  > /tmp/vllm-ssm-test/eval-fp8.log 2>&1 &

# INT8 SSM — GPU 1, port 8001, served as nemotron-int8ssm
/my_home/venvs/eval_venv/bin/python -m lm_eval \
  --model local-completions \
  --model_args "model=nemotron-int8ssm,tokenizer=${HF_ID},tokenizer_backend=huggingface,trust_remote_code=True,base_url=http://localhost:8001/v1/completions,api_key=EMPTY,num_concurrent=50,timeout=45000" \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs temperature=0.0,top_p=0.95,do_sample=true,seed=1 \
  --output_path /my_home/vllm/results/gsm8k-int8ssm --log_samples \
  > /tmp/vllm-ssm-test/eval-int8.log 2>&1 &

# FP16 + Stochastic Rounding baseline — GPU 2, port 8002, served as nemotron-fp16sr
/my_home/venvs/eval_venv/bin/python -m lm_eval \
  --model local-completions \
  --model_args "model=nemotron-fp16sr,tokenizer=${HF_ID},tokenizer_backend=huggingface,trust_remote_code=True,base_url=http://localhost:8002/v1/completions,api_key=EMPTY,num_concurrent=50,timeout=45000" \
  --tasks gsm8k --num_fewshot 5 \
  --gen_kwargs temperature=0.0,top_p=0.95,do_sample=true,seed=1 \
  --output_path /my_home/vllm/results/gsm8k-fp16sr --log_samples \
  > /tmp/vllm-ssm-test/eval-fp16sr.log 2>&1 &
```

The three runs can execute in parallel — each server is on its own GPU, and
the harness is CPU-light (tokenizer + small sync queue).

Keep `temperature=0.0` so output is deterministic and any score delta
between variants is attributable to the SSM-cache representation, not
sampling noise. `num_concurrent=50` keeps the server busy, `timeout=45000`
is generous enough to survive first-request compile/graph-capture stalls.

Observed request rate at `num_concurrent=50`: ~6 req/s per server, so
gsm8k (1319 items × 5-shot) takes **~3–4 minutes per variant**.

### Scoreboard — gsm8k 5-shot (temperature=0.0, seed=1, 1319 items)

**Run 3** (after fixing the fp8x4 byte-order bug — see §7a):

| Variant | Served name | Port | strict-match | flexible-extract |
|---|---|---|---|---|
| FP8 SSM (RN)                    | `nemotron-fp8ssm`      | 8000 | 0.9128 ± 0.0078 | 0.9265 ± 0.0072 |
| INT8 SSM + SR                   | `nemotron-int8ssm-sr`  | 8001 | 0.9128 ± 0.0078 | 0.9212 ± 0.0074 |
| FP16 SSM + SR (baseline)        | `nemotron-fp16sr`      | 8002 | **0.9272 ± 0.0072** | **0.9333 ± 0.0069** |
| **FP8 SSM + SR (PTX, FIXED)**   | `nemotron-fp8ssm-sr`   | 8003 | 0.8643 ± 0.0094 | 0.8825 ± 0.0089 |

FP8+SR recovered from the previous catastrophic collapse (0.0023 → 0.8643
strict-match, ~380× improvement). It still trails baseline by about
6 pp. This is plausible on the merits — see §7a for the root cause of
the collapse, and the discussion below for why pure SR noise can hurt
fp8 inference where it would help fp8 training.

Raw outputs: `/my_home/vllm/results/gsm8k-run3-{fp8ssm,int8ssm-sr,fp16sr,fp8ssm-sr}/`.

---



| Variant | Served name | Port | `exact_match` (strict-match) | `exact_match` (flexible-extract) |
|---|---|---|---|---|
| FP16 SSM + SR (baseline) | `nemotron-fp16sr` | 8002 | **0.9219 ± 0.0074** | **0.9280 ± 0.0071** |
| FP8 SSM | `nemotron-fp8ssm` | 8000 | 0.9090 ± 0.0079 | 0.9242 ± 0.0073 |
| INT8 SSM | `nemotron-int8ssm` | 8001 | **0.9196 ± 0.0075** | **0.9295 ± 0.0071** |

Raw outputs: `/my_home/vllm/results/gsm8k-{fp8ssm,int8ssm,fp16sr}/`.

#### Reading the numbers

- **INT8 ≈ FP16+SR baseline.** Both metrics are within <1 stderr of the
  baseline; INT8 flexible-extract (0.9295) is even nominally higher than
  baseline (0.9280). Block-scaled int8 is a viable drop-in for the SSM state.
- **FP8 is marginally worse.** Flexible-extract is within noise
  (−0.38 pp, ~0.5 σ). Strict-match is −1.29 pp (~1.2 σ) — borderline
  significant but small in practice. Likely attributable to FP8 E4M3's
  tiny sub-normal range combined with the small magnitudes typical of
  SSM state values; the per-`(head, dim)` block scale partly but not
  fully compensates.
- **Both quantized variants stay well above any reasonable quality floor**
  for gsm8k at this model size.

Additional tasks worth adding next: `mmlu` (5-shot, short-decode breadth),
`arc_challenge` (25-shot), `hellaswag` (10-shot). With warm caches each
server can be recycled quickly.

## 5. Planned Next Steps

1. Wait for all three servers to finish flashinfer cubin JIT + CUDA-graph
   capture and bind to their ports, then hit `/v1/models` + a small
   `/v1/completions` on each to confirm decoding works end-to-end with each
   SSM dtype (no kernel NaNs, no shape errors, no `tl.store` / scale-stride
   regressions).
2. Run the three **gsm8k 5-shot** evals above; record per-variant `acc` and
   compare FP8 / INT8 to the FP16+SR baseline.
3. Record peak per-request SSM-cache memory for each variant (the whole point
   of the change) — can be read off the engine init logs (mamba page size).
4. If quality is within noise of FP16+SR, clean up the two WIP commits into a
   single upstreamable change with tests (at minimum a unit test for
   `quantize_scaled()` round-trip and a small end-to-end hybrid-model test
   with `--mamba-ssm-cache-dtype=int8`).

## 6. Next-time quick-start (reuse caches built today)

The caches are all on `/my_home/vllm-scratch/` and will survive reboots.
To bring the three servers back up without re-paying flashinfer JIT,
`torch.compile`, or the 75 GB HF download:

```bash
# Exports — put these in front of every vLLM process on this box
export FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi
export VLLM_CACHE_ROOT=/my_home/vllm-scratch/vllm
export HF_HOME=/my_home/vllm-scratch/hf

# FP8 SSM cache — GPU 0, port 8000
CUDA_VISIBLE_DEVICES=0 /my_home/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --port 8000 --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype fp8_e4m3fn \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --served-model-name nemotron-fp8ssm \
  > /tmp/vllm-ssm-test/fp8.log 2>&1 &

# INT8 SSM cache — GPU 1, port 8001
CUDA_VISIBLE_DEVICES=1 /my_home/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --port 8001 --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype int8 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --served-model-name nemotron-int8ssm \
  > /tmp/vllm-ssm-test/int8.log 2>&1 &

# FP16 SSM cache + stochastic rounding (baseline) — GPU 2, port 8002
CUDA_VISIBLE_DEVICES=2 /my_home/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --port 8002 --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --served-model-name nemotron-fp16sr \
  > /tmp/vllm-ssm-test/fp16sr.log 2>&1 &
```

Expected per-server startup when caches are warm:

- weight load from Lustre → ~55 s
- `torch.compile` → ~30 s (cached; still replays the fx graph)
- flashinfer JIT → **skipped** (cache hit on `fused_moe_trtllm_sm100` etc.)
- CUDA-graph capture + warmup → a few seconds
- Total: **under 2 min per server** (vs ~12 min cold for the first one).

All three can be launched in parallel now — flashinfer JIT is already
cached, so there is no lock contention.

### Extra venv deps installed today

Both venvs needed one missing package each:

- `/my_home/venvs/vllm/bin/python -m pip install pandas`
  (imported transitively by `vllm._aiter_ops`)
- `/my_home/venvs/eval_venv/bin/python -m pip install transformers`
  (needed by `lm_eval.models.api_models.LocalCompletionsAPI`)

Both should persist across reboots since the venvs live on `/my_home`.

## 7. Persistent caches on scratch (reference)

First-time startup is dominated by three caches that **by default live on tmpfs
and disappear on reboot**:

| Cache | Default path | Size observed | Controlling env var |
|---|---|---|---|
| FlashInfer JIT (nvcc-compiled `.so`/`.o` for trtllm MoE) | `/root/.cache/flashinfer/<ver>/<arch>/cached_ops/` | ~199 MB | `FLASHINFER_WORKSPACE_BASE` (the base; flashinfer appends `.cache/flashinfer/<ver>/<arch>/cached_ops`) |
| FlashInfer cubin headers (from `edge.urm.nvidia.com`) | `$VENV/lib/.../flashinfer_cubin/cubins/` (ships with the `flashinfer-cubin` wheel) | ~1.2 GB | `FLASHINFER_CUBIN_DIR` |
| vLLM `torch.compile` cache (Dynamo + AOT fx graph) | `/root/.cache/vllm/torch_compile_cache/` | ~301 MB | `VLLM_CACHE_ROOT` |
| HuggingFace hub (model weights, ~74.80 GiB for this model) | `/root/.cache/huggingface/` | ~75 GB | `HF_HOME` |

`/root` is tmpfs (846 GB) — gone on reboot. `/my_home` is Lustre (500 TB,
effectively unlimited) — persistent. Move everything to `/my_home`:

```bash
mkdir -p /my_home/vllm-scratch/{fi,vllm,hf}
export FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi
export VLLM_CACHE_ROOT=/my_home/vllm-scratch/vllm
export HF_HOME=/my_home/vllm-scratch/hf
```

Resulting on-disk layout after first warm-up:

- FlashInfer JIT: `/my_home/vllm-scratch/fi/.cache/flashinfer/0.6.7/100a/cached_ops/`
- vLLM compile cache: `/my_home/vllm-scratch/vllm/torch_compile_cache/`
- HF hub: `/my_home/vllm-scratch/hf/hub/`

Subsequent launches skip the ~3–10 min flashinfer nvcc JIT phase and the ~30 s
`torch.compile` phase per engine.

### Optional: preserve today's already-warmed caches

```bash
mkdir -p /my_home/vllm-scratch/fi/.cache \
         /my_home/vllm-scratch/vllm \
         /my_home/vllm-scratch/hf
rsync -a /root/.cache/flashinfer/  /my_home/vllm-scratch/fi/.cache/flashinfer/
rsync -a /root/.cache/vllm/        /my_home/vllm-scratch/vllm/
rsync -a /root/.cache/huggingface/ /my_home/vllm-scratch/hf/   # skip if OK to re-download
```

### Known gotcha (observed today)

When three engines start simultaneously and share the same flashinfer JIT
cache, they serialize on the same file locks and can deadlock. Workarounds:

- Bring up the first engine alone, let it warm the cache, then start the
  others. (What we're doing now.)
- Or give each engine its own scratch, e.g.
  `FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi-gpu0` etc. This costs
  disk space (each ~200 MB + ~1.2 GB cubins if you also override cubin dir)
  but avoids any lock contention.

## 7a. FP8 + SR PTX byte-order bug (found & fixed)

The first attempt at wiring `cvt.rs.satfinite.e4m3x4.f32` into
`_selective_scan_update_kernel` produced a catastrophic gsm8k collapse
(**0.0023** strict-match vs. ~0.92 baseline — effectively random output)
while still passing single-prompt smoke tests (`13 × 27 = 351` worked
fine).

### Root cause

Triton's `inline_asm_elementwise` with `pack=N` stores packed outputs
**little-endian** inside the destination register: lane `i` occupies
byte (or 16-bit word) index `i` starting at the low bits.

The `cvt.rs.*.{fNxM}.fSS` family of PTX instructions does the opposite:
the **leftmost** source argument in `{a, b, c, d}` lands in the **high**
bits of the destination register.  So the naive mapping

```ptx
cvt.rs.satfinite.e4m3x4.f32 $0, {$1, $2, $3, $4}, $5;   // WRONG
```

silently reverses every group of 4 contiguous fp32 inputs inside the
4-byte packed output.  Because `BLOCK_SIZE_DSTATE` is the innermost
kernel axis (`dstate = 128` here), this permutes the SSM temporal state
in place: `state[..., 0..4] = rev(state[..., 0..4])`, and similarly for
every 4-element group along dstate.  A spatial shuffle of the entire
SSM state is catastrophic for quality but does not produce NaN/Inf or
touch memory bounds — so it passes every sanity check except actual
evals.

The fix is a literal argument-order reversal, matching the pattern the
pre-existing `convert_rs_fp16x2` already uses (`$0, $2, $1, $3` swaps
the two fp32 sources):

```ptx
cvt.rs.satfinite.e4m3x4.f32 $0, {$4, $3, $2, $1}, $5;   // CORRECT
```

The same pitfall is documented in upstream Triton issue
[`triton-lang/triton#8822`](https://github.com/triton-lang/triton/issues/8822);
the official answer from `peterbell10` matches the fix we landed.

### Minimal unit test (kept for regression)

`/tmp/test_fp8_sr.py` drives the helper on 16 off-grid fp32 values and
checks the stochastic output flips between the two neighbouring E4M3
grid points; e.g. `0.128` bounces between `0.125` and `0.140625`, and
`4.1` bounces between `4.0` and `4.5`.  Before the fix, all 16 outputs
would come back as `0.0` (or, equivalently, all map to the same grid
point) because the 4-element shuffle collapses the information.  After
the fix the distribution matches the expected on-grid / neighbouring
grid cycling.

### Why the `13×27=351` smoke test passed

Greedy decoding with tiny prompts barely exercises the SSM decode
store path — only a few tokens, very short state.  The arithmetic
answer for `13 × 27` is low-entropy and reachable from a nearly any
mildly corrupted state.  gsm8k 5-shot, by contrast, forces 256 new
tokens of chain-of-thought per question across 1319 questions, so
state corruption compounds step-over-step and total accuracy collapses
to noise.  **Moral: always gate quantized-cache changes on a real
multi-step eval, not on a single prompt.**

## 8. Open Questions / TODOs

- Confirm the kernel's `state_scales` stride passes are valid when
  `state_scale=None` (dummy slice of `state[:1,:1,:1,:1]`). Current guard uses
  `QUANT_MAX > 0.0` — worth a unit test to be sure the fp32 path is
  byte-identical before/after.
- Spec-decoding write path (`IS_SPEC_DECODING=True`) is not currently
  quantized — only the non-spec store branch writes scales. Decide whether
  spec decoding needs its own quant-store branch or whether it's fine to
  always decode from fp32 there.
- `int16` is wired up end-to-end but not exercised in this sweep; add once
  the int8 / fp8 numbers look reasonable.
- AGENTS.md says to use `uv pip` rather than bare `pip`; we used
  `/my_home/venvs/vllm/bin/python -m pip install pandas` to add a missing
  runtime dep in the user venv. Consider re-resolving with `uv` before any
  upstream PR.
