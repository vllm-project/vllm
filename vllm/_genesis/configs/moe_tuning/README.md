# MoE Tuning Configs (community-contributed, not in upstream vLLM)

This directory holds pre-tuned Triton kernel configurations for the fused-MoE
kernel on hardware/shape combinations that vLLM upstream does **not** ship.

vLLM looks up tuned configs by filename at runtime:
`vllm/model_executor/layers/fused_moe/configs/E={experts},N={intermediate},device_name={gpu},dtype={dtype},block_shape={shape}.json`

If the matching file is present, `fused_moe` loads it and skips the on-the-fly
Triton autotuner path.

---

## ⚠ Important — Triton vs Marlin backend selection on Ampere

**On Ampere SM 8.x (A5000, A6000, 3090, 3080) with FP8 weights, vLLM uses
the MARLIN kernel by default — NOT Triton.** The Triton FP8 MoE kernel
rejects our quantization scheme on SM 8.x (no hardware FP8 support):

```
ValueError: FP8 MoE backend TRITON does not support the deployment configuration
since kernel does not support quantization scheme
QuantKey(f8e4m3fn,scale(f32,static,GroupShape(128,128)),symmetric)
```

Source: closed PR [vllm-project/vllm#40129](https://github.com/vllm-project/vllm/pull/40129)
review by [@mgoin](https://github.com/mgoin) (2026-04-17). Confirmed by the
PR author after `select_fp8_moe_backend` traces.

**Practical implication for the bundled A5000 file below:** the JSON sits in
the configs directory but **vLLM never reads it on our A5000+FP8 PROD** —
the Marlin path is taken instead. The +16% number in the original PR body
was measured before this backend-selection mismatch was understood.

**Where the A5000 JSON DOES help:** any future deployment that explicitly
forces the Triton MoE backend (`VLLM_FORCED_FP8_MOE_BACKEND=triton`) AND
patches the QuantKey check to accept our scheme. Neither is the default.

**Where this DOES help on consumer Ampere:** the Marlin MoE path has its
own per-SM tuning surface ([`vllm/_genesis/kernels/marlin_tuning.py`](../../kernels/marlin_tuning.py))
with our P17/P18/PN64 patches feeding `BLOCK_SIZE_M`, `num_warps`, `num_stages`.
That's the path Genesis actively tunes for Ampere consumer; the Triton JSON
files in THIS directory are for the ORTHOGONAL Triton backend path used on
Ada (4090) / Hopper (H100) / Blackwell datacenter where native FP8 exists.

### Per-arch backend selection (FP8 MoE)

| Arch | SM | FP8 native? | Default backend | Where to tune |
|---|---:|:-:|:-:|:-:|
| Ampere consumer (3090, A5000, A6000) | 8.6 | NO | **MARLIN** | Genesis P17/P18 + `kernels/marlin_tuning.py` |
| Ampere datacenter (A100) | 8.0 | NO | MARLIN | same |
| Ada (4070, 4080, 4090) | 8.9 | YES | TRITON | this directory (.json) |
| Hopper (H100, H200) | 9.0 | YES | TRITON | this directory (.json) |
| Blackwell datacenter (B100, B200) | 10.0 | YES | TRITON | this directory (.json) |
| Blackwell consumer (5070, 5080, 5090) | 12.0 | YES | TRITON | this directory + Marlin PN64 placeholder |

---

## Files here

### `E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json`

- **Source**: upstream PR [vllm-project/vllm#40129](https://github.com/vllm-project/vllm/pull/40129)
  (community-contributed by Sandermage, **CLOSED 2026-04-19** —
  reviewer flagged that on SM 8.x the FP8 MoE backend is MARLIN, not Triton,
  so the tuned Triton config is never read at runtime on Ampere).
- **Target shape**: MoE with `E=256` experts, `N=512` intermediate size,
  FP8 W8A8 weights, block scaling `[128,128]`.
- **Target hardware**: NVIDIA RTX A5000 (Ampere, SM 86, 24 GB).
- **Target model**: Qwen3.6-35B-A3B-FP8 — exactly our production model on
  VM 100 (192.168.1.10).
- **Batch sizes covered**: 1, 2, 4, 8, 16, 32, 64, 128, 256.
- **Original measured uplift** (from PR body): +16% generation tok/s
  (~125 → ~145 tok/s) — but see runtime caveat above. The number was
  measured against the Triton path that turned out to be inactive on
  our Ampere FP8 stack. The bundled JSON is preserved here for the rare
  Ampere user who explicitly forces the Triton backend (operator setting
  `VLLM_FORCED_FP8_MOE_BACKEND=triton` AND patching the SM 8.x QuantKey
  reject), and for forward-compat with any future vLLM that re-routes.
- **Tuning methodology**: vLLM's built-in Triton autotuner, 100 iterations per
  batch size, best config by lowest kernel time.

Key observations from the PR author for SM 86:
- `BLOCK_SIZE_M=16` is optimal — MoE routes few tokens per expert.
- `BLOCK_SIZE_K` varies 64-256 with batch size.
- `num_stages` kept conservative (1-4) because of SM 86 shared-memory limits.

---

## How to use it (operator runbook)

The file must land inside the vLLM container at the path vLLM probes:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/`
(exact Python version depends on the container image — adjust if different).

### Step 1 — find the vLLM configs directory inside the container

```bash
docker exec vllm-qwen python -c \
  "import vllm, os; \
   p = os.path.join(os.path.dirname(vllm.__file__), \
   'model_executor/layers/fused_moe/configs'); \
   print(p)"
```

### Step 2 — copy the file in

From the repo root on the host (VM 100, where this repo is checked out):

```bash
docker cp \
  "vllm/_genesis/configs/moe_tuning/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json" \
  vllm-qwen:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
```

Replace `vllm-qwen` with the actual container name (`docker ps | grep vllm`) and
`/usr/local/lib/python3.12/...` with the path printed by Step 1 if different.

### Step 3 — restart the container so vLLM re-reads the configs dir

```bash
docker compose restart vllm-qwen
```

Or, if your compose file is separate for the vLLM service:

```bash
cd /opt/genesis/vllm && docker compose restart
```

### Step 4 — verify it loaded

Tail the logs and look for a `Using configuration from ...A5000...` line:

```bash
docker logs -f vllm-qwen 2>&1 | grep -i "configuration from\|fused_moe"
```

If you see `A5000,dtype=fp8_w8a8,block_shape=[128,128].json` referenced — it's live.
If you see `Using default MoE config` — the filename didn't match exactly
(check `E=`, `N=`, `device_name=NVIDIA_RTX_A5000` spelling; GPU names from
`torch.cuda.get_device_name(0)` must match byte-for-byte).

---

## Persistence warning

`docker cp` writes into the container's read/write layer. A `docker compose
restart` or `stop && start` keeps it. A `docker compose down && up -d` (or a
`docker rm`) **wipes it** and you must re-copy.

For permanent install, bake it into a custom image:

```Dockerfile
FROM vllm/vllm-openai:<tag>
COPY "E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json" \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
```

Or mount the configs directory as a volume in `docker-compose.yml`.

---

## Why we carry this locally

Upstream closed PR #40129 without merging — reviewer ([@mgoin](https://github.com/mgoin),
2026-04-17) flagged that on SM 8.x FP8 MoE goes through MARLIN, not Triton, so
the bundled Triton-tuned JSON is **never read at runtime on our Ampere FP8 PROD**.
The "+16%" number from the PR body was measured against the Triton path that
turned out to be inactive on our stack — keeping the file is for forward-compat
(rig that explicitly forces `VLLM_FORCED_FP8_MOE_BACKEND=triton` AND patches
the SM 8.x QuantKey reject) and as the Genesis-side home for any other
community-contributed MoE configs we find useful.

For the path actually exercised on Ampere FP8 PROD see Genesis P17/P18/PN64
in [`kernels/marlin_tuning.py`](../../kernels/marlin_tuning.py) — that's where
real A5000/A6000 uplift comes from.

---

## Cross-rig contributions wanted (2026-05-05)

We currently bundle ONE config (A5000 + 35B-A3B-FP8). Coverage gap is large.
Estimated uplift from a tuned config on a card class that currently autotunes
inline: **2-20%** depending on batch shape.

### High-value targets (community-tunable, no hardware here)

⚠ **Read the backend table above first.** Triton MoE configs only matter on
Ada/Hopper/Blackwell where vLLM picks the Triton FP8 MoE backend. On Ampere
consumer (3090, A5000), FP8 MoE goes through Marlin — see Genesis P17/P18 +
PN64 in `vllm/_genesis/kernels/marlin_tuning.py` instead.

| GPU | SM | Backend on FP8 MoE | Target shape | Status |
|---|---:|:-:|---|---|
| **RTX 4090** (Ada consumer) | 8.9 | TRITON (native FP8) | E=256, N=512, fp8_w8a8, block_shape=[128,128] | **HIGH-VALUE — needed** |
| **RTX 5090** (Blackwell consumer) | 12.0 | TRITON (native FP8) | E=256, N=512, fp8_w8a8, block_shape=[128,128] | **HIGH-VALUE — needed** (apnar club-3090#51) |
| **RTX 4070 Ti SUPER** | 8.9 | TRITON | E=256, N=512, fp8_w8a8, block_shape=[128,128] | nice-to-have (16 GB) |
| **H100 / H200** | 9.0 | TRITON (upstream-bundled) | E=256, N=512, fp8_w8a8, block_shape=[128,128] | upstream covers — verify |
| **RTX 3090** (Ampere consumer) | 8.6 | **MARLIN (not Triton!)** | n/a — see `kernels/marlin_tuning.py` | tune Marlin instead |
| **RTX A5000 / A6000** (Ampere) | 8.6 | **MARLIN (not Triton!)** | n/a — see `kernels/marlin_tuning.py` | tune Marlin instead |

For Qwen3-Next-80B style MoE (E=128, N=2048) — different shape entirely;
generate separately when that model lands in PROD.

### Marlin MoE tuning on Ampere (the actually-relevant path for our PROD)

Genesis tunes Marlin MoE per-arch via dispatcher patches:
- **P17 / P18** — Marlin MoE per-SM tuning (per `kernels/marlin_tuning.py`)
- **P87** — Marlin W4A16/W8A16 sub-tile output dim pad-on-load (vllm#40361)
- **P91** — AutoRound row-parallel cdiv (vllm#39460)
- **P95** — Marlin TP cudagraph cap on Ampere (vllm#40385)
- **PN64** — Marlin MoE SM 12.0 placeholder for Blackwell consumer (env-gated)

These work on Ampere SM 8.x for FP8 MoE. The Triton JSONs in this directory
are orthogonal — same model, different kernel backend.

### Tuning workflow

Use the Genesis-bundled wrapper:

```bash
# On the target rig (cross-rig collaborator runs this on their box):
GPU_OVERRIDE=NVIDIA_GeForce_RTX_3090 ./scripts/moe_lookup_helper.sh
```

The wrapper detects the GPU, runs vLLM's autotuner inside the container,
and writes the resulting JSON to `vllm/_genesis/configs/moe_tuning/`.
Then PR the file back to Genesis (or attach to a club-3090 issue) — we
bundle it into the next release.

### What to expect (per-card uplift heuristic)

| Card class | Expected uplift | Why |
|---|---|---|
| Ampere consumer (3090, A5000) | +10-20% on small batch (1-8) | SM 86 SMEM tight; autotuner default picks suboptimal BLOCK_SIZE_K |
| Ada consumer (4070-4090) | +5-15% | More SMEM than Ampere; autotuner default closer to optimal |
| Blackwell consumer (5080/5090) | +2-10% | Massive SMEM headroom; autotuner default usually within 10% |
| Datacenter (H100, R6000) | +0-5% | Autotuner default already well-tuned by upstream |

So consumer Ampere cards are the highest-ROI targets for cross-rig contribution.
