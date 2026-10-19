# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Self-Maintenance Instructions

**You MUST keep this file and your memory (`MEMORY.md`) up to date as you work.** Do not wait for the user to ask:
- After completing a significant task (optimization, config change, benchmark), update the relevant sections of CLAUDE.md and MEMORY.md immediately
- Update performance numbers when you measure new benchmarks
- Record what you tested and rejected (with reasons) so you don't repeat failed experiments
- Update the "Current State" section at the bottom when the deployment changes
- Before ending a session, verify both files reflect the current state of the project

## Project Overview

This is a **vLLM fork** with a custom `myia_vllm/` directory for self-hosting LLMs on **3x RTX 4090 GPUs** (72GB total VRAM). The project provides OpenAI-compatible API endpoints for LLMs, accessible via reverse proxy at `*.text-generation-webui.myia.io`.

**Current deployment (2026-10-05)**: **Swift-1.5-Qwen3.8-27B W4A16-AWQ + FP8 KV on STOCK vLLM `v0.31.0`** — GPUs 0,1, TP=2, image `vllm/vllm-openai:v0.31.0` (a plain `docker pull`, **no patch tree**; bumped from v0.30.0 on 2026-10-05, PR #72), profile `medium-swift15-27b.yml`. **Model source `ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ` (compressed-tensors AWQ, pack-quantized W4A16; vision blocks + `lm_head` unquantized), pinned by `--revision 9dba8a05877150d587215a519ce6befec3c9978a` since 2026-10-08 (PR #88) — the cached `refs/main` was verified byte-equal to the repo's HEAD at pin time, so the pin is inert and costs no re-download.** gpu-util 0.70, **batch 8192 + max-num-seqs 48 (both raised 2026-10-08 evening, PR #104, user green light; fresh-vs-fresh A/B: N=32 718 t/s = +35%, N=48 907 t/s stable, prefill 157K 1595-1608 vs 1316 t/s = +21%, KV −5.9%/−1.3% to 424,610, VRAM 19,937/18,930 MiB; the 16K-chunks rejection (−17% KV) does NOT extend to 8192; freshness confound eliminated by a 4096-fresh counter-test — old engines measured ~8% above fresh ones that night, see `myia_vllm/_iteration_log/swift15_batch8192_seqs48/NOTES.md`)**, 262K context, vision + thinking + `preserve_thinking`, **dual served names `qwen3.6-35b-a3b` (legacy alias, consumers unchanged) + `swift-1.5-27b` (honest name)**. **KV offload RAM tier active at 24 GiB, ALWAYS UNPINNED** (`--kv-offloading-size 24 --kv-offloading-backend native` + `VLLM_WSL2_ENABLE_PIN_MEMORY=0`; issue #65): pin=1 is FALSIFIED on this host — `cudaHostRegister` of the /dev/shm offload region crashes under WSL2 chunked or not (v0.30.0 monolithic at `is_padding.fill_`, v0.31.0+#51081 chunked at `copy_event.synchronize`; the engine then walks dead: `/health` 200 while workers are gone — **only a real generation proves liveness**). Load path CONFIRMED (bench v3, 05/10): gold 17.4K tok evicted from GPU → replay **0.32 s vs 16.3 s cold (51×)** via **655 MB CPU→GPU load**, answer correct (hybrid GDN correctness end-to-end). **Sizing rule: tier ≥ 1.5× GPU KV** — a 12 GiB tier (< GPU cache) is a write-only sink (capacity LRU evicts every prefix before replay); 24 GiB = ~685K tok at 37.6 KB/tok = 1.48×. RAM cost: 25.77 GB tmpfs, pages committed on touch. `kv_offload_cpu_cache_usage_perc` is an in-flight gauge, NOT fill. **GPU 2 free** for CoursIA training jobs. The profile runs an **entrypoint HF-cache phantom-mount guard** (2026-08-12) that refuses to start if the WSL HF bind resolved empty. **Replaced the Qwen3.6-35B-A3B MoE on 2026-09-25** (user pivot to the quality tier — the 27B line is where the ecosystem iterates: Swift-1.5 finetunes, Qwen 4 27B announced at Apsara as the permanent open-weight local tier; the 35B-A3B MoE has no announced successor; window-based access was ruled out). Known trade: prefill on the dense 27B is ~3-4× slower than the MoE (1,812 t/s @101K vs 5,448-8,316), decode ~0.5-0.6×, and **any request ≥~100K tokens starves new arrivals for 130-170 s** (watchdog GEN_TIMEOUT now 90 s, PR #61). 16K chunks tested and rejected (−17% KV, no wedge relief). Full record: `myia_vllm/_iteration_log/swift15_promotion/NOTES.md`. **Rollback (armed)**: `docker compose -f medium-swift15-27b.yml down && docker compose -f medium-qwen36-stock-tq.yml --env-file myia_vllm/.env up -d` (MoE + TQ k8v4, KV 934,660, compile-cache volume kept).

**MTP-3 on Swift-1.5-27B is BLOCKED by an upstream loader bug** (v0.30.0): `ValueError: There is no module or parameter named 'fc.weight' in Qwen3_5MultiTokenPredictor` — the checkpoint ships the MTP head BF16-dense but vLLM builds the draft Linears from the target's pack-quantized quant config (the ignore list omits `mtp.*`). Cousin of #51581. **Reported upstream as [vllm#58807](https://github.com/vllm-project/vllm/issues/58807) (2026-09-25, ours); maintainer he-yufeng took it 09-26 (fix in `qwen3_5_mtp.py`, safetensors-index detection); HyperQwen-datapoint posted 09-29** — re-test at the fix PR's merge. No spec-dec until fixed (and never TurboQuant KV with any spec-dec anyway, #53180).

**Genesis was retired 2026-08-11** after phase 2 established that the crash which forced its adoption ([vllm#41726](https://github.com/vllm-project/vllm/issues/41726), `AssertionError turboquant_attn.py:720:_continuation_prefill`) no longer reproduces on stock: a **253,503-token chunked prefill** on the real MoE model passed in 58.6 s with a clean survival request, alongside 13/13 functional gates. Upstream fixes #44053/#47609/#39988/#50533 are verified git ancestors of the release. Three further reasons: the image becomes **reproducible** (a release tag is not subject to the ~5-day nightly GC that made `vllm-qwen36-genesis-tq:v7.72.5-vllm01d4d1ad3` unbuildable); a same-night A/B showed stock **faster** than Genesis (+14% at N=16, +29% single-stream); and GPU 0 gains ~1.8 GiB of headroom (Genesis prealloc pools P28/P37/TQ-dequant gone), the direct lever against the boot-OOM history. Cost accepted: −17% KV (1,238,046 → 1,030,407), immaterial at 2–7% observed occupancy. Full record: `myia_vllm/_iteration_log/stock_tq_phase2/NOTES.md`.

**v0.29.0 attempt (2026-09-09, REJECTED on WSL2)**: final image `vllm/vllm-openai:v0.29.0` (digest `aa542864…`, release commit `98dff2a81…`) selects **Model Runner V2 by default**. Both TP workers fail before model loading at `UvaBuffer` with `RuntimeError: UVA is not available`; this is the WSL2 UVA boundary previously found with DSpark, now on the normal production path. No OOM and no TurboQuant runtime was reached. The supported `VLLM_USE_V2_MODEL_RUNNER=0` override exists in the release image but was deliberately **not tested in production** because it adds a fourth configuration change beyond the pre-authorized image-only bump; evaluate it only in a dedicated maintenance window. Rolled back to v0.28.0: health/decode 200, RC=0, KV 934,374, 3/3 containers running, profile diff empty. Record: `myia_vllm/_iteration_log/v0290_bump/NOTES.md`. **RESOLVED 2026-09-11: the boundary was a default-off gate, not an environmental limit — see the promotion section in Current State above; v0.29.0 is now in production on this same digest.**

**v0.27.1 → v0.28.0 (2026-09-01)**: release tag of 2026-08-26 (584 commits, Transformers 5.15, Ubuntu 24.04 runtime). Relevant: prefix caching default-on for Mamba/GDN (#50991), KV-cache layout refactor touching `turboquant_attn.py` (#51704/#51718 — hence the full 250K re-gate), Qwen3.5 GDN MTP kernels, DFlash2 + DSpark spec-dec. 13/13 gates re-passed, prefill 5,448–8,316 tok/s, same-night A/B vs v0.27.1: N=16 **828 vs 813 tok/s**, no regression at any N. Boot ~4.5 min on a cold per-image compile-cache volume (`vllm-compile-cache-qwen36-stock-tq-v0280`; the v0.27.1 volume is kept for rollback). KV 1,038,194 (+0.8%) at batch 4096; the current batch 8192 uses 934,374. VRAM after gates GPU 0 20,006 / GPU 1 19,002 MiB. Same-day investigation: our TQ×MTP issue #53180 is a duplicate of #52475 with fix PRs #53406/#53410 open unreviewed; a DSpark drafter exists for our target but is independently blocked by the same WSL2 UVA boundary. Record: `myia_vllm/_iteration_log/v0280_bump/NOTES.md`.

**v0.27.0 → v0.27.1 (2026-08-14)**: micro-patch (5 commits: 3 CI, 1 docs, 1 additive feature for quantized DSpark Markov heads — touches neither TurboQuant nor the Qwen3.6 MoE path). 13/13 gates re-passed; decode at N=16 measured **956 tok/s** aggregate (vs 524.6 on v0.27.0 the same machine on 08-11), prefill 5,346–8,177 tok/s — the machine appears to have recovered throughput since the 08-11 trough, but treat as a same-day measurement, not a fixed baseline. Fork synced to upstream/main same day (merge `75b227d85f`, 0 commits behind, `starlette >= 1.0.1` CVE-2026-48710 Host-injection fix pulled in via `requirements/common.txt`).

**⚠️ The May performance figures in this file are no longer reproducible on this machine.** The same-night A/B put Genesis at 461 tok/s aggregate at N=16 and stock at 525, against **829** documented in May — *both* stacks ~45% down, so the loss is **the machine, not the software**. Cause not established (GPU clocks / P8 states, driver, Windows power plan, desktop load on GPU 0); tracked as a separate investigation. Until it is resolved, treat every perf number dated May 2026 as historical, and compare only against a same-night A/B.

**Rollback (Genesis, still armed)**: `docker compose -f medium-qwen36-stock-tq.yml down && docker compose -f medium-qwen36-genesis-tq.yml up -d`. The Genesis image tars in `D:\vllm_image_backups\` remain production assets until the stock soak concludes — do not delete them.

**Genesis promotion path (2026-05-16 → 2026-05-17, historical)**:
1. v2f (Genesis-TQ + vision) PASSED active load benches but regressed on +55min idle soak (shm_broadcast.py:733 deadlock). Rolled back to FP8 baseline 05:35Z.
2. v2g = v2f + `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer sampler had its own JIT autotune path corrupting CPython `_thread.lock` descriptor under load). Deployed 2026-05-16 06:29Z.
3. v2g soak 35h+ clean — promoted as new baseline 2026-05-17.

**Why this path beats the 2026-05-06 attempt**: 2026-05-06 used `vllm-qwen36-tq:nightly-e47c98ef-patched1` (stock nightly + transformers/shm patches), which crashed on first ~30K chunked-prefill continuation (`AssertionError turboquant_attn.py:720`). Genesis patch tree includes P22+P38 which fix exactly that crash (externally confirmed in [vllm#41726](https://github.com/vllm-project/vllm/issues/41726) by `xyehya`). Upstream candidate fix PR [#40798](https://github.com/vllm-project/vllm/pull/40798) remains OPEN+BLOCKED, so Genesis was the actionable Option B.

**Prior baseline** (`vllm-qwen36-shmpatched:nightly-f6983f01d-patched1` Apr 06 + shm patch + FP8 KV) and **TQ image** (`vllm-qwen36-tq:nightly-e47c98ef-patched1`) retained on disk for instant rollback. Rollback command: `docker compose -f medium-qwen36-genesis-tq.yml down && docker compose -f medium-qwen36-moe.yml up -d`.

## Key Directories

```
myia_vllm/                    # PRIMARY - all customizations live here
├── configs/docker/           # Docker configs
│   ├── profiles/             # Docker-compose deployment profiles
│   └── Dockerfile.glm-flash  # Custom image for GLM-4.7-Flash
├── scripts/                  # PowerShell & Python deployment scripts
│   ├── quantization/         # Model quantization (W4A16, FP8)
│   └── testing/              # Config validation & benchmarks
├── docs/                     # Documentation (missions, guides)
├── qwen3_benchmark/          # Benchmarking framework
├── archives/                 # Archived legacy configurations
└── entrypoints/              # Custom tool parsers
```

## Common Commands

### Deployment

```powershell
# Start Swift-1.5-Qwen3.8-27B + FP8 KV (primary model, GPUs 0,1, prod since 2026-09-25)
docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b.yml --env-file myia_vllm/.env up -d
docker logs -f myia_vllm-medium-swift15-27b

# Rollback to the MoE (armed): swap profiles, same port/keys/alias
docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b.yml --env-file myia_vllm/.env down
docker compose -f myia_vllm/configs/docker/profiles/medium-qwen36-stock-tq.yml --env-file myia_vllm/.env up -d

# Legacy: Qwen3.6-35B-A3B MoE + FP8 KV (pre-08-11 baseline, superseded)
docker compose -f myia_vllm/configs/docker/profiles/medium-qwen36-moe.yml --env-file myia_vllm/.env up -d
docker logs -f myia_vllm-medium-qwen36-moe

# To re-test TurboQuant once upstream PR #40798 merges:
# 1. Edit medium-qwen36-moe.yml: image → vllm-qwen36-tq:nightly-e47c98ef-patched1
#    and --kv-cache-dtype fp8 → --kv-cache-dtype turboquant_k8v4
# 2. docker compose down + up -d
# (Image already local; rolled back 2026-05-06 due to upstream issue #41726)

# Legacy: GLM-4.7-Flash (archived 2026-04-24, requires custom image build first)
# docker compose -f myia_vllm/archives/2026/profiles_legacy/medium-glm.yml build
# docker compose -f myia_vllm/archives/2026/profiles_legacy/medium-glm.yml --env-file myia_vllm/.env up -d
```

### Testing

```powershell
# Benchmark GLM-4.7-Flash (A/B benchmark tool, supports any backend name)
python myia_vllm/scripts/testing/benchmark_llamacpp_vs_vllm.py --backend vllm
python myia_vllm/scripts/testing/benchmark_llamacpp_vs_vllm.py --compare

# Benchmark Qwen3.5 MoE
python myia_vllm/scripts/testing/benchmark_coder_next.py --model qwen3.5-35b-a3b

# Run all tests
.\myia_vllm\run_all_tests.ps1

# Quick API test
python myia_vllm/scripts/python/tests/test_qwen3_tool_calling.py
```

### Grid Search Optimization

```powershell
# Run configuration optimization
.\myia_vllm\grid_search_optimization.ps1
```

## Architecture

### Docker Deployment Pattern

Qwen3.5-35B-A3B uses the official `vllm/vllm-openai:nightly` image directly (no custom Dockerfile needed).

GLM-4.7-Flash (legacy) used a custom Dockerfile (`Dockerfile.glm-flash`) with `transformers >= 5.0` for `glm4_moe_lite` architecture support.

### GPU Assignment

| Service | GPUs | Port | Model | Profile |
|---------|------|------|-------|---------|
| **medium-swift15-27b** | **0,1** | **5002** | **Swift-1.5-Qwen3.8-27B W4A16 + FP8 KV** | **medium-swift15-27b.yml** |
| **embeddings-qwen3-4b-awq** (fleet embeddings, other machine) | **po-2023 GPU 0** (3080 Ti, pinned) | 8005 (public `embeddings.myia.io` via sidecar :8004) | Qwen3-Embedding-4B W4A16-AWQ, v0.23.0 pooling runner | embeddings-qwen3-4b-awq-po2026.yml (deployed on po-2023 since 2026-10-14 after po-2026 HW failure; runbook `myia_vllm/docs/embeddings-po2026-runbook.md`, seat doc `docs/seats/po2023-embeddings.md`) |
| medium-qwen36-moe | 0,1 | 5002 | Qwen3.6-35B-A3B-AWQ + FP8 KV | medium-qwen36-moe.yml (legacy baseline) |
| medium-qwen36-stock-tq | 0,1 | 5002 | Qwen3.6-35B-A3B + TurboQuant k8v4 | medium-qwen36-stock-tq.yml (**armed rollback**) |
| **GPU 2 — FULLY FREED 2026-05-01** | **2** | — | training jobs (CoursIA) | 78 MiB driver baseline only |
| medium-qwen36-27b | 0,1 | 5002 | Qwen3.6-27B-AWQ-INT4 + TQ K8V4 | archives/2026/medium-qwen36-27b.yml.rejected-2026-05-06 (rejected: -50% decode) |
| kokoro-tts (migrated to po-2023) | — | — | Kokoro TTS (67 voices) | now at `https://tts.myia.io/kokoro/v1` (po-2023, sleep mode) |
| mini-omnicoder | 2 | 5001 | OmniCoder-9B-AWQ-4bit | archived 2026-04-30 (GPU 2 freed for trainings) |
| medium-qwen35-moe | 0,1 | 5002 | Qwen3.5-35B-A3B-AWQ | archived 2026-04-17 (replaced by 3.6) |
| mini-zwz | 2 | 5001 | ZwZ-8B-AWQ-4bit | mini-zwz.yml (replaced by OmniCoder) |
| medium-glm | 0,1 | 5002 | GLM-4.7-Flash-AWQ | archives/2026/profiles_legacy/medium-glm.yml |
| medium-qwen35-dense | 0,1 | 5002 | Qwen3.5-27B-AWQ | medium-qwen35-dense.yml (rejected: too slow) |
| mini-solo | 2 | 5001 | Qwen3-VL-8B-Thinking-AWQ | mini-solo.yml (fallback) |

GPUs 0,1 are on faster PCIe bus. **GPU 2 fully freed 2026-05-01** (78 MiB driver baseline) for CoursIA training jobs (queue 5-7d, sprint sustainable + Sudoku large + RL extension). Previously: OmniCoder-9B + Kokoro TTS. OmniCoder archived 2026-04-30, Kokoro TTS migrated to po-2023 (`https://tts.myia.io/kokoro/v1`, sleep mode) on 2026-05-01 — 7/7 OWUI tenants switched, ai-01 compose service `kokoro-tts` removed.

### Environment Variables

Configuration via `myia_vllm/.env` (not tracked) based on `.env.example`:
- `HUGGING_FACE_HUB_TOKEN` - Required for model downloads
- `VLLM_API_KEY_*` - API keys per service
- `VLLM_MODEL_QWEN36_MOE` - Qwen3.6 MoE model (default: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`)
- `VLLM_PORT_GLM` - Medium model port (default: 5002, shared by all medium profiles)
- `VLLM_MODEL_ZWZ` - ZwZ-8B model path (default: `./models/ZwZ-8B-AWQ-4bit`)

## OmniCoder-9B Deployment (Current GPU 2 Model)

### Model Overview

**OmniCoder-9B** ([Tesslate/OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B)) is the **current model** on GPU 2 (port 5001) since 2026-03-28. A Qwen3.5-9B finetune specialized for agentic coding. Key features:
- **Architecture**: Qwen3.5-9B dense, hybrid Gated Delta Networks + standard attention
- **Training**: 425K+ agentic coding trajectories (Claude Opus, GPT-5.3, Gemini 3.1 Pro)
- **Thinking mode**: `<think>...</think>` reasoning chains
- **Vision**: Images, documents (vision encoder preserved in BF16)
- **Tool calling**: `qwen3_coder` parser (XML format: `<function=name><parameter=key>value</parameter>`)
- **Context**: 262K native, configured at 131072 (VRAM constraint)
- **License**: Apache 2.0
- **Quantization**: AWQ 4-bit from [cyankiwi/OmniCoder-9B-AWQ-4bit](https://huggingface.co/cyankiwi/OmniCoder-9B-AWQ-4bit)

### Custom Dockerfile Required

Qwen3.5 dense models use `model_type: qwen3_5` and `TokenizersBackend`, which require `transformers >= 5.0`. vLLM nightly ships `transformers 4.57.6`. Custom `Dockerfile.omnicoder` adds the layer:
```dockerfile
FROM vllm/vllm-openai:nightly
RUN pip install --no-cache-dir "transformers>=5.0" "tokenizers>=0.21" "huggingface_hub>=0.30"
```

### Key vLLM Flags
```yaml
--model cyankiwi/OmniCoder-9B-AWQ-4bit
--served-model-name omnicoder-9b
--gpu-memory-utilization 0.85          # 0.88 OOM with Kokoro TTS on same GPU
--max-model-len 131072
--kv-cache-dtype fp8
--dtype auto                           # MUST be auto for BF16 vision encoder
--tool-call-parser qwen3_coder         # XML format, NOT hermes
--reasoning-parser qwen3
--trust-remote-code
--enable-prefix-caching
--skip-mm-profiling
```

### Performance (Benchmark 2026-03-28, fresh compile cache)

| Metric | OmniCoder-9B | ZwZ-8B (previous) |
|--------|:---:|:---:|
| Decode tok/s | **96-107** | 90-118 |
| Vision tok/s | **90-105** | 90-118 |
| Concurrent 5 text | **293 tok/s agg** | N/A |
| Tool call latency | **1.09s** | N/A (hermes) |
| Thinking mode | Yes | No |

**CRITICAL**: torch.compile cache corruption causes 10-15x slowdown (7-10 tok/s instead of 90-107). Fix: `docker volume rm profiles_vllm-compile-cache-omnicoder` and restart. Fresh compile takes ~150s.

### Quality Benchmarks (2026-03-28, vs ZwZ-8B)

| Benchmark | OmniCoder-9B | ZwZ-8B | Delta |
|-----------|:---:|:---:|:---:|
| **MME Total** | **1258.5** | 1248.1 | **+10.4** |
| MME Perception | 907.8 | 889.5 | +18.3 |
| MME Cognition | 350.7 | 358.6 | -7.9 |
| **MMStar** | 58.5% | **63.0%** | -4.5 pts |
| OCR | **97.5%** | 82.5% | **+15 pts** |
| code_reasoning | 87.5% | **95.0%** | -7.5 pts |

### Deployment (ARCHIVED 2026-04-30 — GPU 2 freed for CoursIA trainings)

OmniCoder profile + Dockerfile archived to `myia_vllm/archives/2026/mini-omnicoder.yml.archived-2026-04-30` and `Dockerfile.omnicoder.archived-2026-04-30`. To restore (NOT recommended without coordinating with CoursIA training queue):

```powershell
# Restore profile from archive first
git mv myia_vllm/archives/2026/mini-omnicoder.yml.archived-2026-04-30 myia_vllm/configs/docker/profiles/mini-omnicoder.yml
git mv myia_vllm/archives/2026/Dockerfile.omnicoder.archived-2026-04-30 myia_vllm/configs/docker/Dockerfile.omnicoder
docker compose -f myia_vllm/configs/docker/profiles/mini-omnicoder.yml --env-file myia_vllm/.env up -d
docker logs -f myia_vllm-mini-omnicoder
```

## ZwZ-8B (Replaced by OmniCoder-9B)

### Model Overview

**ZwZ-8B** ([inclusionAI/ZwZ-8B](https://huggingface.co/inclusionAI/ZwZ-8B)) was the vision model on GPU 2 (port 5001) from 2026-02-16 to 2026-03-28. Replaced by OmniCoder-9B. A Qwen3-VL-8B-Instruct finetune specialized for fine-grained visual perception. Key features:
- **Single-pass inference**: No iterative zooming like "Thinking-with-Images" methods
- **Region-to-Image Distillation**: Trained with Qwen3-VL-235B and GLM-4.5V as teachers
- **Training data**: 74K VQA samples from inclusionAI/ZwZ-RL-VQA
- **License**: Apache 2.0

### Differences from Qwen3-VL-8B-Thinking

| Feature | Qwen3-VL-8B-Thinking | ZwZ-8B |
|---------|---------------------|--------|
| Reasoning mode | ✅ deepseek_r1 parser | ❌ None |
| Fine-grained vision | Standard | Optimized |
| Tool calling | ✅ hermes | ✅ hermes |
| Single-pass inference | Yes | Yes |

### Quantization (Required)

ZwZ-8B is only available in BF16 (~17GB). Must create AWQ 4-bit for deployment:

```bash
# Create llmcompressor environment
conda create -n llmcompressor python=3.11 -y
conda activate llmcompressor
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install "llmcompressor>=0.9.0" "transformers>=4.48.0" accelerate datasets

# Run quantization (30-60 min)
python myia_vllm/scripts/quantization/quantize_zwz_8b.py \
  --model-id inclusionAI/ZwZ-8B \
  --output-dir ./models/ZwZ-8B-AWQ-4bit \
  --num-samples 512
```

**Critical**: Vision encoder (ViT) is excluded from quantization - kept in BF16 to preserve visual accuracy.

### Deployment

```powershell
# Switch from mini-solo (Qwen3-VL-Thinking) to ZwZ
docker compose -f myia_vllm/configs/docker/profiles/mini-solo.yml down
docker compose -f myia_vllm/configs/docker/profiles/mini-zwz.yml --env-file myia_vllm/.env up -d

# Monitor startup
docker logs -f myia_vllm-mini-zwz
```

### Key vLLM Flags for ZwZ-8B
```yaml
--model ./models/ZwZ-8B-AWQ-4bit
--served-model-name zwz-8b
--gpu-memory-utilization 0.88
--max-model-len 131072
--kv-cache-dtype fp8
--tool-call-parser hermes
# NO --reasoning-parser (ZwZ has no thinking mode)
```

## Critical Configuration Notes

1. **Do NOT add `--enable-chunked-prefill` or `--num-scheduler-steps`** - These flags force V0 engine fallback. V1 engine (default on nightly) handles chunked prefill automatically.

2. **CUDA graphs (PIECEWISE mode) work well at 0.85 gpu-memory-utilization** - Piecewise CUDA graphs have minimal overhead. **gpu-memory-utilization reduced 0.92→0.88→0.85**: Marlin MoE `fused_marlin_moe.py` needs 852-994 MiB variable temporary allocations (`intermediate_cache13`). At 0.85: 335K tokens KV cache (still >262K max-model-len), ~2.3 GiB headroom. Bug tracked in vLLM RFC [#27951](https://github.com/vllm-project/vllm/issues/27951) — no fix as of Feb 27 2026. **Do NOT use `--enforce-eager`** (tested: 3-4x slower on all metrics - 12 tok/s vs 45 tok/s decode).

3. **MLA backends on RTX 4090 don't support FP8 KV cache** - Use `--kv-cache-dtype auto` (not fp8). TRITON_MLA is the only working MLA backend on Ada Lovelace (SM89).

4. **MTP speculative decoding doesn't work with AWQ 4-bit** - 0% acceptance rate at 4-bit precision (tested on GLM-4.6-AWQ). MTP heads are part of the target model and get AWQ-quantized along with it; at 4-bit the auxiliary heads have insufficient capacity → noise predictions. **Only viable with FP8 or FP16 models.**

   **Correction (2026-09-23): the served checkpoint does carry an MTP layer.** Snapshot `00fcea2d` (23/07) of `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` has `mtp_num_hidden_layers: 1` and 2,321 `mtp.*` tensors. Attention and the shared expert are BF16; the 256 routed experts are AWQ INT4. The May note that the heads were stripped is obsolete. Acceptance on this checkpoint is **unmeasured**. The isolated MTP k=3 spike (roo-extensions #2433) was closed on 23/09 under the user's target of maximum concurrent access. Every spec-dec method measured on this host lost aggregate throughput under load (DFlash −15 % at 5 users, DSpark −33 % at N=16, MTP acceptance ÷1.6 under load on the 27B). TQ × MTP also degenerates silently (#53180), so MTP would force fp8 KV.

   **DSpark speculative decoding is NOT viable on this host (v0.28.0 / WSL2, verified 2026-09-02)**: the drafter for our exact target (`RedHatAI/Qwen3.6-35B-A3B-speculator.dspark`) loads, but DSpark selects the V2 model runner, which requires UVA (unified virtual addressing) — unavailable under WSL2 GPU passthrough (`RuntimeError: UVA is not available` at `buffer_utils.py:47`; same family as `pin_memory=False as WSL is detected`). Prod (TQ, batch 8192) runs the V1 runner and is unaffected. This is a WSL2-boundary environmental limit, not a vLLM bug; the V2-runner selection itself (what triggers it, whether a V1 override exists) is a separate open question. DSpark is not a candidate on this machine until that is resolved. **DFlash speculative decoding is architecturally different and DOES work with AWQ** — drafter is a separate 0.5B BF16 model with its own quantization config (vLLM `get_draft_quant_config()`), independent of target's quantization. Block diffusion approach generates K candidates per step. **Locally validated 2026-04-24**: 26-47% acceptance rate per-position on Qwen3.6-35B-A3B-AWQ-4bit target (4-7 tokens accepted per draft step). Single-user decode +23 to +94% but loses 5-user concurrent throughput (-15%) and context capacity (262K → 160K, since flash_attn rejects fp8 KV cache). See "Current State" DFlash bullet for full bench + decision.

5. **Use `qwen3_coder` tool parser for Qwen3.5** and `qwen3` reasoning parser. Legacy: `glm47`/`glm45` for GLM-4.7-Flash.

6. **Use `--dtype auto` for Qwen3.5 models** - `--dtype half` causes dtype mismatch with BF16 vision encoder. `auto` resolves correctly.

7. **Credentials in `.env` were compromised** - Regenerate HuggingFace token and API keys before production deployment.

## TurboQuant K8V4 Migration Attempt (2026-05-06, REJECTED)

### Why we tried
PR [#39931](https://github.com/vllm-project/vllm/pull/39931) "TurboQuant: support hybrid models and uniform quantization" merged upstream 2026-05-05 00:14 UTC (commit `4f2af1a7c`). This unblocks `--kv-cache-dtype turboquant_k8v4` for hybrid GDN+attention models — Qwen3.6 family included.

### What we tested

**Attempt 1: Qwen3.6-27B Dense + TurboQuant K8V4** — Booted cleanly. Bench tripped all 3 of the migration plan's "consider rollback" thresholds:

| Metric | 27B Dense + TurboQuant | MoE 35B-A3B (baseline) | Δ |
|---|:---:|:---:|:---:|
| Decode tok/s (single, no thinking) | 52-54 | 107 | **-50%** |
| Decode tok/s (thinking) | 50.5 | 116.5 | **-57%** |
| 5 concurrent (aggregate) | 189 | 369 | **-49%** |
| Tool call latency | 0.66s | 0.47s | +40% |
| KV cache size | 516K tokens | 322K | +60% |
| Repetition 4-gram | 0.044 | 0.028 | **+59% worse** |
| GSM8K (300 samples) | 91.7% | 87.6% (full 1319) | +4.1pts (CIs overlap) |
| IFEval (300 samples) | 91.0% | 87.6% (full 541) | +3.4pts (CIs overlap) |

Quality gains within sampling noise on the benchmarks we ran locally. Profile archived to `myia_vllm/archives/2026/medium-qwen36-27b.yml.rejected-2026-05-06`.

**Attempt 2: Qwen3.6-35B-A3B MoE + TurboQuant K8V4** (same image, switch model + add `--enable-expert-parallel`) — Booted cleanly with **1,494,999 tokens KV cache (+4.6× vs FP8 baseline 322K)**, hybrid layer detection `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]` (10/40), Marlin MoE active, EP=2 working.

EngineCore crashes on first chunked-prefill continuation request (~30K-token prompt from real OWUI traffic):
```
AssertionError: Workspace is locked but allocation from
'turboquant_attn.py:720:_continuation_prefill' requires 29.73 MB,
current size is 16.31 MB. Workspace growth is not allowed after locking.
```

Already filed upstream as [vllm#41726](https://github.com/vllm-project/vllm/issues/41726) (filed 2026-05-05 by `jhsmith409`). Bug is **not hybrid-specific** and **predates #39931** — bisected to pre-merge nightly on plain dense Qwen3-4B and Llama-3.1-8B. Candidate fix: PR [#40798](https://github.com/vllm-project/vllm/pull/40798) "Share decode scratch workspace across layers" (open). [Our reproduction comment](https://github.com/vllm-project/vllm/issues/41726#issuecomment-4389387531) confirms the bug persists post-#39931 merge on hybrid MoE + Ada hardware.

### Rollback completed (2026-05-06)
Restored MoE + FP8 KV with proven baseline image `vllm-qwen36-shmpatched:nightly-f6983f01d-patched1` (Apr 06 nightly, stable since 2026-04-19). The TQ image (`vllm-qwen36-tq:nightly-e47c98ef-patched1`) and Dockerfile (`Dockerfile.qwen36-tq`) **retained** for re-test once #40798 merges.

To re-test TurboQuant later:
1. Edit [medium-qwen36-moe.yml](myia_vllm/configs/docker/profiles/medium-qwen36-moe.yml) line 36 → `image: vllm-qwen36-tq:nightly-e47c98ef-patched1`
2. Edit line 51 → `--kv-cache-dtype turboquant_k8v4`
3. `docker compose down && docker compose up -d`

## Qwen3.6-35B-A3B Deployment (Current — production since 2026-04-17, FP8 KV)

### Model Specifications
- **Architecture**: 35B MoE with 3B active parameters per token (256 experts, 9 active: 8 routed + 1 shared, 40 layers)
- **Attention**: Hybrid GatedDeltaNet (30 layers, linear fixed state) + Gated Attention (10 layers, standard KV cache)
- **Vision**: Images, videos, documents (vision encoder preserved in BF16)
- **Thinking**: `<think>...</think>` modulation via `chat_template_kwargs`
- **NEW in 3.6**: `preserve_thinking: True` retains reasoning across multi-turn conversations — **ENABLED BY DEFAULT SERVER-SIDE** via `--default-chat-template-kwargs`
- **VRAM**: ~11 GiB per GPU with AWQ 4-bit + TP=2
- **KV cache**: ~322K tokens at 0.85 GPU util with FP8 KV cache
- **Context window**: 262K native (YaRN extensible to 1M), configured at 262K (full native)
- **Quantization**: AWQ 4-bit (compressed-tensors/pack-quantized, group_size=32) with Marlin MoE kernels
- **Model source**: [cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)
- **vLLM class**: `Qwen3_5MoeForConditionalGeneration` (same class as 3.5 — HF config reports this)
- **Engine**: V1 (async scheduling, piecewise CUDA graphs, automatic chunked prefill)
- **Image**: `vllm/vllm-openai:nightly-f6983f01de2bf2e92ab468fa735ebac39cddd670` (Apr 06 nightly, v0.19.1.dev45+gf6983f01d — proven stable; Apr 15/16 nightlies have init-time bugs)
- **Stability investigation (2026-04-19)**: 9 crashes in 50h on Apr 06 nightly. **Real root cause**: `SystemError: PyCFunction with class but no METH_METHOD flag` at `shm_broadcast.py:72` in `with _memory_fence_lock:` (a vanilla `threading.Lock`). Same bug as our own [issue #35104](https://github.com/vllm-project/vllm/issues/35104) (filed Feb 2026 for GLM-4.7-Flash idle crashes), but now fires UNDER LOAD on Qwen3.6 — keepalive sidecar not enough. Surface symptom (`shm_broadcast.py:681 No available shared memory broadcast block found in 60 seconds` → 3x → `EngineDeadError`) **looks identical** to the FlashInfer GDN deadlock from #37729 / #36921 / #35465 but the actual stack trace shows a different bug. `memory_fence()` was added by PR #30407 (Dec 2025) and extended by PR #32022 (Jan 2026). **PR #28053 did NOT fix this** (only removed busy-loop in idle reader — memory note correction). **No upstream fix in flight** as of 2026-04-19; design (per-op `threading.Lock` acquire as memory barrier) is fragile under runtime C-extension loads. Mitigations applied: (1) `--gdn-prefill-backend triton` (kept defensively, doesn't fix the crash), (2) **`--no-enable-flashinfer-autotune`** added 2026-04-19 16:05 UTC — hypothesis: FlashInfer JIT autotune dlopens new `.so` mid-runtime → corrupts CPython `_thread.lock` descriptor. Watch 24-48h. NOTE: there is NO env var for autotune — only the CLI flag on KernelConfig (we tried `VLLM_USE_FLASHINFER_AUTOTUNE=0` first, vLLM reported "Unknown environment variable").

### Deployment

```powershell
docker compose -f myia_vllm/configs/docker/profiles/medium-qwen36-moe.yml --env-file myia_vllm/.env up -d
docker logs -f myia_vllm-medium-qwen36-moe
```

### Key vLLM Flags
```yaml
--model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
--served-model-name qwen3.6-35b-a3b
--tensor-parallel-size 2
--enable-expert-parallel          # EP=2: 128/256 experts per GPU
--gdn-prefill-backend triton      # Workaround shm_broadcast deadlock (vLLM #37729, added 2026-04-19 after 8 crashes/48h)
--gpu-memory-utilization 0.85      # 0.92/0.88 OOM: Marlin MoE needs 852-994 MiB variable temp allocs
--max-model-len 262144            # Full native 262K context
--kv-cache-dtype fp8              # FP8 KV: 322K tokens (2x vs auto)
--dtype auto                      # MUST be auto (not half) for vision encoder BF16 compat
--max-num-batched-tokens 32768
--max-num-seqs 64
--enable-prefix-caching
--tool-call-parser qwen3_coder    # Qwen3.6 function calling (XML format)
--reasoning-parser qwen3          # <think>...</think> extraction
--distributed-executor-backend mp
--default-chat-template-kwargs '{"preserve_thinking":true}'  # Server-side default: multi-turn thinking retention
--limit-mm-per-prompt '{"image":4,"video":0}'
--mm-processor-kwargs '{"max_pixels":774000}'
--skip-mm-profiling               # Required: avoids dtype mismatch during profiling
```

### Environment Variables
```yaml
VLLM_MARLIN_USE_ATOMIC_ADD=1      # Optimized Marlin kernel accumulation
VLLM_USE_DEEP_GEMM=0              # Not needed for AWQ
OMP_NUM_THREADS=4                 # CPU parallelism for scheduling
VLLM_USE_FLASHINFER_MOE_FP16=1    # INERT on this deployment — see correction below
```

**Correction (2026-08-11)**: `VLLM_USE_FLASHINFER_MOE_FP16` is read **only** by
`model_executor/layers/fused_moe/oracle/unquantized.py` — it applies to the **unquantized** MoE path.
Our model is AWQ INT4, so this variable has **never had any effect on this deployment**, under
Genesis or stock. The "+110% MoE" figure previously attributed to it came from a different context
and is not reproducible here. Verified by reading the source inside our own Genesis image. Dropped
from the stock profile along with `VLLM_MARLIN_MOE_BLOCK_SIZE_M` and `VLLM_USE_FLASHINFER_MOE_FP8`
(v0.27.0 selects the MoE path via `KernelConfig.moe_backend`, not env vars).

### Thinking Modulation
To disable thinking per-request (clean, direct responses):
```json
{
  "model": "qwen3.6-35b-a3b",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```
**IMPORTANT**: `chat_template_kwargs` must be a **top-level** field in the request body, NOT inside `extra_body`.

**NEW `preserve_thinking` (Qwen3.6)**: Retains `<think>...</think>` blocks from assistant message history across turns — enables iterative reasoning without redo overhead. **Enabled by default server-side** via `--default-chat-template-kwargs '{"preserve_thinking":true}'` so ALL clients benefit automatically. To opt out per-request: `"chat_template_kwargs": {"preserve_thinking": false}`.

**Reasoning field**: With thinking enabled, the parser separates reasoning into the `reasoning` field (NOT `reasoning_content`, which is always `null`). Both streaming and non-streaming work correctly.

### Performance (Benchmark 2026-04-17, FP8 KV, 262K context, Apr 06 nightly)

| Metric | Qwen3.6-35B-A3B | Qwen3.5-35B-A3B (previous) | Improvement |
|--------|:---:|:---:|:---:|
| Decode speed | **107.0 tok/s** | 86.2 tok/s | **+24%** |
| Thinking decode | **116.5 tok/s** | ~96 tok/s | **+21%** |
| Concurrent 5 users | **369.4 tok/s** | 311.2 tok/s | **+19%** |
| Tool calling | **0.47s** | 0.91s | **-48%** |
| KV cache tokens | 322K (0.85) | 335K | -4% |
| Context | 262K | 262K | = |
| Vision | Yes | Yes | = |
| Thinking | Yes + preserve | Yes | + preserve_thinking |

**Upstream quality improvements (Qwen Team blog)**:
- SWE-bench: 70.0% → **73.4%** (+3.4 pts)
- Terminal-Bench: 40.5% → **51.5%** (+11 pts)
- NL2Repo: 20.5% → **29.4%** (+8.9 pts)
- QwenWebBench: 978 → **1397** (+43%)

### Locally Measured Quality (Benchmark 2026-04-18, AWQ 4-bit)

| Benchmark | Qwen3.6-35B-A3B | Qwen3.5-35B-A3B | Δ |
|---|:---:|:---:|:---:|
| **GSM8K** (math, 1319 q) | 87.6% (1155/1319) | 88.0% (1132/1286) | -0.4 pts (3.6 has **0 errors** vs 33) |
| **IFEval** (instruction, 541 q) | 87.6% (474/541) | 88.5% (476/541) | -0.9 pts (3.6 has **0 errors** vs 3) |
| **MMStar** (vision, 1500 q) | **55.7%** (836/1500) | 53.2% (798/1500) | **+2.5 pts** |
| **MME total** (vision, 2374 q) | 1273.6 | 1294.7 | -1.6% (text_translation -15 pts is the main contributor) |
| MME perception | 918.6 | 926.8 | -0.9% |
| MME cognition | 355.0 | 367.9 | -3.5% |
| **Tool calling accuracy** (12 scenarios) | 83.3% (10/12, 2 intelligent refusals) | not measured | new |
| **Repetition 4-gram** (instruct preset) | 0.028 | 0.042 | **-33%** (less repetition) |
| **TTR diversity** (instruct preset) | 0.78 | 0.54 | **+44%** (more diverse) |

**Key findings:**
- Math/instruction quality essentially unchanged (-0.4 / -0.9 pts within statistical noise) but with **0 errors** vs 33+3 for 3.5 → much more reliable
- Vision: MMStar gains 2.5 pts; MME slightly down (-1.6%) driven by `text_translation` regression (95% → 80%, single category, 40 questions only — investigate if multilingual workloads affected)
- Repetition / lexical diversity meaningfully improved (instruct preset)
- Tool calling: model shows judgment (refused to call `get_weather` when temp already given, refused `calculator` for `sqrt(144)`)
- Upstream SWE-bench / Terminal-Bench / NL2Repo gains NOT yet locally verified — those require dedicated agent harnesses

### Comparison with All Previous Deployments

| | **Qwen3.6-35B-A3B** | Qwen3.5-35B-A3B | GLM-4.7-Flash | Qwen3-Coder-Next |
|---|---|---|---|---|
| Single user tok/s | **107** | 86 | 55 | 5-6 |
| Concurrent tok/s | **369** | 270 | 216 | 21.6 |
| Tool call | **0.47s** | 0.91s | 1.44s | N/A |
| GPUs used | 2 | 2 | 2 | 3 |
| Context | **262K** | 262K | 128K | 65K |
| KV cache | 322K | 335K | 222K | ~65K |
| SWE-bench (upstream) | **73.4%** | 69.2% | 59.2% | 70.6% |
| GSM8K (local AWQ) | 87.6% | 88.0% | n/a | n/a |
| IFEval (local AWQ) | 87.6% | 88.5% | n/a | n/a |
| MME total (local AWQ) | 1273.6 | 1294.7 | n/a | n/a |
| MMStar (local AWQ) | **55.7%** | 53.2% | n/a | n/a |
| Vision | Yes | Yes | No | No |
| preserve_thinking | **Yes** | No | No | No |

### Claude Code Integration

Students connect via `ANTHROPIC_BASE_URL`:
```bash
export ANTHROPIC_BASE_URL="https://api.medium.text-generation-webui.myia.io"
claude --model qwen3.6-35b-a3b
```

## Qwen3.5-35B-A3B (Archived 2026-04-17)

Replaced by Qwen3.6-35B-A3B on 2026-04-17. Profile archived to `myia_vllm/archives/2026/medium-qwen35-moe.yml.archived-2026-04-17`. Key specs: 35B MoE (3B active), hybrid GDN+Gated Attention, AWQ 4-bit, 86 tok/s decode, 269 tok/s concurrent, FP8 KV 335K tokens, SWE-bench 69.2%, IFEval 88.5%, GSM8K 88.0%, MME 1294.7, MMStar 53.2%. Tool parser: `qwen3_coder`, reasoning: `qwen3`. Image: `nightly-f6983f01de2bf2e92ab468fa735ebac39cddd670` (Apr 06).

## GLM-4.7-Flash (Archived)

Replaced by Qwen3.5-35B-A3B on 2026-02-25. Profile archived 2026-04-24 at `myia_vllm/archives/2026/profiles_legacy/medium-glm.yml`. Requires custom Dockerfile for `transformers >= 5.0`. Key specs: 31B MoE, 3B active, MLA attention (~54 KB/token KV), 56 tok/s decode, 197 tok/s concurrent, SWE-bench 59.2%. No vision support. Tool parser: `glm47`, reasoning: `glm45`.

## Qwen3-Coder-Next (Archived)

Archived to `medium-coder.yml`. Pipeline Parallelism (PP=3) caused severe pipeline bubbles (~66% GPU idle), limiting throughput to 5-6 tok/s. Key issues:
- TP=3 fails: `intermediate_size=8192` not divisible by 3
- TP=2 OOM: 46GB model doesn't fit in 48GB (2x24GB)
- PP=3: only viable option but pipeline bubbles destroy autoregressive performance

## Project History & Context

This repository has been maintained primarily by Roo (another AI agent) through 20+ documented missions. Key milestones:
- Missions 1-15: Initial setup, Qwen3 integration, optimization
- Missions 16-17: Vision model support (Qwen3-VL-32B)
- Missions 18-21: FP8 investigations, structure consolidation
- Mission 22+: Migration to Qwen3-Coder-Next (archived)
- Mission 23+: Migration to GLM-4.7-Flash for better performance

Legacy `myia-vllm/` directory has been archived to `myia_vllm/archives/legacy_myia-vllm_*/`.

## Logging Middleware (DISABLED in production)

ASGI middleware (`myia_vllm/middleware/logging_middleware.py`) that intercepts `/v1/chat/completions` and logs full request/response content + timing as JSONL at `/logs/chat_completions.jsonl`. **Disabled since 2026-03-13** due to -40-65% throughput impact. Available for temporary debugging.

- **Captures**: model, messages_count, last_user_message, tools_count, all sampling params, chat_template_kwargs, response_text, reasoning_text, tool_calls, finish_reason, prompt_tokens, completion_tokens, ttft_s, e2e_s
- **Handles both streaming (SSE) and non-streaming** responses
- **Config**: `VLLM_LOG_DIR` (default `/logs`), `VLLM_LOG_REQUESTS_CONTENT` (default `1`)
- **To enable**: add `--middleware logging_middleware.RequestResponseLogger` + `PYTHONPATH=/middleware` to Docker profile
- Volume-mounted read-only: `myia_vllm/middleware:/middleware:ro`

## SK Agent MCP Server

Semantic Kernel-based MCP proxy (`myia_vllm/mcp/sk_agent.py`) exposing any OpenAI-compatible model as MCP tools for Claude Code / Roo Code. Uses pluggable MCP servers for tool calling.

### Tools exposed
- `ask(prompt, system_prompt?)` -- text query with auto tool use
- `analyze_image(image_source, prompt?)` -- vision, converts local paths to base64
- `list_tools()` -- introspection

### Architecture
```
Claude/Roo --stdio--> FastMCP server --> Semantic Kernel
                                          ├── OpenAI Chat Completion --> vLLM
                                          ├── MCPStdioPlugin: SearXNG
                                          └── MCPStdioPlugin: ... (from config)
```

### Config (`sk_agent_config.json`)
- `model.base_url` -- endpoint (vLLM, Open-WebUI, etc.)
- `model.api_key_env` -- env var name for API key
- `model.vision` -- enable image support
- `mcps[]` -- list of MCP servers to plug in (command, args, env)
- Adding a new tool = adding an MCP entry to config, zero code changes

### Registration
```bash
claude mcp add sk-agent --transport stdio \
  -e SK_AGENT_CONFIG="d:/vllm/myia_vllm/mcp/sk_agent_config.json" \
  -e VLLM_API_KEY_MEDIUM="..." \
  -- python d:/vllm/myia_vllm/mcp/sk_agent.py
```
Note: Config points to `qwen3.5-35b-a3b` on port 5002 (updated 2026-02-25).

### Dependencies
```
semantic-kernel[mcp]>=1.39  (requires openai>=1.109)
mcp>=1.7
```

## Sampling Parameter Optimization (2026-03-08)

### Problem
Repetition and language mixing (Chinese in French responses) observed in Roo Code. Root cause: Roo sends only `temperature` (was 0.1, quasi-greedy) with no `presence_penalty`. The `presence_penalty` is critical for Qwen3.5 to avoid repetition loops.

### Qwen Official Sampling Recommendations

| Mode | temp | top_p | top_k | presence_penalty | repetition_penalty |
|------|:----:|:-----:|:-----:|:----------------:|:-----------------:|
| **Thinking General** | 1.0 | 0.95 | 20 | **1.5** | 1.0 |
| **Thinking Coding** | 0.6 | 0.95 | 20 | 0.0 | 1.0 |
| **Instruct General** | 0.7 | 0.8 | 20 | **1.5** | 1.0 |
| **Instruct Reasoning** | 1.0 | 1.0 | 40 | **2.0** | 1.0 |

### vLLM Server-Side Defaults
`--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}'`
Supported params in `--override-generation-config` (`vllm/config/model.py:1395-1402`): `repetition_penalty`, `temperature`, `top_k`, `top_p`, `min_p`, `max_new_tokens`.
Note: `presence_penalty` is NOT in this list — default is 0.0, must be injected client-side or via OWUI wrappers.

### OWUI Model Wrappers (Sampling Injection, calibrated 2026-03-21)
8 models in OWUI inject sampling params optimized for AWQ Q4 quantization. Adjusted from official Qwen BF16 recommendations based on Reddit community feedback + local benchmarks.

**Qwen_* preset wrappers:**

| OWUI Model | Usage | temp | pp | rp | top_p | top_k | min_p | thinking |
|------------|-------|:----:|:--:|:--:|:-----:|:-----:|:-----:|:--------:|
| `Qwen_think` | General | 0.7 | 1.5 | — | 0.95 | 20 | — | yes |
| `Qwen_think-code` | Coding | 0.6 | 0.0 | — | 0.95 | 20 | — | yes |
| `Qwen_think-reason` | Reasoning | 1.0 | 1.5 | — | 1.0 | 40 | — | yes |
| `Qwen_instruct` | Chat | 0.7 | 1.5 | 1.1 | 0.8 | 20 | 0.01 | no |

**Original model wrappers (aligned 2026-03-21):**

| OWUI Model | Usage | temp | pp | rp | top_p | top_k | min_p | thinking |
|------------|-------|:----:|:--:|:--:|:-----:|:-----:|:-----:|:--------:|
| `Local.qwen3.5-35b-a3b` | Chat général | 0.7 | 1.5 | — | 0.95 | 20 | — | yes |
| `Local.qwen3.5-35b-a3b-fast` | Bots/FAQ | 0.6 | 0.5 | 1.1 | 0.85 | 20 | 0.01 | no |
| `expert-analyste` | Analyse/coding | 0.6 | 0.0 | — | 0.95 | 20 | — | yes |
| `redacteur-technique` | Rédaction | 0.8 | 0.5 | 1.05 | 0.95 | 20 | 0.05 | yes |

**Q4 adjustments vs official BF16**: temp 1.0→0.7 (thinking general), pp 2.0→1.5 (reasoning, language mixing risk), rp 1.05-1.1 (anti "reasoning bleed-through"), min_p 0.01-0.05 (quantization artifact filter).

**OWUI endpoint for Roo**: `https://open-webui.myia.io/api` (NOT /v1)
**API**: `POST /api/v1/models/model/update` to modify params (full replace, not partial)
**Mechanism**: `params` (native: temp, top_p, min_p, pp, fp) + `custom_params` (top_k, rp, chat_template_kwargs) deep-merged into request body. `ModelParams` uses `extra="allow"`.

### Repetition Benchmark Results (2026-03-08, AWQ 4-bit)

| Preset | 4gram | 8gram | TTR | RepLine | tok/s |
|--------|:-----:|:-----:|:---:|:-------:|:-----:|
| baseline (0.3, pp=0) | 0.104 | 0.041 | 0.450 | 0.028 | 110.7 |
| roo-current (0.6, pp=0) | 0.108 | 0.035 | 0.441 | 0.036 | 102.1 |
| think-code (0.6, pp=0) | 0.077 | 0.024 | 0.482 | 0.028 | 117.9 |
| **think-general (1.0, pp=1.5)** | 0.071 | 0.030 | 0.522 | 0.012 | 111.3 |
| **think-reason (1.0, pp=2.0)** | 0.050 | 0.015 | 0.555 | 0.016 | 116.6 |
| **instruct (0.7, pp=1.5)** | **0.042** | **0.013** | **0.540** | **0.013** | 106.9 |

**Key findings**:
- `presence_penalty` 1.5-2.0 reduces 4-gram repetition by **2-3x** vs pp=0
- No speed impact from penalties (~100-118 tok/s across all presets)
- `instruct` (pp=1.5, no thinking) has lowest repetition + highest diversity
- These results apply to AWQ 4-bit quant (BF16 may differ)

### SK Agent Sampling Support
SK Agent (`sk_agent.py`) now reads sampling params from `sk_agent_config.json`:
```json
"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 4096}
```
Passed via `OpenAIChatPromptExecutionSettings` to `ChatCompletionAgent.get_response()`.
Non-standard params (top_k, min_p) sent via `extra_body`.

## Current State (2026-10-08: **prod on Swift-1.5-Qwen3.8-27B + FP8 KV on stock `vllm/vllm-openai:v0.31.0` + KV offload RAM tier (unpinned)** — v0.31.0 bumped 05/10 (PR #72: pin=1 re-falsified live on #51081's chunked registration, crash at `copy_event.synchronize` ~14 s post-startup, walking-dead engine); compile-cache volume `…-v0310`; **batch 8192 + max-num-seqs 48 since 22:14Z 08/10 (PR #104, user green light; fresh-vs-fresh A/B: N=32 718 t/s = +35%, N=48 907 t/s stable 2 passes, prefill 157K 1595-1608 vs 1316 t/s = +21%, KV 424,610 = −5.9% then −1.3%, per-stream 23 t/s at N=48)**; **VRAM at 8192/48: GPU 0 19,937 / GPU 1 18,930 MiB** (below the 4096/32 reference — the 05/10 21,388 figure is obsolete; ~3 GiB under the 23,000 alert line). Offload tier: `--kv-offloading-size 24` (load path CONFIRMED 05/10 bench v3 — replay 0.32 s vs 16.3 s cold via 655 MB CPU→GPU load; rule: tier ≥ 1.5× GPU KV) + `native` backend + `VLLM_WSL2_ENABLE_PIN_MEMORY=0` — **pin=1 never on this host**; `--kv-offloading-size 0` does NOT disable (config tests `is None`); `kv_offload_cpu_cache_usage_perc` is an in-flight gauge, NOT a fill gauge. Swift itself promoted 2026-09-25 on user GO; profile `medium-swift15-27b.yml`, dual served names (`qwen3.6-35b-a3b` alias + `swift-1.5-27b`), VRAM ~19.9/18.8 GiB — same envelope as the MoE. 9/13 gates pass, the 4 failures all cascade from the watchdog wedge-restart during the 235K prefill gate (long-prefill starvation of new requests, engine itself healthy — see the promotion NOTES; watchdog GEN_TIMEOUT 90 s since PR #61). nbaudit-hourly re-enabled. Rollback armed: `medium-qwen36-stock-tq.yml`. Previous: v0.30.0 (promoted 09-22, MoE warm A/B N=16 834-847 vs 678.1, +23-25%); v0.29.0 (09-11, PR #39, gate env pin-memory); windows A+B PR #37; container serving the NEW HF token since 08-31)

### 2026-09-22 — K2-Horizon serving eval: not adoptable today; resume harness validated

Record: `myia_vllm/_iteration_log/k2_horizon_serving_eval/NOTES.md`. Three serving paths tried in one window (prod restored healthy): the PR #56637 overlay is **XPU-only** (dead on CUDA); the cyankiwi fork serves correctly but only in `--enforce-eager` at 65K context, **6-8 t/s single-stream vs ~120 for prod** (N=16 112-123 vs 762 same evening); the Siladrim plugin only demonstrates TP=1. Our own W4A16 recipe keeps the MoVA `v_experts` in BF16 (the only form stock vLLM serves, `k2_horizon.py:812`), i.e. ~15.1 GiB/GPU at TP=2 against ~16.8 GiB of budget at gpu-util 0.70 → almost no KV. **Parked by user decision on 2026-09-23** (registry Q3, option b): the in-house quant is not run; re-open when upstream gains CUDA support for quantized `v_experts`. All artifacts are kept. The per-layer **checkpoint/resume harness** of `quantize_k2_horizon.py` is validated for a long run: bit-exact comparison is the wrong criterion (GPTQ on GPU differs between two fresh runs: ~10 % of elements per MoE layer, zero-points identical), and a resumed run sits inside that fresh-vs-fresh noise on every layer, including the two recomputed after the resume.

### 2026-09-22 — notebook audit on the local model through sk-agent (dry pilot, PR #47)

Adoption work (user mandate: delegate real work to the idle prod). A dedicated sk-agent agent `notebook-auditor` (Qwen3.6 + desktop-commander + jupyter-papermill + searxng; local config, sanitized copy in `myia_vllm/scripts/adoption/nb_audit/agent_entry.example.json`) driven by `nb_audit_driver.py`, a deterministic 3-stage harness: the model reads the notebook and writes verification scripts, **the harness executes them** (dedicated venv `%LOCALAPPDATA%\nbaudit\verif-venv`), the model judges against the outputs in the same conversation. Pilot 1 showed the local model does **not** verify on its own (tool calls on 1 notebook in 9, "exact" claimed without executing anything), hence stage B. Pilot 2 on the 11 notebooks the CoursIA #17073 bots audited on 09-22: 11/11 parsed, 20/21 scripts clean, 15 valid findings, **8/25 bot findings recovered**, ~10/15 likely true; **~50 notebooks/h at concurrency 6** (the two bots: 2/h), prod healthy. Binding limit to scale further: sk-agent's hardcoded **300 s per LLM turn** (stage A peaks at 272 s) — proposed to roo-extensions with a `thinking_token_budget` passthrough. Nothing is posted to GitHub; integration mode (third partition vs pre-audit records for Hermes/NanoClaw) awaits the CoursIA coordinator. Record: `myia_vllm/_iteration_log/nb_audit_pilot/NOTES.md`.

**2026-09-23 follow-up**: the three readers chose mode (b) pre-audit, and the user gave a conditional GO ("bien cadré avec les consommateurs"). The runner (`nb_audit_runner.py`, PR #53) and the deliverer (`nb_audit_deliver.py`, PR #54) are built. NanoClaw gets DM attachments; Hermes gets base64 blocks in the DM body. roo-extensions #3797 gives notebook-auditor a 600 s turn, so concurrency is now 6. A first local run on #17107 produced 3/3 records in 284 s and delivered nothing. The scheduled task was held until Hermes gave its reading rule and C3 confirmation, and CoursIA gave C1/C6. Second background workload: CoursIA #1453 Lean prover passes run **on the MoE** from a dedicated clone. Demo 0 took 151.5 s with every role local; record `_iteration_log/prover_1453/NOTES.md` (PR #55).

**2026-09-23 night: live.** Every consumer gave its green light: NanoClaw C2/C3/C5, Hermes C3 with rule C2 (ascending checklist position, skipping anything checked or already audited), and CoursIA C1/C6 tacitly. First deliveries at 22:53Z and 23:00Z: 3 dossiers to NanoClaw (#17107) and 6 to Hermes (#17239 Sudoku, #17357 Lean). The deliverer now re-reads the checklist just before sending and drops records the bot audited in the meantime (PR #57). The user-level scheduled task `nbaudit-hourly` (no UAC, `register_hourly.ps1`; the dry-run was posted on global first) runs `nb_audit_hourly.ps1` at **:05Z every hour**: up to 3 notebooks per series, concurrency 6, about 4-8 min of load on :5002. Its manual test at 23:00Z audited 9/9 notebooks, superseded 3 stale records and sent 5/5 DMs. For #1453, `run_prover_target.py` points the unmodified harness at a file:theorem target by injecting a transient DEMOS id; pass 1 targets `Folk.lean:folk_theorem_discounted`. Scheduling lesson: a prover pass that overlaps the runner gets about 50 tok/s per stream, and the harness's 240 s per-call client cap (×2 with one retry for `local`) then kills the coordinator's thinking turn. Start prover passes at **:30**, clear of the runner. **Outage 2026-09-24 00:26Z→01:34Z**: the full `knot_lean` prebuild (`lake build`, 32 threads on the Windows host) pushed the host commit charge to 97.8 % and Docker Desktop died, taking every container down for ~68 min. The hourly Docker watchdog relaunched it. Check the commit headroom (`reclaim-wsl-memory.log`, below ~85 %) before any heavy host-side job. Single-file prover passes are fine.

### 2026-09-22 — Qwen3.8-27B + DFlash2 eval: BLOCKED UPSTREAM at boot (window 15:22→15:59Z, GO user)

The reopened 27B line (see `myia_vllm/_iteration_log/qwen38_dflash2_v0300/NOTES.md`, PR #45) hit a deterministic boot crash on **both** AWQ quants: `AttributeError: 'QKVParallelLinear' object has no attribute 'weight'` — `qwen3_dflash.py:472` `_build_context_kv_buffers` slices the **target's** `qkv_proj.weight` to build the drafter's fused KV GEMM, and pack-quantized targets expose `weight_packed`. An fp8 target would also be silently wrong there (per-channel scale not sliced), and 27B BF16 (54 GB) doesn't fit 2×24 GB — so **DFlash/DFlash2 on v0.30.0 requires an unquantized target; no path exists on this host**. Already tracked upstream: fix PRs vllm-project/vllm#51684 (active 09-15) / #51620, bug #51581 — re-test at #51684's merge (drafter `syvai/Qwen3.8-27B-DFlash2-W4A16` + both quants cached, profile `medium-qwen38-27b-dflash2.yml` kept runnable). Validated before the crash: `DFlash2DraftModel` auto-resolution, fp8 KV, V2 gate held, spec scheduler engaged. Only viable spec-dec on an AWQ target remains **MTP-3** (registry Q6 arbitration). Baseline A1 captured on the live MoE: N=1 86.3 / N=16 815.9 t/s. Operator gotcha: `hf download` through the `\\wsl.localhost` 9p bind with Xet storage leaves a ~0-byte stub and prints "✓ Downloaded" — use `HF_HUB_DISABLE_XET=1` and verify `du -sb` inside WSL; also `-v '//wsl...'` in Git Bash silently creates a phantom bind (use the exact `.env` HF_CACHE_PATH via PowerShell). Prod restored healthy 15:59:22Z (~37 min outage).

### 2026-09-22 — Qwen3.8-27B + DFlash2 eval: window 1 blocked on stock, window 2 PASSES on PR-#51684 overlay

Full record: `myia_vllm/_iteration_log/qwen38_dflash2_v0300/NOTES.md` (PRs #45, #46). **Window 1** (stock v0.30.0): deterministic crash on both AWQ quants — `qwen3_dflash.py:472` `_build_context_kv_buffers` slices the target's `qkv_proj.weight`, nonexistent on pack-quantized targets (bug #51581, fix PRs #51684/#51620). **Window 2** (user GO "build a dedicated image with the PR"): overlay `vllm-openai-v0300-s51684:v1` = stock v0.30.0 + the PR's 4 runtime files (apply verified clean: base is an ancestor, files unchanged in between) — **boots healthy in 5.5 min where stock crash-looped 37×**, battery **16/17 PASS**, **degenerate-output canary 0.00** (correct drafts on a quantized target), acceptance 0.48 single / 0.21 under N=16 (DSpark pattern attenuated). KV 262,504 tokens = **1.0× the window** (drafter + lookahead eat August's 383,696 surplus; capping max-model-len to 150K would buy ~1.75 concurrent full-context conversations; the claudish per-client histogram of 09-23 settles it: Hermes and NanoClaw run at a 121K / 103K median and a third of claude-* requests exceed 100K, so only sk-agent's short turns would fit — the MoE stays). **Ratios vs same-day MoE brackets: single 0.5-0.66×, N=16 0.22-0.40×, prefill ~4×** — the community 134-154 t/s single-stream was 1×GPU/TP=1 on an optimized fork; on our TP=2 the hoped-for inversion does not materialize, the 27B stays a pure quality trade (SWE-bench Pro +8.2). Validation datapoint posted on the upstream PR ([comment](https://github.com/vllm-project/vllm/pull/51684#issuecomment-5780401133)) to help its merge; re-test on stock when it lands. Operator gotchas persisted: `hf download` through the 9p bind with Xet leaves a silent ~0-byte stub (`HF_HUB_DISABLE_XET=1` + verify `du -sb` in WSL); `-v '//wsl...'` in Git Bash creates a phantom bind (use the exact `.env` HF_CACHE_PATH); `.gitignore`'s `models/` pattern shadows nested overlay paths (`git add -f`). Prod healthy after both windows (outages 37 + 17 min).

### 2026-09-22 — v0.30.0 PROMOTED (GO user; record `myia_vllm/_iteration_log/v0300_bump/NOTES.md`)

Window 11:25→11:47Z: 3 profile lines only (image, compile-cache mount `…-v0300`, volume declaration; `…-v0290`/`…-v0280` kept for rollback). Boot ~4.5 min, `Using V2 Model Runner`, pin-memory gate held. Config byte-identical: `prefix_match_unit 16`, mamba fp16, **KV 934,660 tokens**, VRAM GPU 0 19,344 / GPU 1 18,294 MiB. **13/13 gates PASS** (prefill 8,089/7,345/5,673 tok/s, concurrent gate 386→507 on repeat = prefix cache live), FORCING 6×4,618-tok concurrent prefills → forwards in (4096,8192], 0 error signatures. Bench same-window: first pass pays the documented JIT warm-up tax (N=16 609.6), warm passes **847.0/834.1 vs baseline 678.1 → +23-25 %, better at every N** (all-time record stays 869.5, 09-11 — cross-day comparisons are their own baseline). What v0.30.0 brings us: #56908 (V2 tolerates missing pinned memory), #51700 (FULL cudagraphs under V2 — new territory, clearance not soak), #55133 (DSpark Qwen3 padded-vocab fix). #52676/#52789 verified already ancestors of v0.29.0 — no action. New boot log suggests explicit `--kv-cache-memory` (recorded, not acted on). TQ×MTP #53406/#53410 still OPEN, frozen since 08-22. **DSpark EVALUATED same day (GO user) and REJECTED** — see `myia_vllm/_iteration_log/dspark_v0300_eval/NOTES.md`: after 4 boot attempts (EP×dense-drafter validation, an operator flag-truncation incident, KV-memory → 131K), acceptance measured **7.3 %** (0.58 tok/step; pos-0 ~33 %) vs DFlash's 26-47 %, N=16 **−33 %**, KV 181K/1.38× at 131K, EP lost. Likely root cause: **#53929** (adaptive DSpark verification for Qwen GDN) still OPEN — re-test at its merge. Boot-blocking lesson persisted: never place `#` comment lines inside a backslash-continued `command:` block (continuation joins the comment → command terminates → `exec` swallows every following flag; signature: `speculative_config=None` in the banner).

### 2026-09-11 — v0.29.0 PROMOTED (the 09-09 rejection was configuration, not a wall)

**Same image, same digest** (`sha256:aa542864b00a…`, verified against the running container). One environment line changed: `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. `UvaBuffer` raises when `is_uva_available()` is false, and that helper (`utils/platform_utils.py:51-57`) is literally `is_pin_memory_available() or is_cpu()`; under WSL2 `is_pin_memory_available()` (`platforms/cuda.py:299-318`) returns that env var, **default False**. Both TP workers now load 11.52 GiB each; health 200 in 2.9 ms, RC=0.

Post-boot: `block_size=1424`, `prefix_match_unit=16`, `mamba_ssm_cache_dtype=float16`, KV **934,660**, blocks 681, VRAM **GPU 0 19,127 / GPU 1 18,296 MiB** (below v0.28.0's 19,744 / 18,830 — consistent with #53955 releasing cudagraph memory before the KV allocation, our boot-OOM lever). **13/13 gates PASS**, error scan 0, prefill 8,737 / 7,670 / 5,827 tok/s (+2 to +4.5%), N=16 **869.5 tok/s** — above every previous same-machine figure (828 on 09-01, 794.9 on 09-09). The concurrent gate's shared-prompt repeat stayed at 1,103 tok/s, so the A+B prefix-cache win carries over.

**Watch item**: the V2 model runner is new on this host — gates + bench are a correctness clearance, not a soak. Rollback is four lines (image, compile-cache volume mount, volume declaration, env var). DSpark was blocked by the same UVA boundary and is now unblocked for re-evaluation. Record: `myia_vllm/_iteration_log/v0290_bump/NOTES.md`. **First WEDGE under V2 on 2026-09-12 17:02Z** (decode stuck >40 s while health 200): watchdog v5 restarted the engine cleanly (17:04Z, RC=0, healthy since). It coincided with heavy 9p weight-loading IO from a GPU-2 eval container (16-20 GB × 12 boots through `\\wsl.localhost`) — different GPU, so a host IO/CPU-side correlation, not VRAM. Keep watching; the machine-throughput investigation already documented CPU starvation as a wedge-adjacent mechanism.


### 2026-09-11 — prefix cache granularity + fp16 mamba state (PR #37, issue #36)

**The production prefix cache was structurally dead** (`prefix_cache_queries_total=520,101`, `hits=0`). Root cause: `mamba_cache_mode` defaults to `"align"` as soon as prefix caching is enabled (`model_executor/models/config.py:622-649`), pinning `mamba_block_size` to `block_size`, and without `--prefix-match-unit` the cache keys are computed **per block (2768 tokens)** — nothing shorter than a full block could ever match.

Two flags added: `--prefix-match-unit 16` (2768 = 16 × 173, divisibility rule holds) and `--mamba-ssm-cache-dtype float16` (GDN ssm state only; conv stays `auto` — the HF config asks float32 and vLLM logs that the user value wins, `model_executor/models/config.py:799`). **Not** taken: `--enable-mamba-fine-grained-prefix-cache` — inert without EAGLE/MTP (`config/cache.py:198-203`).

Same-night paired probes: 403-token prompt **0 → 400 hits**; 7,699-token prompt 5,536 → **7,696 hits** (whole prompt) with repeat **0.412 s → 0.078 s**. The gate battery's 16 concurrent requests sharing one 3,200-token prompt went **527 → 1,008 tok/s** on the repeat run (the syv-ai "shared system prompt" case). Side effect: `block_size` re-derives **2768 → 1424**, `num_gpu_blocks` 360 → 688, KV **934,374 → 944,267 (+1.1%)**. 13/13 gates PASS, prefill 8,363 / 7,312 / 5,704 tok/s, error scan 0, bench N=16 766 vs 799 same-evening (machine variance, no systematic regression). Record: `myia_vllm/_iteration_log/prefix_granularity/NOTES.md`.


**2026-08-31 incident + recovery (record for successors):** around 08:18Z a WSL/Docker Desktop degradation killed the stack twice over — first the 9p bind mounts (`runc create failed … failed to fulfil mount request`, exit 127), and after a `--force-recreate` fixed the mounts, the **GPU passthrough** turned out broken too (`RuntimeError: Failed to infer device type`, `libnvidia-ml.so` missing in-container, GPUs at 0 MiB). Root cause was the host GPU driver/WSL layer — **fixed by an NVIDIA driver update + host reboot** (user action). Recovery validated 09:36Z: health 200, decode probe OK, 33h+ clean soak since. Three lessons now encoded in the `/vllm-surveillance` command and `feedback_docker_recovery_recipes.md` (Patterns 3+4): (1) `docker logs --since` is unreliable since 08-30 (log split-brain — full-read path sees only the dead segment; use small `--tail`), (2) relaunching sidecars must use `docker start`, NEVER `docker compose up` (config-hash drift recreates the engine), (3) GPU-passthrough loss needs driver+reboot, not profile changes.
**Token rotation status (from the 27/08 security incident):** the container was force-recreated 08-31 and **serves fingerprint `769d40e66497`** (the new fine-grained token). The old in-service token `04368a9965aea735` is still valid — **revocation pending the user**; after revocation, purge the 13 unpushed local commits on `feature/secure-configs` that contain it. **No push from d:/vllm until that revocation is done.**

**`VLLM_API_KEY_MEDIUM` rotated 2026-09-06 18:18Z** (user-mandated; the May key was exposed and the external friend access window closes with it). Stack recreated (~5.5 min outage), old key → 401 verified, watchdog decode OK. **claudish is the single holder of the new key**: ai-01 sidecar patched + restarted ✅, and the claudish **hub on po-2025** (`192.168.0.50:3000`, publicly `models.myia.io` via the po-2023 IIS frontend + `:18182` TCP relay — the real provider-calling hub moved there in the 05/09 cutover; the deployment doc in `D:\claudish/docs` still saying "hub on po-2023" is **stale**) patched + restarted by po-2025:claudish at ~19:55Z after a workspace-level DM re-route (a machine-level DM is invisible to workspace lanes). **Rotation complete end-to-end, verified 20:03Z**: real `qwen3.6-35b-a3b` completion → 200 on all three paths (hub direct, po-203 relay NOMINAL, models.myia.io). Earlier same-evening false closure retracted: a 200 measured through a sidecar during a hub restart is the sidecar serving AUTONOMOUSly — never validate the relay path via a sidecar in a hub-restart window. Local consumers patched with the raw key (legitimate OpenAI-protocol direct exceptions): `~/.claudish/config.json`, `d:/roo-extensions/.env`, RSM `.env` (×2), sk-agent config (×2 entries on :5002) — MCP processes spawned before 18:18Z keep the old key in memory until restarted (condensation falls back to cloud, nominal). **Consumption standard going forward: through claudish** (`x-proxy-key` header, Anthropic; `models.myia.io/v1`, OpenAI) — the raw key is distributed only by private DM attachment for documented exceptions, never on GitHub/dashboards. Fleet migration tracked by roo-extensions issue #3401 (inventory: `docs/investigation/issue-3401-inventory-local-model-consumers.md`). Discriminator learned from `proxy-auth.ts`: claudish 401 body "invalid proxy authentication" = proxy auth failed; Anthropic-shaped "Unauthorized" = the upstream vLLM rejected the key. Full record: memory `project_vllm_api_key_rotation_20260906.md`.

**`VLLM_API_KEY_MEDIUM` rotated 2026-09-30 13:53Z** (rule secrets n°5: web1 published the RAW VALUE as a fake "32-char fingerprint" on roo-extensions#3958 c.5911224566 at 12:23Z — masked 12:38Z but recoverable from GitHub edit history; repo public + endpoint exposed via `api.medium.text-generation-webui.myia.io`). Fingerprint sha256[:12] old `299ad00b21b5` → new `db2738ab1328`. Announced on the global dashboard BEFORE applying (13:52Z); engine+watchdog **recreated** via `compose up -d` (explicit recreate, ~2 min warm boot, KV 465,423 unchanged); verified **200 new / 401 old / 401 no-key** on BOTH direct `:5002` and the public URL. ai-01 patched by the vllm lane (9 live files: `myia_vllm/.env`, `roo-extensions/.env`, RSM ai-01 `.env` — its `OPENAI_API_KEY` condenser line carried the VL key `7711C3D0…`, a latent 401 per memo #1147, aligned to MEDIUM — `sk_agent_config.json` ×3, `~/.claudish/config.json`, CoursIA `master.env` + GenAI `.env` + `vllm-zimage/.env`, `Open-WebUI/myia-open-webui/.env`); exhaustive `docker inspect` showed NO container env holding the old key. Key distributed by DM attachment (`destruct_after` 24 h) to web1 (RSM-seat propagator, #3401), po-2026, po-2025:claudish; confirmations by fingerprint only on #3958 (c.5912929249). Lessons: publish `sha256(value)[:12]`, NEVER a hex slice of the value; the Grep tool (ripgrep) skips gitignored `.env` files — key sweeps need `rg --no-ignore --hidden`; key rotation requires compose RECREATE, not `docker restart`. Full record: memory `project_vllm_api_key_rotation_20260930.md`.

### Stock v0.27.0 — what changed vs the Genesis profile

Beyond `image`, the `GENESIS_ENABLE_*` variables and a separate compile-cache volume, exactly four differences:

1. **`--prefix-caching-hash-algo sha256`** (Genesis used `xxhash`). The stock image **does not ship the `xxhash` module**, and vLLM imports it **lazily** inside `request_block_hasher` — so short prompts pass and only requests that fill a complete KV block fail, with an HTTP 500 (`ModuleNotFoundError: xxhash is required...` at `vllm/utils/hashing.py:63`). That masquerades convincingly as the TurboQuant crash we were hunting. `sha256` is the upstream default and keeps the image a pure `docker pull`; the alternative (an overlay image adding `pip install xxhash`) was rejected for exactly that reason.
2. **`labels: ["autoheal=False"]`** — see the autoheal section below. Added to the **Genesis profile too**, so a rollback keeps the fix.
3. **`start_period: 1800s`** (was 900s). A cold boot with an empty compile-cache volume exceeds 900 s: weights 262 s, torch.compile 209 s, engine init 474 s, then a **multi-modal warmup on the API-server side that is new in v0.27.0 and not covered by `--skip-mm-profiling`**. A warm boot is ~8 min.
4. **Three env vars dropped**: `VLLM_USE_FLASHINFER_MOE_FP16`, `VLLM_MARLIN_MOE_BLOCK_SIZE_M`, `VLLM_USE_FLASHINFER_MOE_FP8`. v0.27.0 selects the MoE path via `KernelConfig.moe_backend`, not env vars — and the first of those never did anything here anyway (see the Environment Variables correction above).

Everything else is byte-identical to the Genesis profile: gpu-util 0.70, batch 4096 / max-num-seqs 16, `turboquant_k8v4`, 262144 context, `VLLM_USE_FLASHINFER_SAMPLER=0`, `--no-enable-flashinfer-autotune`, `NCCL_P2P_DISABLE=1`, watchdog v5, telemetry, same `container_name` (only one stack runs at a time).

5. **Entrypoint HF-cache phantom-mount guard (2026-08-12).** The `create_host_path: false` on the HF bind (#2 of the fixes) only fails loud at container **create** (`up -d`). A restart-policy restart after a Windows reboot reuses the mount spec without re-validating the source — so on 2026-08-12 Docker silently substituted an empty dir for the WSL HF bind and vLLM began re-downloading 19 GB while every signal said "patient boot". The profile now overrides the image entrypoint with a guard that, on every start, waits up to 3 min for a >4 GB blob in the model cache and `exit 1`s loud otherwise (making `RestartCount` climb → the watchdog CRASH-LOOP detector surfaces it in ~9 min, instead of an hour of silent re-download). Validated both branches: 5.01 GiB → proceed; 2 bytes → wait+fail.

### ⚠️ `autoheal` — a competing restart authority with almost no signature

This host runs `qdrant_autoheal` (`willfarrell/autoheal`) with **`AUTOHEAL_CONTAINER_LABEL=all`**: it `docker restart`s **every** container that goes `unhealthy`. It came from the Qdrant stack and was never meant for vLLM, but `all` does not discriminate — so it has been restarting the engine **in competition with our watchdog**, cancelling the watchdog's deliberate "patient boot" design (watchdog v5 never restarts during `starting`; autoheal fires the moment `start_period` expires).

**Its signature is almost nil**, which is why it hid for months: `ExitCode=0`, `OOMKilled=false`, and crucially **`RestartCount` unchanged** — `docker restart` does not increment it, so the watchdog's CRASH-LOOP detector (which reads `RestartCount`) is *structurally blind* to it. On the vLLM side the only trace is a `KeyboardInterrupt: terminated` from `_interrupt_init`.

Observed: it killed the first cold boot of the stock image at 2026-08-11 00:03:04Z, **and** it fired at 2026-08-10 13:40:15 and 14:40:50 on the prod container — inside that day's incident window, which **revises** that incident's analysis (previously attributed to watchdog + WDDM paging alone).

Fix: `labels: ["autoheal=False"]` — capital F is mandatory, the filter is `select(.Labels["autoheal"] != "False")`, an exact string comparison. **General rule: before analysing any unexplained container restart, check whether another restart authority exists on the host (`docker ps | grep -i heal`) — a flat `RestartCount` does not prove no restart happened.**

### ⚠️ The Genesis image is IRREPRODUCIBLE — keep the backups until the stock soak concludes

This is what motivated the migration, and it still governs the rollback path.

`vllm-qwen36-genesis-tq:v7.72.5-vllm01d4d1ad3` **cannot be rebuilt**. Its base, `vllm/vllm-openai:nightly-01d4d1ad375dc5854779c593eee093bcebb0cada`, now **404s on Docker Hub** — nightly tags are retained only ~5 days. The pinned Genesis commit `fbecee31fd42…` *does* still resolve (200 via the repo-rename redirect), so **only the base layer is unobtainable**. Until 2026-08-06 the image existed in exactly one place: the local Docker store.

Backed up to `D:\vllm_image_backups\` (outside the git repo): `vllm-qwen36-genesis-tq_v7.72.5-vllm01d4d1ad3.tar` (8,770,002,944 B, manifest verified) and `vllm-qwen36-shmpatched_nightly-f6983f01d-patched1.tar` (9,872,095,232 B — the FP8 rollback image, same risk). Restore with `docker load -i <tar>`. **Treat these tars as production assets**: losing them means losing the ability to run *or* roll back this deployment.

### ⚠️ Phantom bind mount — Docker's short syntax auto-creates a missing source

**Docker's SHORT bind syntax (`- src:/dst`) CREATES a missing source path instead of failing.** On 2026-08-05/06 a WSL GPU-passthrough failure forced a `wsl --shutdown`; on the next `up -d` the `\\wsl.localhost\Ubuntu\...` HF-cache source was momentarily unreachable, Docker substituted an **empty directory**, and vLLM started **re-downloading the 19 GB model at 0 MiB/min** — for about an hour, while every surface signal said *patient boot* (container `Up`, docker health `starting`, `RestartCount` FLAT, watchdog `BOOTING`, logs stuck after `Starting to load model`).

**Discriminator (one command):**
```powershell
docker exec myia_vllm-medium-qwen36-moe du -sh /root/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit
# ~47G = real cache.  ~300M or missing = PHANTOM MOUNT.
```
**Recovery:** confirm `wsl -l -v` shows Ubuntu **Running**, then `compose down && up -d`. A plain `docker restart` re-mounts the same empty dir.

**Fix in the profile:** the HF-cache bind uses the long syntax with `bind: { create_host_path: false }`, which turns the silent substitution into a loud `up -d` failure. `./models` is deliberately left on the short syntax — `--model` is an HF repo id, so that bind carries nothing (host dir is empty) and hardening it would make `up -d` fail on a fresh clone over a directory git cannot track.

**Boot timing, corrected:** on Genesis, container start → `/health` 200 was **~6 min**, health staying **000** for the last ~3. **On stock v0.27.0 a warm boot is ~8 min** and a cold one (empty compile-cache volume) exceeds 15 — hence `start_period: 1800s`. `health=000` at T+4 min is normal, not a hang.

- **Qwen3.6-35B-A3B MoE + TurboQuant k8v4 on stock `vllm/vllm-openai:v0.27.0`** on port 5002 (GPUs 0,1) — migrated off Genesis 2026-08-11, gates 13/13 (see the migration record in `myia_vllm/_iteration_log/stock_tq_phase2/NOTES.md`)
  - Image: `vllm/vllm-openai:v0.28.0` (bumped from v0.27.1 on 2026-09-01 — 13/13 re-gated, prefill 5,448–8,316 tok/s, N=16 828 tok/s same-night vs 813; KV 1,038,194; VRAM 20,006 / 19,002 MiB; per-image compile-cache volume `…-v0280`, v0.27.1 volume kept for rollback; record `myia_vllm/_iteration_log/v0280_bump/NOTES.md`). Previously v0.27.1 (bumped from v0.27.0 on 2026-08-14 — micro-patch: CI/docs + additive DSpark Markov heads; 13/13 re-gated). A plain `docker pull`, **reproducible**, no patch tree, with an **entrypoint HF-cache phantom-mount guard** (refuses to start if the WSL HF bind resolved empty after a host reboot). Prefill throughput measured at gate: 31,417 tok in 7.2 s, 101,838 in 21.1 s, **253,503 in 58.6 s** (~4,300–4,800 tok/s on v0.27.0; v0.27.1 same gate 5,346–8,177 tok/s on a warmer machine), each followed by a passing survival request.
  - Profile: `medium-qwen36-stock-tq.yml`. Previous: `medium-qwen36-genesis-tq.yml` (Genesis patches P3/P4/P6/P22[skipped]/P26[skipped]/P37/P38B/P40/P67/P78/P98/P101 + PN33 by-default + custom env) — retained as the armed rollback.
  - **KV cache: 1,030,407 tokens @ gpu-util 0.70** (3.93× the 262K window), down 17% from Genesis's 1,238,046 — immaterial at 2–7% observed occupancy. VRAM **GPU 0 19,336 MiB / GPU 1 19,040** vs Genesis 21,145 / 19,340: the Genesis prealloc pools (P28 GDN / P37 MoE / TQ-dequant) lived *outside* the gpu-util budget, so dropping them returns ~1.8 GiB of headroom to the desktop-shared GPU 0. **The `/vllm-surveillance` VRAM reference is therefore ~19,300–20,000 MiB on GPU 0 (20,006 measured on v0.28.0 after the gate battery), not the ~21,150 written for Genesis**; the 23,000 MiB alert threshold is unchanged.
  - **Do not raise gpu-util to recover the lost KV** until the soak concludes — the out-of-pool budget changed with the pools gone and would need re-measuring.
  - **`--gpu-memory-utilization 0.70`** (0.82 → 0.78 on 2026-07-14 → **0.70 on 2026-08-04**). GPU 0 is shared with the Windows desktop (explorer/VSCode/Edge, fluctuating VRAM baseline); GPU 1 has none. A desktop spike between memory-profiling and KV-cache-tensor allocation pushes GPU 0 over its own gpu-util pool cap → **boot-time CUDA-OOM crash-loop** at `_allocate_kv_cache_tensors`. Three occurrences in 19 days: 2026-07-13 @0.82 (RestartCount 53→59+, OOM on a 34 MiB alloc with "9.91 GiB free" = pool-cap not true exhaustion), then **again at 0.78** on 07-28 and 08-01 (686 MiB alloc failing with 2.66 GiB free) — the 0.82→0.78 step had moved the failure point by only ~0.2 GiB. **Why the nominal budget lies:** vLLM's real footprint exceeds `gpu-util × 24564` by **+2 768 MiB @0.78 and +2 145 MiB @0.70** (measured on GPU 1, desktop-free) because Marlin-MoE variable temp allocs (RFC #27951), the Genesis prealloc pools (P28 GDN / P37 MoE / TQ-dequant) **and the CUDA graphs** all live outside it — vLLM warns about the last one at boot, since we run `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. Pre-change GPU 0 was at 23,745/24,564 MiB = **819 MiB free**. At 0.70: GPU 0 **21,145 MiB**, GPU 1 **19,340 MiB**, KV **1,736,379 → 1,238,046 tokens (−29%, still 4.72× the 262K window; prod occupancy is 2–7%)**, no decode regression. **Untested alternative if KV is ever needed back:** re-enable `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` and run gpu-util **0.7349** as vLLM's own boot warning suggests. **Watchdog v3 blind spot — FIXED in v4 (2026-07-15, deployed 07-28, field-proven correct-negative 08-01):** a crash-loop re-enters Docker `starting` on each restart → the DOWN-path treated `starting` as boot-safe (never-restart) → the watchdog never intervened (07-13 crash-loop was completely silent). Watchdog **v4** now reads `RestartCount` inside the `starting` branch: a patient cold boot keeps it flat, a crash-loop makes it climb → after 3 restart-increments (~6 min) it emits a loud greppable `CRASH-LOOP` line. No auto-restart (futile while Docker is already relooping the container; the real cure is the gpu-util drop or freeing GPU-0 VRAM) — detection + visibility only. Simulated (flat RC → BOOTING, climbing RC → CRASH-LOOP at churn≥3, single restart → no false positive) and `sh -n` clean; deployed 07-28 and field-proven on 08-01 (flat `RestartCount=2` → patient `BOOTING`, no false `CRASH-LOOP`). Shipped in [PR jsboige/vllm#11](https://github.com/jsboige/vllm/pull/11) (merged `d6342acc40`) with the gpu-util change. Both the gpu-util value and the watchdog comment live OUTSIDE the `command: >` folded scalar (a `#` inside it becomes a literal vLLM arg). **Watchdog v5 (2026-08-06)** closes two more gaps: (a) a **post-boot warm-up grace** — a freshly booted engine answers its first decodes slowly (measured 52 s → 16 s → 0.49 s) while `/health` already returns 200, so v4's WEDGE path counted two fails and **restarted a perfectly healthy engine** at 00:30:12Z; v5 gives the first 3 probes after any boot a 120 s timeout and never counts them as a wedge, re-arms that grace on entering `starting` and after a watchdog-issued restart, and spends it the instant one decode succeeds; (b) **BOOT-STALL** — `docker=starting` for ≥12 min with `RestartCount` FLAT (so v4's churn detector structurally cannot fire) and `Starting to load model` present without `Model loading took` ⇒ the phantom-mount case above, flagged with the `du` check and the recovery command inline, detection-only (an auto-restart would re-mount the same empty dir). v5 also stops the crash-loop message asserting CUDA-OOM unconditionally — it names OOM only when the log shows it.
  - Marlin MoE (`MarlinExperts` + `MoEPrepareAndFinalizeNoDPEPModular`), Expert Parallelism (EP=2), CUDA graphs (full+piecewise), prefix caching (**sha256** — see divergence #1 above; Genesis used xxhash), chunked prefill, async scheduling
  - ✅ Vision (images, documents) + Thinking modulation + `preserve_thinking` server-side default
  - ✅ `--override-generation-config` defaults (temp 0.6, top_p 0.95, top_k 20, min_p 0.0, rp 1.0)
  - ✅ Watchdog sidecar **v5** (dual-ping host + Docker DNS; 24-token decode probe; four failure modes covered: decode-WEDGE, hard-HANG, boot CRASH-LOOP, BOOT-STALL; post-boot warm-up grace)
  - ✅ `error_source_capture` ASGI middleware active (logs body_head + body_tail 1500B to /logs/error_sources.jsonl)
  - **KV cache: 1,238,046 tokens @ gpu-util 0.70 (×3.8 vs prior FP8 322K; was 1,736,379 @0.78 and ~2.0M @0.82)** — `--kv-cache-dtype turboquant_k8v4` + Genesis P22/P38B continuation-prefill workspace fix. Max concurrency 4.72× at 262K.
  - **Single-user decode**: 120 tok/s no-think (+12% vs FP8), 124 tok/s thinking (+6%)
  - **Concurrent**: N=12 → 625 tok/s, N=16 → **829 tok/s aggregate (+125%)** — main reason for promotion. Prior FP8 saturated at ~N=5.
  - **Context**: 262K native, max-model-len 262144
  - **Critical env**: `VLLM_USE_FLASHINFER_SAMPLER=0` (v2f had idle deadlock at +55min; v2g fix). `--no-enable-flashinfer-autotune`. `NCCL_P2P_DISABLE=1`.
  - **Batch raised to `--max-num-batched-tokens 8192` / `--max-num-seqs 16` on stock (2026-09-02, PR #24). The 4096 ceiling was a Genesis P28 buffer artifact, not a stock constraint (v0.28.0 default is 16384); stock has no size-limited GDN buffer. 13/13 gates + the FORCING test (6x 2702-token concurrent prefills forming forwards in (4096, 8192]) pass; prefill slightly faster; no `Workspace is locked`/Traceback/OOM; KV 934,374 (3.56x, immaterial at 2-7% occupancy). Same-window A/B throughput is confounded by the engine-restart freshness — the turnaround is correctness clearance + ceiling headroom, NOT a measured throughput gain. The 2026-06-24 Genesis P72+P74 attempt that crashed is a different path (Genesis P28 buffer bug, not stock).** **Mechanism CORRECTED 2026-06-28 (verified on our own image via read-only `docker exec`; Sandermage, the patch author, confirmed on [genesis #33](https://github.com/Sandermage/genesis-vllm-patches/issues/33#issuecomment-4811423542)): the overflowing GDN buffer is patch **P28** (`_genesis_gdn_core_attn_buf` prealloc), NOT P72.** P72 (`GENESIS_PROFILE_RUN_CAP_M=4096`) only caps the *profiling* forward at M=4096 (which dodges the 2026-05-24 Dynamo boot crash), letting the engine *boot* at batch 8192 — it does **not** size the runtime buffer. The P28 buffer is sized by the P73 budget resolver, which — with no `GENESIS_PREALLOC_TOKEN_BUDGET` set — falls to its **default 4096** (the `scheduler_config` auto-probe returns None at resolve time, an init-ordering race; every boot logs `[Genesis P73] token budget resolved → 4096 (default fallback)`). P72's cap and P28's default were *coincidentally* both 4096, so it looked like the profile cap coupled the buffer — it did not. A runtime forward >4096 overflows the 4096-sized P28 buffer and killed the worker at 15:23:56Z (~1h25 after deploy): `RuntimeError: setStorage: sizes [5536, 16, 128] … requiring a storage size of 22675456 are out of bounds for storage of size 16777216` (16777216 = 4096×16×128×2; 22675456 = 5536×16×128×2) at `qwen3_next.py:495` via the inductor/cuda-graph `as_strided` path. This crash was the "condensation no longer works" red alert (roo-state-manager :5002 calls got `Connection error` ×3). The 2026-06-24 bench that "validated" 8192 could not trigger it: the 80K crash test is P74-chunked to ≤4096 and N=16 decode is ~16 tok/step — neither produces a forward batch in (4096, 8192]. **4096 IS the effective ceiling** (the P28 default). To run 8192 safely: `GENESIS_PREALLOC_TOKEN_BUDGET=8192` (GLOBAL resolver override — sizes the P28 GDN, P37 MoE, and TQ-dequant pools) + **keep** P72 at 4096 + `--max-num-batched-tokens 8192` + a FORCING test (several concurrent ~2-3K prefills summing one forward into (4096,8192]) off-peak. UNTESTED, DEFERRED — do not retry on prod without that test. Watchdog flapping is NOT fixable by ramping batch (tune the watchdog: done — MAX_FAIL 3→5, --max-time 30→60). Full detail: MEMORY.md `project_genesis_tq_batch_ceiling.md`.
  - **Oversize-request 413 guard REMOVED 2026-06-24** (`VLLM_MAX_REQUEST_BODY_BYTES=0`, kept after the batch revert — independent and safe): the 512 KB HTTP 413 reject turned slow z.ai→qwen fallback overflow into hard-fails → **cluster-wide cascade outage**. Real lever is the claudish-side concurrency cap (no-fallback routing), NOT body-size rejection. `error_source_capture` middleware stays ON for logging; set the env >0 to re-enable the reject if ever needed.
  - Quality upgrades over Qwen3.5 (upstream): SWE-bench 70→73.4, Terminal-Bench 40.5→51.5, NL2Repo 20.5→29.4, QwenWebBench 978→1397
- **Why Genesis (Option B) was the actionable path**: The 2026-05-06 stock attempt with `vllm-qwen36-tq:nightly-e47c98ef-patched1` + `--kv-cache-dtype turboquant_k8v4` crashed on first chunked-prefill continuation ([vllm#41726](https://github.com/vllm-project/vllm/issues/41726): `AssertionError turboquant_attn.py:720`, requires 29.73 MB / has 16.31 MB). Upstream candidate fix [PR #40798](https://github.com/vllm-project/vllm/pull/40798) was OPEN+BLOCKED with no ETA, so we adopted [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) v7.72.x downstream patch tree whose P22/P38 family fixes exactly that crash (externally confirmed by `xyehya` in vllm#41726).
- **Promotion path 2026-05-16 → 2026-05-17**:
  - v2a/v2d/v2e iterations: tuned KV/concurrent tradeoffs
  - v2f (Genesis-TQ + vision): PASSED active load benches but regressed on +55min IDLE soak (shm_broadcast.py:733 deadlock at 05:29Z). Auto-rolled back to FP8 baseline 05:35Z per user rule.
  - v2g = v2f + `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer sampler had its own JIT autotune path corrupting CPython `_thread.lock` descriptor under load). Deployed 06:29Z. **35h soak clean → promoted 2026-05-17.**
  - Iteration log: `myia_vllm/_iteration_log/genesis_tq_night_log.md`
- **Prior FP8 baseline retained for instant rollback**: `vllm-qwen36-shmpatched:nightly-f6983f01d-patched1` (Apr 06 + shm_broadcast.py patch + transformers>=5.0). Image still local, profile `medium-qwen36-moe.yml` untouched. Rollback: `docker compose -f medium-qwen36-genesis-tq.yml down && docker compose -f medium-qwen36-moe.yml up -d`.
- **Spec-decode experiments — RETIRED to documentation-only (2026-05-17/18)**: 4 attempts across DFlash + MTP, 4 crashes, 2 distinct upstream-trackable bugs + 1 deferred deepcopy poison family. Stay on v2g (829 tok/s aggregate N=16 is excellent for our multi-user workload anyway). Profiles preserved on disk for re-test once upstream patches land.
  - **Att1 — DFlash-v2g Attempt 1** (`medium-qwen36-genesis-tq-dflash.yml`, K=15, with `--default-chat-template-kwargs`): deepcopy/segfault in tokenizer `init_kwargs` (`copy.py:128 memo.get()` with memo being a `code` object). Hypothesis at the time: chat-template-kwargs injected a code object. **Falsified later by Att4**.
  - **Att2 — DFlash-v2g Attempt 2** (K=7, dropped chat-template-kwargs, max-num-batched-tokens 8192): torch.compile fake-tensor mismatch dim 0 (`mul(FakeTensor(65536,128), FakeTensor(16*s59,128))`).
  - **Att3 — DFlash-v2g Attempt 3** (K=7, max-num-batched-tokens 4096): `RuntimeError: Worker failed with error 'Page size mismatch after block_size adjust: 1136064 != 3408192'` (ratio 3:1 exact). TurboQuant k8v4 page (target) ↔ FLASH_ATTN page (drafter) interop gap. Genesis PN9 self-retires when upstream #39930 merged but doesn't close the page-layout gap. **Same family as upstream [vllm#41559](https://github.com/vllm-project/vllm/issues/41559)** — datapoint posted [issuecomment-4472523463](https://github.com/vllm-project/vllm/issues/41559#issuecomment-4472523463). Fix path: PR [#39995](https://github.com/vllm-project/vllm/pull/39995) (DFlash + FlashInfer + FP8 KV) but doesn't cover TurboQuant.
  - **Att4 — FP8+MTP Attempt 4** (`medium-qwen36-fp8-mtp.yml`, `Qwen/Qwen3.6-35B-A3B-FP8` model, MTP method): SAME deepcopy/segfault as Att1, different call site (`transformers/configuration_utils.py:678 → copy.deepcopy(kwargs)` in AutoConfig.from_pretrained). **Falsifies the Att1 chat-template-kwargs hypothesis** (Att4 doesn't have that flag and still crashes). Real pattern: spec-decode + Qwen3.6 multimodal + Genesis image triggers dual-config-load deepcopy poison (similar to vllm#35104 PyCFunction family). **NOT FILED YET** — must repro on stock vLLM `01d4d1ad3` without Genesis layer first to disambiguate venue (vllm-project vs Sandermage).
  - **v2g working datapoint posted**: [vllm#41726 issuecomment-4472611221](https://github.com/vllm-project/vllm/issues/41726#issuecomment-4472611221) — confirms xyehya's 2026-05-09 finding (Genesis P22+P38 fixes the `_continuation_prefill` crash) on Ada SM89 with 35h+ production soak. Mentions `VLLM_USE_FLASHINFER_SAMPLER=0` trap.
  - Profiles kept on disk as documentation: `medium-qwen36-genesis-tq-dflash.yml` (Att2/Att3 config), `medium-qwen36-genesis-tq-mtp.yml` (INERT — AWQ target drops MTP heads), `medium-qwen36-fp8-mtp.yml` (Att4 — re-test if Genesis ships a patch for the dual-config-load deepcopy).
  - Iteration logs: `myia_vllm/_iteration_log/dflash_v2g_attempt{1,2,3}/`, `myia_vllm/_iteration_log/fp8_mtp_attempt4/`, `myia_vllm/_iteration_log/v2g_promotion_followup/`.
- **GPU 2 — FULLY FREED 2026-05-01** for CoursIA training jobs (queue 5-7d : QC ML strategies, Sudoku large, RL extension). State: **78 MiB / 24564 MiB** (driver baseline only, no resident process). OmniCoder-9B archived 2026-04-30 (`myia_vllm/archives/2026/mini-omnicoder.yml.archived-2026-04-30`, `Dockerfile.omnicoder.archived-2026-04-30`). Reasoning: ai-01 piloting double-track (coordination + training) sprint sustainable. Issue CoursIA #626.
  - ⚠️ **GPU-2 container facts (measured 2026-09-12, eval window #35)** — apply to any future eval on this slot (K2 re-test, DSpark, sanity servers):
    1. **WSL2/DXG paravirtualization tax = 2.02 GiB**: a BARE CUDA context in a `--gpus device=2` container reports 21.97/23.99 GiB free while the host `nvidia-smi` shows 45 MiB. vLLM's startup free-memory check caps **gpu-util ≈ 0.91** deterministically (0.92 fails).
    2. **Never `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` here**: deterministic `RuntimeError: CUDA driver error: device not ready` inside the Marlin repack (`marlin_utils.py:518`, advanced indexing during `process_weights_after_loading`); a plain alloc passes, the kernel+VMM combo does not.
    3. **HF hub nests one extra level under WSL**: models live in `\\wsl.localhost\Ubuntu\home\user\vllm\.cache\huggingface\hub\hub\models--…`. Prod mounts the PARENT onto `/root/.cache/huggingface` (HF_HOME); mounting the parent onto `…/huggingface/hub` puts models one level too deep → `LocalEntryNotFoundError` under `HF_HUB_OFFLINE=1`. Mount `hub\hub` → `/root/.cache/huggingface/hub`.
    4. `docker rm -f` on an engine → its CUDA context needs ~10 s to release VRAM; an immediate re-run trips the free-memory check (false "insufficient free memory"). Settle delay in the serve script.
    5. Reusable skeleton: `myia_vllm/_iteration_log/qwen38_27b_gpu2/serve_qwen38.sh` (strict pinning, port 5022 keyless, all traps in comments) + `quality_27b.py`/`bench_27b.py`/`paired_27b.py` (env-keyed paired runners).
- **Kokoro TTS migrated to po-2023** (2026-05-01): final endpoint `https://tts.myia.io/kokoro/v1` (path-strip via IIS, bearer auth required, sleep-mode like Orpheus). 7/7 OWUI tenants switched (myia, epf, epf-genai, ece, esg, epita, pauwels — synthesis tested HTTP 200 with `ff_siwis` voice, MP3 ID3 valid). Service `kokoro-tts` removed from `myia-open-webui` compose stack on ai-01, volume `kokoro-data` purged. Quirks noted: po-2023 endpoint serves `/v1/voices` (200) but NOT `/v1/audio/voices` (404, non-standard); `/v1/audio/speech` works fine (used by OWUI). Commit `d37df601b` (myia-open-webui workspace).
- **Orpheus TTS moved to po-2023** (2026-03-18): `https://orpheus-tts.myia.io/v1/audio/speech`
- **OWUI sampling calibration** (2026-03-21): 8 model wrappers calibrated for AWQ Q4 (Reddit + HF + local benchmarks). Key Q4 adjustments: temp 1.0→0.7, pp capped at 1.5 (not 2.0), rp 1.05-1.1 anti-bleed, min_p 0.01-0.05. Bug fixed: `-fast` had missing `enable_thinking: false`.
- **SK Agent MCP server** uses Qwen3.6-35B-A3B (port 5002, updated 2026-04-17)
- **roo-state-manager condensation** uses Qwen3.6-35B-A3B via `OPENAI_CHAT_MODEL_ID` env var
- **Roo "simple" apiConfig** uses Qwen3.6-35B-A3B (in `roo-extensions/roo-config/model-configs.json`)
- **vLLM versions** (updated 2026-04-17):
  - GPUs 0,1 (Qwen3.6 MoE): **pinned to nightly `f6983f01de2bf2e92ab468fa735ebac39cddd670`** (Apr 06, v0.19.1.dev45+gf6983f01d). Apr 15 nightly hangs at compile init; Apr 16 nightly has broken transformers import (missing pandas). Apr 04 nightly has shm_broadcast PyCFunction bug.
  - GPU 2 (OmniCoder): nightly Apr 04 (v0.19.1) + transformers 5.5.0
- **API keys rotated** (2026-03-13): all 3 keys regenerated after accidental git exposure, hardcoded keys removed from 13 files
- **Sampling optimization** (2026-03-08): presence_penalty 1.5 reduces repetition 2-3x with no speed impact
- **OWUI routing for Roo: ABANDONED** (2026-03-10): 83+ MCP tools overwhelm OWUI pipe. OWUI wrappers exist for direct OWUI users only.
- **Models rejected**: Qwen3.5-27B Dense (2026-02-25), GPTQ-Int4 (2026-03-03), BNB NF4 distill (2026-03-13), Qwen3.5-27B-Claude-Opus-Distilled-v2 AWQ (2026-04-05: 56 tok/s decode -36%, tool calling broken with qwen3_coder, concurrent -53%, KV cache 106K vs 324K), **Qwen3.8-27B AWQ — GPU-2 TP=1 slot INFEASIBLE (2026-09-12, issue #35 closed)**: 12 boots, 7 failure modes; best case leaves **+0.15 GiB KV (~5K tokens)** on one 4090. Measured: 2.02 GiB DXG tax, ~16 GiB weights, ~2.4 activations, ~3.1 GiB cudagraph estimate, and MTP-3 adds a fixed 2.37 GiB BF16 drafter embedding (`qwen3_5_mtp.py:244`) — structurally impossible. Combined with the 08-20 TP=2 verdict (N=16 5.7×-6.3× slower than prod MoE), **the 27B line is fully explored and closed** on this host (quality tier was never in question: SWE-bench Pro +8.2). Record + reusable harness: `myia_vllm/_iteration_log/qwen38_27b_gpu2/NOTES.md`, PR [#41](https://github.com/jsboige/vllm/pull/41)., **Qwen3.6-27B dense AWQ INT4 single-GPU** (2026-04-24: 19.06 GB weights don't fit single RTX 4090 24GB with KV cache headroom. Universal upstream guard, not Ampere-specific. Profile `mini-qwen36-27b.yml` retained for TP=2 deployment or future 4B/9B variants), **Qwen3.6-27B dense + TurboQuant K8V4 TP=2** (2026-05-06 same-day cutover attempt: decode -50%, 5-concurrent -49%, tool +40% — all 3 rollback thresholds tripped; quality gains within sampling noise. Profile archived to `archives/2026/medium-qwen36-27b.yml.rejected-2026-05-06`), **Qwen3.6-35B-A3B MoE + TurboQuant K8V4** (2026-05-06: booted with 1.49M-token KV but EngineCore crashes on first chunked-prefill continuation — upstream bug [vllm#41726](https://github.com/vllm-project/vllm/issues/41726), candidate fix [PR #40798](https://github.com/vllm-project/vllm/pull/40798). Re-test when #40798 merges)
- **DFlash speculative decoding** (evaluated 2026-04-24, NOT deployed in prod): drafter `z-lab/Qwen3.6-35B-A3B-DFlash` (8-layer Qwen3 dense, BF16, ~0.5B). Block diffusion, block_size=16, target_layer_ids=[1,10,19,28,37]. Profile `medium-qwen36-moe-dflash.yml` retained as opt-in. **Empirical bench vs baseline 3.6**: single-user code +23% (131 vs 107 tok/s), single-user reasoning **+94%** (226 vs 116.5 tok/s), single-user code long +59% (170 tok/s); 5-user concurrent **-15%** (315 vs 369 tok/s aggregate). **Trade-offs**: requires `--attention-backend flash_attn` which **rejects fp8 KV cache** → max-model-len capped at 160K (vs baseline 262K), KV cache 93K tokens (vs baseline 322K), max concurrency 1.15× (vs baseline 4.69×). **Acceptance rate confirmed compatible with AWQ target**: 26-47% per-position, 4-7 tokens accepted per draft step. **The "0% acceptance with AWQ" claim previously in MEMORY.md was a hallucination** — only MTP (multi-token prediction, vLLM `--speculative-config method=mtp`) shows 0% with AWQ (tested on GLM-4.6-AWQ, see CLAUDE.md "Critical Configuration Notes" #4). DFlash drafter is a separate BF16 model with its own quant config via `vllm.model_executor.models.utils.get_draft_quant_config`, hence AWQ target compatibility. **Decision (2026-04-24)**: rollback to baseline — concurrent throughput + 262K context matter more for our Roo orchestrator + multi-student workload than single-user speedup. Profile retained at `myia_vllm/configs/docker/profiles/medium-qwen36-moe-dflash.yml` for benchmarks or future single-user-heavy use cases.
- **TurboQuant migration — Option B is the actionable path (2026-05-14 review)**: Workload is **multi-user long context** (OWUI/Roo multi-tenant + Claude Code routing) → KV cache **capacity** is the bottleneck → TurboQuant (×4.6 KV: 1.49M vs 322K tokens) is the priority lever (not DFlash, which helps single-user but costs −15% concurrent). **Two paths:**
  - **Option A (upstream) — NOT VIABLE**: PR [#39931](https://github.com/vllm-project/vllm/pull/39931) merged 2026-05-05 unblocks hybrid models but exposes the `_continuation_prefill` workspace crash ([#41726](https://github.com/vllm-project/vllm/issues/41726), hit on our MoE 2026-05-06). Candidate fix [PR #40798](https://github.com/vllm-project/vllm/pull/40798) is **OPEN + BLOCKED** (verified 2026-05-14, no ETA — `gaby`: "TurboQuant does not work at all without this").
  - **Option B (ACTIONABLE)**: [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) downstream patch tree — v7.72.x, 126 patches, 1958 tests, explicitly targets Qwen3.6-35B-A3B + TurboQuant k8v4 + 256K context. Patches **P22+P38** fix our exact crash — externally confirmed by `xyehya` in issue #41726 (2026-05-09: "patches P38 and P22 ... fixed my issue"). **Next step**: build a genesis-patched image (base nightly ≥2026-05-05 + genesis installer + our transformers>=5.0/shm patches), re-run the 2026-05-06 MoE+TQ attempt against real ~30K-prefix OWUI traffic. Full detail in MEMORY.md `project_turboquant_hybrid_status.md`.
  - Both 27B Dense + TQ (perf regression, rejected) and MoE 35B-A3B + TQ (the crash above) were tried 2026-05-06. Issue [#40807](https://github.com/vllm-project/vllm/issues/40807) (TurboQuant + spec-dec MTP + chunked-prefill) does NOT affect us — no spec-dec in prod.

## Related Resources

- [vLLM Documentation](https://docs.vllm.ai)
- [Qwen3.6-35B-A3B-AWQ on HuggingFace](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)
- [Qwen3.6 Official Blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- [Qwen3.5-35B-A3B-AWQ on HuggingFace](https://huggingface.co/cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit) (previous model)
- [GLM-4.7-Flash on HuggingFace](https://huggingface.co/zai-org/GLM-4.7-Flash) (legacy)
- [Unsloth vLLM Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide)
