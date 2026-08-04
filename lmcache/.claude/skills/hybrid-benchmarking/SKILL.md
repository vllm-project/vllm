---
name: hybrid-benchmarking
description: Benchmark a hybrid-attention model across the LMCache performance ladder (vLLM no-hybrid-allocator → hybrid allocator + prefix caching → hybrid allocator + LMCache) and produce a decode-throughput / TTFT / cache-hit-rate comparison. Use when asked to benchmark, demo, or quantify the value of LMCache caching for a hybrid model.
allowed-tools: Bash, Read, Grep, Glob, Write
argument-hint: "<hf-model-id> [--gpus N] [--tp N] [--output-dir DIR]"
---

# Hybrid-Model Benchmarking — the LMCache performance ladder

Given a hybrid-attention model + hardware, calibrate a `long-doc-qa` workload, run the vLLM
config ladder, and write `report.md` (launch commands + decode-throughput / TTFT / TPOT /
hit-rate table + a rationale for every tuned number). Validated on `google/gemma-4-31B-it`
(sliding-window), `Qwen/Qwen3.6-27B` (Mamba/GDN), and `deepseek-ai/DeepSeek-V4-Flash`
(sparse-MLA + fp8).

> **Run fully autonomously.** Log each decision and keep going through all runs and the
> report — do **not** pause for approval/calibration confirmation. Only stop on a hard
> blocker (model not given, gated weights missing). The user interrupts if something's off.

## Inputs

`$ARGUMENTS`: positional model id (required, e.g. `Qwen/Qwen3.6-27B`); `--gpus N` / `--tp N`
(default: detect via `nvidia-smi -L`, tp = gpus); `--output-dir DIR` (default
`./hybrid-bench-<slug>/`). On a shared box, pick free GPUs / non-default ports and never kill
processes you didn't start.

## The ladder

Same `long-doc-qa` workload, four vLLM configs — each isolates one variable:

| Run | config | what it shows |
|--|--|--|
| **A** | allocator off, prefix caching off | baseline: smallest batch → lowest decode tput, worst TTFT |
| **B** | allocator on, prefix caching off | isolates allocator: packs ~8× more tokens → much higher decode tput |
| **C** | + `--enable-prefix-caching` (no LMCache) | GPU prefix cache ~0 % hit under batch saturation → ≈ Run B |
| **D** | + `LMCacheMPConnector` + `lmcache server` | CPU pool serves prefixes → ~100 % hit → lowest TTFT, highest decode tput |

Two counterintuitive facts: **(1)** for sliding-window hybrids the allocator gives the
*larger* batch (sliding layers keep only a window-sized KV slice → ~`num_layers /
num_full_layers`× more tokens, ≈8× for Gemma-4), so Run A (all layers full) is the
*smallest*/slowest. **(2)** Run C's GPU prefix cache is ~0 % hit because the active batch's KV
(`batch × (doc+output)`) fills the whole pool — no blocks left to *retain* prefixes. That
saturation gap is exactly what LMCache's CPU pool fills.

> **Mandatory-allocator models → 3 runs (A ≡ B).** Mamba/GDN (Qwen3.5/3.6, Qwen3-Next) and
> DeepSeek-V4 sparse-MLA have no allocator-off baseline — `--disable-hybrid-kv-cache-manager`
> is rejected (Mamba) or fails at runtime (DeepSeek: CuTe sparse-attn needs C128/block_size=8).
> Drop Run A; the story is the cache (C→D). The full 4-run structure applies only to
> sliding-window hybrids (Gemma 3/4, gpt-oss).

## Make the criteria visible

Default `long-doc-qa` is prefill-bound, which *hides* the decode gap. **Always** pass
`--ignore-eos --ldqa-max-output-length <N>` (≈2048 at L=24 000) → decode-bound *and* deterministic
(identical total output tokens/run, so decode-throughput numbers are reproducible). Size the
working set to **overflow the GPU pool but fit the LMCache pool with margin** (step 3).

## Procedure

**0 — Versions + env.** Record for the report (benchmarks are meaningless without them):
`vllm --version` (→ commit); LMCache `git rev-parse --short HEAD` + `git status --short` (note
any local patch, e.g. the `--ignore-eos` / `--ldqa-max-output-length` bench patch → record "base
commit + with `<patch>`"). Export `HF_HOME` if weights live off-default — for **every** command
(env doesn't persist across tool calls). Read the model's `docs/source/recipes/*.rst` for
validated flags + quirks (else `docs/source/mp/hybrid_models.rst`, and warn it's unvalidated).
Confirm weights are present.

**1 — Classify → decide Run A.**

| Family | Examples | Run A |
|--|--|--|
| Sliding-window + full | Gemma 3/4, gpt-oss | `--disable-hybrid-kv-cache-manager --no-enable-prefix-caching` |
| Mamba / GDN | Qwen3.5/3.6, Qwen3-Next | allocator mandatory → no Run A; caching is non-bit-exact (GDN) |
| Sparse-MLA + fp8 | DeepSeek-V4 | allocator mandatory (CuTe needs C128/block_size=8) → no Run A |
| Dense | Llama, Mistral | not a hybrid; allocator distinction is moot |

Classify via the recipe, else HF `config.json` `architectures`/`layer_types`.

**2 — Probe pools.** Launch with `--max-model-len auto` (**never pin small** — it shrinks the
sliding-window pool ~5×). From the startup log record GPU pool tokens for hybrid (`P_B`) and
hybrid-off (`P_A`); their ratio is the allocator gain. For LMCache, once the server+connector
engine is up read `cache_size_per_token` from `/status` → `tokens_per_gb = 1024³ / it`; pass it
via `--tokens-per-gb-kvcache` so `num_documents` is identical across runs. (Real on-pool storage
≈ 1.4× this — size against that, not the raw number.)

**3 — Calibrate.** Util < 0.9; lower it so GPU memory is the binding constraint (0.50 on H200;
**0.35 for fp8 DeepSeek** — fp8 makes the GPU pool huge). Pick `L` (≤ `P_A`, a sane hybrid batch
`P_B/(L+output)`, a meaningful prefix). `num_documents = floor(kv_cache_volume_gb × tokens_per_gb
/ L)` chosen so it **overflows GPU** (`num_docs × L > P_B`) and **fits LMCache with margin**
(real storage ≈ 1.4×(tokens / tokens_per_gb) ≤ ~0.6 × `l1_size_gb`, watermark 0.95). `l1_size_gb`
≤ host RAM and /dev/shm. Set `--ldqa-max-output-length` so decode dominates (≈2048 @ L=24 000). Log
the operating point and proceed.

**4 — Freeze one shared config.**
```bash
lmcache bench engine --engine-url http://localhost:8000 --workload long-doc-qa \
    --model <model> --tokens-per-gb-kvcache <N> --kv-cache-volume <GB> \
    --ldqa-document-length <L> --ldqa-query-per-document 1 \
    --ldqa-num-inflight-requests <≥ hybrid batch> --ldqa-max-output-length <out> --ignore-eos \
    --no-interactive --export-config "$OUT/shared.json"
```
Replay every run with `--config "$OUT/shared.json"` (so `num_documents` is byte-identical, even in A/B with no server).

**5 — Run.** Start `lmcache server` **first** so its pinned pool pre-expands (lazy ~80 GiB/min) before D:
```bash
lmcache server --port 5560 --http-port 8090 --prometheus-port 9099 \
    --l1-size-gb <GB> --eviction-policy LRU --eviction-trigger-watermark 0.95 [--chunk-size N]
```
Each run: launch engine → poll `curl -sf :8000/health` → snapshot metrics (step 6) → bench → teardown. `<base>` = recipe flags (`--tensor-parallel-size`, `--trust-remote-code`, …).
```bash
# Run-flags appended to <base>:
#   A  --disable-hybrid-kv-cache-manager --no-enable-prefix-caching   (skip for mandatory-allocator families)
#   B  --no-enable-prefix-caching
#   C  --enable-prefix-caching
#   D  C + --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":5560,"lmcache.mp.mq_timeout":900}}'
vllm serve <model> <base> --gpu-memory-utilization <U> --max-model-len auto --port 8000 <run-flags>
lmcache bench engine --engine-url http://localhost:8000 --config "$OUT/shared.json" \
    --no-interactive --json --output-dir "$OUT/run-<A|B|C|D>"
```
**Run D at steady state:** bench **twice** — run #1 primes, run #2 measures; report run #2's delta. First confirm the pool is pinned (`/status memory_total_bytes ≈ l1`) and the engine registered (`cache_context_meta` non-empty).
**Teardown by PID, not pattern:** `pkill -f "vllm serve …"` self-matches (exit 144) and orphans EngineCore/Workers that keep holding GPU memory. Kill the main PID + worker PIDs from `nvidia-smi --query-compute-apps`; kill `lmcache server` cleanly; verify GPUs return to ~0 MiB.

**6 — Metrics.** Bench JSON `results`: `mean_ttft_ms`, `p90/p99_ttft_ms`, `output_throughput`
(decode), `input_throughput` (prefill), `mean_decode_speed` (TPOT = 1000/it),
`total_output_tokens`. **Hit rate from vLLM `:8000/metrics`** (LMCache Prometheus is often
disabled in MP builds): `prefix_cache_*` (GPU-local), `external_prefix_cache_*` (LMCache).
Snapshot before/after each bench; for D report the measure-run delta and prefer the
`external_prefix_cache_*` token counters.

**7 — Report `$OUT/report.md`.** Header: versions, hardware, TP. **(a)** exact launch commands
(server + runs + bench; note D's prime+measure double-run). **(b)** table: decode tput,
mean/p90/p99 TTFT, TPOT, prefill tput, hit rate, wall-clock, GPU pool tokens, effective batch.
**(c)** rationale + findings, validating the criteria explicitly: A→B decode jumps (≈8× batch);
B→C ≈ no change (0 % hit, saturated); C→D TTFT collapses + decode rises (hit ~100 %).

## Family turnkey configs

**Mamba/GDN** (Qwen3.5/3.6, …): every run uses `--mamba-cache-mode align
--max-num-batched-tokens 2N-1`, where N is the unified block size (vLLM logs `Setting attention
block size to N tokens`; also set server `--chunk-size N`). **Use 2N-1, not N** — at exactly N a
decoding request consumes the per-step budget, so a new request's block-aligned prefill rounds
to 0 tokens → serial execution (Running:1, ~7× slower; verified by single-variable A/B). Also
`--max-num-seqs ≤ #Mamba cache blocks` vLLM reports, or CUDA-graph capture fails.

**DeepSeek-V4 sparse-MLA + fp8**: `export CUDA_HOME=/usr/local/cuda-<ver>
PATH=$CUDA_HOME/bin:$PATH` (sparse-attn JIT fails on the default toolchain) for **every** launch;
flags `--kv-cache-dtype fp8_ds_mla --trust-remote-code --tokenizer-mode deepseek_v4
--enable-expert-parallel`. **Util ≈0.35** (at 0.80 the fp8 GPU pool swallows the whole working
set → nothing for LMCache to serve); large `--l1-size-gb` (full per-token state ~62 KB,
`tokens_per_gb ≈ 16 800`).

## Troubleshooting

| Symptom | Fix |
|--|--|
| B decode ≈ A (not ≫) | run is prefill-bound — add `--ignore-eos`, raise `--ldqa-max-output-length` |
| D hit ≪ 100 % | working set hit the watermark mid-fill — pre-pin before D; raise `--l1-size-gb`; shrink working set (real ≈1.4× est.) |
| D aborts "register_kv_caches within 300s" | big pinned pool → set `lmcache.mp.mq_timeout` ≥ 900 |
| reg hangs "Wrapping N KV tensors for IPC" | server/connector transfer-mode mismatch — use default (auto) on both |
| pool ~5× too small | `--max-model-len` pinned — use `auto` |
| `exit 144` / leaked GPU after kill | `pkill` self-matched / orphaned workers — kill by PID (main + `nvidia-smi` workers) |
| "CUDA compiler … headers are incompatible" (DeepSeek) | set `CUDA_HOME` matching torch's CUDA, every launch |
| "CuTe DSL … only supports C128 block_size=8" | tried to disable the allocator on DeepSeek — don't; it's mandatory (3 runs) |
| fp8: D ~100 % but no saturation / nothing to serve | GPU pool fits the whole working set — lower util (≈0.35) |

## Flag cheat-sheet

- Allocator off: `--disable-hybrid-kv-cache-manager` (sliding-window/dense only).
- Prefix caching: `--no-enable-prefix-caching` / `--enable-prefix-caching`.
- Deterministic decode-bound: `--ignore-eos --ldqa-max-output-length <N>`.
- Pools: `--max-model-len auto`; `lmcache server --l1-size-gb <GB> --eviction-trigger-watermark 0.95` (start early to pre-pin).
- LMCache wiring: `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":5560,"lmcache.mp.mq_timeout":900}}'`.
- Mamba/GDN: `--mamba-cache-mode align --max-num-batched-tokens 2N-1`, server `--chunk-size N`.
- DeepSeek sparse-MLA: `CUDA_HOME` set; `--kv-cache-dtype fp8_ds_mla --tokenizer-mode deepseek_v4 --enable-expert-parallel`; util ≈0.35.
- Hit rate: vLLM `:8000/metrics` (`prefix_cache_*` GPU-local, `external_prefix_cache_*` LMCache).
- Teardown: by PID; verify GPUs ~0 MiB.
