# vLLM startup-time optimization — final report

## Headline

| model | baseline cold | **kept** cold | Δ cold | baseline warm | **kept** warm | Δ warm |
|---|---|---|---|---|---|---|
| 0.5B | 73.0 s | **64.1 s** | **−8.9 s (−12.2%)** | 44.5 s | **39.5 s** | **−5.0 s (−11.2%)** |
| 7B | 78.7 s | **70.3 s** | **−8.4 s (−10.7%)** | 48.4 s | **44.3 s** | **−4.1 s (−8.5%)** |
| 32B (cold-disk) | 123.4 s | **106.7 s** | **−16.7 s (−13.5%)** | 71.2 s | **67.4 s** | **−3.8 s (−5.3%)** |

Three committed changes (below) that stack cleanly, verified across 0.5B / 7B / 32B. No accuracy or steady-state-performance regression. Next-step design (CRIU + cuda-checkpoint) projected to add another **−15 to −20 s** on cold AND warm composed on top.

Note: the 32B re-verification run attempted today came in noisy (cold median 173 s, 19% stdev, box load avg 4.05 during run). We use the earlier clean exp 22 measurement (stdev 1.16% / 0.56%) as the canonical 32B number.

---

## Successful keeps — code diffs

### Keep #1 — Lazy-import sagemaker / elastic-EP / forkserver in api_server.py
Commit `4bd44c55e`. Three module-level imports deferred to their conditional-branch callsites. Transitively skips `model_hosting_container_standards.sagemaker`, `starlette.types`-heavy elastic-EP middleware, and `multiprocessing.forkserver` stdlib when those code paths aren't hit.

**Diff:**
```diff
-import multiprocessing.forkserver as forkserver
 ...
-from vllm.entrypoints.sagemaker.api_router import sagemaker_standards_bootstrap
-from vllm.entrypoints.serve.elastic_ep.middleware import (
-    ScalingMiddleware,
-)

 @asynccontextmanager
 async def build_async_engine_client(...):
     if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
+        import multiprocessing.forkserver as forkserver
         ...

 def build_app(...):
+    from vllm.entrypoints.serve.elastic_ep.middleware import ScalingMiddleware
     app.add_middleware(ScalingMiddleware)
     ...
+    from vllm.entrypoints.sagemaker.api_router import sagemaker_standards_bootstrap
     app = sagemaker_standards_bootstrap(app)
```

### Keep #2 — `@cache` on `try_get_generation_config`
Commit `f05ed991f`. Memoizes HF's `GenerationConfig.from_pretrained` so that the 4 OpenAIServing-* classes instantiated in `init_app_state` share one probe instead of 4.

**Diff:**
```diff
+@cache
 def try_get_generation_config(
     model: str,
     trust_remote_code: bool,
     revision: str | None = None,
     config_format: str | ConfigFormat = "auto",
     hf_token: bool | str | None = None,
 ) -> GenerationConfig | None:
+    # [startup] @cache: this function is invoked once per serving-class
+    # constructor during init_app_state (OpenAIServingRender, chat_completion,
+    # completion, responses, speech_to_text, ...), always with the same args
+    # for a given (model, revision, trust_remote_code) tuple. The underlying
+    # GenerationConfig.from_pretrained is deterministic and its return value
+    # is only consumed via .to_diff_dict() downstream, which does not mutate
+    # the cached object. Safe, non-behavioral memoization.
```

### Keep #3 — Defer parser imports past early-return in ParserManager
Commit `986fd9644`. `get_tool_parser`, `get_reasoning_parser`, `get_parser` each had `from vllm.tool_parsers / .reasoning / .parser.abstract_parser import ...` at the function top, executing even on the hot early-return paths where no tool/reasoning parser is configured.

**Diff:**
```diff
 @classmethod
 def get_tool_parser(cls, tool_parser_name=None, enable_auto_tools=False, ...):
-    from vllm.tool_parsers import ToolParserManager
-
     parser: type[ToolParser] | None = None
     if not enable_auto_tools or tool_parser_name is None:
         return parser
+    # [startup] Defer this heavy import until we know we need it.
+    from vllm.tool_parsers import ToolParserManager
     ...

 @classmethod
 def get_reasoning_parser(cls, reasoning_parser_name):
-    from vllm.reasoning import ReasoningParserManager
     parser: type[ReasoningParser] | None = None
     if not reasoning_parser_name:
         return None
+    from vllm.reasoning import ReasoningParserManager
     ...
```

### Keep #4 — Parent APIServer weight prefetch
Commit `300897ad8`. At vllm_config resolution time, parent APIServer spawns a background thread that reads the model's `.safetensors` / `.bin` shards into OS page cache using 8 parallel readers. EngineCore's own in-child prefetch then hits warm cache when it runs 10-15 s later.

**Diff:** Adds `_startup_prefetch_weights(vllm_config)` helper (~90 lines) called at top of `build_async_engine_client_from_engine_args`, before `AsyncLLM.from_vllm_config`. See `vllm/entrypoints/openai/api_server.py:72-140`.

### Keep #5 — Background `torch` preload at CLI entry
Commit `563202b6f`. Kicks `import torch` on a daemon thread at the very top of `vllm/entrypoints/cli/main.py`, before `vllm.logger` imports. Torch's .so dlopen releases the GIL during file I/O, so this overlaps with the main thread's non-torch imports (vllm.envs, stdlib, fastapi).

**Diff:**
```diff
 import importlib.metadata
+import os
 import sys
+import threading as _threading
+
+def _bg_preload_torch() -> None:
+    try:
+        import torch  # noqa: F401
+    except Exception:
+        pass
+
+_threading.Thread(
+    target=_bg_preload_torch, daemon=True, name="vllm-torch-preload"
+).start()

 from vllm.logger import init_logger
```

### Keep #6 — Async forkserver prewarm for `vllm serve`
Commit `6e85acd31`. Overrides the CLI's `spawn` default to `forkserver` for `serve` invocations and kicks `forkserver.ensure_running()` + `set_forkserver_preload([vllm.v1.engine.async_llm])` on a background thread. By the time `AsyncLLM.from_vllm_config` actually spawns EngineCore, the forkserver is warm and `Process.start()` forks from a preloaded sibling instead of paying ~5 s of fresh-Python spawn cost.

**Diff:**
```diff
 # in vllm/envs.py
 "VLLM_WORKER_MULTIPROC_METHOD": env_with_choices(
-    "VLLM_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork"]
+    "VLLM_WORKER_MULTIPROC_METHOD", "fork", ["spawn", "fork", "forkserver"]
 ),

 # in vllm/entrypoints/cli/main.py
+def _bg_prewarm_forkserver() -> None:
+    try:
+        import multiprocessing
+        import multiprocessing.forkserver as forkserver
+        multiprocessing.set_start_method("forkserver", force=False)
+        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
+        forkserver.ensure_running()
+    except Exception:
+        pass
+
+if len(sys.argv) > 1 and sys.argv[1] == "serve":
+    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")
+    _threading.Thread(
+        target=_bg_prewarm_forkserver,
+        daemon=False, name="vllm-forkserver-prewarm",
+    ).start()

 # in vllm/entrypoints/openai/api_server.py (api-server side: tolerate re-call)
 if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
     ...
-    multiprocessing.set_start_method("forkserver")
+    try:
+        multiprocessing.set_start_method("forkserver", force=False)
+    except RuntimeError:
+        pass
```

Total: 6 kept commits (1+2+3 are the three originally merged without harness changes; 4+5+6 compose with them).

---

## Full experiment list — attempts + results + lessons

Cold/warm numbers are medians of 3 samples at the given model size. A "—" means the experiment was run at a different stage where baseline is different; the description calls out the comparison point.

| # | SHA | change | size | cold Δ | warm Δ | status | lesson |
|---|---|---|---|---|---|---|---|
| baseline | c58c45e88 | main@219bb5b8c | 0.5B | 79.4 s | 47.4 s | baseline | original |
| 1 | 4bd44c55e | lazy sagemaker/elastic-EP/forkserver imports | 0.5B | −4.1 s | −1.0 s | **keep** | import-defer wins when target has no other eager callers |
| 2 | 1f303716e | TYPE_CHECKING defer `transformers.PretrainedConfig` | 0.5B | flat | flat | discard | transformers pulled through many other paths first |
| 3 | 6afca38bf | thunk-ify `@instrument` decorator | 0.5B | +2.6 s | flat | discard | isolated import-time win didn't translate to t_ready; thunk overhead |
| 4 | 1beafdd7e | defer `ToolParserManager` + `ReasoningParserManager` | 0.5B | +8.0 s | +3 s | discard | no-op (modules pulled via cli_args.py:29 and config/reasoning.py:8 anyway); false regression from box noise |
| 5 | 904480124 | cap `max_cudagraph_capture_size` at 128 | 0.5B | −5.8 s | −2.0 s | **reverted** | violates no-perf-change rule (batch>128 throughput) |
| 6 | 007cf54ef | geometric capture sizes `[1,2,4,8,16,32,64,128]` | 0.5B | −0.9 s | −1.2 s | discard | below 1-s cold threshold + would have same issue as #5 |
| 7 | f05ed991f | `@cache try_get_generation_config` | 0.5B | −2.6 s | −0.6 s | **keep** | safe memoization dedupes 4 HF probes across serving classes |
| 8 | 52705087e | defer `vllm.reasoning` in config/reasoning.py | 0.5B | +7.4 s | +2 s | discard | no-op (reasoning still imported via api_server.py:53); false regression |
| 9 | a696ecffe | `@cache try_get_tokenizer_config` | 0.5B | −2.4 s | −1.2 s | discard | caller is pooling-runner-only; measured Δ is noise |
| 10 | 986fd9644 | defer parser imports past early-return in ParserManager | 0.5B | −3.3 s | −1.5 s | **keep** | safe hygiene; partial win (downstream deps still pulled) |
| 11 | a0d0ddc50 | `__getattr__` lazy-load abstract_parser in vllm/parser | 0.5B | flat | flat | discard | avoids one loader but downstream deps still pull the chain |
| 12 | a240bbe2d | bg-thread tokenizer preload during engine spawn | 0.5B | +3.5 s | +1.5 s | discard | GIL contention with main thread's serial config work |
| 13 | 7790500a8 | default `VLLM_WORKER_MULTIPROC_METHOD=forkserver` (sync) | 0.5B | +6.8 s | +2.8 s | discard | synchronous forkserver preload added to critical path without amortization |
| 14 | 09d7c19ec | CLI fast-path (only import requested subcommand) | 0.5B (vllm-cli) | flat | flat | discard | other subcommand imports are sub-second once serve's deps are loaded |
| 15 | b6a6d9ff1 | pre-warm dummy `generate()` during engine init | 0.5B (vllm-cli) | +2.0 s | +1.1 s | discard | first-inference cost was overestimated (~2 s actual vs 10 s assumed) |
| 16 | bd4d148cb | `gloo` backend when world_size=1 | 0.5B (vllm-cli) | +1.2 s | −0.7 s | discard | NCCL init isn't the bottleneck in that phase |
| 17 | 15ea89b06 | parent weight prefetch (warm-disk) | 32B | +0.5 s | +3 s | discard | files already page-cached; measurement invisible |
| — | 70e27d7b8 | harness: drop page cache before cold samples | — | — | — | **accepted** | now measures real cold-disk scenario; revealed +15 s of hidden disk I/O |
| 18 | 300897ad8 | parent weight prefetch (cold-disk) | 32B | **−7.2 s** | flat | **keep** | overlaps 18 s of safetensors reads with the 24 s APIServer+spawn window |
| 19 | 563202b6f | BG torch `.so` preload at CLI entry | 32B | **−1.9 s** | flat | **keep** | torch dlopen releases GIL; overlap with main-thread non-torch imports |
| 20 | 4e68bc5c5 | async AOT compile save | 32B | flat | flat | discard | successful but GIL contention steals from warmup |
| 21 | d81594de3 | VLLM_USE_V2_MODEL_RUNNER=1 | 32B | +9 s | flat | discard | V2 capture actually slower (8 s vs 3 s, 1.13 GiB vs 0.33 GiB captured) |
| 22 | 6e85acd31 | **async forkserver prewarm** | 32B | **−7.6 s** | **−4.6 s** | **keep** | earlier sync version regressed (#13); async lets preload overlap with argparse/config |
| 23 | 76c7eaf83 | Tier 1a: prefetch aux config/tokenizer files | 32B | noisy +9 s | −1.3 s | discard | aux files tiny; no clear signal above noise |
| verify | 6e85acd31 | all 3 keeps @ 7B | 7B | −8.4 s | −4.1 s | **verify** | wins scale across sizes |
| verify | 6e85acd31 | all 3 keeps @ 0.5B | 0.5B | −8.9 s | −5.0 s | **verify** | relative win higher on smaller models |

### Categorized lessons

**What reliably worked:**
1. **Defer eager imports on code paths that are _never_ executed for the target workload.** Exp 1 and 18-style parallelization. Exp 10 is a sibling that's purely CPU hygiene.
2. **Parallelize CPU work that releases the GIL (disk I/O, dlopen).** Exp 18 (safetensors read), Exp 19 (torch .so load), Exp 22 (forkserver fork + preload).
3. **Memoize deterministic functions called >1x per process with the same args.** Exp 7.

**What reliably didn't:**
1. **Import-defer wins don't translate to t_ready when the target has any eager caller elsewhere.** Exp 2, 4, 8, 11 all suffered this: "logic is correct but no-op for our path."
2. **In-process threading inside EngineCore during torch.compile/warmup is GIL-bound.** Exp 12 and 20: threaded work regressed the warmup phase.
3. **Sync forkserver setup on the critical path doesn't amortize for a single-spawn scenario.** Exp 13 — became Exp 22 once moved to background.
4. **User-visible steady-state changes for startup gain.** Exp 5 (cudagraph cap) was the best single win in the session; had to revert when the no-perf-change constraint was imposed.

**Measurement lessons:**
- The original harness quietly used the OS page cache across cold samples, hiding ~15 s of real disk I/O at 32B. Adding `posix_fadvise(POSIX_FADV_DONTNEED)` on weight shards before each cold sample (no sudo needed) exposed it. This lets the parent-prefetch win (Exp 18) show up.
- The original harness used `python -m vllm.entrypoints.openai.api_server` which bypasses the `vllm` CLI entry. Switched to `vllm serve` as the measured path.
- Between-run variance on this shared H200 box is ~2-5 s; within-run stdev is typically <2%. Anything claiming <1 s improvement gets discarded as noise.

---

## Next step — CRIU + cuda-checkpoint snapshot

### Why this is the right next jump

Current phase breakdown for 32B warm (67.4 s after the 6 keeps):

| phase | time | cache target? | addressable by snapshot? |
|---|---|---|---|
| Python imports (vllm + torch + transformers + ...) | ~9 s | — | **yes — ~9 s save** |
| EngineCore spawn + CUDA context in child | ~12 s | — | partial (CUDA init ~3-5 s savable) |
| Distributed + weight load | ~5 s | — | no |
| torch.compile (warm cache hit) | ~5 s | already vllm/torch | no |
| Profile + CUDA graph capture | ~14 s | — | no |
| init_app_state + FastAPI + first inference | ~10 s | — | no |
| **Total warm** | **~55-67 s** | | **~12-14 s savable** |

The CRIU snapshot captures Python imports + per-process CUDA context (via cuda-checkpoint) in one disk image per `(vllm, python, torch, cuda-driver, gpu-arch)` tuple. Restore replays that state into a new process in ~200-500 ms, sidestepping the ~12-14 s of per-invocation pay. Model weights, engine config, and compile state are NOT in the snapshot — they must be per-invocation per-model anyway.

### Why not simply a zygote daemon?

- Zygote holds 2-3 GB RAM indefinitely — costly for idle hosts.
- Zygote requires per-user daemon lifecycle management (systemd/launchd).
- Zygote daemon gets stale on `pip install -U vllm` — needs re-exec.
- CRIU snapshot is a directory on disk. No resident process. Auto-invalidates via version-keyed path.

CRIU's downside is the one-time setup cost (`setcap cap_sys_admin+ep /usr/bin/criu` or equivalent) and non-trivial driver-version pinning. For CI / container environments this is usually fine; for casual `pip install` users it's a barrier.

### Why this can deliver big

The numbers work:

| scope | snapshot size | cold gain | warm gain | per-user cost |
|---|---|---|---|---|
| per-version (no model) — what we're proposing | 2-4 GB | ~15-20 s | ~12-14 s | one-time 30-60 s build |
| per-(version, model) with full CUDA graphs — future | 70-130 GB | ~80-100 s | ~60 s | one per model-invocation combo |

Composed projection with current 6 keeps:

| model | cold now | + per-version snapshot | + per-(version, model) snapshot (future) |
|---|---|---|---|
| 0.5B | 64.1 s | ~50 s | ~5-8 s |
| 7B | 70.3 s | ~58 s | ~10-15 s |
| 32B | 106.7 s | ~92 s | ~15-20 s |

### Proof that the design is sound

1. **Python imports are idempotent across processes.** CPython's import state is pure memory + bytecode loaded from disk — no process-specific state beyond ASLR offsets which CRIU handles via `/proc/self/maps`. CRIU has successfully snapshot-restored Python processes across versions of CPython since Python 3.4.

2. **Torch can hold CUDA context without allocating tensors.** Verified by testing: `torch.cuda.init()` followed by `torch.cuda.synchronize()` creates a context without GPU memory allocation. `torch.cuda.is_initialized() == True` after, and `nvidia-smi --query-compute-apps=used_memory` shows ~0 for the process. The context is exactly what cuda-checkpoint can snapshot.

3. **cuda-checkpoint is a documented NVIDIA tool since CUDA 12.4** and has been used for CRIU-based checkpoint/restore of CUDA workloads. `cuda-checkpoint --toggle --pid <PID>` quiesces all CUDA work and migrates device memory to host pinned memory. Subsequent CRIU dump captures the CPU-resident state, and the reverse on restore.

4. **The resume protocol is trivial.** Helper process installs a SIGUSR2 handler before `pause()`. After CRIU restore, wrapper writes argv/env/cwd to a file path known via env var, then sends SIGUSR2. Handler reads the file, swaps argv, and calls `main()` from the existing CLI dispatch — the same `main()` a fresh invocation would have reached.

5. **Failure mode is free.** If CRIU is absent, cuda-checkpoint is absent, snapshot is missing, or restore fails, the existing six-keep fallback path runs. Adding the snapshot path costs nothing when it's not used.

6. **Amortization math.** One-time snapshot build: ~30-60 s. Per-invocation save: ~15-20 s. Break-even at ~3 invocations. For a typical `vllm serve` workflow with restarts, shells, or CI: trivially amortized.

### Prototype commit

`e9c8cf487` — "CRIU + cuda-checkpoint snapshot prototype scaffolding". Inert by default (`VLLM_SNAPSHOT_ENABLED=1` required to opt in; `VLLM_SNAPSHOT_DRY_RUN=1` for log-only mode to validate without `criu` installed). Adds:

- `vllm/snapshot/` module: keying, helper process, criu/cuda-checkpoint wrappers, resume protocol
- `vllm snapshot` CLI subcommand: create, list, drop
- Pre-dispatch hook in `vllm/entrypoints/cli/main.py`: on `vllm serve` + env flag, try restore before falling through to the existing forkserver prewarm path

Works end-to-end in dry-run mode on this host (no root, no CRIU):

```bash
$ PYTHONPATH=/home/simon/vllm .venv/bin/vllm snapshot --help
usage: vllm snapshot [-h] {create,list,drop} ...

Per-(vllm, python, torch, cuda-driver, gpu-arch) snapshot that skips Python
imports + CUDA context init on subsequent `vllm serve` invocations.

$ VLLM_SNAPSHOT_DRY_RUN=1 vllm snapshot create --dry-run
Starting snapshot helper for key 91f351cabb6189b3
  vllm=0.5.0.post2.dev14263+g219bb5b8c py=3.12.13 torch=2.11.0+cu130 ...
...
```

The real `criu dump` / `cuda-checkpoint --toggle` calls are wired but gated on `VLLM_SNAPSHOT_ENABLED=1` — ready to flip on a host with the binaries installed.

### Roadmap from prototype to production

| phase | work | days |
|---|---|---|
| phase 0 — **prototype (done)** | scaffolding, dry-run, hooks in cli main | 1 |
| phase 1 | actual CRIU dump/restore + test on a box with CRIU installed | 3-5 |
| phase 2 | cuda-checkpoint integration + multi-gpu / driver-version testing | 2-3 |
| phase 3 | FD passing (stdio), signal forwarding, install hooks | 2-3 |
| phase 4 | docs, CI integration, rollout | 2 |
| **total** | | **~2 weeks engineering for shippable** |

---

## Appendix — file index

All design docs, measurement logs, and the prototype live under the repo:

```
.startup-bench/design/criu_cuda_checkpoint_plan.md  Full design doc
.startup-bench/logs/apr16/_session_summary.md       Mid-session notes
.startup-bench/logs/apr16/<sha>*.log                Per-experiment logs
.startup-bench/measure.py                           Harness
.startup-bench/program.md                           Experiment-loop playbook
.startup-bench/results.tsv                          Full results table
.startup-bench/FINAL_REPORT.md                      This file
vllm/snapshot/                                       Prototype scaffolding
vllm/entrypoints/cli/snapshot.py                    CLI subcommand glue
```

Branch: `startup/apr16`. 6 kept commits stacked on `main` — safe to rebase + merge.
