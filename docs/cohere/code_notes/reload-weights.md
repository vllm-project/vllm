# Code Notes: `/collective_rpc reload_weights`

This note captures the behavior, integration boundaries, and the
`recapture_cudagraphs` shim required to make vLLM's in-place weight
reload (`reload_weights` invoked via `/collective_rpc`) coexist with
CUDA graphs. Tested end-to-end in
`tests/cohere/test_collective_rpc_reload.py` (test group
`collective_rpc_reload`).

## 1) Round-Trip Surface

The reload round-trip is a pure HTTP control plane (no `torch` / vLLM
imports needed on the client):

1. `POST /pause?mode=wait` — drains in-flight generations and clears KV
   caches via `_reset_caches`.
2. `POST /collective_rpc` with
   `{"method": "reload_weights", "kwargs": {"weights_path": "/path"}}` —
   loads weights from disk into the running engine, layer-by-layer.
3. `POST /collective_rpc` with `{"method": "recapture_cudagraphs"}` —
   **Cohere fork addition** on `GPUWorker`. Drops every
   `CUDAGraphWrapper`'s `concrete_cudagraph_entries` and re-runs
   `capture_model` against the freshly reloaded weights. No-op when the
   server runs with `--enforce-eager` or `cudagraph_mode=NONE`.
4. `POST /resume` — re-enables generation.

All endpoints require `VLLM_SERVER_DEV_MODE=1`. Server log capture is
wired through `RELOAD_SERVER_LOG` so failures surface the engine crash
window without re-running.

## 2) Layerwise Reload Internals

`reload_weights` (in `vllm/v1/worker/gpu_model_runner.py`) delegates to
`vllm/model_executor/model_loader/reload/layerwise.py`:

- `initialize_layerwise_reload(model)`:
    - Snapshots current parameters/buffers as `info.kernel_tensors` (the
      "post-`process_weights_after_loading`" tensors that the live model
      is using).
    - Restores layers to their pre-load (meta) state.
    - Wraps weight loaders so loading is deferred until each leaf has all
      its weights, then `_layerwise_process` runs.
- `_layerwise_process(layer, info)`:
    - Materializes fresh tensors, replays loaded weights, re-runs
      `process_weights_after_loading`, then **copies the post-process
      values back into `info.kernel_tensors`** so the original storage is
      preserved (this is the design intent for keeping CUDA graphs valid
      for parameters captured at startup).
    - `_place_kernel_tensors` re-registers `info.kernel_tensors` as the
      layer's parameters/buffers, dropping anything new that
      `process_weights_after_loading` may have added.

This mechanism is **storage-preserving for tensor `data_ptr`s**:
verified empirically with a one-off `snapshot_tensor_pointers` worker
probe walking `Parameters`, `Buffers`, top-level tensor attrs, and
`quant_method` tensor attrs — across reload, all 1455 tensors on
`c5-3a30t_fp8` keep the same `data_ptr()`. The earlier hypothesis that
quant backends were stranding "auxiliary derived tensors" was wrong —
those derived tensors are tracked in `info.kernel_tensors` and the
`param.data.copy_()` step copies the new values into the original
storage.

What **is** stale after reload is a small set of **Python helper
objects** on the FP8 MoE quant method. The same probe extended with
`id(...)` of non-tensor attributes shows that each PWAL call rebuilds
`self.moe_kernel` and `self.moe_quant_config` on every
`Fp8MoEMethod` / `CompressedTensorsW8A8Fp8MoEMethod` instance (96 fresh
identities for 48 MoE layers on `c5-3a30t_fp8`). The CUDA graphs
captured at startup bind to method addresses on the original
`moe_kernel`; once that object is garbage-collected the next forward
pass hits an illegal memory access.

The same rebuild pattern exists on every MoE quant backend
(`modelopt`, `mxfp4`, the W4A4 / W4A8 / W8A8 / WNA16 variants in
`compressed_tensors_moe/`), upstream and on this fork — none of them
guard against re-running the kernel setup. Fixing each one with an
idempotency check is possible but model/backend-specific, while the
recapture shim works regardless of which backend the model uses.

## 3) Why CUDA-Graph Recapture Is Required

Without an explicit recapture step, the first forward pass after reload
faults with `CUDA error: an illegal memory access was encountered` and
takes the engine down (`EngineDeadError`).

Two failed alternatives tried before landing the current shim:

- **Disable CUDA graphs entirely (`--enforce-eager`)**. Works, but
  forfeits the throughput benefits of graph dispatch and is overly
  pessimistic.
- **Call `compile_or_warm_up_model` via `/collective_rpc`**. Crashes
  the same way, because its outer
  `for size in warmup_sizes: self.model_runner._dummy_run(size)` loop
  leaves `cudagraph_runtime_mode=None`, which falls back to the
  dispatcher and *replays the stale graphs* before recapture can run.
  By contrast, `_warmup_and_capture` (called from inside
  `capture_model`) explicitly passes `CUDAGraphMode.NONE` for its
  warmup runs, so recapture itself is safe.

Prefix caching is **safe** end-to-end — `_reset_caches` runs during
`/pause` and clears the KV cache manager before reload begins.

## 4) `recapture_cudagraphs` Worker Method

`vllm/v1/worker/gpu_worker.py` adds a Cohere-fork worker method (under
`# cohere start` / `# cohere end` markers):

```python
def recapture_cudagraphs(self) -> int:
    if self.model_config.enforce_eager:
        return 0
    if self.vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return 0
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    CUDAGraphWrapper.clear_all_graphs()
    self.model_runner.capture_model()
    return sum(
        len(w.concrete_cudagraph_entries)
        for w in CUDAGraphWrapper._all_instances
    )
```

Behavior:

- `CUDAGraphWrapper.clear_all_graphs()` drops
  `concrete_cudagraph_entries` on every wrapper instance. The next
  capture pass will create fresh `CUDAGraphEntry` objects.
- `model_runner.capture_model()` walks
  `cudagraph_dispatcher.get_capture_descs()` (already populated at
  startup, untouched by reload) and runs `_warmup_and_capture` for each
  shape. Internal warmup runs eagerly (`CUDAGraphMode.NONE`); the
  capture pass uses the actual mode (`PIECEWISE` / `FULL`) and writes
  into the same global graph pool.
- The return value sums `len(concrete_cudagraph_entries)` across every
  live `CUDAGraphWrapper` instance after recapture — a deterministic
  "graphs actually recaptured" signal. We intentionally do **not**
  return `capture_model()`'s byte delta (computed from
  `torch.cuda.mem_get_info` and intended for upstream warmup logging):
  CUDA graphs share a global memory pool (`get_global_graph_pool` is a
  class-level singleton, never released between captures), so on
  recapture the delta is often `0` or negative even though graphs were
  successfully recaptured. A `> 0` assertion on the byte delta would
  spuriously fail. The entry count avoids that pitfall while still
  going to zero when CUDA graphs were misconfigured off.
- Skipped automatically when CUDA graphs are off (eager / NONE), so the
  test/control plane code can call it unconditionally.

`capture_model` is invocable a second time without other plumbing
because:

- `set_cudagraph_capturing_enabled(True/False)` is bracketed inside
  `capture_model` itself.
- `cudagraph_dispatcher.cudagraph_keys` are populated by
  `initialize_cudagraph_keys` at startup and not cleared by reload.
- `lock_workspace()` is idempotent — the recapture pass uses the same
  shapes that already fit pre-reload.

Observed timings on `c5-3a30t_fp8` (TP=1, GB200): pause + reload +
recapture round-trip ≈ 19 s (reload ~13 s, recapture ~6 s for 38
graphs across PIECEWISE + FULL modes).

## 5) Relationship to Upstream's `finish_weight_update`

Upstream (`vllm-project/vllm` `main`, HEAD `4ff865c38`) added an
IPC/NCCL weight-transfer state machine on the GPU worker:
`start_weight_update` → one or more `update_weights` → `finish_weight_update`.
`finish_weight_update` is a thin wrapper that calls
`finalize_layerwise_reload(model, model_config)` →
`finalize_layerwise_processing`, which handles deferred attention
layers (kv-cache scale reload via `_finalize_attention_layer`) and
runs `_layerwise_process` for any "no weights loaded" layers.

`_layerwise_process` upstream calls `quant_method.process_weights_after_loading(layer)`
and then `_copy_and_restore_kernel_tensors` to put the original
parameter storage back. The new helper carries the explicit comment
"Preserves cudagraph references" — confirming the same data-pointer
preservation our probe verified.

What upstream does **not** do, and why this RPC is still required:

- No worker-side `recapture_cudagraphs` method exists upstream.
- No FP8 MoE method guards against rebuilding `self.moe_kernel` /
  `self.moe_quant_config` on the second PWAL call. The exact same
  unconditional rebuild lives in
  `Fp8MoEMethod._setup_kernel` (line 785) and every backend under
  `compressed_tensors/compressed_tensors_moe/` (e.g.
  `compressed_tensors_moe_w8a8_fp8.py` line 335).
- Every upstream IPC/NCCL example and test pins `enforce_eager=True`
  (`tests/entrypoints/weight_transfer/test_weight_transfer_llm.py`,
  `examples/rl/rlhf_ipc*.py`, `examples/rl/rlhf_nccl*.py`,
  `examples/rl/rlhf_async_new_apis.py`,
  `examples/rl/skip_loading_weights_in_engine_init.py`). The
  CUDA-graph staleness is invisible to upstream because the
  combination is not exercised.

`recapture_cudagraphs` is therefore the general fix — independent of
which quant backend the model uses — and would be necessary for the
upstream NCCL/IPC path as well once CUDA graphs get enabled there.

## 6) Test Coverage

See [`docs/cohere/tests/features/weight_reload.md`](../tests/features/weight_reload.md)
for the full test documentation (How it runs, Checks, Measurements,
Compatibility, Implementation).

`tests/cohere/test_collective_rpc_reload.py` (test group
`collective_rpc_reload`) is a *behavior* test (broken → fixed), not a
preservation test:

- Pre-test setup (`run_tests.sh`):
  `tests/cohere/scripts/zero_safetensor_param.py` builds a corrupted
  mirror of the checkpoint by symlinking every file except the shard
  containing `model.language_model.embed_tokens.weight`, which is
  rewritten with that tensor zeroed. In this checkpoint
  `embed_tokens` is tied to the LM head, so zeroing it kills both
  input encoding and output projection — guaranteed to break the
  model. (Single-layer ablations like `layers.N.self_attn.o_proj`
  don't degrade the score because residual skip-connections route
  signal around the zeroed layer; verified empirically.)
- Phase 1: server booted from the corrupted mirror → run `infovqa`,
  require `avg_score <= RELOAD_BROKEN_MAX_SCORE` (0.20). `infovqa`
  is used because `ocrbench`'s `compute_soft_accuracy` returns 1.0
  whenever either string contains the other, and an empty generation
  is a substring of every label — so a fully broken model would
  spuriously score 1.0 there. `infovqa` uses ANLS only.
- Phase 2: `/pause?mode=wait` → `reload_weights` (pointing at the
  *original*, good `RELOAD_MODEL_PATH`) → `recapture_cudagraphs` →
  `/resume`.
- Phase 3: rerun `infovqa`, require `avg_score >= RELOAD_MIN_SCORE`
  (0.40). The score delta (`fixed - broken`) is logged.
- Server log path is forwarded as `RELOAD_SERVER_LOG`; the test dumps
  the tail on HTTP errors or when every Phase 3 request fails (typical
  signal of an engine crash mid-reload).

`tests/cohere/scripts/run_tests.sh::run_collective_rpc_reload` starts
the server with `VLLM_SERVER_DEV_MODE=1` and **without** `--enforce-eager`
— CUDA graphs are exercised on both sides of the reload. The corrupted
mirror is removed after the test exits.

## 7) Files of Interest

- `vllm/v1/worker/gpu_worker.py` — `reload_weights`, `recapture_cudagraphs`.
- `vllm/v1/worker/gpu_model_runner.py` — `capture_model`,
  `_warmup_and_capture`, `_dummy_run`, `reload_weights` (model-runner side).
- `vllm/compilation/cuda_graph.py` — `CUDAGraphWrapper`,
  `clear_all_graphs`, `clear_graphs`, `concrete_cudagraph_entries`.
- `vllm/model_executor/model_loader/reload/layerwise.py` — layerwise
  reload core (`_layerwise_process`, `_place_kernel_tensors`).
- `vllm/entrypoints/serve/rpc/api_router.py` — `/collective_rpc`.
- `vllm/entrypoints/serve/rlhf/api_router.py` — `/pause`, `/resume`,
  `/init_weight_transfer_engine`, `/update_weights`,
  `/finish_weight_update` (upstream IPC/NCCL counterpart).
- `tests/cohere/test_collective_rpc_reload.py`,
  `tests/cohere/scripts/run_tests.sh` (`run_collective_rpc_reload`).

## 8) Possible Upstream Path

Two ways upstream could remove the need for this shim:

- Adopt the same `clear_all_graphs` + `capture_model` pattern inside
  `reload_weights` (and `finish_weight_update`) so callers never see
  the stale-graph state. Backend-agnostic; matches what this RPC does
  externally.
- Add an idempotency guard to every backend's `_setup_kernel` /
  `process_weights_after_loading` so the helper kernel object is
  built once and reused across reloads. Lower runtime cost, but
  per-backend churn (FP8 MoE, MXFP8, MXFP4, INT8, NVFP4, W4A4, W4A8,
  W8A8, WNA16 / Marlin variants).

Until one of those lands, the worker-side `recapture_cudagraphs`
remains the integration point for the fork.
