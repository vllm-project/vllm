# Weight Reload (`collective_rpc_reload`)

Hot-reload of model weights via `/collective_rpc reload_weights` with
CUDA graph recapture. Validates that a deliberately broken model can be
fixed in place without restarting the vLLM server.

See [`docs/cohere/code_notes/reload-weights.md`](../../code_notes/reload-weights.md)
for the full technical investigation.

<details>
<summary>Test case 1: Broken model → reload_weights → passing score</summary>

## How it runs

1. `run_collective_rpc_reload` in `run_tests.sh` creates a corrupted
   checkpoint mirror (zeroes `model.language_model.embed_tokens.weight`)
   and boots `vllm serve` from it with `VLLM_SERVER_DEV_MODE=1`.
   - [`tests/cohere/scripts/run_tests.sh`](../../../tests/cohere/scripts/run_tests.sh) — `run_collective_rpc_reload` function
   - [`tests/cohere/scripts/zero_safetensor_param.py`](../../../tests/cohere/scripts/zero_safetensor_param.py)
2. Phase 1: runs `infovqa` bee-eval task against the broken server,
   expects a low score (≤ 0.20).
   - [`tests/cohere/test_collective_rpc_reload.py`](../../../tests/cohere/test_collective_rpc_reload.py) — `_run_task`
3. Phase 2 setup: submits a long-running `chat/completions` request
   in a background thread so the engine has an in-flight request when
   `/pause?mode=wait` is called. This forces `_pause` to time out and
   fall back to `/pause?mode=abort`, exercising the fallback path end
   to end against the real server.
   - [`tests/cohere/test_collective_rpc_reload.py`](../../../tests/cohere/test_collective_rpc_reload.py) — `_start_long_generation`, `_pause`
4. Phase 2: POSTs `/pause` (wait → abort fallback) → `/collective_rpc
   reload_weights` (good checkpoint) → `/collective_rpc recapture_cudagraphs`
   → `/resume`.
   - [`tests/cohere/test_collective_rpc_reload.py`](../../../tests/cohere/test_collective_rpc_reload.py) — `_reload_weights`
   - [`vllm/v1/worker/gpu_worker.py`](../../../vllm/v1/worker/gpu_worker.py) — `recapture_cudagraphs`
5. Phase 3: reruns the same `infovqa` task, expects a passing score
   (≥ 0.40).
   - [`tests/cohere/test_collective_rpc_reload.py`](../../../tests/cohere/test_collective_rpc_reload.py) — `_run_task`
6. Test group `collective_rpc_reload` in `run_tests.sh`; invoked via
   `TEST_GROUP=collective_rpc_reload`.

## Checks

1. **Broken model scores low** (avg_score ≤ 0.20) — validates the
   corruption actually degraded the model.
   - `test_reload_weights_fixes_broken_model` (Phase 1 assertion)
2. **Pause fallback fired** (`_reload_weights` returned `"abort"`) —
   given the injected in-flight load, `/pause?mode=wait` must have
   timed out and triggered `mode=abort`. Guards against the fallback
   path silently regressing (e.g. timeout raised but not caught,
   `_PAUSE_WAIT_TIMEOUT` accidentally bumped too high for hardware).
   - `test_reload_weights_fixes_broken_model` (Phase 2 pause assertion)
3. **`recapture_cudagraphs` returned non-zero on every worker** —
   guards against silent no-op when the server was misconfigured with
   `--enforce-eager` or `cudagraph_mode=NONE`, which would make the
   test trivially pass without exercising the recapture path.
   - `test_reload_weights_fixes_broken_model` (Phase 2 RPC assertion)
4. **Reloaded model scores high** (avg_score ≥ 0.40) — validates
   `reload_weights` swapped in the correct weights and the engine
   remained healthy through the reload + recapture + abort cycle.
   - `test_reload_weights_fixes_broken_model` (Phase 3 assertion)

## Measurements

N/A — this test group does not upload artifacts to CI. JUnit XML is
generated to `$OUTPUT_DIR` for local inspection only.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Image (compatible — `infovqa` is a document VQA task)
2. **Cohere Feature**: Weight Reload (compatible)
3. **Model Architecture**: C5 Arch (compatible)
   - [`tests/cohere/scripts/download_checkpoints.sh`](../../../tests/cohere/scripts/download_checkpoints.sh) — `c5-3a30t_fp8`
4. **Quantization**: FP8 (compatible)
   - Model checkpoint `c5-3a30t_fp8` uses `compressed-tensors` W8A8 FP8
5. **Hardware**: GB200 (compatible)
   - Tested on GB200 (TP=1); no `runner_map.json` entry yet (not hooked into CI)
6. **vLLM Feature**: CUDA Graphs (compatible)
   - Server boots **without** `--enforce-eager`; CUDA graphs captured at
     startup and recaptured after reload via `recapture_cudagraphs`

## Implementation

Primary test: [`tests/cohere/test_collective_rpc_reload.py`](../../../tests/cohere/test_collective_rpc_reload.py)
Runtime path: [`vllm/v1/worker/gpu_worker.py`](../../../vllm/v1/worker/gpu_worker.py) — `recapture_cudagraphs`
Shell orchestration: [`tests/cohere/scripts/run_tests.sh`](../../../tests/cohere/scripts/run_tests.sh) — `run_collective_rpc_reload`
Corruption helper: [`tests/cohere/scripts/zero_safetensor_param.py`](../../../tests/cohere/scripts/zero_safetensor_param.py)
Code notes: [`docs/cohere/code_notes/reload-weights.md`](../../code_notes/reload-weights.md)

### Setup

1. `VLLM_SERVER_DEV_MODE=1` — enables `/collective_rpc`, `/pause`,
   `/resume` endpoints.
2. `VLLM_ENABLE_COHERE_AUTO_CONFIG=1` — hardware profiles applied
   (CUDA graphs enabled by default on supported GPUs).
3. Server booted from the corrupted mirror (symlinked checkpoint with
   `embed_tokens.weight` zeroed); no `--enforce-eager`.
4. `infovqa` task (16 samples) — ANLS scoring; chosen because
   `ocrbench` scores empty generations as 1.0 due to a substring bug.
5. `_PAUSE_WAIT_TIMEOUT=5s` — kept intentionally short so the
   injected long-generation reliably exceeds it. The pause-fallback
   path is exercised every test run; without injection the engine
   would drain immediately and the fallback would never execute.
6. Client uses only `requests` + `openai` — no torch/vLLM dependencies
   on the test side.

</details>
