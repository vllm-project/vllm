# DSv4 weight-offloading upstreaming record

## Scope

- Target branch: `tizheng/weight-offloading-upstream`
- Base: `origin/main` at `6f00a1ae3bd4b86168667bce673998218f461c0f`
- Source change: `f0645921e9cd6063d706ea4a28b00bd2448ce5c0`
  (`[Offloader] Add prefetch-based weight offloading for DSv4-Flash`)

The change adds asynchronous, group-based H2D prefetch for model weights,
including semantic parameter selectors and the static runtime-buffer management
needed by the DSv4-Flash MoE workload.

## Cherry-pick conflicts and resolutions

### `vllm/config/vllm.py`

`origin/main` had added the `validate_mamba_cached_kernel` model validator in
the same location where the source change added the prefetch-offloading/EPLB
validation.

Resolution: retain the upstream Mamba validation unchanged and add
`validate_prefetch_offload_eplb` as a separate subsequent validator. This
preserves both safety checks: ReplaySSM compatibility and the unsupported
prefetch-offloading + EPLB combination.

### `vllm/v1/worker/gpu_worker.py`: weight wake-up

The source change called `CuMemAllocator.wake_up()` directly, whereas
`origin/main` has moved the operation behind `SleepModeBackend.resume()`.

Resolution: retain `SleepModeBackend.resume(tags)` and invoke
`_reset_offloader_after_weight_wake(tags)` immediately afterward. The runtime
prefetch state is therefore reset only when weights are woken, while retaining
the current sleep-mode abstraction.

### `vllm/v1/worker/gpu_worker.py`: memory profiling

The source change added `static_runtime_buffer_bytes` explicitly to the model
memory-profile result and to the suggested KV-cache accounting. Since the
source commit, upstream changed `memory_profiling()` to calculate
`total_consumed` from the free-memory delta between model creation and the end
of profiling. The offloader allocates its static runtime buffers during model
loading, before profiling, so those bytes are already included.

Resolution: retain the upstream profiling calculation and do not add the
static buffer bytes a second time. Adding them would understate the available
KV cache and double-count GPU memory.

## Additional upstream compatibility fixes

The source change's tests relied on older upstream interfaces. The following
test-only updates keep their behavioral coverage while matching `origin/main`:

- EPLB validation tests explicitly mock two visible GPUs. They construct a
  tensor-parallel size of two to exercise EPLB, so this removes a dependency on
  the host's physical GPU count.
- The FusedMoE lookup test uses a `MoERunnerInterface`-spec mock. Current
  upstream correctly validates that resolved layers implement this interface.
- The CUDA graph capture test accepts the current `warmup` factory argument
  and returns the forward function directly, matching the current
  `CudaGraphManager.capture()` contract.

## Environment notes

The editable install intentionally uses no build isolation. Its build
requirements must therefore already exist in `.venv`; in particular, PyTorch,
`setuptools-rust`, CMake, Ninja, and the remaining packages from
`requirements/build/cuda.txt`. `VLLM_USE_PRECOMPILED=1` selects the matching
precompiled CUDA extension wheel while keeping Python sources editable.

## Verification

```bash
.venv/bin/python -m pytest tests/weight_offload -q
```

Result on the two-GPU validation host: `112 passed, 1 skipped`.
