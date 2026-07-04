# Issue #47548 — Sleep mode + speculative decoding crash (ROCm / MI250)

## 1. Root cause

Combining `--enable-sleep-mode` with `--speculative-config '{"method":"qwen3_next_mtp",...}'`
crashes at startup with:

```
AttributeError: .../site-packages/tilelang/lib/libcudart_stub.so: undefined symbol: hipSetDevice
```

The crash happens in the device-allocator initialization chain:

```
gpu_worker.py  _maybe_get_memory_pool_context -> get_mem_allocator_instance()
device_allocator/cumem.py:40  libcudart = CudaRTLibrary()
distributed/device_communicators/cuda_wrapper.py  find_loaded_library(...)
```

Why the *combination* is required:

- **Sleep mode** forces the cumem allocator on, which imports `vllm/device_allocator/cumem.py`.
  That module constructs `CudaRTLibrary()`, which calls
  `find_loaded_library("libamdhip64" | "libcudart")` to locate the CUDA/HIP runtime.
- **Speculative decoding with MTP** (`qwen3_next_mtp` → `mtp`) pulls in the optional
  `tilelang` package, which loads its own `tilelang/lib/libcudart_stub.so` into the process.

The real bug is in `find_loaded_library()` (`vllm/utils/system_utils.py`). It scanned
`/proc/self/maps` with a **loose substring match** and accepted the first line containing the
search string:

```python
if lib_name in line: ...        # "libcudart" is a substring of "libcudart_stub.so"
...
assert filename.rpartition(".so")[0].startswith(lib_name)   # "libcudart_stub".startswith("libcudart") == True
```

So when `tilelang`'s `libcudart_stub.so` is present, `find_loaded_library("libcudart")`
returns the **stub** instead of the real runtime, and the sanity `assert` does *not* catch it
(`"libcudart_stub".startswith("libcudart")` is `True`). `CudaRTLibrary` then binds against the
stub, which does not export the HIP symbols (`hipSetDevice`, …), and crashes. Without
speculative decoding the stub is never loaded, so sleep mode works; without sleep mode the
allocator is never created, so speculative decoding works. Only the combination triggers it.

I reproduced the matching bug deterministically (no GPU needed): searching `libcudart`
against a maps line for `.../libcudart_stub.so` returned the stub path and passed the old
assertion.

## 2. The fix and why

Harden `find_loaded_library()` so it only matches a *real* library file. A candidate is
accepted only when the mapped file's basename is `lib_name` followed by a version/build
separator — `.` (`libcudart.so`, `libcudart.so.12`, `cumem_allocator.cpython-312-….so`) or
`-` (`libcudart-<hash>.so.11.0`). Names like `libcudart_stub.so` (an `_` follows) or
`libcudarter.so` are rejected. The scan now **skips** non-matching lines and keeps looking
(returning the real runtime even if a lookalike is mapped first) instead of breaking on the
first substring hit and asserting.

This fixes the problem at its source: the same function is used by both `cumem.py` and
`cuda_wrapper.py`, so the fix covers the ROCm path (`libamdhip64`) and the CUDA path
(`libcudart`), and it is robust regardless of load order. The matching logic is extracted into
a small pure helper `_matching_library_path()` so it can be unit-tested without a GPU.

## 3. Files changed

- `vllm/utils/system_utils.py` — rewrote `find_loaded_library()`; added helper
  `_matching_library_path()`; removed the too-loose substring match + `startswith` assertion.
- `tests/utils_/test_system_utils.py` — added a parametrized test for
  `_matching_library_path` (real variants match; `libcudart_stub.so` and `libcudarter.so`
  are rejected) and a `find_loaded_library` test proving the stub is skipped when it is
  mapped before the real runtime.

## 4. Risk / uncertainty

- **Behavior change:** matching is now stricter and the previous `assert` was removed. Callers
  already handle a `None` return (fallback to `VLLM_CUDART_SO_PATH`, then a clear assertion),
  so a missed match degrades to a readable error instead of a confusing symbol crash. I checked
  all real search terms in the repo (`libcudart`, `libamdhip64`, `cumem_allocator`) and their
  actual on-disk filename shapes still match.
- **Environment:** the reporter runs a bleeding-edge ROCm 7.x / `_rocm_sdk_core` stack that I
  cannot run here, so I could not execute the exact `vllm serve` repro. On current `main` the
  ROCm branch already searches `libamdhip64` (which the stub does not match), so this change is
  primarily a robustness/root-cause fix that also protects the CUDA + tilelang path and any
  ROCm fallback where `libcudart` is searched. The core matching bug is verified by the new
  tests.

## 5. How I verified

- Extracted the old matching logic and confirmed `find_loaded_library("libcudart")` returned
  `.../libcudart_stub.so` and the old assertion passed — reproducing the root cause.
- Ran the new `_matching_library_path` logic against a table of real/stub/lookalike filenames
  and an ordered-scan case (stub mapped before the real lib): all pass, the stub is rejected,
  and the real runtime is returned.
- `python -m py_compile` on both changed files; verified changed lines stay within the repo's
  88-char limit.

Note: the full pytest suite could not run in this environment (vLLM runtime deps such as
`torch`/`tblib` are not installed), so the added tests were validated via the standalone logic
harness above rather than through `pytest`.

AI assistance (Claude) was used to investigate and implement this change.
