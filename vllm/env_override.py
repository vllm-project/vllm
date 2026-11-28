# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
import importlib.util
import os
from collections.abc import Callable


def _get_torch_root():
    """Locate the installed torch package without importing it."""
    spec = importlib.util.find_spec("torch")
    if not spec:
        return None
    if spec.origin:
        return os.path.dirname(spec.origin)
    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None


def _get_torch_version_attr(attr):
    """Read an attribute of torch.version without importing torch.

    PyTorch version must not be determined by importing directly
    because it will trigger the CUDA initialization, losing the
    chance to set the LD_LIBRARY_PATH beforehand.
    """
    try:
        torch_root = _get_torch_root()
        if not torch_root:
            return None
        version_path = os.path.join(torch_root, "version.py")
        if not os.path.exists(version_path):
            return None
        # Load the version module without importing torch
        ver_spec = importlib.util.spec_from_file_location("torch.version", version_path)
        if not ver_spec or not ver_spec.loader:
            return None
        module = importlib.util.module_from_spec(ver_spec)
        # Avoid registering in sys.modules to not confuse future imports
        ver_spec.loader.exec_module(module)
        return getattr(module, attr, None)
    except Exception:
        return None


def _get_torch_cuda_version():
    """Peripheral function to _maybe_set_cuda_compatibility_path()."""
    return _get_torch_version_attr("cuda")


def _maybe_set_cuda_compatibility_path():
    """Set LD_LIBRARY_PATH for CUDA forward compatibility if enabled.

    Must run before 'import torch' since torch loads CUDA shared libraries
    at import time and the dynamic linker only consults LD_LIBRARY_PATH when
    a library is first loaded.

    CUDA forward compatibility is only supported on select professional and
    datacenter NVIDIA GPUs. Consumer GPUs (GeForce, RTX) do not support it
    and will get Error 803 if compat libs are loaded.
    """
    enable = os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower() in (
        "1",
        "true",
    )
    if not enable:
        return

    cuda_compat_path = os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH", "")
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_compat = os.path.join(conda_prefix, "cuda-compat")
        if conda_prefix and os.path.isdir(conda_compat):
            cuda_compat_path = conda_compat
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):
        torch_cuda_version = _get_torch_cuda_version()
        if torch_cuda_version:
            default_path = f"/usr/local/cuda-{torch_cuda_version}/compat"
            if os.path.isdir(default_path):
                cuda_compat_path = default_path
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):
        return

    norm_path = os.path.normpath(cuda_compat_path)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = existing.split(os.pathsep) if existing else []

    if ld_paths and ld_paths[0] and os.path.normpath(ld_paths[0]) == norm_path:
        return  # Already at the front

    new_paths = [norm_path] + [
        p for p in ld_paths if not p or os.path.normpath(p) != norm_path
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_paths)


def _maybe_promote_torch_symbols_for_rocm():
    """Put libtorch_cpu.so in the global symbol scope on ROCm, for GPU profiling.

    Must run before 'import torch'. libkineto advertises itself to
    rocprofiler-sdk by exporting rocprofiler_configure from libtorch_cpu.so,
    and rocprofiler-sdk looks for that symbol exactly once, from a lazy
    initializer that runs *during* 'import torch'. CPython loads extension
    modules RTLD_LOCAL, so unless something has already placed the symbol in
    the global scope the lookup misses and no client is ever registered.

    The failure is silent and total: no queue interception, so roctracer
    yields no GPU records, while torch.profiler still writes a complete
    CPU-only trace and reports success.

    Best effort. If the library cannot be loaded early we leave the process
    exactly as it would have been, losing only GPU tracing.

    Skipped when HSA_TOOLS_LIB is set (an external HSA profiling tool is
    active).  Loading libtorch_cpu.so with RTLD_GLOBAL pulls
    libhsa-runtime64 into the global symbol scope, which prevents the
    external tool's OnLoad callback from firing — resulting in zero GPU
    kernel records captured.
    """
    if _get_torch_version_attr("hip") is None:
        return
    if os.environ.get("HSA_TOOLS_LIB"):
        return
    try:
        import ctypes
        import glob

        torch_root = _get_torch_root()
        if not torch_root:
            return
        lib = os.path.join(torch_root, "lib", "libtorch_cpu.so")
        if not os.path.exists(lib):
            return
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Some ROCm wheel layouts leave a stale RUNPATH in libtorch_cpu.so,
            # so a dependency that resolves once torch has set things up does
            # not resolve for a standalone load. Satisfy it by search, then
            # retry; give up quietly if it is not where we expect.
            site_root = os.path.dirname(torch_root)
            deps = glob.glob(
                os.path.join(
                    site_root, "_rocm_sdk_*", "lib", "*", "lib", "librocm-openblas.so.*"
                )
            )
            if not deps:
                return
            ctypes.CDLL(deps[0], mode=ctypes.RTLD_LOCAL)
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


_maybe_set_cuda_compatibility_path()
_maybe_promote_torch_symbols_for_rocm()

import torch

from vllm.logger import init_logger
from vllm.utils.torch_version import is_torch_equal_or_newer

logger = init_logger(__name__)

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

# see https://github.com/vllm-project/vllm/issues/10480 and
# https://github.com/vllm-project/vllm/issues/10619.
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# Enable Triton autotuning result caching to disk by default.
# Without this, Triton re-runs autotuning on every process restart,
# adding significant latency to the first inference request.
# This writes autotuning results to TRITON_CACHE_DIR.
# It can still be overridden by setting TRITON_CACHE_AUTOTUNING=0
# in the environment.
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

# When unset, TileLang routes JIT temp dirs through a world-shared
# /tmp/tvm-debug-mode-tempdirs/ whose ownership is pinned to whichever
# user compiled first, breaking every other user on a shared host.
# Opt into per-process tempdirs unless the user explicitly chose the
# debug layout (see https://github.com/vllm-project/vllm/issues/41410).
os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")


def _run_after_module_import(module_name: str, callback: Callable[[], None]) -> None:
    """Run a backport after the target module is imported naturally.

    Deferring until first import keeps `import vllm` from pulling in the
    Inductor stack (slow on startup and order-sensitive: in CPU CI eager C++
    codegen imports can fail with "import of torch._inductor.async_compile
    halted; None in sys.modules").
    """
    import sys

    if module_name in sys.modules:
        callback()
        return

    import importlib.abc

    class _PatchFinder(importlib.abc.MetaPathFinder):
        _resolving = False

        def find_spec(self, fullname, path, target=None):
            if fullname != module_name or self._resolving:
                return None
            self._resolving = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._resolving = False
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            original_exec = loader.exec_module

            def _exec_then_patch(module):
                spec.loader = module.__loader__ = loader
                original_exec(module)
                callback()
                if self in sys.meta_path:
                    sys.meta_path.remove(self)

            # A finder can reuse its loader. Wrap only this spec's execution.
            class _Loader(importlib.abc.Loader):
                exec_module = staticmethod(_exec_then_patch)  # type: ignore[assignment]

                def __getattr__(self, name):
                    return getattr(loader, name)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _PatchFinder())


# ===================================================
# torch < 2.13 Inductor implicit-fallback log message
# ===================================================
# Before torch 2.13, GraphLowering.call_function builds
#   log.info("Creating implicit fallback for:\n%s",
#            error.operator_str(target, args, kwargs))
# eagerly, even when INFO logging is off. operator_str stringifies every IR
# argument and IRNode.__str__ has no depth or cycle protection, so deep IR
# with shared nodes can effectively hang compilation (see #42129, #43111).
# Only print the target for vllm ops. Remove once every platform pins
# torch >= 2.13.

_VLLM_FALLBACK_NAMESPACE_PREFIXES = ("vllm::", "vllm_aiter::")


def _apply_inductor_operator_str_patch() -> None:
    from torch._inductor.exc import OperatorIssue

    original = OperatorIssue.operator_str

    def operator_str(target, args, kwargs) -> str:
        name = target.name() if hasattr(target, "name") else str(target)
        if name.startswith(_VLLM_FALLBACK_NAMESPACE_PREFIXES):
            return f"  target: {target}"
        return original(target, args, kwargs)

    OperatorIssue.operator_str = staticmethod(operator_str)


if not is_torch_equal_or_newer("2.13.0.dev"):
    _run_after_module_import("torch._inductor.exc", _apply_inductor_operator_str_patch)


def _apply_inductor_pattern_matcher_patch() -> None:
    """Allow custom ops and functionalization wrappers with unsupported dtypes."""
    from torch._inductor import pattern_matcher
    from torch._inductor.lowering import fallback_node_due_to_unsupported_type

    def fallback_for_builtin(node, allow_cpu_inputs=True):
        return (
            isinstance(node.target, torch._ops.OpOverload)
            and torch._library.utils.is_builtin(node.target)
            and fallback_node_due_to_unsupported_type(node, allow_cpu_inputs)
        )

    pattern_matcher.fallback_node_due_to_unsupported_type = fallback_for_builtin


# Remove once the minimum supported torch includes
# https://github.com/pytorch/pytorch/pull/196013.
if not is_torch_equal_or_newer("2.16.0.dev"):
    _run_after_module_import(
        "torch._inductor.pattern_matcher", _apply_inductor_pattern_matcher_patch
    )


# ===================================================
# torch >= 2.12 Inductor best_config path of a copied compile cache
# ===================================================
# A Triton autotuner restored from a compile cache entry keeps the absolute
# best_config path of the directory it was compiled in. In a cache directory
# that was copied elsewhere it misses its saved config and saves a new one to
# the old location. Point it at the cache directory in use, as Inductor already
# does for the kernel binary. Remove once the minimum supported torch includes
# a fix for https://github.com/pytorch/pytorch/issues/206320.
def _apply_inductor_autotune_cache_path_patch() -> None:
    from functools import wraps
    from pathlib import Path

    from torch._inductor.runtime.runtime_utils import cache_dir
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    original = CachingAutotuner.recheck_autotune_cache

    @wraps(original)
    def recheck_autotune_cache(self, reload_kernel_from_src):
        cache = self.save_cache_hook and self.save_cache_hook.__self__
        if self.filename and cache and cache.local_cache:
            old = Path(self.filename)
            new = Path(cache_dir(), old.parent.name, old.name)
            # Keep the original location if the copy omitted the kernel source.
            if new != old and new.is_file():
                backend, key = cache.local_cache
                self.filename = str(new)
                cache.local_cache = (backend, str(new.parent / Path(key).name))
        return original(self, reload_kernel_from_src)

    CachingAutotuner.recheck_autotune_cache = recheck_autotune_cache


if is_torch_equal_or_newer("2.12.0.dev"):
    _run_after_module_import(
        "torch._inductor.runtime.triton_heuristics",
        _apply_inductor_autotune_cache_path_patch,
    )
