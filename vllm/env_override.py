# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
import importlib.util
import os


def _get_torch_cuda_version():
    """Peripheral function to _maybe_set_cuda_compatibility_path().
    PyTorch version must not be determined by importing directly
    because it will trigger the CUDA initialization, losing the
    chance to set the LD_LIBRARY_PATH beforehand.
    """
    try:
        spec = importlib.util.find_spec("torch")
        if not spec:
            return None
        if spec.origin:
            torch_root = os.path.dirname(spec.origin)
        elif spec.submodule_search_locations:
            torch_root = spec.submodule_search_locations[0]
        else:
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
        return getattr(module, "cuda", None)
    except Exception:
        return None


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


_maybe_set_cuda_compatibility_path()

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import is_torch_equal_or_newer

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

# ===================================================
# torch <2.12 GraphCaptureOutput.get_runtime_env monkeypatch
# ===================================================
# PyTorch's AOT compile path omits builtins from used_globals, causing
# 'Missing required external references' errors for refs like 'type'.
# (which happens in transformers code)
# This mirrors the fix in https://github.com/pytorch/pytorch/pull/177558
# and can be removed once torch >=2.12 is the minimum supported version.

# ===================================================
# torch >= 2.11 Inductor constrain_to_fx_strides monkeypatch
# ===================================================
# Inductor's constrain_to_fx_strides calls .stride() on every FX arg's meta
# value, which crashes on FakeScriptObject (the compile-time proxy for hoisted
# opaque types). The patched version skips args whose meta value is not a
# torch.Tensor.
# Upstream issue: https://github.com/pytorch/pytorch/issues/175973


_constrain_to_fx_strides_patched = False


def _apply_constrain_to_fx_strides_patch():
    """Patch lowering.constrain_to_fx_strides globally. Safe to call
    multiple times; only the first call does anything.
    Only applies for torch >= 2.11 and < 2.12."""
    global _constrain_to_fx_strides_patched
    if _constrain_to_fx_strides_patched:
        return
    _constrain_to_fx_strides_patched = True

    if not is_torch_equal_or_newer("2.11.0.dev") or is_torch_equal_or_newer(
        "2.12.0.dev"
    ):
        return

    import torch._inductor.ir as _ir
    import torch._inductor.lowering as _lowering
    from torch._inductor.virtualized import V as _V

    def _patched(fx_node, *args, **kwargs):
        def apply_constraint(arg, fx_arg):
            if isinstance(arg, _ir.IRNode):
                meta_val = fx_arg.meta.get("val")
                if isinstance(meta_val, torch.Tensor):
                    stride_order = _ir.get_stride_order(
                        meta_val.stride(), _V.graph.sizevars.shape_env
                    )
                    return _ir.ExternKernel.require_stride_order(arg, stride_order)
                return arg
            if isinstance(arg, dict):
                return {key: apply_constraint(arg[key], fx_arg[key]) for key in arg}
            return arg

        args = tuple(
            apply_constraint(arg, fx_arg) for arg, fx_arg in zip(args, fx_node.args)
        )
        kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for k, v in kwargs.items()}
        return args, kwargs

    _lowering.constrain_to_fx_strides = _patched


if is_torch_equal_or_newer("2.10.0") and not is_torch_equal_or_newer("2.12.0.dev"):
    import builtins as _builtins
    import pickle

    from torch._dynamo.convert_frame import GraphCaptureOutput

    _original_get_runtime_env = GraphCaptureOutput.get_runtime_env

    def _safe_builtins_dict(builtins_dict: dict) -> dict:
        """Filter a builtins dict to only picklable entries for serialization."""
        result = {}
        for k, v in builtins_dict.items():
            try:
                pickle.dumps(v)
                result[k] = v
            except Exception:
                pass
        return result

    def _patched_get_runtime_env(self):  # type: ignore[no-untyped-def]
        runtime_env = _original_get_runtime_env(self)
        for ref in runtime_env.external_refs:
            if ref not in runtime_env.used_globals:
                if ref.startswith("__builtins_dict__") and ref in self.f_globals:
                    runtime_env.used_globals[ref] = _safe_builtins_dict(
                        self.f_globals[ref]
                    )
                elif hasattr(_builtins, ref):
                    runtime_env.used_globals[ref] = getattr(_builtins, ref)
        return runtime_env

    GraphCaptureOutput.get_runtime_env = _patched_get_runtime_env

# ===================================================
# torch 2.10 FxGraphCachePickler.dumps ValueError fix
# ===================================================
# PyTorch 2.10's FxGraphCachePickler.dumps() doesn't catch ValueError,
# causing torch.compile cache failures when tensors with non-standard
# layouts (e.g. blocked-layout prepacked weights) are serialized.
# PyTorch mainline fixed this in pytorch/pytorch#176557 (merged 2026-03-04).
# This is a thin backport for 2.10 users; remove once 2.10 is dropped.


def _apply_fxgraphcache_pickle_patch(pickler_cls, bypass_cls):
    """Wrap pickler_cls.dumps to convert ValueError into bypass_cls.

    Idempotent: sets `_vllm_fxgraph_dumps_patched` on the class after the
    first apply to prevent re-application. The wrapper function is also
    marked with `_vllm_patched` as an additional safeguard.
    """
    if getattr(pickler_cls, "_vllm_fxgraph_dumps_patched", False):
        return

    original_dumps = pickler_cls.dumps
    if hasattr(original_dumps, "_vllm_patched"):
        return

    def patched_dumps(self, obj):
        try:
            return original_dumps(self, obj)
        except ValueError as e:
            raise bypass_cls("Failed to pickle cache key") from e

    patched_dumps._vllm_patched = True  # type: ignore[attr-defined]
    pickler_cls.dumps = patched_dumps
    pickler_cls._vllm_fxgraph_dumps_patched = True  # type: ignore[attr-defined]


def _patch_fxgraphcache_pickle_if_needed():
    """Apply FxGraphCachePickler.dumps ValueError backport when on torch 2.10.x."""
    from vllm.utils.torch_utils import is_torch_equal_or_newer

    if not is_torch_equal_or_newer("2.10.0") or is_torch_equal_or_newer("2.11.0"):
        return

    from torch._inductor.codecache import BypassFxGraphCache, FxGraphCachePickler

    _apply_fxgraphcache_pickle_patch(FxGraphCachePickler, BypassFxGraphCache)


_patch_fxgraphcache_pickle_if_needed()

# ===================================================
# torch 2.11 Inductor cpp codegen indirect_assert scalar-mask fix
# ===================================================
# CppVecKernel.indirect_assert wraps a scalar mask with
# `VecMask<...>(scalar)`, which is not a valid constructor and triggers a
# C++ compile error during torch.compile of any model that does indirect
# indexing inside a tail-vectorized loop (e.g. Qwen3-VL).
# Failure looks like:
#   no matching function for call to 'VecMask<int64_t,2>::VecMask(int&)'
# Upstream fix in PyTorch mainline replaces the call with
# `VecMask<...>::from(scalar)`, see pytorch/pytorch#178148 (lands in 2.12).
# This is a thin backport for torch >= 2.11 and < 2.12; remove once the
# minimum supported torch is 2.12.


def _apply_cpp_indirect_assert_patch():
    """Replace CppVecKernel.indirect_assert with a fixed copy that uses
    `VecMask<...>::from(scalar)` for scalar masks.

    Idempotent: marks the class with `_vllm_indirect_assert_patched` after
    the first apply.
    """
    from torch._inductor.codegen.cpp import CppVecKernel

    if getattr(CppVecKernel, "_vllm_indirect_assert_patched", False):
        return

    from torch._inductor.codegen.cpp import CppCSEVariable, cexpr_index

    def patched_indirect_assert(self, var, lower, upper, mask=None):
        assert isinstance(var, CppCSEVariable)
        assert var.dtype is not None
        if not var.is_vec:
            if isinstance(mask, CppCSEVariable) and mask.is_vec:
                mask = f"({mask}).all_masked()"
            return super(CppVecKernel, self).indirect_assert(var, lower, upper, mask)
        lower_scalar = lower
        upper_scalar = upper
        if lower:
            lower = f"{self._get_vec_type(var.dtype)}({lower})"
        if upper:
            upper = f"{self._get_vec_type(var.dtype)}({upper})"
        if lower and upper:
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower_scalar} <= {var} < {upper_scalar}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = f"{lower_scalar} <= {var}"
        else:
            assert upper
            cond = f"{var} < {upper}"
            cond_print = f"{var} < {upper_scalar}"
        cond = f"{self._get_mask_type(var.dtype)}({cond})"
        if mask:
            if not mask.is_vec:
                # Backport of pytorch/pytorch#178148 -- use ::from for
                # scalar masks so g++ picks the correct overload.
                mask = f"{self._get_mask_type(var.dtype)}::from({mask})"
            cond = f"({cond}) | ~({mask})"
        if self.tail_size:
            cond = (
                f"{self._get_mask_type(var.dtype)}::set("
                f"{self._get_mask_type(var.dtype)}::from(1)"
                f", ({cond}), {cexpr_index(self.tail_size)})"
            )
        cond = f"({cond}).all_masked()"
        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    CppVecKernel.indirect_assert = patched_indirect_assert
    CppVecKernel._vllm_indirect_assert_patched = True  # type: ignore[attr-defined]


def _patch_cpp_indirect_assert_if_needed():
    """Apply cpp codegen indirect_assert backport when on torch 2.11.x.

    Defers application until torch._inductor.codegen.cpp is naturally
    imported by Inductor. Importing it eagerly during vllm.__init__ pulls
    in torch._inductor.scheduler, whose top-level
    `import torch._inductor.async_compile` can fail with
    `ModuleNotFoundError: import of torch._inductor.async_compile halted;
    None in sys.modules` depending on the import order on the runner
    (observed in vLLM CPU CI).
    """
    if not is_torch_equal_or_newer("2.11.0") or is_torch_equal_or_newer("2.12.0.dev"):
        return

    import sys

    target_name = "torch._inductor.codegen.cpp"
    if target_name in sys.modules:
        _apply_cpp_indirect_assert_patch()
        return

    import importlib.abc

    class _CppCodegenPatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != target_name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def _exec_then_patch(module):
                original_exec(module)
                _apply_cpp_indirect_assert_patch()

            spec.loader.exec_module = _exec_then_patch  # type: ignore[method-assign]
            return spec

    sys.meta_path.insert(0, _CppCodegenPatchFinder())


_patch_cpp_indirect_assert_if_needed()

# ============================================================
# Inductor FALLBACK_ALLOW_LIST fast-path for vllm::*/vllm_aiter::* ops
# ============================================================
# When Inductor encounters a custom op without a registered lowering or
# decomposition (e.g. vllm::all_reduce, vllm_aiter::fused_add_rms_norm) it
# correctly creates an implicit fallback that calls into the eager Python
# impl. However, unless `base_name` (e.g. "vllm::all_reduce") is in
# torch._inductor.lowering.FALLBACK_ALLOW_LIST, GraphLowering.call_function
# (torch/_inductor/graph.py:~1283) takes the slow path that emits
#   log.info("Creating implicit fallback for:\n%s",
#            error.operator_str(target, args, kwargs))
# `operator_str` eagerly recurses through __str__ on every input TensorBox;
# for deep MoE/TP graphs (e.g. Kimi-K2.6 at TP=8) the IR provenance tree
# behind a TP all-reduce input or a residual-fed RMSNorm input is hundreds
# of layers deep, and stringifying it consumes many minutes of CPU per call,
# effectively hanging compilation.
#
# Patching FALLBACK_ALLOW_LIST membership to also match any "vllm::*" or
# "vllm_aiter::*" base_name routes our custom ops through the fast path
# `make_fallback(target, warn=False, override_decomp=True)` instead. This
# preserves all downstream behaviour (allreduce_rms_fusion still pattern-
# matches them, partitioning still works, fallback semantics identical) but
# skips the expensive log formatting on the FIRST encounter of each op.
#
# We wrap the OrderedSet in a thin proxy that:
#   - Returns True from __contains__ for any vllm::*/vllm_aiter::* op
#   - Otherwise delegates to the underlying set (preserving membership of
#     the standard entries like "torchvision::roi_align", "aten::index_add")
#   - Forwards add()/__iter__()/__len__()/etc. so other Inductor code paths
#     that mutate or iterate the set keep working.

_VLLM_FALLBACK_NAMESPACE_PREFIXES = ("vllm::", "vllm_aiter::")


class _VllmFallbackAllowList:
    """Membership proxy that auto-allows vllm::*/vllm_aiter::* base_names."""

    _vllm_patched = True

    def __init__(self, inner):
        self._inner = inner

    def __contains__(self, item):
        if isinstance(item, str) and item.startswith(_VLLM_FALLBACK_NAMESPACE_PREFIXES):
            return True
        return item in self._inner

    def add(self, item):
        self._inner.add(item)

    def discard(self, item):
        self._inner.discard(item)

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)

    def __repr__(self):
        return f"_VllmFallbackAllowList({self._inner!r})"

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch_inductor_fallback_allow_list() -> None:
    """Wrap torch._inductor.lowering.FALLBACK_ALLOW_LIST so any custom op in
    the ``vllm::`` or ``vllm_aiter::`` namespaces is treated as a member.

    Idempotent: a sentinel attribute on the proxy prevents re-wrapping.
    """
    try:
        from torch._inductor import lowering as _lowering
    except ImportError:
        return

    base = getattr(_lowering, "FALLBACK_ALLOW_LIST", None)
    if base is None or getattr(base, "_vllm_patched", False):
        return

    _lowering.FALLBACK_ALLOW_LIST = _VllmFallbackAllowList(base)

    # torch/_inductor/graph.py imports the symbol at module load time:
    #   from torch._inductor.lowering import FALLBACK_ALLOW_LIST
    # so we also need to overwrite the local binding in the graph module if
    # it has already been imported.
    try:
        from torch._inductor import graph as _graph

        if hasattr(_graph, "FALLBACK_ALLOW_LIST"):
            _graph.FALLBACK_ALLOW_LIST = _lowering.FALLBACK_ALLOW_LIST
    except ImportError:
        pass


_patch_inductor_fallback_allow_list()

# ============================================================
# Triton Autotuner determinism
# ============================================================
# Replace the Autotuner.run so it always pick the first running configuration.
# Useful to eliminate autotune variability leading to non determinism.
if os.environ.get("VLLM_TRITON_FORCE_FIRST_CONFIG", "0").strip().lower() in (
    "1",
    "true",
):
    from vllm.triton_utils.force_first_config import (
        install as _install_force_first_config,
    )

    _install_force_first_config()
