# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import os


def _get_torch_cuda_version():
    """Read Torch's CUDA version without importing Torch.

    Importing Torch may initialize CUDA before the compatibility library path
    can be set.
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
        ver_spec = importlib.util.spec_from_file_location("torch.version", version_path)
        if not ver_spec or not ver_spec.loader:
            return None
        module = importlib.util.module_from_spec(ver_spec)
        # Avoid registering in sys.modules to not confuse future imports.
        ver_spec.loader.exec_module(module)
        return getattr(module, "cuda", None)
    except Exception:
        return None


def _maybe_set_cuda_compatibility_path(get_torch_cuda_version=None):
    """Set LD_LIBRARY_PATH for CUDA forward compatibility if enabled.

    This must run before importing Torch because the dynamic linker only
    consults LD_LIBRARY_PATH when CUDA libraries are first loaded.
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
        get_torch_cuda_version = get_torch_cuda_version or _get_torch_cuda_version
        torch_cuda_version = get_torch_cuda_version()
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
        return

    new_paths = [norm_path] + [
        path for path in ld_paths if not path or os.path.normpath(path) != norm_path
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_paths)


def apply_pre_torch_environment() -> None:
    """Apply vLLM process defaults that must precede importing Torch."""
    _maybe_set_cuda_compatibility_path()

    # Avoid unintentional CUDA initialization from torch.cuda.is_available().
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

    # See https://github.com/vllm-project/vllm/issues/10480 and
    # https://github.com/vllm-project/vllm/issues/10619.
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

    # Cache Triton autotuning results to disk across process restarts.
    os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

    # Avoid TileLang's world-shared /tmp debug directory on shared hosts.
    os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")


def apply_runtime_environment() -> None:
    """Apply Torch-dependent overrides at supported runtime boundaries.

    Import-light paths call apply_pre_torch_environment() only. The CLI
    execution path, public lazy API, configuration, compiler, and model-runner
    paths call this function before entering the runtime. Python's module cache
    makes successful repeated calls idempotent; import failures propagate.
    """
    importlib.import_module("vllm.env_override")
