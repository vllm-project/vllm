# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Assemble and link the generated gfx950 kernels, and reach HIP through ctypes.

Kernels are generated per process at first use (some depend on the TP rank or the model's max length), assembled
with clang and linked with ld.lld into code objects cached by source hash under $VLLM_CACHE_ROOT/dsv41_asm.
vLLM's TP workers build the same kernels at once: every build writes per-process files and renames atomically,
and the runner construction is serialised with ``build_lock()`` so one rank builds and the others hit the cache."""

import contextlib
import ctypes
import fcntl
import glob
import hashlib
import importlib.util
import os
import shutil
import subprocess
import tempfile

import vllm.envs as envs


def cache_dir() -> str:
    d = os.path.join(envs.VLLM_CACHE_ROOT, "dsv41_asm")
    os.makedirs(d, exist_ok=True)
    return d


@contextlib.contextmanager
def build_lock():
    """Cross-process lock around kernel builds (vLLM's TP ranks build the same kernels at once)."""
    with open(os.path.join(cache_dir(), ".build.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _llvm_dirs() -> list[str]:
    """ROCm's LLVM: the pip ROCm SDK (vLLM's ROCm images), $ROCM_PATH, a system
    ROCm; PATH only after them (a host LLVM may lack gfx950)."""
    dirs = []
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec is not None and spec.submodule_search_locations:
        dirs += [
            os.path.join(p, "lib/llvm/bin") for p in spec.submodule_search_locations
        ]
    if rocm := os.environ.get("ROCM_PATH"):
        dirs += [os.path.join(rocm, "lib/llvm/bin"), os.path.join(rocm, "llvm/bin")]
    return dirs + ["/opt/rocm/llvm/bin", *sorted(glob.glob("/opt/rocm-*/llvm/bin"))]


def _tool(name: str) -> str | None:
    for d in _llvm_dirs():
        p = os.path.join(d, name)
        if os.access(p, os.X_OK):
            return p
    return shutil.which(name)


# gfx950-only instructions the kernels use (an LLVM without them can't build them)
_PROBE = """.text
  v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3], v20, v21
  v_permlane32_swap_b32 v0, v1
"""

_tools: dict = {}


def _assemble(src_path: str, obj: str, co: str) -> str | None:
    """Clang + ld.lld for gfx950 → None, or the first error lines."""
    for cmd in (
        [_tools["clang"], "-x", "assembler", "-target", "amdgcn-amd-amdhsa"]
        + ["-mcpu=gfx950", "-c", src_path, "-o", obj],
        [_tools["lld"], "-shared", obj, "-o", co],
    ):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return f"{cmd[0]}: " + " ".join(r.stderr.splitlines()[:4])
    return None


def toolchain_missing() -> str | None:
    """None if clang and ld.lld assemble and link gfx950 code, else what is
    missing (checked once per process)."""
    if "missing" not in _tools:
        _tools.update(clang=_tool("clang"), lld=_tool("ld.lld"))
        missing = " and ".join(
            n for k, n in (("clang", "clang"), ("lld", "ld.lld")) if _tools[k] is None
        )
        if not missing:
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "p.s"), "w") as f:
                    f.write(_PROBE)
                err = _assemble(*(os.path.join(d, n) for n in ("p.s", "p.o", "p.co")))
            if err:
                missing = f"an LLVM that assembles gfx950 ({err})"
        _tools["missing"] = missing or None
    return _tools["missing"]


def build(src: str, name: str) -> bytes:
    """Assemble + link ``src`` → code object bytes, cached by source hash."""
    if toolchain_missing():
        raise FileNotFoundError(
            f"building the asm decode kernels needs {toolchain_missing()}"
        )
    h = hashlib.sha1(src.encode()).hexdigest()[:12]
    base = os.path.join(cache_dir(), f"{name}-{h}")
    co = base + ".co"
    if not os.path.exists(co):
        # per-process files, then an atomic rename (several ranks may build the same new kernel at once)
        tmp = f"{base}.{os.getpid()}"
        with open(tmp + ".s", "w") as f:
            f.write(src)
        if err := _assemble(tmp + ".s", tmp + ".o", tmp + ".co"):
            raise RuntimeError(f"building {tmp}.s failed: {err}")
        os.remove(tmp + ".s")
        os.remove(tmp + ".o")
        os.replace(tmp + ".co", co)
    with open(co, "rb") as f:
        return f.read()


_hip = None


def hip():
    """The HIP runtime torch loaded (found in the process's maps, so it is the same library instance)."""
    global _hip
    if _hip is None:
        import torch

        torch.zeros(1, device="cuda")  # make torch load and initialise the runtime
        lib = None
        with open("/proc/self/maps") as f:
            for line in f:
                if "libamdhip64.so" in line:
                    lib = line.split()[-1]
                    break
        _hip = ctypes.CDLL(lib or "libamdhip64.so")
        _hip.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )
        _hip.hipGetErrorString.restype = ctypes.c_char_p
    return _hip


def _check(err: int, what: str) -> None:
    if err:
        raise RuntimeError(
            f"{what}: hip error {err} {hip().hipGetErrorString(err).decode()}"
        )
