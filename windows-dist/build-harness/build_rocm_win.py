"""Build vLLM's _rocm_C extension for Windows (path-corrected for this machine)."""
import os, shutil, sys, time

import torch
from torch.utils import cpp_extension
from torch.utils.hipify import hipify_python as _hp

_orig = _hp.hipify
def _no_none(*a, **k):
    r = _orig(*a, **k)
    for key, v in r.items():
        if getattr(v, "hipified_path", None) is None:
            v.hipified_path = key
    return r
_hp.hipify = _no_none

ROCM = r"E:\ROCM-7.13.0-Windows"
os.environ["HIP_PATH"] = ROCM
os.environ["ROCM_HOME"] = ROCM
os.environ["ROCM_PATH"] = ROCM
VLLM_ROOT = r"C:\Users\rr\Desktop\vllm"
VLLM_CSRC = os.path.join(VLLM_ROOT, "csrc")
HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "shim")
HIPDIR = os.path.join(HERE, "_hip")
BUILD_DIR = os.path.join(HERE, "_build")
DEVICE_LIB = os.path.join(ROCM, r"lib\llvm\amdgcn\bitcode")

os.makedirs(SHIM, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "ATen/cuda/Exceptions.h": "#include <ATen/hip/Exceptions.h>\n",
    "c10/cuda/CUDAGuard.h": "#include <c10/hip/HIPGuard.h>\n#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAStream.h": "#include <c10/hip/HIPStream.h>\n#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
}
for rel, body in SHIMS.items():
    dst = os.path.join(SHIM, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write("#pragma once\n" + body)

print(f"torch {torch.__version__} hip {torch.version.hip}")
print(f"=== hipify {VLLM_ROOT}\\vllm\\csrc -> {HIPDIR} ===")
shutil.rmtree(HIPDIR, ignore_errors=True)
shutil.copytree(VLLM_CSRC, HIPDIR)

from torch.utils.hipify.hipify_python import RE_PYTORCH_PREPROCESSOR, PYTORCH_MAP

def _pt(mo):
    return str(PYTORCH_MAP[mo.group(1)])

n = 0
for dp, _, fns in os.walk(HIPDIR):
    for fn in fns:
        if not fn.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc")):
            continue
        p = os.path.join(dp, fn)
        s = open(p, encoding="utf-8", errors="ignore").read()
        s2 = RE_PYTORCH_PREPROCESSOR.sub(_pt, s)
        if s2 != s:
            open(p, "w", encoding="utf-8", newline="\n").write(s2)
            n += 1
print(f"rewrote {n} files")

src = [
    os.path.join(HIPDIR, "rocm", "skinny_gemms.cu"),
    os.path.join(HERE, "win_rocm_bindings.cu"),
]
print("=== compiling vllm_win_rocm_C ===")
sys.stdout.flush()
t0 = time.perf_counter()
ext = cpp_extension.load(
    name="_rocm_C",
    sources=src,
    extra_include_paths=[SHIM, HIPDIR],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-DTORCH_HIP_VERSION=0",
        "-DUSE_ROCM=1",
        f"-I{SHIM}",
        f"-I{HIPDIR}",
    ],
    extra_ldflags=[
        f"/LIBPATH:{ROCM}\\lib",
        "hipblas.lib",
        "rocblas.lib",
        "amdhip64.lib",
    ],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")

# Copy into vLLM site-packages so torch.ops._rocm_C resolves
out_dir = os.path.join(HERE, "_build")
for f in os.listdir(out_dir):
    if "vllm_win_rocm_C" in f and f.endswith((".pyd", ".dll")):
        src_f = os.path.join(out_dir, f)
        dst_f = os.path.join(VLLM_ROOT, "vllm", f.replace("vllm_win_rocm_C", "_rocm_C"))
        shutil.copy2(src_f, dst_f)
        print(f"Copied {src_f} -> {dst_f}")
