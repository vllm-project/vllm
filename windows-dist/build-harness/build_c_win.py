"""Build vLLM _C extension for Windows (libtorch_stable ABI, path-corrected)."""
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
STABLE_SRC = os.path.join(VLLM_CSRC, "libtorch_stable")
HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "shim")
HIPDIR = os.path.join(HERE, "_hip_cext")
BUILD_DIR = os.path.join(HERE, "_build_cext")
DEVICE_LIB = os.path.join(ROCM, r"lib\llvm\amdgcn\bitcode")
os.makedirs(BUILD_DIR, exist_ok=True)

SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n"
                               "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "ATen/cuda/Exceptions.h": "#include <ATen/hip/Exceptions.h>\n",
    "c10/cuda/CUDAGuard.h": "#include <c10/hip/HIPGuard.h>\n"
                            "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAStream.h": "#include <c10/hip/HIPStream.h>\n"
                             "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
}
for rel, body in SHIMS.items():
    dst = os.path.join(SHIM, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write("#pragma once\n" + body)

from torch.utils.hipify.hipify_python import RE_PYTORCH_PREPROCESSOR, PYTORCH_MAP

print("torch", torch.__version__, "hip", torch.version.hip)

# Hipify: copy entire csrc, then replace cuda->hip
print(f"=== hipify csrc -> {HIPDIR} ===")
shutil.rmtree(HIPDIR, ignore_errors=True)
shutil.copytree(VLLM_CSRC, HIPDIR)

def _pt(mo):
    return str(PYTORCH_MAP[mo.group(1)])

n = 0
for dp, _, fns in os.walk(HIPDIR):
    for fn in fns:
        if not fn.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc")):
            continue
        p = os.path.join(dp, fn)
        try:
            s = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        s2 = RE_PYTORCH_PREPROCESSOR.sub(_pt, s)
        if s2 != s:
            open(p, "w", encoding="utf-8", newline="\n").write(s2)
            n += 1
print("rewrote", n, "files")

# Source files from libtorch_stable/ (upstream stable ABI)
HIP_STABLE = os.path.join(HIPDIR, "libtorch_stable")

# Avoid duplicate object names: rename int8 per_token_group_quant.cu
int8_ptg = os.path.join(HIP_STABLE, "quantization", "w8a8", "int8", "per_token_group_quant.cu")
int8_ptg_renamed = os.path.join(HIP_STABLE, "quantization", "w8a8", "int8", "per_token_group_quant_int8.cu")
if os.path.exists(int8_ptg) and not os.path.exists(int8_ptg_renamed):
    os.rename(int8_ptg, int8_ptg_renamed)
src = [
    os.path.join(HERE, "win_c_bindings_shim.cpp"),
    os.path.join(HIP_STABLE, "activation_kernels.cu"),
    os.path.join(HIP_STABLE, "layernorm_kernels.cu"),
    os.path.join(HIP_STABLE, "layernorm_quant_kernels.cu"),
    os.path.join(HIP_STABLE, "pos_encoding_kernels.cu"),
    os.path.join(HIP_STABLE, "quantization", "gptq", "q_gemm.cu"),
    os.path.join(HIP_STABLE, "sampler.cu"),
    os.path.join(HIP_STABLE, "permute_cols.cu"),
    os.path.join(HIPDIR, "cuda_view.cu"),
    os.path.join(HIP_STABLE, "cuda_utils_kernels.cu"),
    os.path.join(HIP_STABLE, "topk.cu"),
    os.path.join(HIP_STABLE, "cache_kernels.cu"),
    os.path.join(HIPDIR, "custom_quickreduce.cu"),
    os.path.join(HIP_STABLE, "custom_all_reduce.cu"),
    # Quant kernels
    os.path.join(HIP_STABLE, "quantization", "w8a8", "fp8", "per_token_group_quant.cu"),
    os.path.join(HIP_STABLE, "quantization", "w8a8", "fp8", "common.cu"),
    os.path.join(HIP_STABLE, "quantization", "w8a8", "int8", "scaled_quant.cu"),
    os.path.join(HIP_STABLE, "quantization", "w8a8", "int8", "per_token_group_quant_int8.cu"),
    os.path.join(HIP_STABLE, "quantization", "fused_kernels", "fused_layernorm_dynamic_per_token_quant.cu"),
    os.path.join(HIP_STABLE, "quantization", "fused_kernels", "fused_silu_mul_block_quant.cu"),
]

# Verify all source files exist
existing = []
for s in src:
    if os.path.exists(s):
        existing.append(s)
    else:
        print(f"SKIP (not found): {os.path.relpath(s, HIPDIR)}")

print(f"=== compiling _C with {len(existing)} source files ===")
sys.stdout.flush()
t0 = time.perf_counter()

extra_include_dirs = [
    SHIM,
    HIP_STABLE,  # before HIPDIR so libtorch_stable/ops.h wins over csrc/ops.h
    HIPDIR,
    os.path.join(HIPDIR, "core"),
]

cpp_extension.load(
    name="_C",
    sources=existing,
    extra_include_paths=extra_include_dirs,
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        "-D_USE_MATH_DEFINES",
        "-DENABLE_FP8",
        "-DTORCH_HIP_VERSION=0",
        "-DUSE_ROCM=1",
    ] + [f"-I{p}" for p in extra_include_dirs],
    extra_ldflags=[
        f"/LIBPATH:{ROCM}\\lib",
        "hipblas.lib", "rocblas.lib", "amdhip64.lib",
    ],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")

# Copy _C.pyd into vLLM
for f in os.listdir(BUILD_DIR):
    if f.endswith(".pyd") and "_C" in f:
        shutil.copy2(os.path.join(BUILD_DIR, f), os.path.join(VLLM_ROOT, "vllm", "_C.pyd"))
        print(f"Copied -> vllm/_C.pyd")
