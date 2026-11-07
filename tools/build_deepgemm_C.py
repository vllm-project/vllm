# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's TORCH_LIBRARY extension and copy vendored artifacts.

DeepGEMM now registers ops via TORCH_LIBRARY into ``deep_gemm._C_extension``
(abi3) and exposes the legacy API through ``deep_gemm/_C.py``. This driver
delegates to DeepGEMM's ``setup.py build_ext --inplace`` and copies the shim
plus extension into the cmake output directory.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR> [--launcher CMD]

``--launcher`` puts a compiler launcher such as ccache in front of the compiler
setuptools uses, as CMAKE_CXX_COMPILER_LAUNCHER does for the rest of the build.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("src", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--launcher", default="")
args = parser.parse_args()

src = args.src.resolve()
out = args.out.resolve()
_pkg = src / "deep_gemm"
out.mkdir(parents=True, exist_ok=True)

if not (_pkg / "_C.py").is_file():
    sys.exit(
        f"DeepGEMM source at {src} is missing deep_gemm/_C.py; "
        "expected TORCH_LIBRARY migration layout"
    )

# Avoid DeepGEMM's clean-git assertion when vendoring a local dirty tree.
env = os.environ.copy()
env.pop("DG_SKIP_CUDA_BUILD", None)
if args.launcher.strip():
    for var in ("CC", "CXX"):
        compiler = env.get(var) or sysconfig.get_config_var(var)
        if compiler and not shlex.split(compiler)[0].endswith("ccache"):
            env[var] = f"{args.launcher} {compiler}"

print(f"[build_deepgemm_C] building in {src} with {sys.executable}", flush=True)
# CMake decides when to rebuild; setuptools would only check python_api.cpp.
subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
    cwd=src,
    env=env,
)

shim = _pkg / "_C.py"
shutil.copy2(shim, out / shim.name)

so = _pkg / "_C_extension.abi3.so"
if not so.is_file():
    sys.exit(f"DeepGEMM build did not produce {so}")
for stale in out.glob("_C_extension*.so"):
    stale.unlink()
shutil.copy2(so, out / so.name)
print(f"[build_deepgemm_C] installed {so.name} -> {out}", flush=True)
