# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's TORCH_LIBRARY extension and copy vendored artifacts.

DeepGEMM now registers ops via TORCH_LIBRARY into ``deep_gemm._C_extension``
(abi3) and exposes the legacy API through ``deep_gemm/_C.py``. This driver
delegates to DeepGEMM's ``setup.py build_ext --inplace`` and copies the shim
plus extension into the cmake output directory.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR>
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
_pkg = src / "deep_gemm"
out.mkdir(parents=True, exist_ok=True)

if not (_pkg / "_C.py").is_file():
    sys.exit(
        f"DeepGEMM source at {src} is missing deep_gemm/_C.py; "
        "expected TORCH_LIBRARY migration layout"
    )

# Avoid DeepGEMM's clean-git assertion when vendoring a local dirty tree.
env = os.environ.copy()
env.setdefault("DG_USE_LOCAL_VERSION", "0")
env.pop("DG_SKIP_CUDA_BUILD", None)

print(f"[build_deepgemm_C] building in {src} with {sys.executable}", flush=True)
subprocess.check_call(
    [sys.executable, "setup.py", "build_ext", "--inplace"],
    cwd=src,
    env=env,
)

shim = _pkg / "_C.py"
shutil.copy2(shim, out / shim.name)

so_files = sorted(_pkg.glob("_C_extension*.so"))
if not so_files:
    sys.exit(f"DeepGEMM build did not produce deep_gemm/_C_extension*.so under {src}")
for so in so_files:
    shutil.copy2(so, out / so.name)
    print(f"[build_deepgemm_C] installed {so.name} -> {out}", flush=True)
