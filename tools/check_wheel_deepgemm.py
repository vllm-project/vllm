# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify the wheel ships a `_C.cpython-X.Y-*.so` for every required Python
under `vllm/third_party/deep_gemm/`. Wheels without the deep_gemm package
(non-SM90/100 builds) are skipped.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import regex as re

REQUIRED = ("3.10", "3.11", "3.12", "3.13", "3.14")
SO_RE = re.compile(r"vllm/third_party/deep_gemm/_C\.cpython-(\d)(\d+)-")
PKG_MARKER = "vllm/third_party/deep_gemm/__init__.py"


def check(wheel: Path, required: list[str]) -> int:
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
    if PKG_MARKER not in names:
        print(f"  {wheel.name}: no DeepGEMM package — skipping")
        return 0
    found = {f"{m[1]}.{m[2]}" for n in names if (m := SO_RE.match(n))}
    missing = [v for v in required if v not in found]
    print(f"  {wheel.name}: found {sorted(found)}, missing {missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--required", default=",".join(REQUIRED))
    args = ap.parse_args()
    required = [v for v in args.required.split(",") if v]
    rc = 0
    for wheel in sorted(args.directory.glob("*.whl")):
        rc |= check(wheel, required)
    sys.exit(rc)
