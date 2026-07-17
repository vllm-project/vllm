# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assert the vendored DeepGEMM package has the TORCH_LIBRARY binding layout.

Expects ``deep_gemm/_C.py`` plus a single ``_C_extension*.so`` (abi3) under
``vllm.third_party.deep_gemm``. Run after vLLM is installed, e.g. the H100
deepgemm kernel tests in .buildkite/test_areas/kernels.yaml.
"""

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    spec = importlib.util.find_spec("vllm.third_party.deep_gemm")
    if spec is None or spec.origin is None:
        print(
            "vllm.third_party.deep_gemm not importable; is vllm installed?",
            file=sys.stderr,
        )
        return 1
    pkg_dir = Path(spec.origin).parent

    shim = pkg_dir / "_C.py"
    so_files = sorted(pkg_dir.glob("_C_extension*.so"))
    missing = []
    if not shim.is_file():
        missing.append("_C.py")
    if not so_files:
        missing.append("_C_extension*.so")

    print(
        f"deepgemm vendored binding: shim={shim.is_file()}, "
        f"extensions={[p.name for p in so_files]}"
    )
    if missing:
        print(f"missing: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
