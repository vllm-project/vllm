"""
utils/compile.py

Compiles all CUDA kernels and C++ bindings into a shared cuda_extension
module using PyTorch's C++ extension infrastructure.

Do NOT modify this file.

Usage:
    python3 -m utils.compile
    # or via shell wrapper:
    TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent


def find_sources() -> list[str]:
    """Locate all .cu and .cpp source files in ROOT and ROOT/kernels/."""
    sources: list[str] = []
    for directory in [_ROOT, _ROOT / "kernels"]:
        for suffix in (".cu", ".cpp"):
            sources.extend(str(p) for p in directory.glob(f"*{suffix}"))
    return sorted(set(sources))


def compile_kernels() -> int:
    """Build the cuda_extension shared library. Returns 0 on success, 1 on failure."""
    try:
        from torch.utils import cpp_extension as cpp_ext
    except ImportError:
        print("[ERROR] PyTorch is not installed.", file=sys.stderr)
        return 1

    build_dir = str(_ROOT / "build" / "forced_compile")
    output_so = str(_ROOT / "cuda_extension.so")

    sources = find_sources()
    if not sources:
        print("[ERROR] No .cu / .cpp source files found.", file=sys.stderr)
        return 1

    print(f"[INFO] Compiling {len(sources)} source file(s):")
    for s in sources:
        print(f"       {s}")

    # Clean previous build artefacts
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    try:
        module = cpp_ext.load(
            name="cuda_extension",
            sources=sources,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
            ],
            build_directory=build_dir,
            verbose=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Compilation failed: {exc}", file=sys.stderr)
        return 1

    # Locate the compiled .so and copy to agent_workdir root
    so_path = Path(module.__file__)  # type: ignore[attr-defined]
    if not so_path.exists():
        print(f"[ERROR] Compiled .so not found at {so_path}", file=sys.stderr)
        return 1

    shutil.copy(str(so_path), output_so)
    print(f"[INFO] cuda_extension.so written to {output_so}")
    return 0


def main() -> int:
    return compile_kernels()


if __name__ == "__main__":
    sys.exit(main())
