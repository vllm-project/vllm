# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a portable vLLM wheel for load-bench, without native extensions.

Requires build and the build dependencies listed in pyproject.toml.
The source build configuration and native build remain unchanged.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=source / "dist")
    parser.add_argument("--version", default="0.28.1rc0+cpuload1")
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vllm-cpu-wheel-") as tmp:
        staged = Path(tmp)
        for name in ("vllm", "tools", "requirements"):
            shutil.copytree(
                source / name,
                staged / name,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.egg-info",
                    "*.so",
                    "*.so.*",
                    "*.dylib",
                    "*.pyd",
                    "*.dll",
                    "vllm-rs",
                    "build",
                    "dist",
                ),
            )
        for name in ("pyproject.toml", "setup.py", "README.md", "LICENSE"):
            shutil.copy2(source / name, staged / name)

        setup = staged / "setup.py"
        config = setup.read_text()
        replacements = [
            (
                'sys.platform.startswith("darwin") and VLLM_TARGET_DEVICE != "cpu"',
                (
                    'sys.platform.startswith("darwin") '
                    'and VLLM_TARGET_DEVICE not in ("cpu", "empty")'
                ),
            ),
            (
                (
                    "rust_extensions = rust_build.rust_extensions(\n"
                    "    optional=not should_require_rust_frontend()\n)"
                ),
                "rust_extensions = []",
            ),
            # The empty-device build uses common.txt, which omits PyTorch.
            (
                "install_requires=get_requirements(),",
                'install_requires=[*get_requirements(), "torch==2.14.0"],',
            ),
        ]
        for old, new in replacements:
            if config.count(old) != 1:
                raise RuntimeError(f"Expected exactly one setup.py fragment: {old}")
            config = config.replace(old, new)
        setup.write_text(config)

        env = os.environ.copy()
        env.update(
            VLLM_TARGET_DEVICE="empty",
            VLLM_USE_PRECOMPILED="0",
            VLLM_USE_PRECOMPILED_RUST="0",
            VLLM_PRECOMPILED_WHEEL_LOCATION="",
            VLLM_REQUIRE_RUST_FRONTEND="0",
            VLLM_VERSION_OVERRIDE=args.version,
            SETUPTOOLS_SCM_PRETEND_VERSION=args.version,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(outdir),
                str(staged),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
