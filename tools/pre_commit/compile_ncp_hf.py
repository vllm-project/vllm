# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lock the HF-only NCP environment against the main CUDA test dependencies."""

import os
import subprocess
import tempfile
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # Relax only these two test-lock pins. Resolve normally, without overrides.
    pins = re.findall(
        r"^([\w.-]+)==([^\s\\]+)",
        (ROOT / "requirements/test/cuda.txt").read_text(),
        re.MULTILINE,
    )
    with tempfile.TemporaryDirectory() as directory:
        constraints = Path(directory) / "constraints.txt"
        constraints.write_text(
            "".join(
                f"{name}=={value}\n"
                for name, value in pins
                if name not in {"transformers", "huggingface-hub"}
            )
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "compile",
                "--generate-hashes",
                "--no-annotate",
                "requirements/test/ncp_hf.in",
                "-c",
                str(constraints),
                "-o",
                "requirements/test/ncp_hf.txt",
                "--torch-backend",
                "cu130",
                "--python-platform",
                "x86_64-manylinux_2_28",
                "--python-version",
                "3.12",
                "--custom-compile-command",
                ".venv/bin/python tools/pre_commit/compile_ncp_hf.py",
            ],
            cwd=ROOT,
            env={k: v for k, v in os.environ.items() if k != "UV_OVERRIDE"},
            check=True,
        )


if __name__ == "__main__":
    main()
