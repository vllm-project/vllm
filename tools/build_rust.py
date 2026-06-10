# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rust build support shared by `setup.py` and the standalone `build_rust.sh`.

This module is the single source of truth for the Rust artifacts shipped in
the vllm package: which crates are built, where their artifacts land, and how
precompiled artifacts are detected.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension
from setuptools_rust.build import build_rust

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT_DIR / "vllm"


def rust_extensions(*, optional: bool = False) -> list[RustExtension]:
    return [
        RustExtension(
            target="vllm.vllm-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "vllm-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
        RustExtension(
            target="vllm._rust_tool_parser",
            path="rust/src/tool-parser/python/Cargo.toml",
            features=["pyo3/abi3-py38"],
            binding=Binding.PyO3,
            optional=optional,
            py_limited_api=True,
        ),
    ]


def should_require_rust_frontend() -> bool:
    value = os.getenv("VLLM_REQUIRE_RUST_FRONTEND", "")
    return value.lower() not in ("", "0", "false", "no")


def _expected_artifacts() -> list[tuple[str, Binding]]:
    """(basename, binding) of each artifact installed into the package."""
    artifacts = []
    for extension in rust_extensions():
        for target in extension.target.values():
            package, _, name = target.rpartition(".")
            assert package == "vllm", f"unexpected Rust target: {target}"
            artifacts.append((name, extension.binding))
    return artifacts


def _is_artifact_file(filename: str, name: str, binding: Binding) -> bool:
    # setuptools-rust installs Exec binaries under their bare name, and PyO3
    # modules as `<module>.<ext-suffix>` where the suffix ends with `.so` on
    # Linux and macOS alike (e.g. `_rust_foo.abi3.so`).
    if binding == Binding.Exec:
        return filename == name
    return filename.endswith(".so") and filename.split(".", 1)[0] == name


def find_precompiled_artifacts() -> list[Path]:
    """Rust artifacts already present in the package directory."""
    return sorted(
        path
        for path in PACKAGE_DIR.iterdir()
        if any(_is_artifact_file(path.name, *spec) for spec in _expected_artifacts())
    )


def missing_precompiled_artifacts() -> list[str]:
    """Expected-but-absent artifacts, as file patterns for diagnostics."""
    present = [path.name for path in find_precompiled_artifacts()]
    return [
        str(PACKAGE_DIR / (name if binding == Binding.Exec else f"{name}.*.so"))
        for name, binding in _expected_artifacts()
        if not any(_is_artifact_file(filename, name, binding) for filename in present)
    ]


def is_precompiled_artifact_member(member_name: str) -> bool:
    """Whether a wheel member is a Rust artifact (e.g. `vllm/vllm-rs`)."""
    package, _, filename = member_name.rpartition("/")
    return package == "vllm" and any(
        _is_artifact_file(filename, *spec) for spec in _expected_artifacts()
    )


class precompiled_build_rust(build_rust):
    """Skips the local Rust build when all precompiled artifacts are present."""

    def run(self) -> None:
        missing = missing_precompiled_artifacts()
        if not missing:
            logger.info(
                "Skipping local Rust build: using precompiled %s",
                find_precompiled_artifacts(),
            )
            return

        logger.warning(
            "Precompiled Rust artifacts missing (%s); "
            "falling back to local Rust build.",
            ", ".join(missing),
        )
        super().run()


def build_artifacts(build_rust_args: list[str]) -> None:
    os.chdir(ROOT_DIR)
    PACKAGE_DIR.mkdir(exist_ok=True)
    setup(
        name="vllm-rust-frontend-build",
        packages=[],
        rust_extensions=rust_extensions(optional=False),
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )


def main() -> None:
    build_artifacts(sys.argv[1:])


if __name__ == "__main__":
    main()
