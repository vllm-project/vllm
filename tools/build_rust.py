# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension
from setuptools_scm import get_version

ROOT_DIR = Path(__file__).resolve().parents[1]
VLLM_RS_BUILD_VERSION = "VLLM_RS_BUILD_VERSION"

# Select vLLM release tags, excluding crate tags such as "proto-v0.3.0".
VLLM_GIT_DESCRIBE_COMMAND = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--abbrev=40",
    "--match",
    "v[0-9]*",
]


def prepare_build_environment() -> str | None:
    """Set the device-independent vLLM source version for Rust artifacts."""
    version = os.getenv(VLLM_RS_BUILD_VERSION) or None
    if version is None:
        try:
            version = get_version(
                root=ROOT_DIR,
                git_describe_command=VLLM_GIT_DESCRIBE_COMMAND,
            )
        except LookupError:
            return None

    os.environ[VLLM_RS_BUILD_VERSION] = version
    return version


def rust_extensions(*, optional: bool = False) -> list[RustExtension]:
    freethreaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    return [
        RustExtension(
            target="vllm.vllm-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "vllm-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
        # Silero VAD and Smart Turn for /v1/realtime turn detection.
        RustExtension(
            target="vllm._rust_turn_detection",
            path="rust/src/turn-detection/python/Cargo.toml",
            # Free-threaded CPython has no stable ABI.
            features=[] if freethreaded else ["pyo3/abi3-py311"],
            binding=Binding.PyO3,
            optional=optional,
            py_limited_api=not freethreaded,
        ),
    ]


def rust_py_extension_module_names() -> list[str]:
    module_names = []
    for extension in rust_extensions():
        if extension.binding != Binding.PyO3:
            continue

        for target_name in extension.target.values():
            if target_name.startswith("vllm._rust_"):
                module_names.append(target_name.rsplit(".", 1)[-1])

    return module_names


def build_binary(build_rust_args: list[str]) -> None:
    os.chdir(ROOT_DIR)
    (ROOT_DIR / "vllm").mkdir(exist_ok=True)
    setup(
        name="vllm-rust-frontend-build",
        packages=[],
        rust_extensions=rust_extensions(optional=False),
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )


def main() -> None:
    prepare_build_environment()
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
