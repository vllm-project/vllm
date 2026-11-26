# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension
from setuptools_scm import get_version

ROOT_DIR = Path(__file__).resolve().parents[1]
VLLM_RS_BUILD_VERSION = "VLLM_RS_BUILD_VERSION"
# Must match `vllm-build-info`, which reads the slot at run time.
VERSION_SLOT_MAGIC = b"@@VLLM_RS_VERSION_SLOT@@"
VERSION_SLOT_LEN = 128

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
    return [
        RustExtension(
            target="vllm.vllm-rs",
            path="rust/src/cmd/Cargo.toml",
            args=["--bin", "vllm-rs"],
            features=["native-tls-vendored"],
            binding=Binding.Exec,
            optional=optional,
        ),
        # Example PyO3 module (the removed tool-parser bridge):
        # RustExtension(
        #     target="vllm._rust_tool_parser",
        #     path="rust/src/parser/python/Cargo.toml",
        #     features=["pyo3/abi3-py38"],
        #     binding=Binding.PyO3,
        #     optional=optional,
        #     py_limited_api=True,
        # ),
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


def stamp_version(binary: Path, version: str) -> None:
    """Write `version` into the version slot of a linked Rust binary.

    Docker builds stamp the cached binary instead of relinking the crates that
    embed the version: ~130 s -> ~1 s per commit.
    """
    payload = version.encode()
    if len(payload) > VERSION_SLOT_LEN:
        raise ValueError(f"Version does not fit the version slot: {version!r}")
    data = bytearray(binary.read_bytes())
    count = data.count(VERSION_SLOT_MAGIC)
    if count != 1:
        raise ValueError(f"Expected 1 version slot in {binary}, found {count}")
    start = data.index(VERSION_SLOT_MAGIC) + len(VERSION_SLOT_MAGIC)
    data[start : start + VERSION_SLOT_LEN] = payload.ljust(VERSION_SLOT_LEN, b"\0")
    binary.write_bytes(data)
    if platform.system() == "Darwin":
        # Patching invalidates the linker's ad hoc signature on arm64.
        subprocess.run(["codesign", "--force", "--sign", "-", binary], check=True)


def main() -> None:
    version = prepare_build_environment()
    if sys.argv[1:2] == ["--stamp"]:
        if version is None:
            sys.exit("No vLLM version to stamp: no Git metadata found")
        for binary in sys.argv[2:]:
            stamp_version(Path(binary), version)
        return
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
