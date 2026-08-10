# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared setuptools-rust build entry for Rust artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools_rust import Binding, RustExtension

ROOT_DIR = Path(__file__).resolve().parents[1]


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
            path="rust/src/parser/python/Cargo.toml",
            features=["pyo3/abi3-py38"],
            binding=Binding.PyO3,
            optional=optional,
            py_limited_api=True,
        ),
    ]


def write_coverage_objects(extensions: list[RustExtension], output: Path) -> None:
    artifacts = []
    for extension in extensions:
        for target in sorted(set(extension.target.values())):
            target_path = ROOT_DIR.joinpath(*target.split("."))
            if extension.binding == Binding.Exec:
                matches = [target_path]
            else:
                matches = sorted(target_path.parent.glob(f"{target_path.name}*.so"))
            if len(matches) != 1 or not matches[0].is_file():
                raise RuntimeError(f"unable to locate Rust artifact for {target}")
            artifacts.append(matches[0].relative_to(ROOT_DIR).as_posix())

    output.write_text("\n".join(artifacts) + "\n")


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
    extensions = rust_extensions(optional=False)
    setup(
        name="vllm-rust-frontend-build",
        packages=[],
        rust_extensions=extensions,
        script_args=["build_rust", "--quiet", "--inplace", *build_rust_args],
    )
    if output := os.getenv("VLLM_RUST_COVERAGE_OBJECTS"):
        write_coverage_objects(extensions, Path(output))


def main() -> None:
    build_binary(sys.argv[1:])


if __name__ == "__main__":
    main()
