# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL import canaries for CUDA wheel selection."""

import importlib
import importlib.metadata as metadata
import pathlib
import subprocess
import sys

import pytest
import torch


def _print_distribution_version(package_name: str) -> None:
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package_name}: {version}")


@pytest.mark.skipif(torch.version.cuda is None, reason="CUDA NIXL EP canary")
def test_nixl_and_nixl_ep_imports() -> None:
    """Verify both core NIXL and the NIXL EP extension import successfully."""
    print(f"torch cuda: {torch.version.cuda}")
    for package_name in ("nixl", "nixl-cu12", "nixl-cu13"):
        _print_distribution_version(package_name)

    nixl = importlib.import_module("nixl")
    print(f"nixl: {nixl.__file__}")

    # Exercise the core NIXL bindings used by NixlConnector.
    importlib.import_module("nixl._api")
    importlib.import_module("nixl._bindings")

    # Exercise the NIXL EP extension used by fused MoE expert parallelism.
    nixl_ep = importlib.import_module("nixl_ep")
    print(f"nixl_ep: {nixl_ep.__file__}")

    assert nixl_ep.__file__ is not None
    extension_dir = pathlib.Path(nixl_ep.__file__).parent
    extension_files = sorted(extension_dir.glob("nixl_ep_cpp*.so"))
    assert extension_files, f"No nixl_ep_cpp extension found in {extension_dir}"

    extension_file = extension_files[0]
    completed = subprocess.run(
        ["ldd", str(extension_file)],
        capture_output=True,
        check=False,
        text=True,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)

    assert completed.returncode == 0
    if torch.version.cuda is not None:
        cuda_major = torch.version.cuda.split(".", maxsplit=1)[0]
        expected_cudart = f"libcudart.so.{cuda_major}"
        assert expected_cudart in completed.stdout
        assert f"{expected_cudart} => not found" not in completed.stdout
