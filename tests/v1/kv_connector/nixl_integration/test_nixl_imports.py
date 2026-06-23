# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL import canaries for CUDA wheel selection."""

import importlib
import importlib.metadata as metadata
import subprocess
import sys
import types

import pytest
import torch


def _print_distribution_version(package_name: str) -> None:
    try:
        version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package_name}: {version}")


def _import_nixl_ep_cpp(nixl_ep: types.ModuleType) -> types.ModuleType:
    candidate_module_names = []

    config_module_name = getattr(getattr(nixl_ep, "Config", None), "__module__", None)
    if config_module_name and config_module_name.endswith("nixl_ep_cpp"):
        candidate_module_names.append(config_module_name)

    if torch.version.cuda is not None:
        cuda_major = torch.version.cuda.split(".", maxsplit=1)[0]
        candidate_module_names.append(f"nixl_ep_cu{cuda_major}.nixl_ep_cpp")

    # Keep compatibility with the pre-dispatcher wheel layout.
    candidate_module_names.append("nixl_ep.nixl_ep_cpp")

    for module_name in dict.fromkeys(candidate_module_names):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_module = exc.name
            if missing_module not in (module_name, module_name.split(".", 1)[0]):
                raise

    raise AssertionError(
        "No nixl_ep_cpp extension module found; tried "
        f"{', '.join(dict.fromkeys(candidate_module_names))}"
    )


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

    nixl_ep_cpp = _import_nixl_ep_cpp(nixl_ep)
    assert nixl_ep_cpp.__file__ is not None
    extension_file = nixl_ep_cpp.__file__
    print(f"nixl_ep_cpp: {extension_file}")

    completed = subprocess.run(
        ["ldd", extension_file],
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
