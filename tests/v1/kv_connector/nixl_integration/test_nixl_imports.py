# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL import canaries for CUDA wheel selection."""

import importlib
import importlib.metadata as metadata
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
    assert nixl_ep.__file__ is not None

    # Check that the NIXL EP extension is loaded.
    assert nixl_ep.Config is not None
