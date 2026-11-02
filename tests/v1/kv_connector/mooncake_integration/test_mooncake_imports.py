# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake import canary for CUDA wheel selection."""

import importlib.metadata as metadata


def test_mooncake_engine_imports():
    """Fail here rather than as "Mooncake is not available" at engine startup.

    The PyPI mooncake-transfer-engine wheel is built against CUDA 12, so on the
    CUDA 13 image install-kv-connectors.sh replaces it with the cuda13 variant.
    A mismatch surfaces as an ImportError on the compiled extension.
    """
    for package_name in ("mooncake-transfer-engine", "mooncake-transfer-engine-cuda13"):
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package_name}: {version}")

    import mooncake.engine  # noqa: F401


def test_pp_prefix_handoff_with_native_mooncake_tcp(tmp_path):
    """Real native engine registration/writes retain cached pages and padding."""
    import torch

    from tests.v1.worker.test_pp_buffer import _prefix_handoff_worker

    torch.multiprocessing.spawn(
        _prefix_handoff_worker,
        args=(f"file://{tmp_path / 'native'}", "mooncake", True),
        nprocs=2,
        join=True,
    )
