# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake import and startup error canaries."""

import gc
import importlib.metadata as metadata
import sys

import pytest

from vllm import LLM
from vllm.config import KVTransferConfig


def test_mooncake_engine_imports():
    """Fail here rather than as "Mooncake is not available" at engine startup.

    The PyPI mooncake-transfer-engine wheel is built against CUDA 12, so the
    CUDA 13 CI image ships the cuda13 variant instead.
    A mismatch surfaces as an ImportError on the compiled extension.
    """
    for package_name in ("mooncake-transfer-engine", "mooncake-transfer-engine-cuda13"):
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package_name}: {version}")

    import mooncake.engine  # noqa: F401


def test_extensible_cache_rejects_peermem_without_cleanup_error(monkeypatch):
    """Report the incompatible registration mode without a destructor error."""
    monkeypatch.setenv("WITH_NVIDIA_PEERMEM", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    unraisable_errors: list[str] = []
    monkeypatch.setattr(
        sys,
        "unraisablehook",
        lambda error: unraisable_errors.append(str(error.exc_value)),
    )

    with pytest.raises(RuntimeError, match="WITH_NVIDIA_PEERMEM=1 is incompatible"):
        LLM(
            model="vllm-project/tiny-random-LlamaForCausalLM",
            enforce_eager=True,
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            gpu_memory_utilization=0.1,
            enable_extensible_kv_cache=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeConnector",
                kv_role="kv_consumer",
            ),
        )

    gc.collect()
    assert not unraisable_errors, unraisable_errors
