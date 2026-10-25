# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.v1.attention.backends.harness import KernelSpy
from vllm.platforms import current_platform


def _rocm_indexer_runnable() -> str | None:
    if not current_platform.is_rocm():
        return "ROCm only"
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        return "AITER not installed or unsupported on this GPU"
    return None


SKIP_REASON = _rocm_indexer_runnable()


def pytest_collection_modifyitems(config, items):
    if SKIP_REASON is None:
        return
    marker = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if item.path.parent.name == "rocm_dsv32_indexer":
            item.add_marker(marker)


@pytest.fixture
def spy(monkeypatch):
    from tests.v1.attention.backends.rocm_dsv32_indexer.idx_harness import (
        watch_all_kernels,
    )

    kernel_spy = KernelSpy(monkeypatch)
    watch_all_kernels(kernel_spy)
    return kernel_spy


@pytest.fixture
def env(dist_init, workspace_init):
    """Distributed groups and the workspace manager."""
    yield


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
