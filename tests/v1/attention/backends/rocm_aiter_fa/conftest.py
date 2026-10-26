# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.v1.attention.backends.harness import KernelSpy
from tests.v1.attention.backends.rocm_aiter_fa.known_defects import (
    mark_known_defects,
)
from vllm.platforms import current_platform


def _aiter_fa_runnable() -> str | None:
    if not current_platform.is_rocm():
        return "ROCm only"
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.platforms.rocm import get_cdna_version

    if not is_aiter_found_and_supported():
        return "AITER not installed or unsupported on this GPU"
    if get_cdna_version() <= 2:
        return "ROCM_AITER_FA needs CDNA3 or newer"
    return None


SKIP_REASON = _aiter_fa_runnable()


def pytest_collection_modifyitems(config, items):
    mark_known_defects(items)
    if SKIP_REASON is None:
        return
    marker = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if item.path.parent.name == "rocm_aiter_fa":
            item.add_marker(marker)


@pytest.fixture
def spy(monkeypatch):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
        watch_all_kernels,
    )

    kernel_spy = KernelSpy(monkeypatch)
    watch_all_kernels(kernel_spy)
    return kernel_spy


@pytest.fixture(params=[False, True], ids=["packed", "shuffle"])
def shuffle(request, monkeypatch):
    from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import set_shuffle

    set_shuffle(monkeypatch, request.param)
    return request.param


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
