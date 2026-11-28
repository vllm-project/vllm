# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform


def _runnable() -> str | None:
    if not current_platform.is_rocm():
        return "ROCm only"
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        return "AITER not installed or unsupported on this GPU"
    return None


SKIP_REASON = _runnable()


def pytest_collection_modifyitems(config, items):
    if SKIP_REASON is None:
        return
    marker = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        if item.path.parent.name == "rocm_flashmla_sparse_dsv41":
            item.add_marker(marker)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
