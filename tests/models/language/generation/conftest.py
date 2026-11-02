# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest configuration for vLLM language generation tests."""

import pytest

from vllm.model_executor.model_loader.weight_utils import drop_checkpoint_cache
from vllm.platforms import current_platform

_previous_model: str | None = None


@pytest.fixture(autouse=True)
def release_previous_checkpoint_cache(request):
    """Free the previous test's checkpoint page cache when the model changes.

    Page cache counts against the container's memory cgroup, so on ROCm CI the
    checkpoints these tests read would otherwise fill the pod's memory limit.
    """
    global _previous_model
    if not current_platform.is_rocm():
        return
    callspec = getattr(request.node, "callspec", None)
    model = callspec.params.get("model") if callspec else None
    if not isinstance(model, str):
        return
    if _previous_model is not None and _previous_model != model:
        drop_checkpoint_cache(_previous_model)
    _previous_model = model
