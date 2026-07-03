# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from vllm.utils.gpu_sync_debug import (
    SYNC_ERROR_MESSAGE,
    gpu_sync_allowed,
    with_gpu_sync_check,
)

from ..utils import create_new_process_for_each_test

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _no_sync():
    # Pure on-GPU compute, no implicit CPU sync...
    x = torch.ones(4, device="cuda") + 1
    # ...plus a sync that we explicitly allow.
    with gpu_sync_allowed():
        return x.cpu()


def _causes_sync():
    x = torch.ones(4, device="cuda")
    # An allowed sync (suppressed)...
    with gpu_sync_allowed():
        x.cpu()
    # ...then an un-allowed sync that should trip the check.
    return x.cpu()


@pytest.mark.parametrize("mode", ["warn", "error"])
@create_new_process_for_each_test()
def test_with_env_set(monkeypatch, mode):
    # Env set + gate flipped on: the unguarded sync is detected.
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", mode)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    # Guarded syncs always pass.
    with_gpu_sync_check(_no_sync)()

    if mode == "error":
        # "error" mode turns the stray sync into a RuntimeError.
        with pytest.raises(RuntimeError, match=SYNC_ERROR_MESSAGE):
            with_gpu_sync_check(_causes_sync)()
    else:
        # "warn" mode only warns, so the call still succeeds.
        with_gpu_sync_check(_causes_sync)()


@create_new_process_for_each_test()
def test_without_env_set(monkeypatch):
    # Env unset: the decorator is a pass-through, no sync is detected.
    monkeypatch.delenv("VLLM_GPU_SYNC_CHECK", raising=False)
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)

    with_gpu_sync_check(_no_sync)()
    with_gpu_sync_check(_causes_sync)()
