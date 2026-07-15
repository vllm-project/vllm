# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu_worker import Worker


def _module_with_buffer(name: str, value: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    module.register_buffer(name, value.clone())
    return module


def test_level_two_sleep_restores_target_and_drafter_buffers():
    target = _module_with_buffer("target_cache", torch.arange(8, dtype=torch.float32))
    drafter = _module_with_buffer(
        "draft_cache", torch.arange(8, dtype=torch.float32) + 10
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        model=target,
        drafter=SimpleNamespace(model=drafter),
    )
    worker._sleep_saved_buffers = {}

    worker._save_buffers_before_sleep()
    target.target_cache.zero_()
    drafter.draft_cache.zero_()
    worker._restore_buffers_after_sleep()

    torch.testing.assert_close(
        target.target_cache, torch.arange(8, dtype=torch.float32)
    )
    torch.testing.assert_close(
        drafter.draft_cache, torch.arange(8, dtype=torch.float32) + 10
    )
    assert worker._sleep_saved_buffers == {}
