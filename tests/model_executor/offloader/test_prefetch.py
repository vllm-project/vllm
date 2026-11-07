# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.offloader.base import should_pin_memory
from vllm.model_executor.offloader.prefetch import (
    _CpuParamOffloader,
    _get_next_prefetch_index,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("module_count", "prefetch_step", "expected_targets"),
    [
        (3, 1, [1, 2, 0]),
        (3, 2, [2, 1, 0]),
        (4, 2, [2, 3, 0, 1]),
        (5, 2, [2, 3, 4, 1, 0]),
        (3, 3, [0, 1, 2]),
        (3, 4, [1, 2, 0]),
    ],
)
def test_next_prefetch_index_preserves_slot_ownership(
    module_count: int,
    prefetch_step: int,
    expected_targets: list[int],
) -> None:
    targets = [
        _get_next_prefetch_index(index, prefetch_step, module_count)
        for index in range(module_count)
    ]

    assert targets == expected_targets
    if prefetch_step < module_count:
        assert all(
            target % prefetch_step == index % prefetch_step
            for index, target in enumerate(targets)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sync_cpu_storage_from_reshaped_weight_left_on_device() -> None:
    # A model-level process_weights_after_loading() hook runs outside
    # device_loading_context and may leave a reshaped weight on the device.
    module = nn.Module()
    module.weight = nn.Parameter(torch.arange(64.0, device="cuda"))
    offloader = _CpuParamOffloader(module, "weight")
    module.weight = nn.Parameter(torch.arange(32.0, device="cuda").view(4, 8))

    offloader.sync_cpu_storage()

    storage = offloader._cpu_storage
    assert storage is not None
    torch.testing.assert_close(storage, torch.arange(32.0).view(4, 8))
    assert storage.is_pinned() == should_pin_memory()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and should_pin_memory()),
    reason="needs CUDA pinned memory",
)
def test_device_loading_context_does_not_double_pinned_weights() -> None:
    layers, nbytes = 4, 16 * 2**20
    torch.cuda.memory.reset_peak_host_memory_stats()
    baseline = torch.cuda.memory.host_memory_stats()["active_bytes.current"]

    modules, offloaders = [], []
    for _ in range(layers):
        module = nn.Module()
        module.weight = nn.Parameter(
            torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        )
        offloaders.append(_CpuParamOffloader(module, "weight"))
        modules.append(module)
    for module in modules:
        with device_loading_context(module, torch.device("cuda")):
            module.weight = nn.Parameter(module.weight * 2, requires_grad=False)
    for offloader in offloaders:
        offloader.sync_cpu_storage()
        storage = offloader._cpu_storage
        assert storage is not None and storage.is_pinned()
        assert torch.all(storage == 2)

    peak = torch.cuda.memory.host_memory_stats()["active_bytes.peak"] - baseline
    assert peak < 1.5 * layers * nbytes, (peak, layers * nbytes)
