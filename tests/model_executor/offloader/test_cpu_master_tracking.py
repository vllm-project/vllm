# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader import base as offloader_base
from vllm.model_executor.offloader.prefetch import _CpuParamOffloader


def _make_initialized_param_offloader(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[nn.Linear, _CpuParamOffloader]:
    monkeypatch.setattr(offloader_base, "is_pin_memory_available", lambda: False)
    module = nn.Linear(4, 4, bias=False)
    offloader = _CpuParamOffloader(module, "weight")

    runtime_buffer = torch.empty_strided(
        size=module.weight.data.size(),
        stride=module.weight.data.stride(),
        dtype=module.weight.data.dtype,
        device="cpu",
    )
    offloader.assign_static_buffer(runtime_buffer)
    runtime_buffer.copy_(offloader._cpu_storage)
    offloader.mark_cpu_master_synced()
    return module, offloader


def test_cpu_master_tracking_accepts_unchanged_runtime_param(
    monkeypatch: pytest.MonkeyPatch,
):
    _, offloader = _make_initialized_param_offloader(monkeypatch)

    offloader.ensure_cpu_master_freshness()
    assert offloader._cpu_master_stale is False


def test_cpu_master_tracking_detects_in_place_runtime_weight_mutation(
    monkeypatch: pytest.MonkeyPatch,
):
    _, offloader = _make_initialized_param_offloader(monkeypatch)

    assert offloader._gpu_buffer is not None
    offloader._gpu_buffer.add_(1)

    with pytest.raises(
        RuntimeError,
        match="was mutated after CPU master synchronization",
    ):
        offloader.ensure_cpu_master_freshness()
    assert offloader._cpu_master_stale is True


def test_cpu_master_tracking_detects_runtime_buffer_replacement(
    monkeypatch: pytest.MonkeyPatch,
):
    module, offloader = _make_initialized_param_offloader(monkeypatch)

    module.weight.data = module.weight.data.clone()

    with pytest.raises(
        RuntimeError,
        match="no longer points to the managed runtime buffer",
    ):
        offloader.ensure_cpu_master_freshness()
    assert offloader._cpu_master_stale is True


def test_cpu_master_tracking_detects_explicit_external_stale_mark(
    monkeypatch: pytest.MonkeyPatch,
):
    _, offloader = _make_initialized_param_offloader(monkeypatch)

    offloader.release_runtime_buffer_tracking()
    offloader.mark_cpu_master_stale("external weight mutation")

    with pytest.raises(RuntimeError, match="external weight mutation"):
        offloader.ensure_cpu_master_freshness()
    assert offloader._cpu_master_stale is True


def test_cpu_master_tracking_can_resync_explicit_external_stale_mark(
    monkeypatch: pytest.MonkeyPatch,
):
    module, offloader = _make_initialized_param_offloader(monkeypatch)

    module.weight.data.add_(2)
    offloader.mark_cpu_master_stale("external weight mutation")
    offloader.sync_cpu_master_from_runtime()

    offloader.ensure_cpu_master_freshness()
    assert offloader._cpu_master_stale is False
    assert offloader._cpu_master_stale_reason is None
    assert torch.equal(offloader._cpu_storage, module.weight.data)


def test_later_prefetch_uses_runtime_mutated_weight_after_cpu_master_resync(
    monkeypatch: pytest.MonkeyPatch,
):
    module, offloader = _make_initialized_param_offloader(monkeypatch)
    assert offloader._cpu_storage is not None

    original_cpu_master = offloader._cpu_storage.clone()
    module.weight.data.add_(7)
    rebalanced_runtime_weight = module.weight.data.clone()

    offloader.sync_cpu_master_from_runtime()

    module.weight.data.zero_()
    module.weight.data.copy_(offloader._cpu_storage)

    assert not torch.equal(module.weight.data, original_cpu_master)
    assert torch.equal(module.weight.data, rebalanced_runtime_weight)


def test_later_prefetch_rejects_nonresident_runtime_mutated_weight(
    monkeypatch: pytest.MonkeyPatch,
):
    _, offloader = _make_initialized_param_offloader(monkeypatch)

    offloader.release_runtime_buffer_tracking()
    offloader.mark_cpu_master_stale(
        "external weight mutation; offload runtime slot is not resident"
    )

    with pytest.raises(RuntimeError, match="offload runtime slot is not resident"):
        offloader.ensure_cpu_master_freshness()


def test_cpu_master_tracking_ignores_expected_slot_reuse_after_release(
    monkeypatch: pytest.MonkeyPatch,
):
    _, offloader = _make_initialized_param_offloader(monkeypatch)

    assert offloader._gpu_buffer is not None
    offloader.release_runtime_buffer_tracking()
    offloader._gpu_buffer.add_(1)

    offloader.ensure_cpu_master_freshness()
    assert offloader._cpu_master_stale is False


def test_cpu_master_tracking_accepts_shared_slab_views_after_group_sync(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offloader_base, "is_pin_memory_available", lambda: False)
    module = nn.Linear(4, 4, bias=True)
    weight_offloader = _CpuParamOffloader(module, "weight")
    bias_offloader = _CpuParamOffloader(module, "bias")

    slab = torch.empty(20, dtype=module.weight.dtype)
    weight_offloader.assign_static_buffer(slab[:16].view(4, 4))
    bias_offloader.assign_static_buffer(slab[16:20].view(4))

    assert weight_offloader._cpu_storage is not None
    assert bias_offloader._cpu_storage is not None
    weight_offloader._gpu_buffer.copy_(weight_offloader._cpu_storage)
    bias_offloader._gpu_buffer.copy_(bias_offloader._cpu_storage)

    weight_offloader.mark_cpu_master_synced()
    bias_offloader.mark_cpu_master_synced()

    weight_offloader.ensure_cpu_master_freshness()
    bias_offloader.ensure_cpu_master_freshness()
    assert weight_offloader._cpu_master_stale is False
    assert bias_offloader._cpu_master_stale is False
