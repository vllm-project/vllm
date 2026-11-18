# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed import localization


@pytest.fixture
def cuda_locality(monkeypatch):
    """Model partition ownership without requiring CUDA hardware."""
    cuda = localization.torch.cuda
    state = SimpleNamespace(device=0, remainder=SimpleNamespace(sm_count=12))
    partitions = [SimpleNamespace(sm_count=n) for n in (60, 68)]
    resource = SimpleNamespace(sm_count=140, split=Mock())
    resource.split.return_value = (partitions, state.remainder)

    def make_context(*, sm_partition):
        return SimpleNamespace(
            sm_count=sm_partition.sm_count,
            sm_partition=sm_partition,
            Stream=lambda: SimpleNamespace(partition=sm_partition),
        )

    green = SimpleNamespace(
        SMPartition=SimpleNamespace(from_device=Mock(return_value=resource)),
        GreenContext=Mock(side_effect=make_context),
        get_num_locality_domains=lambda device: 2,
        is_localization_supported=lambda device: True,
    )
    monkeypatch.setitem(sys.modules, "torch.cuda.green_contexts", green)

    @contextmanager
    def device(index):
        previous = state.device
        state.device = index
        try:
            yield
        finally:
            state.device = previous

    monkeypatch.setattr(localization.torch.accelerator, "device_index", device)
    monkeypatch.setattr(
        localization.torch.accelerator, "current_device_index", lambda: state.device
    )
    monkeypatch.setattr(
        cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(multi_processor_count=140),
    )
    monkeypatch.setattr(
        cuda, "Stream", lambda *, device: SimpleNamespace(device=device)
    )
    monkeypatch.setattr(
        cuda,
        "LocalizedAllocator",
        lambda domain, *, device: SimpleNamespace(allocator=lambda: (domain, device)),
        raising=False,
    )
    monkeypatch.setattr(
        cuda,
        "MemPool",
        lambda *, allocator: SimpleNamespace(allocator=allocator, device=state.device),
    )
    # Model a torch/FlashInfer stack that provides the localized MoE APIs.
    monkeypatch.setattr(localization, "localized_moe_unsupported_reason", lambda: None)
    monkeypatch.setattr(localization.envs, "VLLM_USE_LOCALITY_DOMAINS", True)
    monkeypatch.setattr(localization.envs, "VLLM_MOE_LOCALIZE_BACKFILL", False)
    monkeypatch.setattr(localization, "_LOCALIZED_RESOURCES", {})
    return resource, green, state


def test_partition_counts_remainder_and_pool_device(cuda_locality):
    resource, green, state = cuda_locality
    device = SimpleNamespace(index=1)
    result = localization.maybe_init_localized_resources(device)

    assert result is not None
    # Resource diagnostics must not disclose the hardware partition layout.
    assert str(result) == repr(result) == "LocalizedResources()"
    # Unequal partitions must retain their actual counts, excluding remainder SMs.
    assert result.split_sm_counts == [60, 68]
    assert result.sm_per_resource == [60, 68, 12]
    assert result.aux_stream.partition is state.remainder
    assert [pool.device for pool in result.pools[:2]] == [1, 1]
    assert [pool.allocator for pool in result.pools[:2]] == [(0, 1), (1, 1)]
    assert state.device == 0
    assert localization.maybe_init_localized_resources(device) is result
    assert resource.split.call_count == 1


@pytest.mark.parametrize("failure", ["split", "counts"])
def test_backfill_failure_retries_node_local_partitions(
    cuda_locality, monkeypatch, failure
):
    resource, green, state = cuda_locality
    monkeypatch.setattr(localization.envs, "VLLM_MOE_LOCALIZE_BACKFILL", True)
    node_local = resource.split.return_value
    failed = RuntimeError("unsupported backfill") if failure == "split" else node_local
    resource.split.side_effect = [failed, node_local]

    result = localization.maybe_init_localized_resources(SimpleNamespace(index=0))

    assert result is not None
    assert result.split_sm_counts == [60, 68]
    assert result.aux_stream.partition is state.remainder
    assert [call.kwargs["backfill"] for call in resource.split.call_args_list] == [
        True,
        False,
    ]


@pytest.mark.parametrize("backfill", [False, True])
def test_no_remainder_uses_full_device_aux_stream(cuda_locality, monkeypatch, backfill):
    resource, green, state = cuda_locality
    monkeypatch.setattr(localization.envs, "VLLM_MOE_LOCALIZE_BACKFILL", backfill)
    resource.split.return_value = (
        [SimpleNamespace(sm_count=70) for _ in range(2)],
        None,
    )

    result = localization.maybe_init_localized_resources(SimpleNamespace(index=1))

    assert result is not None
    assert result.split_sm_counts == [70, 70]
    assert result.contexts[-1] is None
    assert result.pools[-1] is None
    assert result.aux_stream.device == 1
    assert result.sm_per_resource[-1] == 140


def test_remainder_creation_failure_preserves_compute_domains(cuda_locality):
    resource, green, state = cuda_locality
    create = green.GreenContext.side_effect

    def fail_remainder(*, sm_partition):
        if sm_partition is state.remainder:
            raise RuntimeError("cannot create auxiliary context")
        return create(sm_partition=sm_partition)

    green.GreenContext.side_effect = fail_remainder
    result = localization.maybe_init_localized_resources(SimpleNamespace(index=0))

    assert result is not None
    assert result.split_sm_counts == [60, 68]
    assert result.aux_stream.device == 0
    assert result.contexts[-1] is None


@pytest.mark.parametrize("missing", ["torch", "flashinfer"])
def test_unsupported_stack_disables_localization(monkeypatch, missing):
    """An old torch or FlashInfer disables localization before any partitioning."""
    import vllm.utils.flashinfer as fi

    cuda = localization.torch.cuda
    monkeypatch.setattr(localization.envs, "VLLM_USE_LOCALITY_DOMAINS", True)
    monkeypatch.setattr(localization, "_LOCALIZED_RESOURCES", {})
    # Provide the allocator/partition APIs so only the MoE-stack check can refuse.
    monkeypatch.setattr(cuda, "LocalizedAllocator", Mock(), raising=False)
    green = SimpleNamespace(
        SMPartition=SimpleNamespace(from_device=Mock()),
        GreenContext=Mock(),
        get_num_locality_domains=lambda device: 2,
        is_localization_supported=lambda device: True,
    )
    monkeypatch.setitem(sys.modules, "torch.cuda.green_contexts", green)

    if missing == "torch":
        monkeypatch.delattr(cuda, "execute_on_streams", raising=False)
        monkeypatch.setattr(fi, "has_flashinfer_localized_moe", lambda: True)
        expected = "execute_on_streams"
    else:
        monkeypatch.setattr(
            cuda, "execute_on_streams", lambda *a, **k: None, raising=False
        )
        monkeypatch.setattr(fi, "has_flashinfer_localized_moe", lambda: False)
        expected = "flashinfer-ai/flashinfer#6488"

    reason = localization.localized_moe_unsupported_reason()
    assert reason is not None and expected in reason

    assert localization.maybe_init_localized_resources(SimpleNamespace(index=0)) is None
    assert green.SMPartition.from_device.call_count == 0
    assert localization.get_localized_resources(0) is None


def test_supported_stack_reports_no_reason(monkeypatch):
    import vllm.utils.flashinfer as fi

    monkeypatch.setattr(
        localization.torch.cuda,
        "execute_on_streams",
        lambda *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(fi, "has_flashinfer_localized_moe", lambda: True)
    assert localization.localized_moe_unsupported_reason() is None
