# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    naive_dp_ep,
    naive_dp_ep_rocm,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    make_moe_prepare_and_finalize_naive_dp_ep,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep_rocm import (
    MoEPrepareAndFinalizeNaiveDPEPModularROCmDBO,
    _comm_overlap_active,
)


class _RegionProbe(MoEPrepareAndFinalizeNaiveDPEPModularROCmDBO):
    def __init__(self) -> None:
        super().__init__()
        self.body_ran = False

    def run_comm(self) -> None:
        with self._comm_region():
            self.body_ran = True


@pytest.fixture
def dbo_mocks():
    return {
        "yield_switch": MagicMock(),
        "switch_back": MagicMock(),
    }


def _patch_dbo_calls(dbo_mocks):
    return (
        patch.object(
            naive_dp_ep_rocm,
            "dbo_yield_and_switch_from_compute_to_comm",
            dbo_mocks["yield_switch"],
        ),
        patch.object(
            naive_dp_ep_rocm,
            "dbo_switch_to_compute_sync",
            dbo_mocks["switch_back"],
        ),
    )


def test_comm_region_noop_without_dbo(dbo_mocks):
    probe = _RegionProbe()
    patches = _patch_dbo_calls(dbo_mocks)
    with (
        patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=False),
        patches[0],
        patches[1],
    ):
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_not_called()
    dbo_mocks["switch_back"].assert_not_called()


def test_comm_region_handoff_when_dbo_active(dbo_mocks):
    probe = _RegionProbe()
    patches = _patch_dbo_calls(dbo_mocks)
    with (
        patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=True),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patches[0],
        patches[1],
    ):
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_called_once_with()
    dbo_mocks["switch_back"].assert_called_once_with()


def test_comm_region_skipped_during_graph_capture(dbo_mocks):
    probe = _RegionProbe()
    patches = _patch_dbo_calls(dbo_mocks)
    with (
        patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=True),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        patches[0],
        patches[1],
    ):
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_not_called()
    dbo_mocks["switch_back"].assert_not_called()


def test_comm_region_restores_compute_stream_on_error(dbo_mocks):
    class _FailingProbe(_RegionProbe):
        def run_comm(self) -> None:
            with self._comm_region():
                raise RuntimeError("collective failed")

    probe = _FailingProbe()
    patches = _patch_dbo_calls(dbo_mocks)
    with (
        patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=True),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patches[0],
        patches[1],
        pytest.raises(RuntimeError, match="collective failed"),
    ):
        probe.run_comm()
    dbo_mocks["yield_switch"].assert_called_once_with()
    dbo_mocks["switch_back"].assert_called_once_with()


def test_comm_overlap_active_requires_dbo_and_eager():
    with (
        patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=False),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
    ):
        assert not _comm_overlap_active()
    with (
        patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=True),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
    ):
        assert not _comm_overlap_active()


def test_modular_subclass_supports_dbo_without_async():
    inst = MoEPrepareAndFinalizeNaiveDPEPModularROCmDBO()
    # supports_async() also drives MK-internal shared-expert overlap, so it
    # must not change with the DBO state.
    for dbo_active in (False, True):
        with patch.object(naive_dp_ep_rocm, "dbo_enabled", return_value=dbo_active):
            assert not inst.supports_async()
            assert inst.supports_dbo()


def test_base_supports_dbo_follows_supports_async():
    base = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular()
    assert not base.supports_async()
    assert not base.supports_dbo()
    with patch.object(type(base), "supports_async", return_value=True):
        assert base.supports_dbo()


def test_factory_returns_rocm_subclass_on_rocm():
    rocm_stub = SimpleNamespace(is_rocm=lambda: True)
    with patch.object(naive_dp_ep, "current_platform", rocm_stub):
        mono = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=True)
        modular = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=False)
    # No ROCm config selects a monolithic expert kernel on this path, so the
    # monolithic variant falls back to the base class.
    assert isinstance(mono, naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPMonolithic)
    assert isinstance(modular, MoEPrepareAndFinalizeNaiveDPEPModularROCmDBO)


def test_factory_returns_base_classes_elsewhere():
    cuda_stub = SimpleNamespace(is_rocm=lambda: False)
    with patch.object(naive_dp_ep, "current_platform", cuda_stub):
        mono = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=True)
        modular = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=False)
    assert isinstance(mono, naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPMonolithic)
    assert isinstance(modular, naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular)
    assert not isinstance(modular, MoEPrepareAndFinalizeNaiveDPEPModularROCmDBO)
