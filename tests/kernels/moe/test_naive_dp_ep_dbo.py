# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
    make_moe_prepare_and_finalize_naive_dp_ep,
)


class _RegionProbe(MoEPrepareAndFinalizeNaiveDPEPModular):
    def __init__(self) -> None:
        super().__init__()
        self.body_ran = False

    def run_comm(self) -> None:
        with self._comm_region():
            self.body_ran = True


@pytest.fixture
def dbo_mocks():
    yield_switch = MagicMock()
    switch_back = MagicMock()
    with (
        patch.object(
            naive_dp_ep, "dbo_yield_and_switch_from_compute_to_comm", yield_switch
        ),
        patch.object(naive_dp_ep, "dbo_switch_to_compute_sync", switch_back),
    ):
        yield {"yield_switch": yield_switch, "switch_back": switch_back}


def _dbo_state(dbo_active: bool, capturing: bool):
    return (
        patch.object(naive_dp_ep, "dbo_enabled", return_value=dbo_active),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
    )


def test_comm_region_noop_without_dbo(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=False, capturing=False)
    with dbo, capturing:
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_not_called()
    dbo_mocks["switch_back"].assert_not_called()


def test_comm_region_handoff_when_dbo_active(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=True, capturing=False)
    with dbo, capturing:
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_called_once_with()
    dbo_mocks["switch_back"].assert_called_once_with()


def test_comm_region_skipped_during_graph_capture(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=True, capturing=True)
    with dbo, capturing:
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_not_called()
    dbo_mocks["switch_back"].assert_not_called()


def test_comm_region_restores_compute_stream_on_error(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=True, capturing=False)
    with (
        dbo,
        capturing,
        pytest.raises(RuntimeError, match="collective failed"),
        probe._comm_region(),
    ):
        raise RuntimeError("collective failed")
    dbo_mocks["yield_switch"].assert_called_once_with()
    dbo_mocks["switch_back"].assert_called_once_with()


@pytest.mark.parametrize("is_rocm", [True, False])
def test_supports_dbo_only_on_rocm(is_rocm):
    inst = MoEPrepareAndFinalizeNaiveDPEPModular()
    platform = SimpleNamespace(is_rocm=lambda: is_rocm)
    with patch.object(naive_dp_ep, "current_platform", platform):
        assert inst.supports_dbo() is is_rocm
        # supports_async() also drives MK-internal shared-expert overlap, so
        # it must not change with DBO support.
        assert not inst.supports_async()


@pytest.mark.parametrize("is_rocm", [True, False])
def test_factory_returns_base_classes(is_rocm):
    platform = SimpleNamespace(is_rocm=lambda: is_rocm)
    with patch.object(naive_dp_ep, "current_platform", platform):
        mono = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=True)
        modular = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=False)
    assert type(mono) is naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPMonolithic
    assert type(modular) is MoEPrepareAndFinalizeNaiveDPEPModular


def test_default_supports_dbo_follows_supports_async():
    # Classes that do not override supports_dbo() keep the old contract:
    # DBO only with async prepare/finalize.
    mono = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPMonolithic()
    assert not mono.supports_async()
    assert not mono.supports_dbo()
    with patch.object(type(mono), "supports_async", return_value=True):
        assert mono.supports_dbo()
