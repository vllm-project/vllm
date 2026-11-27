# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.core import (
    CheckpointState,
    DPEngineCoreProc,
    EngineCore,
    EngineCoreProc,
)


def _make_engine_core(
    cls: type[EngineCore],
    *,
    weight_transfer_config: object = None,
    incomplete_flags: list[bool] | None = None,
) -> EngineCore:
    core = object.__new__(cls)
    core._checkpoint_state = CheckpointState.RUNNING
    core._kv_cache_reclaimable = True
    core.vllm_config = SimpleNamespace(weight_transfer_config=weight_transfer_config)
    core.model_executor = MagicMock()
    core.model_executor.vllm_config.cache_config.enable_extensible_kv_cache = False
    core.model_executor.collective_rpc.return_value = (
        incomplete_flags if incomplete_flags is not None else [False]
    )
    core.scheduler = MagicMock()
    core.scheduler.pause_state = PauseState.PAUSED_ALL
    if issubclass(cls, EngineCoreProc):
        core.reject_new_requests = True
    if issubclass(cls, DPEngineCoreProc):
        core.pending_pause = None
        core.engines_running = False
        core.ignore_start_dp_wave = True
    return core


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc, DPEngineCoreProc])
def test_resume_scheduler_blocks_when_weight_update_incomplete(cls):
    """Refuse to resume scheduling if any worker reports an incomplete weight
    update, leaving the scheduler paused and rejection state intact."""
    core = _make_engine_core(
        cls,
        weight_transfer_config=object(),
        incomplete_flags=[False, True],
    )

    with pytest.raises(RuntimeError, match="weight update is incomplete"):
        core.resume_scheduler()

    core.model_executor.collective_rpc.assert_called_once_with(
        "has_incomplete_weight_update", None, (), None
    )
    core.scheduler.set_pause_state.assert_not_called()
    assert core._kv_cache_reclaimable is True
    if issubclass(cls, EngineCoreProc):
        assert core.reject_new_requests is True
    if issubclass(cls, DPEngineCoreProc):
        assert core.ignore_start_dp_wave is True


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc])
def test_resume_scheduler_succeeds_when_weight_update_complete(cls):
    """Allow resuming scheduling once all workers report no incomplete weight
    updates."""
    core = _make_engine_core(
        cls,
        weight_transfer_config=object(),
        incomplete_flags=[False, False],
    )

    core.resume_scheduler()

    core.model_executor.collective_rpc.assert_called_once_with(
        "has_incomplete_weight_update", None, (), None
    )
    core.scheduler.set_pause_state.assert_called_once_with(PauseState.UNPAUSED)
    assert core._kv_cache_reclaimable is False
    if issubclass(cls, EngineCoreProc):
        assert core.reject_new_requests is False


@pytest.mark.parametrize("cls", [EngineCore, EngineCoreProc])
def test_resume_scheduler_skips_rpc_without_weight_transfer_config(cls):
    """Avoid an extra collective_rpc when weight transfer is not configured."""
    core = _make_engine_core(cls, weight_transfer_config=None)

    core.resume_scheduler()

    core.model_executor.collective_rpc.assert_not_called()
    core.scheduler.set_pause_state.assert_called_once_with(PauseState.UNPAUSED)
    assert core._kv_cache_reclaimable is False
    if issubclass(cls, EngineCoreProc):
        assert core.reject_new_requests is False
