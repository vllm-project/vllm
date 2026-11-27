# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.device_communicators.all2all import (
    NixlEPAll2AllManager,
    _NixlEPBufferState,
)
from vllm.distributed.elastic_ep import elastic_execute


def test_nixl_ep_reconnects_failed_rank_slots_before_commit(monkeypatch):
    buffer = MagicMock()
    state = _NixlEPBufferState(buffer, 4, 4, 4)
    monkeypatch.setattr(NixlEPAll2AllManager, "_buffer", state)
    monkeypatch.setattr(NixlEPAll2AllManager, "_last_mask", torch.tensor([0, 1, 0, 0]))
    monkeypatch.setattr(NixlEPAll2AllManager, "_mask", torch.zeros(4))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    manager = object.__new__(NixlEPAll2AllManager)
    manager.world_size = 4
    manager.tcp_store_group = SimpleNamespace(store=object())

    # FT may replay its cumulative dead set; each stale endpoint is dropped once.
    manager.disconnect_ranks([1])
    manager.disconnect_ranks([1])
    buffer.disconnect_ranks.assert_called_once_with([1])

    manager.stage_ep_size()
    manager.stage_ep_size()
    buffer.connect_ranks.assert_called_once_with([1], activate=False)
    buffer.update_mask_buffer.assert_not_called()

    manager.commit_ep_size()
    buffer.update_mask_buffer.assert_called_once_with(1, mask=False)
    assert not state.ft_disconnected_ranks and not state.pending_activation_ranks
    buffer.query_mask_buffer.side_effect = lambda mask: mask.zero_()
    assert not manager.query_fault().item()


@pytest.mark.parametrize("commit_fails", [False, True])
def test_scale_up_resumes_eplb_only_after_successful_commit(commit_fails):
    runner = SimpleNamespace(eep_eplb_suppressed=True)
    worker = SimpleNamespace(model_runner=runner)
    executor = object.__new__(elastic_execute.ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    executor._can_reuse_fused_moe_kernel = lambda: False
    executor.broadcast_expert_mapping = MagicMock()
    executor.switch_and_prepare = MagicMock(return_value=())
    executor.warm_and_capture = MagicMock()
    executor._perform_eplb_reshuffle = MagicMock()
    executor._start_group_cleanup = MagicMock()

    if commit_fails:
        executor._perform_eplb_reshuffle.side_effect = RuntimeError("reshuffle failed")
        with pytest.raises(RuntimeError, match="reshuffle failed"):
            executor.commit_scale_up(is_existing_worker=True)
        assert runner.eep_eplb_suppressed
    else:
        executor.commit_scale_up(is_existing_worker=True)
        assert not runner.eep_eplb_suppressed
