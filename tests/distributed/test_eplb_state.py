# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for :class:`EplbState`."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.parallel import ParallelConfig
from vllm.distributed.eplb import eplb_state as eplb_state_module
from vllm.distributed.eplb.eplb_state import EplbState


def test_step_logged_metrics_reduce_over_rank_axis(monkeypatch):
    """step()'s logged avg/max tokens must reduce across ranks, then sum
    across layers (per its docstring) - not reduce across layers instead."""
    # CpuGpuEvent wraps torch.cuda.Event, unavailable without CUDA.
    monkeypatch.setattr(eplb_state_module, "CpuGpuEvent", MagicMock())
    state = EplbState(ParallelConfig(), torch.device("cpu"))
    state.expert_rearrangement_step_interval = 1_000
    state.expert_load_window_size = 1

    # Fake 2 EP ranks; skip the all-reduce by preloading cluster-wide load.
    ep = MagicMock()
    ep.device_group.size.return_value = 2
    ep.device_group.rank.return_value = 0
    monkeypatch.setattr(eplb_state_module, "get_ep_group", lambda: ep)
    monkeypatch.setattr(state, "_allreduce_list", lambda t: t)
    logger = MagicMock()
    monkeypatch.setattr(eplb_state_module, "logger", logger)

    # 3 layers x 4 experts; per-layer per-rank loads: 30/70, 3/7, 100/100.
    load = torch.tensor(
        [[10, 20, 30, 40], [1, 2, 3, 4], [50, 50, 50, 50]], dtype=torch.int32
    )
    state.model_states["m"] = MagicMock(
        model_name="m",
        expert_load_pass=load,
        expert_load_window=torch.zeros((1, *load.shape), dtype=torch.int32),
    )

    state.step(log_stats=True)

    avg, max_, balancedness = logger.info.call_args.args[3:6]
    assert avg == pytest.approx(155.0)  # 310 total tokens / 2 ranks
    assert max_ == pytest.approx(177.0)  # 70 + 7 + 100 per-layer rank maxima
    assert balancedness == pytest.approx(155.0 / 177.0)
