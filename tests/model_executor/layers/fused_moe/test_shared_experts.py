# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.layers.fused_moe.runner.shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)


def _make_shared_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> SharedExperts:
    monkeypatch.setattr(
        shared_module,
        "dbo_current_ubatch_id",
        lambda: 0,
    )
    shared = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(shared)
    shared.enable_dbo = False
    shared._output = [None, None]
    shared._precomputed_output = [None, None]
    shared._layer = lambda tensor: tensor + 1
    shared._determine_shared_experts_order = lambda _: SharedExpertsOrder.NO_OVERLAP
    return shared


def test_precomputed_output_bypasses_layer_and_is_consumed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _make_shared_experts(monkeypatch)
    precomputed = torch.randn(2, 4)

    with shared.use_precomputed_output(precomputed):
        shared.maybe_sync_shared_experts_stream(torch.empty_like(precomputed))
        shared(
            torch.empty_like(precomputed),
            SharedExpertsOrder.NO_OVERLAP,
        )
        assert shared.output is precomputed

    assert shared._output == [None, None]
    assert shared._precomputed_output == [None, None]


def test_precomputed_output_rejects_collision_and_cleans_up_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _make_shared_experts(monkeypatch)

    with (
        pytest.raises(RuntimeError, match="downstream failure"),
        shared.use_precomputed_output(torch.zeros(1)),
    ):
        with (
            pytest.raises(RuntimeError, match="already occupied"),
            shared.use_precomputed_output(torch.ones(1)),
        ):
            pass
        raise RuntimeError("downstream failure")

    assert shared._output == [None, None]
    assert shared._precomputed_output == [None, None]


def test_normal_shared_expert_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _make_shared_experts(monkeypatch)
    hidden = torch.randn(2, 4)

    shared(hidden, SharedExpertsOrder.NO_OVERLAP)

    torch.testing.assert_close(shared.output, hidden + 1)
