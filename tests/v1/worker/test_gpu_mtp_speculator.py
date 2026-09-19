# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from torch import nn

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


def test_model_hook_runs_after_target_head_aliasing() -> None:
    target_model = nn.Module()
    target_model.model = nn.Module()
    target_model.model.embed_tokens = nn.Embedding(8, 4)
    target_model.lm_head = nn.Linear(4, 8, bias=False)
    draft_model = nn.Module()
    draft_model.model = nn.Module()
    draft_model.model.embed_tokens = nn.Embedding(8, 4)
    old_draft_head = nn.Linear(4, 8, bias=False)
    draft_model.lm_head = old_draft_head

    def initialize() -> None:
        assert draft_model.lm_head is target_model.lm_head

    draft_model.maybe_init_fp8_proposal_head = Mock(side_effect=initialize)
    speculator = object.__new__(MTPSpeculator)
    speculator.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(index_share_for_mtp_iteration=False)
            ),
            moe_backend=None,
            kv_cache_dtype=None,
            attention_backend=None,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )

    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model",
            return_value=draft_model,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        result = speculator.load_draft_model(target_model, set())

    assert result is draft_model
    assert old_draft_head is not target_model.lm_head
    assert draft_model.lm_head is target_model.lm_head
    draft_model.maybe_init_fp8_proposal_head.assert_called_once_with()


def test_model_runner_rejects_target_reload_before_mutation(monkeypatch) -> None:
    draft_model = Mock()
    draft_model.before_target_model_reload.side_effect = RuntimeError("blocked reload")
    runner = object.__new__(GPUModelRunner)
    monkeypatch.setattr(runner, "get_draft_model", Mock(return_value=draft_model))
    reload_impl = Mock()
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner.reload_weights", reload_impl
    )

    with pytest.raises(RuntimeError, match="blocked reload"):
        runner.reload_weights()

    draft_model.before_target_model_reload.assert_called_once_with()
    reload_impl.assert_not_called()
