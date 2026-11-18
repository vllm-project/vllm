# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 replicated MLP (VLLM_QWEN3_5_REPLICATE_FFN): setup checks.

CPU-only: the layouts the switch refuses, and the cross-rank determinism it turns on
(TP-synced Triton autotune, deterministic Inductor).
"""

from types import SimpleNamespace

import pytest
import torch._inductor.config as inductor_config

import vllm.triton_utils.tp_synced_autotune as tp_synced_autotune
from vllm.model_executor.models.qwen3_5_replicate import setup_replicated_ffn


def _config(model_type="qwen3_5_text"):
    return SimpleNamespace(model_type=model_type)


def _layout(dcp=1, pcp=1, ep=False, sp_moe=False):
    return SimpleNamespace(
        decode_context_parallel_size=dcp,
        prefill_context_parallel_size=pcp,
        enable_expert_parallel=ep,
        use_sequence_parallel_moe=sp_moe,
    )


@pytest.fixture
def installs(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []
    monkeypatch.setattr(tp_synced_autotune, "install", lambda: calls.append(1))
    # Inductor's config module is restored with its own patch(), not monkeypatch.
    with inductor_config.patch(deterministic=False):
        yield calls


def test_dense_turns_on_determinism(installs):
    setup_replicated_ffn(_config(), _layout())
    assert installs == [1]
    assert inductor_config.deterministic is True


def test_moe_is_refused(installs):
    with pytest.raises(NotImplementedError, match="dense"):
        setup_replicated_ffn(_config("qwen3_5_moe_text"), _layout())
    assert installs == []


@pytest.mark.parametrize(
    "layout, reason",
    [
        (_layout(dcp=2), "context parallelism"),
        (_layout(pcp=2), "context parallelism"),
        (_layout(ep=True), "expert parallelism"),
        (_layout(sp_moe=True), "sequence-parallel MoE"),
    ],
)
def test_unsynced_layouts_are_refused(installs, layout, reason):
    # Each of these turns the TP-synced autotune off (tp_synced_autotune.configure).
    with pytest.raises(NotImplementedError, match=reason):
        setup_replicated_ffn(_config(), layout)
    assert installs == []
