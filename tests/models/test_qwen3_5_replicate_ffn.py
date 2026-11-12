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
from vllm.model_executor.models.qwen3_5 import _setup_replicated_ffn


def _config(model_type="qwen3_5_text"):
    return SimpleNamespace(model_type=model_type)


def _layout(dcp=1, pcp=1):
    return SimpleNamespace(
        decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp
    )


@pytest.fixture
def installs(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []
    monkeypatch.setattr(tp_synced_autotune, "install", lambda: calls.append(1))
    # Inductor's config module is restored with its own patch(), not monkeypatch.
    with inductor_config.patch(deterministic=False):
        yield calls


def test_dense_turns_on_determinism(installs):
    _setup_replicated_ffn(_config(), _layout())
    assert installs == [1]
    assert inductor_config.deterministic is True


def test_moe_is_refused(installs):
    with pytest.raises(NotImplementedError, match="dense"):
        _setup_replicated_ffn(_config("qwen3_5_moe_text"), _layout())
    assert installs == []


@pytest.mark.parametrize("layout", [_layout(dcp=2), _layout(pcp=2)])
def test_context_parallel_is_refused(installs, layout):
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _setup_replicated_ffn(_config(), layout)
    assert installs == []
