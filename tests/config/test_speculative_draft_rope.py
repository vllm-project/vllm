# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EAGLE draft RoPE logging and omission warning (#61231).

EAGLE draft rotary embeddings come from the draft ``hf_config`` only. When
the draft ``config.json`` omits rope keys, transformers and ``get_rope``
default to base 10000, which can silently disagree with a long-context
target. ``SpeculativeConfig`` must log the resolved draft RoPE and warn
when keys were omitted and the default differs from the target's base.
Numerics are unchanged.
"""

import logging
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig

# Public Hub repos; only config files are fetched.
EAGLE3_DRAFT_OMITS_ROPE = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
EAGLE1_DRAFT_WITH_THETA = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"
LLAMA31_TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"

_LOGGER = "vllm.config.speculative"
_INFO_MSG = "Draft model RoPE:"
_WARN_MSG = "Draft model config omits rope keys"

_LLAMA3_ROPE = {
    "rope_theta": 500000.0,
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
_DEFAULT_ROPE = {"rope_theta": 10000.0, "rope_type": "default"}


@pytest.fixture
def vllm_caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Make caplog see vLLM logger records (vLLM sets propagate=False)."""
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        yield caplog


def _info_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return any(_INFO_MSG in record.getMessage() for record in caplog.records)


def _warn_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return any(
        record.levelno >= logging.WARNING and _WARN_MSG in record.getMessage()
        for record in caplog.records
    )


def _model_config(hf_config, model: str = "mock") -> MagicMock:
    cfg = MagicMock(spec=ModelConfig)
    cfg.hf_config = hf_config
    cfg.model = model
    return cfg


@pytest.mark.cpu_test
def test_resolve_rope_from_rope_parameters():
    hf_config = SimpleNamespace(rope_parameters=dict(_LLAMA3_ROPE))
    base, rope_type = SpeculativeConfig._resolve_rope_base_and_type(hf_config)
    assert base == 500000.0
    assert rope_type == "llama3"


@pytest.mark.cpu_test
def test_resolve_rope_defaults_when_absent():
    hf_config = SimpleNamespace()
    base, rope_type = SpeculativeConfig._resolve_rope_base_and_type(hf_config)
    assert base == 10000.0
    assert rope_type == "default"


@pytest.mark.cpu_test
def test_checkpoint_declares_rope_false_for_omitting_draft():
    assert SpeculativeConfig._checkpoint_declares_rope(EAGLE3_DRAFT_OMITS_ROPE) is False


@pytest.mark.cpu_test
def test_checkpoint_declares_rope_true_when_theta_present():
    assert SpeculativeConfig._checkpoint_declares_rope(EAGLE1_DRAFT_WITH_THETA) is True


@pytest.mark.cpu_test
def test_log_warns_when_draft_omits_rope_and_base_differs(
    vllm_caplog: pytest.LogCaptureFixture,
):
    draft = _model_config(SimpleNamespace(rope_parameters=dict(_DEFAULT_ROPE)))
    target = _model_config(SimpleNamespace(rope_parameters=dict(_LLAMA3_ROPE)))

    with patch.object(
        SpeculativeConfig, "_checkpoint_declares_rope", return_value=False
    ):
        SpeculativeConfig._maybe_log_draft_rope(draft, target)

    assert _info_logged(vllm_caplog)
    assert _warn_logged(vllm_caplog)


@pytest.mark.cpu_test
def test_log_skips_warning_when_draft_declares_rope(
    vllm_caplog: pytest.LogCaptureFixture,
):
    draft = _model_config(SimpleNamespace(rope_parameters=dict(_DEFAULT_ROPE)))
    target = _model_config(SimpleNamespace(rope_parameters=dict(_LLAMA3_ROPE)))

    with patch.object(
        SpeculativeConfig, "_checkpoint_declares_rope", return_value=True
    ):
        SpeculativeConfig._maybe_log_draft_rope(draft, target)

    assert _info_logged(vllm_caplog)
    assert not _warn_logged(vllm_caplog)


@pytest.mark.cpu_test
def test_log_skips_warning_when_bases_match(vllm_caplog: pytest.LogCaptureFixture):
    draft = _model_config(SimpleNamespace(rope_parameters=dict(_DEFAULT_ROPE)))
    target = _model_config(SimpleNamespace(rope_parameters=dict(_DEFAULT_ROPE)))

    with patch.object(
        SpeculativeConfig, "_checkpoint_declares_rope", return_value=False
    ):
        SpeculativeConfig._maybe_log_draft_rope(draft, target)

    assert _info_logged(vllm_caplog)
    assert not _warn_logged(vllm_caplog)


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["eagle", "eagle3"])
def test_eagle3_omitting_draft_warns_against_llama31_target(
    method: Literal["eagle", "eagle3"],
    vllm_caplog: pytest.LogCaptureFixture,
):
    target_model_config = ModelConfig(LLAMA31_TARGET)
    SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        model=EAGLE3_DRAFT_OMITS_ROPE,
        method=method,
        num_speculative_tokens=3,
    )
    assert _info_logged(vllm_caplog)
    assert _warn_logged(vllm_caplog)


@pytest.mark.cpu_test
def test_eagle1_with_theta_does_not_warn(vllm_caplog: pytest.LogCaptureFixture):
    target_model_config = ModelConfig(LLAMA31_TARGET)
    SpeculativeConfig(
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        model=EAGLE1_DRAFT_WITH_THETA,
        method="eagle",
        num_speculative_tokens=3,
    )
    assert _info_logged(vllm_caplog)
    assert not _warn_logged(vllm_caplog)
