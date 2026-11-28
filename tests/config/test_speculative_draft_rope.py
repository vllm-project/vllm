# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EAGLE draft RoPE base logging/warning (#61231).

An EAGLE/EAGLE-3 draft head builds its rotary embedding from the draft
checkpoint's own config only. When that config carries no rope keys,
transformers fills ``rope_parameters`` with its defaults (base 10000),
which silently mismatches a target running at a different base (e.g. the
``yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`` head against a Llama-3.1 target at
base 500000). ``SpeculativeConfig`` must log the resolved draft rope base
at startup and warn when the draft resolves to the silent default while
the target uses a different base. This changes no numerics.
"""

import logging

import pytest
from transformers import LlamaConfig

from vllm.config.speculative import SpeculativeConfig

_LOGGER = "vllm.config.speculative"
_INFO_MSG = "resolves rope base"
_WARN_MSG = "differs from the target"

# Mirrors a Llama-3.1 target: llama3 scaling at base 500000.
_LLAMA3_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 500000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


@pytest.fixture
def vllm_caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Make caplog see vLLM logger records (vLLM sets propagate=False)."""
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        yield caplog


def _logged(caplog: pytest.LogCaptureFixture, msg: str) -> bool:
    return any(msg in record.getMessage() for record in caplog.records)


@pytest.mark.cpu_test
def test_info_log_always_reports_resolved_base(
    vllm_caplog: pytest.LogCaptureFixture,
):
    """The resolved draft rope base is always logged at startup."""
    draft_hf_config = LlamaConfig()  # no rope keys -> default base 10000
    target_hf_config = LlamaConfig()  # default base 10000
    SpeculativeConfig._log_draft_rope_resolution(draft_hf_config, target_hf_config)
    assert _logged(vllm_caplog, _INFO_MSG)
    assert _logged(vllm_caplog, "10000.0")
    assert not _logged(vllm_caplog, _WARN_MSG)


@pytest.mark.cpu_test
def test_warn_on_silent_default_mismatch(vllm_caplog: pytest.LogCaptureFixture):
    """A draft with no rope keys against a non-default target warns,
    naming both bases."""
    draft_hf_config = LlamaConfig()  # no rope keys -> default base 10000
    target_hf_config = LlamaConfig(rope_parameters=dict(_LLAMA3_ROPE))
    SpeculativeConfig._log_draft_rope_resolution(draft_hf_config, target_hf_config)
    assert _logged(vllm_caplog, _INFO_MSG)
    assert _logged(vllm_caplog, _WARN_MSG)
    assert _logged(vllm_caplog, "10000")
    assert _logged(vllm_caplog, "500000")


@pytest.mark.cpu_test
def test_no_warn_when_draft_matches_target(vllm_caplog: pytest.LogCaptureFixture):
    """A draft that explicitly matches the target's rope base stays quiet."""
    draft_hf_config = LlamaConfig(rope_parameters=dict(_LLAMA3_ROPE))
    target_hf_config = LlamaConfig(rope_parameters=dict(_LLAMA3_ROPE))
    SpeculativeConfig._log_draft_rope_resolution(draft_hf_config, target_hf_config)
    assert _logged(vllm_caplog, _INFO_MSG)
    assert not _logged(vllm_caplog, _WARN_MSG)


@pytest.mark.cpu_test
def test_no_warn_when_draft_explicitly_differs(
    vllm_caplog: pytest.LogCaptureFixture,
):
    """A draft that explicitly pins a non-default base is the checkpoint
    author's choice; only the silent default is warned about."""
    draft_hf_config = LlamaConfig(
        rope_parameters={"rope_type": "default", "rope_theta": 20000.0}
    )
    target_hf_config = LlamaConfig(rope_parameters=dict(_LLAMA3_ROPE))
    SpeculativeConfig._log_draft_rope_resolution(draft_hf_config, target_hf_config)
    assert _logged(vllm_caplog, _INFO_MSG)
    assert not _logged(vllm_caplog, _WARN_MSG)
