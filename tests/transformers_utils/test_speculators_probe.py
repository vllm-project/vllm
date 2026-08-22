# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the speculators probe in ``maybe_override_with_speculators``.

The probe reads a model reference only to look for a ``speculators_config``
key. A reference this loader cannot read is not a speculators model, which is
the same conclusion it draws for a config that reads fine without the key.
Raising instead turned the probe into a gate, rejecting formats that an
out-of-tree config parser resolves -- a single-file GGUF checkpoint, say --
before that parser ever runs.
"""

import json
import logging

import pytest

from vllm.transformers_utils.config import maybe_override_with_speculators

_LOGGER = "vllm.transformers_utils.config"
_SKIPPED_MSG = "Not probing"

SPECULATORS_CONFIG = {
    "speculators_model_type": "eagle",
    "speculators_config": {
        "proposal_methods": [{"speculative_tokens": 3}],
        "verifier": {"name_or_path": "target/model"},
    },
    "transformer_layer_config": {},
}


@pytest.fixture
def vllm_caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Make caplog see vLLM logger records (vLLM sets propagate=False)."""
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        yield caplog


def _probe_skipped(caplog: pytest.LogCaptureFixture) -> bool:
    return any(_SKIPPED_MSG in record.getMessage() for record in caplog.records)


def _probe(model: str, spec_config: dict | None = None):
    return maybe_override_with_speculators(
        model=model,
        tokenizer=None,
        trust_remote_code=False,
        vllm_speculative_config=spec_config,
    )


@pytest.mark.cpu_test
def test_unreadable_reference_is_not_a_speculators_model(tmp_path, vllm_caplog):
    """A single file is what a GGUF checkpoint looks like to the probe."""
    checkpoint = tmp_path / "draft.gguf"
    checkpoint.touch()
    spec_config = {"model": str(checkpoint)}

    model, tokenizer, returned = _probe(str(checkpoint), spec_config)

    assert (model, tokenizer) == (str(checkpoint), None)
    assert returned is spec_config
    assert _probe_skipped(vllm_caplog)


@pytest.mark.cpu_test
def test_readable_config_without_the_key_is_unchanged(tmp_path, vllm_caplog):
    """The pre-existing conclusion, reached without going through the except."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    spec_config = {"num_speculative_tokens": 3}

    model, tokenizer, returned = _probe(str(tmp_path), spec_config)

    assert (model, tokenizer) == (str(tmp_path), None)
    assert returned is spec_config
    assert not _probe_skipped(vllm_caplog)


@pytest.mark.cpu_test
def test_readable_speculators_config_is_still_processed(tmp_path, vllm_caplog):
    """Guards the success path the probe was wrapped in a try for."""
    (tmp_path / "config.json").write_text(json.dumps(SPECULATORS_CONFIG))

    model, tokenizer, returned = _probe(str(tmp_path))

    verifier = SPECULATORS_CONFIG["speculators_config"]["verifier"]["name_or_path"]
    assert (model, tokenizer) == (verifier, verifier)
    assert returned["model"] == str(tmp_path)
    assert returned["num_speculative_tokens"] == 3
    assert not _probe_skipped(vllm_caplog)
