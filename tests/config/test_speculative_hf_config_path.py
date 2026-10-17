# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``SpeculativeConfig.hf_config_path``.

``ModelConfig`` has long resolved its config and its weights from separate
references, and ``EngineArgs`` forwards ``hf_config_path`` for the target. A
draft had no way to say the same thing, so a draft whose weights reference
carries no config of its own -- a single-file checkpoint, or one sitting in
the directory of the model it drafts for -- could only be pointed at its
config by rewriting ``model`` and restoring the weights path afterwards.
"""

import pytest

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig

# Public repo; only config/tokenizer-config files are fetched.
AR_MODEL = "JackFram/llama-68m"


@pytest.fixture
def weights_without_config(tmp_path):
    """A weights reference that carries no config of its own."""
    (tmp_path / "model.safetensors").touch()
    return str(tmp_path)


def _speculative_config(**kwargs) -> SpeculativeConfig:
    return SpeculativeConfig(
        target_model_config=ModelConfig(AR_MODEL),
        target_parallel_config=ParallelConfig(),
        method="draft_model",
        num_speculative_tokens=3,
        **kwargs,
    )


@pytest.mark.cpu_test
def test_config_is_read_from_hf_config_path(weights_without_config):
    draft = _speculative_config(
        model=weights_without_config, hf_config_path=AR_MODEL
    ).draft_model_config

    assert draft.hf_config_path == AR_MODEL
    assert draft.hf_config.model_type == "llama"
    assert draft.hf_config.hidden_size == ModelConfig(AR_MODEL).hf_config.hidden_size


@pytest.mark.cpu_test
def test_weights_reference_is_left_alone(weights_without_config):
    """The point of the field: the config moves, the weights stay put."""
    draft = _speculative_config(
        model=weights_without_config, hf_config_path=AR_MODEL
    ).draft_model_config

    assert draft.model == weights_without_config


@pytest.mark.cpu_test
def test_omitting_it_keeps_the_model_as_the_config_source():
    draft = _speculative_config(model=AR_MODEL).draft_model_config

    assert draft.hf_config_path is None
    assert draft.model == AR_MODEL
    assert draft.hf_config.model_type == "llama"
