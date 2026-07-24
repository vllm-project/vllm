# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import SpeculativeConfig

pytestmark = pytest.mark.skip_global_cleanup


class _FakeModelConfig:
    def __init__(self, vocab_size: int):
        self.hf_text_config = SimpleNamespace(vocab_size=vocab_size)
        self.hf_config = SimpleNamespace(
            vocab_size=vocab_size,
            text_config=self.hf_text_config,
        )
        self.model_arch_config = SimpleNamespace(vocab_size=vocab_size)

    def get_vocab_size(self) -> int:
        return self.model_arch_config.vocab_size

    def verify_with_parallel_config(self, parallel_config):
        pass


def _new_speculative_config(
    *,
    method: str = "draft_model",
    allow_vocab_padding: bool = False,
    use_heterogeneous_vocab: bool = False,
    target_vocab_size: int = 20,
    draft_vocab_size: int = 14,
):
    config = object.__new__(SpeculativeConfig)
    config.tensor_parallel_size = None
    config.num_speculative_tokens = 1
    config.rejection_sample_method = "standard"
    config.synthetic_acceptance_rates = None
    config.synthetic_acceptance_length = None
    config.method = method
    config.allow_draft_model_vocab_padding = allow_vocab_padding
    config.use_heterogeneous_vocab = use_heterogeneous_vocab
    config.draft_sample_method = "greedy"
    config.draft_model_unpadded_vocab_size = None
    config.target_model_config = _FakeModelConfig(target_vocab_size)
    config.draft_model_config = _FakeModelConfig(draft_vocab_size)
    config.draft_parallel_config = None
    return config


def test_draft_model_vocab_mismatch_still_rejected_by_default():
    config = _new_speculative_config()

    with pytest.raises(ValueError, match="same vocabulary size"):
        SpeculativeConfig.verify_equal_vocab_size_if_draft_model(config)


def test_draft_model_vocab_padding_updates_runtime_vocab_size():
    config = _new_speculative_config(allow_vocab_padding=True)

    SpeculativeConfig._maybe_enable_draft_model_vocab_padding(config)

    assert config.draft_model_unpadded_vocab_size == 14
    assert config.draft_model_config.get_vocab_size() == 20
    assert config.draft_model_config.hf_config.vocab_size == 20
    assert config.draft_model_config.hf_text_config.vocab_size == 20
    SpeculativeConfig.verify_equal_vocab_size_if_draft_model(config)


@pytest.mark.parametrize("method", ["eagle", "mtp", "ngram"])
def test_draft_model_vocab_padding_only_supports_draft_model(method):
    config = _new_speculative_config(
        method=method,
        allow_vocab_padding=True,
    )

    with pytest.raises(ValueError, match="method='draft_model'"):
        SpeculativeConfig._maybe_enable_draft_model_vocab_padding(config)


def test_draft_model_vocab_padding_rejects_heterogeneous_vocab_mode():
    config = _new_speculative_config(
        allow_vocab_padding=True,
        use_heterogeneous_vocab=True,
    )

    with pytest.raises(ValueError, match="cannot be enabled together"):
        SpeculativeConfig._verify_args(config)


@pytest.mark.parametrize(
    ("target_vocab_size", "draft_vocab_size"),
    [(20, 20), (14, 20)],
)
def test_draft_model_vocab_padding_requires_larger_target_vocab(
    target_vocab_size,
    draft_vocab_size,
):
    config = _new_speculative_config(
        allow_vocab_padding=True,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
    )

    with pytest.raises(ValueError, match="target model vocabulary to be larger"):
        SpeculativeConfig._maybe_enable_draft_model_vocab_padding(config)
