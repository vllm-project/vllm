# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest

from vllm.inputs.preprocess import InputPreprocessor


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    truncation_side = "right"

    def encode(self, text, **kwargs):
        """Respect truncation kwargs to simulate HF tokenizer behavior."""
        truncation = kwargs.get("truncation")
        max_length = kwargs.get("max_length")
        if truncation and max_length is not None:
            return list(range(max_length))
        # Simulate long input when no truncation is applied.
        return list(range(16))


def make_preprocessor(max_len: int = 8, encoder_cfg: dict | None = None):
    cfg = types.SimpleNamespace(
        max_model_len=max_len,
        is_encoder_decoder=False,
        encoder_config=encoder_cfg or {},
        hf_config=None,
    )
    return InputPreprocessor(cfg, DummyTokenizer())


def test_default_truncation_raises_value_error():
    preprocessor = make_preprocessor()
    with pytest.raises(ValueError):
        preprocessor._tokenize_prompt("x" * 100)


def test_explicit_disable_truncation_allows_long_inputs():
    preprocessor = make_preprocessor()
    tokens = preprocessor._tokenize_prompt("x" * 100, {"truncation": False})
    assert len(tokens) > preprocessor.model_config.max_model_len


def test_custom_max_length_applies_truncation():
    preprocessor = make_preprocessor()
    tokens = preprocessor._tokenize_prompt(
        "x" * 100, {"truncation": True, "max_length": 5}
    )
    assert len(tokens) == 5


def test_truncate_inputs_right_side():
    preprocessor = make_preprocessor()
    preprocessor.tokenizer.truncation_side = "right"
    inputs = list(range(10))

    truncated = preprocessor._truncate_inputs(
        inputs, {"truncation": True, "max_length": 4}
    )

    assert truncated == inputs[:4]


def test_truncate_inputs_left_side():
    preprocessor = make_preprocessor()
    preprocessor.tokenizer.truncation_side = "left"
    inputs = list(range(10))

    truncated = preprocessor._truncate_inputs(
        inputs, {"truncation": True, "max_length": 4}
    )

    assert truncated == inputs[-4:]
