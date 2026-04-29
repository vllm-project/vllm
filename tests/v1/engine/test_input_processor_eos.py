# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.engine.input_processor import InputProcessor


def test_input_processor_uses_model_config_eos_when_tokenizer_has_none():
    processor = InputProcessor.__new__(InputProcessor)
    processor.renderer = SimpleNamespace(get_eos_token_id=lambda: None)
    processor.model_config = SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=3))

    assert processor._get_eos_token_id() == 3
