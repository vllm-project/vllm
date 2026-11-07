# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import ModelConfig
from vllm.config.reasoning import ReasoningConfig
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def test_turn_end_token_ids_include_tokenizer_and_generation_config_eos():
    """Qwen3 ends chat turns with <|im_end|>, the tokenizer's EOS; its
    generation config also lists <|endoftext|>."""
    reasoning_config = ReasoningConfig(reasoning_parser="qwen3")
    reasoning_config.initialize_token_ids(ModelConfig(MODEL_NAME))

    tokenizer = get_tokenizer(MODEL_NAME)
    assert set(reasoning_config.turn_end_token_ids) == {
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
