# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor
from vllm.entrypoints.pooling.scoring.typing import ScoringData
from vllm.pooling_params import PoolingParams


class DummyTokenizeParams:
    def get_encode_kwargs(self):
        return {}

    def apply_post_tokenization(self, tokenizer, engine_prompt):
        pass


class DummyRenderer:
    def __init__(self):
        self.prompts = []

    def process_for_engine(self, prompt, arrival_time):
        self.prompts.append(prompt.copy())
        engine_input = {
            "prompt_token_ids": prompt["prompt_token_ids"],
            "arrival_time": arrival_time,
        }
        if cache_salt := prompt.get("cache_salt"):
            engine_input["cache_salt"] = cache_salt
        return engine_input


def test_cross_encoder_prompt_extras_are_applied_before_engine_processing():
    processor = object.__new__(CrossEncoderIOProcessor)
    processor.model_config = SimpleNamespace(is_encoder_decoder=False)
    processor.tokenizer = object()
    processor.renderer = DummyRenderer()

    def get_score_prompt(**kwargs):
        return None, {
            "prompt_token_ids": [1, 2, 3],
            "token_type_ids": [0, 0, 1],
        }

    processor.get_score_prompt = get_score_prompt

    pooling_params = PoolingParams(task="classify")
    engine_inputs, pooling_params_list = processor._pre_process(
        ScoringData(data_1=["query"], data_2=["document"]),
        DummyTokenizeParams(),
        pooling_params,
        prompt_extras={"cache_salt": "test-salt"},
    )

    assert engine_inputs[0]["cache_salt"] == "test-salt"
    assert processor.renderer.prompts[0]["cache_salt"] == "test-salt"
    assert "compressed_token_type_ids" in pooling_params_list[0].extra_kwargs
