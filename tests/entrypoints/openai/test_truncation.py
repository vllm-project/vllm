# SPDX-License-Identifier: Apache-2.0
import pytest
from fastapi import HTTPException

model_name = "intfloat/multilingual-e5-large"
input = "The lake gets frozen in the winter"
max_model_len = 512


def test_smaller_truncation_size(vllm_runner,
                                 model_name=model_name,
                                 max_model_len=max_model_len):
    with vllm_runner(model_name, task="embed",
                     max_model_len=max_model_len) as vllm_model:
        truncation_size = 10
        response = vllm_model.model.encode(
            prompts=input, truncate_prompt_tokens=truncation_size)

    assert len(response.prompt_tokens) == truncation_size


def test_bigger_truncation_size(vllm_runner,
                                model_name=model_name,
                                max_model_len=max_model_len):

    with vllm_runner(model_name, task="embed",
                     max_model_len=max_model_len) as vllm_model:
        truncation_size = max_model_len + 1

        with pytest.raises(HTTPException):
            assert str(
                vllm_model.model.encode(prompts=input,
                                        truncate_prompt_tokens=truncation_size)
            ) == f"fastapi.exceptions.HTTPException: 400: \
                truncate_prompt_tokens value ({truncation_size}) \
                is greater than max_model_len ({max_model_len}).\
                Please, select a smaller truncation size."


def test_max_truncation_size(vllm_runner,
                             model_name=model_name,
                             max_model_len=max_model_len):
    with vllm_runner(model_name, task="embed",
                     max_model_len=max_model_len) as vllm_model:
        truncation_size = -1
        response = vllm_model.model.encode(
            prompts=input, truncate_prompt_tokens=truncation_size)

    assert len(response.prompt_tokens) == max_model_len
