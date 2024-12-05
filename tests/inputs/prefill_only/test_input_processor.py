# mypy: ignore-errors
import pytest

from vllm.inputs.prefill_only.data import (TextOnlyInputs, TextPrompt,
                                           TokensPrompt, ValidationError)
from vllm.inputs.prefill_only.preprocessor import TextInputProcessor

input_processor = TextInputProcessor()


@pytest.fixture(scope="session")
def request_id():
    return "0"


def test_input_processor_1(request_id):
    prompt = "test"
    request = input_processor(request_id, prompt)

    assert request.inputs == {"prompt": prompt}


def test_input_processor_2(request_id):
    prompt = "test"
    inputs = TextPrompt(prompt=prompt)
    request = input_processor(request_id, inputs)

    assert request.inputs == {"prompt": prompt}


def test_input_processor_3(request_id):
    prompt_token_ids = [0]
    inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
    request = input_processor(request_id, inputs)

    assert request.inputs == {"prompt_token_ids": prompt_token_ids}


def test_input_processor_4(request_id):
    prompt = "test"
    prompt_token_ids = [0]
    inputs = TextOnlyInputs(prompt_token_ids=prompt_token_ids)
    request = input_processor(request_id, inputs)

    assert request.inputs == {"prompt_token_ids": prompt_token_ids}

    inputs = TextOnlyInputs(prompt_token_ids=prompt_token_ids, prompt=prompt)
    request = input_processor(request_id, inputs)

    assert request.inputs == {
        "prompt_token_ids": prompt_token_ids,
        "prompt": prompt
    }


def test_input_processor_5(request_id):
    prompt = "test"
    prompt_token_ids = [0]
    inputs = {"prompt_token_ids": prompt_token_ids, "prompt": prompt}

    request = input_processor(request_id, inputs)

    assert request.inputs == inputs


def test_validation_error(request_id):
    with pytest.raises(ValidationError):
        inputs = {}
        input_processor(request_id, inputs)

    with pytest.raises(ValidationError):
        inputs = {"foo": "bar"}
        input_processor(request_id, inputs)

    with pytest.raises(ValidationError):
        inputs = 0
        input_processor(request_id, inputs)

    with pytest.raises(ValidationError):
        inputs = 0.0
        input_processor(request_id, inputs)
