# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (apply_hf_chat_template,
                                         load_chat_template)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.transformers_utils.tokenizer import get_tokenizer

from ...models.registry import HF_EXAMPLE_MODELS
from ...utils import VLLM_PATH

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()

# Define models, templates, and their corresponding expected outputs
MODEL_TEMPLATE_GENERATON_OUTPUT = [
    ("facebook/opt-125m", chatml_jinja_path, True, False, """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of<|im_end|>
<|im_start|>assistant
"""),
    ("facebook/opt-125m", chatml_jinja_path, False, False, """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of"""),
    ("facebook/opt-125m", chatml_jinja_path, False, True, """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of<|im_end|>
<|im_start|>assistant
The capital of"""),
]

TEST_MESSAGES = [
    {
        'role': 'user',
        'content': 'Hello'
    },
    {
        'role': 'assistant',
        'content': 'Hi there!'
    },
    {
        'role': 'user',
        'content': 'What is the capital of'
    },
]
ASSISTANT_MESSAGE_TO_CONTINUE = {
    'role': 'assistant',
    'content': 'The capital of'
}


def test_load_chat_template():
    # Testing chatml template
    template_content = load_chat_template(chat_template=chatml_jinja_path)

    # Test assertions
    assert template_content is not None
    # Hard coded value for template_chatml.jinja
    assert template_content == """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""  # noqa: E501


def test_no_load_chat_template_filelike():
    # Testing chatml template
    template = "../../examples/does_not_exist"

    with pytest.raises(ValueError, match="looks like a file path"):
        load_chat_template(chat_template=template)


def test_no_load_chat_template_literallike():
    # Testing chatml template
    template = "{{ messages }}"

    template_content = load_chat_template(chat_template=template)

    assert template_content == template


@pytest.mark.parametrize(
    "model,template,add_generation_prompt,continue_final_message,expected_output",
    MODEL_TEMPLATE_GENERATON_OUTPUT)
def test_get_gen_prompt(model, template, add_generation_prompt,
                        continue_final_message, expected_output):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
    )

    # Initialize the tokenizer
    tokenizer = get_tokenizer(
        tokenizer_name=model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code,
    )
    template_content = load_chat_template(chat_template=template)

    # Create a mock request object using keyword arguments
    mock_request = ChatCompletionRequest(
        model=model,
        messages=TEST_MESSAGES + [ASSISTANT_MESSAGE_TO_CONTINUE]
        if continue_final_message else TEST_MESSAGES,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )

    # Call the function and get the result
    result = apply_hf_chat_template(
        tokenizer=tokenizer,
        conversation=mock_request.messages,
        chat_template=mock_request.chat_template or template_content,
        model_config=model_config,
        tools=None,
        add_generation_prompt=mock_request.add_generation_prompt,
        continue_final_message=mock_request.continue_final_message,
    )

    # Test assertion
    assert result == expected_output, (
        f"The generated prompt does not match the expected output for "
        f"model {model} and template {template}")
