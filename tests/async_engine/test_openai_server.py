from argparse import Namespace
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.api_server import *

# Define models, templates, and their corresponding expected outputs
MODEL_TEMPLATE_GENERATON_OUTPUT = [
    ("facebook/opt-125m", None, True,
     "Hello</s>Hi there!</s>What is the capital of</s>"),
    ("facebook/opt-125m", None, False,
     "Hello</s>Hi there!</s>What is the capital of</s>"),
    ("facebook/opt-125m", "../../examples/template_chatml.jinja", True,
     """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of<|im_end|>
<|im_start|>assistant
"""),
    ("facebook/opt-125m", "../../examples/template_chatml.jinja", False,
     """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of""")
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
client = TestClient(app)


@dataclass
class MockTokenizer:
    chat_template = None


def test_load_chat_template():
    # Testing chatml template
    template = "../../examples/template_chatml.jinja"
    mock_args = Namespace(chat_template=template)
    tokenizer = MockTokenizer()

    # Call the function with the mocked args
    load_chat_template(mock_args, tokenizer)

    template_content = tokenizer.chat_template

    # Test assertions
    assert template_content is not None
    # Hard coded value for template_chatml.jinja
    assert template_content == """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""


def test_no_load_chat_template():
    # Testing chatml template
    template = "../../examples/does_not_exist"
    mock_args = Namespace(chat_template=template)
    tokenizer = MockTokenizer()

    # Call the function with the mocked args
    load_chat_template(mock_args, tokenizer=tokenizer)
    template_content = tokenizer.chat_template

    # Test assertions
    assert template_content is not None
    # Hard coded value for template_chatml.jinja
    assert template_content == """../../examples/does_not_exist"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,template,add_generation_prompt,expected_output",
    MODEL_TEMPLATE_GENERATON_OUTPUT)
async def test_get_gen_prompt(model, template, add_generation_prompt,
                              expected_output):
    # Initialize the tokenizer
    tokenizer = get_tokenizer(tokenizer_name=model)

    mock_args = Namespace(chat_template=template)
    load_chat_template(mock_args, tokenizer)

    # Create a mock request object using keyword arguments
    mock_request = ChatCompletionRequest(
        model=model,
        messages=TEST_MESSAGES,
        add_generation_prompt=add_generation_prompt)

    # Call the function and get the result
    result = tokenizer.apply_chat_template(
        conversation=mock_request.messages,
        tokenize=False,
        add_generation_prompt=mock_request.add_generation_prompt)

    # Test assertion
    assert result == expected_output, f"The generated prompt does not match the expected output for model {model} and template {template}"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
