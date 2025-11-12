# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.score_utils import _apply_custom_score_template

from ...utils import VLLM_PATH

score_jinja_path = VLLM_PATH / "examples/template_score_basic.jinja"
assert score_jinja_path.exists()

# Define templates and their corresponding expected outputs
TEMPLATE_OUTPUT = [
    (
        score_jinja_path,
        "What is the capital of France?",
        "Paris is the capital of France.",
        """Query: What is the capital of France?
Document: Paris is the capital of France.""",
    ),
    (
        "Query: {{ query }}\nDocument: {{ document }}",
        "Hello",
        "World",
        "Query: Hello\nDocument: World",
    ),
]


def test_load_score_template():
    # Testing score template loading from file
    template_content = load_chat_template(chat_template=score_jinja_path)

    # Test assertions
    assert template_content is not None
    # Hard coded value for template_score.jinja
    assert (
        template_content
        == """Query: {{ query }}
Document: {{ document }}"""
    )


@pytest.mark.parametrize(
    "template,query,document,expected_output",
    TEMPLATE_OUTPUT,
)
def test_apply_score_template(template, query, document, expected_output):
    template_content = load_chat_template(chat_template=template)

    assert template_content is not None

    # Apply the score template
    result = _apply_custom_score_template(
        score_template=template_content,
        prompt_1=query,
        prompt_2=document,
    )

    # Test assertion
    assert result == expected_output, (
        f"The generated prompt does not match the expected output for "
        f"template {template}"
    )


def test_apply_score_template_invalid_jinja():
    # Test that invalid Jinja2 syntax raises an error
    template = "{{ query }{% invalid %}"

    with pytest.raises(ValueError, match="Error rendering Jinja2 score template"):
        _apply_custom_score_template(
            score_template=template,
            prompt_1="test",
            prompt_2="test",
        )
