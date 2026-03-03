# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.hf import (
    _get_hf_base_chat_template_params,
    _try_extract_ast,
    resolve_chat_template,
    resolve_chat_template_content_format,
    resolve_chat_template_kwargs,
    safe_apply_chat_template,
)
from vllm.tokenizers import get_tokenizer

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import VLLM_PATH

EXAMPLES_DIR = VLLM_PATH / "examples"

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()

# Define models, templates, and their corresponding expected outputs
MODEL_TEMPLATE_GENERATION_OUTPUT = [
    (
        "facebook/opt-125m",
        chatml_jinja_path,
        True,
        False,
        """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of<|im_end|>
<|im_start|>assistant
""",
    ),
    (
        "facebook/opt-125m",
        chatml_jinja_path,
        False,
        False,
        """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of""",
    ),
    (
        "facebook/opt-125m",
        chatml_jinja_path,
        False,
        True,
        """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
<|im_start|>user
What is the capital of<|im_end|>
<|im_start|>assistant
The capital of""",
    ),
]

TEST_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is the capital of"},
]
ASSISTANT_MESSAGE_TO_CONTINUE = {"role": "assistant", "content": "The capital of"}


def test_load_chat_template():
    # Testing chatml template
    template_content = load_chat_template(chat_template=chatml_jinja_path)

    # Test assertions
    assert template_content is not None
    # Hard coded value for template_chatml.jinja
    assert (
        template_content
        == """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""  # noqa: E501
    )


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
    "model",
    [
        "Qwen/Qwen2-VL-2B-Instruct",  # chat_template is of type str
        "NousResearch/Hermes-3-Llama-3.1-8B",  # chat_template is of type dict
    ],
)
@pytest.mark.parametrize("use_tools", [True, False])
def test_resolve_chat_template(sample_json_schema, model, use_tools):
    """checks that chat_template is a dict type for HF models."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    # Build the tokenizer
    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "dummy_function_name",
                    "description": "This is a dummy function",
                    "parameters": sample_json_schema,
                },
            }
        ]
        if use_tools
        else None
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=None,
        tools=tools,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)


@pytest.mark.parametrize(
    "model, expected_kwargs",
    [
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            {
                "add_vision_id",
                "add_generation_prompt",
                "continue_final_message",
                "tools",
            },
        ),
        (
            "Qwen/Qwen3-8B",
            {
                "enable_thinking",
                "add_generation_prompt",
                "continue_final_message",
                "tools",
            },
        ),
    ],
)
def test_resolve_chat_template_kwargs(sample_json_schema, model, expected_kwargs):
    """checks that chat_template is a dict type for HF models."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "dummy_function_name",
                "description": "This is a dummy function",
                "parameters": sample_json_schema,
            },
        }
    ]

    chat_template_kwargs = {
        # both unused
        "unsed_kwargs_1": 123,
        "unsed_kwargs_2": "abc",
        # should not appear
        "chat_template": "{% Hello world! %}",
        "tokenize": True,
        # used by tokenizer
        "continue_final_message": True,
        "tools": tools,
        # both used by Qwen2-VL and Qwen3
        "add_generation_prompt": True,
        # only used by Qwen2-VL
        "add_vision_id": True,
        # only used by Qwen3
        "enable_thinking": True,
    }

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    # Build the tokenizer
    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=None,
        tools=tools,
        model_config=model_config,
    )
    with pytest.raises(
        ValueError, match="Found unexpected chat template kwargs from request"
    ):
        # should raise error if `chat_template_kwargs` contains
        # `chat_template` or `tokenize`
        resolve_chat_template_kwargs(
            tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
    resolved_chat_template_kwargs = resolve_chat_template_kwargs(
        tokenizer,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
        raise_on_unexpected=False,
    )
    assert set(resolved_chat_template_kwargs.keys()) == expected_kwargs

    # Additional test: Verify HF base parameters work with **kwargs tokenizers
    # This validates the fix for tokenizers like Kimi K2 that use **kwargs
    # to receive standard HuggingFace parameters instead of declaring them explicitly
    hf_base_params = _get_hf_base_chat_template_params()
    # Verify common HF parameters are in the base class
    assert {"add_generation_prompt", "tools", "continue_final_message"}.issubset(
        hf_base_params
    ), f"Expected HF base params not found in {hf_base_params}"

    # Test with a mock tokenizer that uses **kwargs (like Kimi K2)
    class MockTokenizerWithKwargs:
        def apply_chat_template(self, conversation, **kwargs):
            return "mocked_output"

    mock_tokenizer = MockTokenizerWithKwargs()
    mock_kwargs = {
        "add_generation_prompt": True,
        "tools": tools,
        "continue_final_message": False,
        "unknown_param": "should_be_filtered",
    }
    resolved_mock = resolve_chat_template_kwargs(
        mock_tokenizer, chat_template, mock_kwargs, raise_on_unexpected=False
    )
    # HF base params should pass through even with **kwargs tokenizer
    assert "add_generation_prompt" in resolved_mock
    assert "tools" in resolved_mock
    assert "continue_final_message" in resolved_mock
    # Unknown params should be filtered out
    assert "unknown_param" not in resolved_mock


# NOTE: Qwen2-Audio default chat template is specially defined inside
# processor class instead of using `tokenizer_config.json`
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [
        ("microsoft/Phi-3.5-vision-instruct", "string"),
        ("Qwen/Qwen2-VL-2B-Instruct", "openai"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "openai"),
        ("fixie-ai/ultravox-v0_5-llama-3_2-1b", "string"),
        ("Qwen/Qwen2-Audio-7B-Instruct", "openai"),
        ("meta-llama/Llama-Guard-3-1B", "openai"),
    ],
)
def test_resolve_content_format_hf_defined(model, expected_format):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=None,
        tools=None,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        None,  # Test detecting the tokenizer's chat_template
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


@pytest.mark.parametrize(
    ("model", "expected_format"),
    [
        ("Salesforce/blip2-opt-2.7b", "string"),
        ("facebook/chameleon-7b", "string"),
        ("deepseek-ai/deepseek-vl2-tiny", "string"),
        ("adept/fuyu-8b", "string"),
        ("google/paligemma-3b-mix-224", "string"),
        ("Qwen/Qwen-VL", "string"),
        ("Qwen/Qwen-VL-Chat", "string"),
    ],
)
def test_resolve_content_format_fallbacks(model, expected_format):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_chat_template(
        tokenizer,
        chat_template=None,
        tools=None,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        None,  # Test detecting the tokenizer's chat_template
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


@pytest.mark.parametrize(
    ("template_path", "expected_format"),
    [
        ("template_alpaca.jinja", "string"),
        ("template_baichuan.jinja", "string"),
        ("template_chatglm.jinja", "string"),
        ("template_chatglm2.jinja", "string"),
        ("template_chatml.jinja", "string"),
        ("template_dse_qwen2_vl.jinja", "openai"),
        ("template_falcon_180b.jinja", "string"),
        ("template_falcon.jinja", "string"),
        ("template_inkbot.jinja", "string"),
        ("template_teleflm.jinja", "string"),
        ("template_vlm2vec_phi3v.jinja", "openai"),
        ("template_vlm2vec_qwen2vl.jinja", "openai"),
        ("tool_chat_template_granite_20b_fc.jinja", "string"),
        ("tool_chat_template_hermes.jinja", "string"),
        ("tool_chat_template_internlm2_tool.jinja", "string"),
        ("tool_chat_template_llama3.1_json.jinja", "openai"),
        ("tool_chat_template_llama3.2_json.jinja", "openai"),
        ("tool_chat_template_mistral_parallel.jinja", "string"),
        ("tool_chat_template_mistral.jinja", "string"),
    ],
)
def test_resolve_content_format_examples(template_path, expected_format):
    model = "Qwen/Qwen2-VL-2B-Instruct"  # Dummy
    model_config = ModelConfig(
        model,
        tokenizer=model,
        trust_remote_code=True,
    )

    dummy_tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )
    dummy_tokenizer.chat_template = None

    chat_template = load_chat_template(EXAMPLES_DIR / template_path)
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        chat_template,
        None,
        "auto",
        dummy_tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


@pytest.mark.parametrize(
    "model,template,add_generation_prompt,continue_final_message,expected_output",
    MODEL_TEMPLATE_GENERATION_OUTPUT,
)
def test_get_gen_prompt(
    model, template, add_generation_prompt, continue_final_message, expected_output
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        trust_remote_code=model_info.trust_remote_code,
        revision=model_info.revision,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
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
        if continue_final_message
        else TEST_MESSAGES,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )

    # Call the function and get the result
    result = safe_apply_chat_template(
        model_config,
        tokenizer,
        mock_request.messages,
        tools=None,
        chat_template=mock_request.chat_template or template_content,
        add_generation_prompt=mock_request.add_generation_prompt,
        continue_final_message=mock_request.continue_final_message,
        tokenize=False,
    )

    # Test assertion
    assert result == expected_output, (
        f"The generated prompt does not match the expected output for "
        f"model {model} and template {template}"
    )
