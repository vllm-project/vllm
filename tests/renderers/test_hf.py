# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import jinja2
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import MultiModalInput
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
from vllm.renderers.chat_utils import (
    ConversationMessage,
    load_chat_template,
    parse_chat_messages,
)
from vllm.renderers.hf import (
    HfRenderer,
    _append_content,
    _build_prompt_embeds_updates,
    _detect_content_format,
    _detect_developer_role_support,
    _ensure_prompt_embeds_placeholder_token,
    _get_hf_base_chat_template_params,
    _lower_developer_messages,
    _resolve_developer_role_mode,
    _template_error_reason,
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


def test_apply_prompt_embeds_keeps_engine_token_ids_in_vocabulary() -> None:
    """Consume OOV sentinels without changing text IDs or embedding offsets."""
    model = "HuggingFaceTB/SmolVLM-256M-Instruct"
    model_config = ModelConfig(model, enable_prompt_embeds=True)
    renderer = HfRenderer(VllmConfig(model_config=model_config), get_tokenizer(model))
    try:
        assert renderer.tokenizer is not None
        sentinel = _ensure_prompt_embeds_placeholder_token(renderer.tokenizer)
        vocab_size = model_config.get_vocab_size()
        assert sentinel >= vocab_size
        tensors = [
            torch.zeros(length, model_config.get_hidden_size()) for length in (2, 1)
        ]
        engine_input: MultiModalInput = {
            "type": "multimodal",
            "prompt_token_ids": [1, sentinel, sentinel, 2, sentinel, 3],
            "mm_kwargs": MultiModalKwargsItems(),
            "mm_hashes": {},
            "mm_placeholders": {},
        }

        renderer._apply_prompt_embeds_to_engine_input(
            engine_input, tensors, _build_prompt_embeds_updates(tensors, sentinel)
        )

        assert all(
            0 <= token_id < vocab_size for token_id in engine_input["prompt_token_ids"]
        )
        assert engine_input["prompt_token_ids"] == [1, 0, 0, 2, 0, 3]
        assert engine_input["mm_placeholders"]["prompt_embeds"] == [
            PlaceholderRange(offset=1, length=2),
            PlaceholderRange(offset=4, length=1),
        ]
    finally:
        renderer.shutdown()


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

    with pytest.raises(VLLMValidationError, match="looks like a file path"):
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
    """Checks that chat_template is a dict type for HF models."""
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


def test_processor_multiple_chat_templates_uses_default():
    """Render with the "default" template of a processor that has several."""
    # Ships chat_template.jinja and additional_chat_templates/reranker.jinja
    model = "Qwen/Qwen3-VL-Reranker-2B"
    model_config = ModelConfig(model)
    tokenizer = get_tokenizer(model)

    prompt = safe_apply_chat_template(
        model_config,
        tokenizer,
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"


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
    """Checks that chat_template is a dict type for HF models."""
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
        "unused_kwargs_1": 123,
        "unused_kwargs_2": "abc",
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


def test_resolve_chat_template_resolves_name():
    """When chat_template is a name, resolve_chat_template should return
    the actual Jinja content so that kwargs detection works correctly."""
    from unittest.mock import MagicMock

    jinja_content = "{{ messages }}{% if tools %}{{ tools }}{% endif %}"
    tokenizer = MagicMock()
    tokenizer.get_chat_template.return_value = jinja_content

    model_config = MagicMock()

    result = resolve_chat_template(
        tokenizer,
        chat_template="tool_use",
        tools=None,
        model_config=model_config,
    )

    assert result == jinja_content
    tokenizer.get_chat_template.assert_called_once_with("tool_use", tools=None)


def test_resolve_chat_template_kwargs_with_template_name():
    """Ensures template kwargs are not silently dropped when chat_template
    was originally a template name that has been resolved to Jinja content."""
    from unittest.mock import MagicMock

    jinja_content = (
        "{% for m in messages %}{{ m }}{% endfor %}"
        "{% if tools %}{{ tools }}{% endif %}"
        "{% if documents %}{{ documents }}{% endif %}"
    )

    tokenizer = MagicMock()
    tokenizer.apply_chat_template = MagicMock()

    kwargs = {
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "documents": [{"title": "doc"}],
        "unknown_param": "should be dropped",
    }

    resolved = resolve_chat_template_kwargs(
        tokenizer,
        chat_template=jinja_content,
        chat_template_kwargs=kwargs,
        raise_on_unexpected=False,
    )

    # template vars "tools" and "documents" should be preserved
    assert "tools" in resolved
    assert "documents" in resolved
    # unknown param should be filtered
    assert "unknown_param" not in resolved


@pytest.mark.parametrize(
    ("chat_template", "expected_format"),
    [
        (
            """
            {% macro render_message(role, message_content) %}
              {% for item in message_content %}{{ item['type'] }}{% endfor %}
            {% endmacro %}
            {% for message in messages %}
              {{ render_message(message['role'], message['content']) }}
            {% endfor %}
            """,
            "openai",
        ),
        (
            """
            {% macro render_message(role, message_content) %}
              {% for item in message_content %}{{ item['type'] }}{% endfor %}
            {% endmacro %}
            {% for message in messages %}
              {{ render_message(
                  role=message['role'],
                  message_content=message['content'],
              ) }}
            {% endfor %}
            """,
            "openai",
        ),
        (
            """
            {% for message in messages %}
              {% for item in message['content'] %}{{ item['type'] }}{% endfor %}
            {% endfor %}
            """,
            "openai",
        ),
        (
            """
            {% macro render_message(content) %}
              {% for item in content %}{{ item['type'] }}{% endfor %}
            {% endmacro %}
            {% for message in messages %}
              {{ render_message(content=['text only']) }}
            {% endfor %}
            """,
            "string",
        ),
        (
            """
            {% for message in messages %}{{ message['content'] }}{% endfor %}
            """,
            "string",
        ),
        (
            """
            {% set ns = namespace(rest=messages) %}
            {% for message in ns.rest %}
              {% for item in message['content'] %}{{ item['type'] }}{% endfor %}
            {% endfor %}
            """,
            "openai",
        ),
        (
            """
            {% set ns = namespace(rest=[]) %}
            {% set ns.rest = messages %}
            {% for message in ns.rest %}
              {% for item in message['content'] %}{{ item['type'] }}{% endfor %}
            {% endfor %}
            """,
            "openai",
        ),
    ],
)
def test_detect_content_format(chat_template, expected_format):
    """Detect content format when content is passed through a macro."""
    assert _detect_content_format(chat_template, default="string") == expected_format


# NOTE: Qwen2-Audio default chat template is specially defined inside
# processor class instead of using `tokenizer_config.json`
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [
        ("microsoft/Phi-3.5-vision-instruct", "string"),
        ("Qwen/Qwen2-VL-2B-Instruct", "openai"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "openai"),
        ("Qwen/Qwen3.5-4B", "openai"),
        ("fixie-ai/ultravox-v0_5-llama-3_2-1b", "string"),
        ("Qwen/Qwen2-Audio-7B-Instruct", "openai"),
        ("meta-llama/Llama-Guard-3-1B", "openai"),
        ("XiaomiMiMo/MiMo-V2.5-Pro", "openai"),
    ],
)
def test_resolve_content_format_hf_defined(model, expected_format):
    """Detect the chat template content format for built-in HF models."""
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
        ("deepseek-ai/deepseek-vl2-tiny", "string"),
        ("google/paligemma-3b-mix-224", "string"),
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
        ("template_chatglm.jinja", "string"),
        ("template_chatglm2.jinja", "string"),
        ("template_chatml.jinja", "string"),
        ("template_falcon_180b.jinja", "string"),
        ("template_falcon.jinja", "string"),
        ("template_inkbot.jinja", "string"),
        ("template_teleflm.jinja", "string"),
        ("pooling/embed/template/dse_qwen2_vl.jinja", "openai"),
        ("pooling/embed/template/vlm2vec_phi3v.jinja", "openai"),
        ("pooling/embed/template/vlm2vec_qwen2vl.jinja", "openai"),
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
    # Clear the HF tokenizer's own template so resolution falls back to the
    # one loaded below; ``chat_template`` is not part of ``TokenizerLike``.
    dummy_tokenizer.chat_template = None  # type: ignore[attr-defined]

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

    # Call the function and get the result. The plain role/content request
    # messages are already in conversation shape, which mypy cannot see
    # through ``ChatCompletionMessageParam``.
    result = safe_apply_chat_template(  # type: ignore[call-overload]
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


class TestLowerDeveloperMessages:
    def test_removes_tools_key(self):
        conversation = [
            {"role": "user", "content": "Hello"},
            {
                "role": "developer",
                "content": "Instructions",
                "tools": [{"type": "function"}],
            },
        ]
        result = _lower_developer_messages(conversation, "system")
        assert result[1] == {"role": "system", "content": "Instructions"}

    def test_leading_run_merged_keeping_parts(self):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "Rule 1."}]},
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Rule 2."}, {"type": "image"}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result = _lower_developer_messages(conversation, "leading_system")
        assert result == [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Rule 1.\n\nRule 2."},
                    {"type": "image"},
                ],
            },
            conversation[2],
        ]

    @pytest.mark.parametrize("mode", ["system", "leading_system"])
    def test_does_not_mutate_original(self, mode):
        conversation = [
            {"role": "developer", "content": "base"},
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "developer", "content": "update", "tools": []},
        ]
        original = copy.deepcopy(conversation)
        _lower_developer_messages(conversation, mode)
        assert conversation == original

    def test_append_content_joins_text_at_the_boundary(self):
        # The cases of the Rust frontend's `ChatContent::append` test.
        image = {"type": "image"}

        def text(value: str) -> dict[str, str]:
            return {"type": "text", "text": value}

        assert _append_content("a", "b") == "a\n\nb"
        assert _append_content("a", [text("b"), image]) == [text("a\n\nb"), image]
        assert _append_content([text("a"), image], "b") == [
            text("a"),
            image,
            text("b"),
        ]
        assert _append_content("", [image]) == [image]
        assert _append_content("a", "") == "a"


# Like Llama 2's and Mistral 7B's templates: user and assistant turns alternate
# after an optional first system message.
ALTERNATING_TURNS_TEMPLATE = (
    "{%- if messages[0].role == 'system' %}"
    "{%- set loop_messages = messages[1:] %}"
    "{{ messages[0].role }}={{ messages[0].content }};"
    "{%- else %}{%- set loop_messages = messages %}{%- endif %}"
    "{%- for m in loop_messages %}"
    "{%- if (m.role == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate "
    "user/assistant/user/assistant/...') }}"
    "{%- endif %}"
    "{{ m.role }}={{ m.content }};"
    "{%- endfor %}"
)


# The templates of the Rust frontend's
# `probes_how_templates_take_developer_messages` test.
@pytest.mark.parametrize(
    ("chat_template", "expected_mode"),
    [
        pytest.param(
            "{%- for m in messages %}{%- if m.role == 'developer' %}<dev>"
            "{%- endif %}{{ m.content }}{%- endfor %}",
            "native",
            id="names_developer",  # gpt-oss
        ),
        pytest.param(
            "{%- for m in messages %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            "system",
            id="any_role",  # Qwen3, Qwen3-Coder
        ),
        pytest.param(
            "{%- for m in messages %}"
            "{{ bos_token + '<' + m.role + '>' + m.content }}{%- endfor %}",
            "system",
            id="needs_bos_token",  # Llama 3
        ),
        pytest.param(
            "{%- if not tools %}{{ raise_exception('tools required') }}"
            "{%- endif %}{%- for m in messages %}<{{ m.role }}>{{ m.content }}"
            "{%- endfor %}",
            "system",
            id="needs_tools",  # a `tool_use` template
        ),
        pytest.param(
            "{%- for m in messages %}"
            "{%- if m.role == 'system' and not loop.first %}"
            "{{ raise_exception('System message must be at the beginning.') }}"
            "{%- endif %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            "leading_system",
            id="leading_system_raises",  # Qwen3.5
        ),
        pytest.param(
            "{%- for m in messages %}{%- if m.role == 'system' and "
            "loop.previtem is defined and loop.previtem.role == 'assistant' %}"
            "{{ raise_exception('Unexpected role system after role assistant') }}"
            "{%- endif %}<{{ m.role }}>{{ m.content }}{%- endfor %}",
            "leading_system",
            id="system_after_user_only",  # Ministral 3
        ),
        pytest.param(
            ALTERNATING_TURNS_TEMPLATE,
            "leading_system",
            id="alternating_turns",  # Llama 2
        ),
        pytest.param(
            "{%- if messages[0].role == 'system' %}<system>"
            "{{ messages[0].content }}{%- endif %}{%- for m in messages %}"
            "{%- if m.role != 'system' %}<{{ m.role }}>{{ m.content }}{%- endif %}"
            "{%- endfor %}",
            "leading_system",
            id="leading_system_only",
        ),
    ],
)
def test_resolve_developer_role_mode(chat_template, expected_mode):
    tokenizer = get_tokenizer("facebook/opt-125m")
    assert _resolve_developer_role_mode(tokenizer, chat_template) == expected_mode


# --- Developer role detection and conversion tests ---

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}"
    "{% if (loop.last and add_generation_prompt) or not loop.last %}"
    "{{ '<|im_end|>' + '\\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

TEMPLATE_WITH_DEVELOPER = (
    "{% for message in messages %}"
    "{% if message['role'] == 'developer' %}"
    "{{'<|im_start|>developer\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'system' %}"
    "{{'<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{{'<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)

STRICT_ROLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{'<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{{'<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% else %}"
    "{{ raise_exception('Unexpected message role: ' + message['role']) }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


class TestDetectDeveloperRoleSupport:
    def test_absent_in_chatml(self):
        assert _detect_developer_role_support(CHATML_TEMPLATE) is False

    def test_present_double_quotes(self):
        assert _detect_developer_role_support(TEMPLATE_WITH_DEVELOPER) is True

    def test_present_single_quotes(self):
        template = TEMPLATE_WITH_DEVELOPER.replace('"developer"', "'developer'")
        assert _detect_developer_role_support(template) is True

    def test_absent_in_strict_template(self):
        assert _detect_developer_role_support(STRICT_ROLE_TEMPLATE) is False


# Codex: instructions, its base prompt as a developer message, and a developer
# context update mid-conversation.
CODEX_MESSAGES = [
    {"role": "system", "content": "instructions"},
    {"role": "developer", "content": "base"},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
    {"role": "developer", "content": "update"},
    {"role": "user", "content": "follow-up"},
]

# The caller's own non-leading system message stays as sent.
CALLER_SYSTEM_MESSAGES = [
    {"role": "developer", "content": "base"},
    {"role": "user", "content": "question"},
    {"role": "system", "content": "update"},
    {"role": "user", "content": "follow-up"},
]

# No user turn follows the developer message: it joins the previous one.
BEFORE_ASSISTANT_MESSAGES = [
    {"role": "user", "content": "question"},
    {"role": "developer", "content": "update"},
    {"role": "assistant", "content": "answer"},
    {"role": "user", "content": "follow-up"},
]


def _render_messages(
    model_config, tokenizer, messages, chat_template=None, add_generation_prompt=True
):
    # Parse the messages for the template's content format, as `HfRenderer` does.
    chat_template = chat_template or tokenizer.chat_template
    content_format = resolve_chat_template_content_format(
        chat_template, None, "auto", tokenizer, model_config=model_config
    )
    conversation, _, _ = parse_chat_messages(messages, model_config, content_format)
    return safe_apply_chat_template(
        model_config,
        tokenizer,
        conversation,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


class TestSafeApplyChatTemplateDeveloperRole:
    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            "facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
        )

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("facebook/opt-125m")

    @pytest.fixture
    def qwen3_5_tokenizer(self):
        return get_tokenizer("Qwen/Qwen3.5-0.8B")

    def test_developer_converted_to_system_for_chatml(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=CHATML_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<|im_start|>system" in result
        assert "You are a helpful assistant." in result
        assert "<|im_start|>developer" not in result

    def test_developer_preserved_when_template_supports_it(
        self, model_config, tokenizer
    ):
        conversation: list[ConversationMessage] = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=TEMPLATE_WITH_DEVELOPER,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<|im_start|>developer" in result
        assert "You are a helpful assistant." in result

    def test_developer_does_not_crash_strict_template(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=STRICT_ROLE_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<|im_start|>system" in result
        assert "You are a helpful assistant." in result

    def test_no_developer_messages_no_overhead(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=CHATML_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<|im_start|>system" in result
        assert "You are helpful." in result

    def test_later_developer_joins_next_user_turn(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=SYSTEM_FIRST_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert result == (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\nHi there!<|im_end|>\n"
            "<|im_start|>user\nBe concise.\n\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def test_developer_without_system_joins_next_user_turn(
        self, model_config, tokenizer
    ):
        conversation: list[ConversationMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=SYSTEM_FIRST_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert result == (
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>user\nBe concise.\n\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # Expected prompts from the Rust frontend's
    # `developer_messages_follow_template_system_support` test. Qwen3 renders a
    # system message anywhere; Qwen3.5 only first, and Llama 2 also requires
    # alternating user and assistant turns.
    @pytest.mark.parametrize(
        ("model", "chat_template", "messages", "expected"),
        [
            (
                "Qwen/Qwen3-0.6B",
                None,
                CODEX_MESSAGES,
                (
                    "<|im_start|>system\ninstructions\n\nbase<|im_end|>\n"
                    "<|im_start|>user\nquestion<|im_end|>\n"
                    "<|im_start|>assistant\nanswer<|im_end|>\n"
                    "<|im_start|>system\nupdate<|im_end|>\n"
                    "<|im_start|>user\nfollow-up<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
            ),
            (
                "Qwen/Qwen3-0.6B",
                None,
                CALLER_SYSTEM_MESSAGES,
                (
                    "<|im_start|>system\nbase<|im_end|>\n"
                    "<|im_start|>user\nquestion<|im_end|>\n"
                    "<|im_start|>system\nupdate<|im_end|>\n"
                    "<|im_start|>user\nfollow-up<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
            ),
            (
                "Qwen/Qwen3.5-0.8B",
                None,
                CODEX_MESSAGES,
                (
                    "<|im_start|>system\ninstructions\n\nbase<|im_end|>\n"
                    "<|im_start|>user\nquestion<|im_end|>\n"
                    "<|im_start|>assistant\nanswer<|im_end|>\n"
                    "<|im_start|>user\nupdate\n\nfollow-up<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                ),
            ),
            (
                "facebook/opt-125m",
                ALTERNATING_TURNS_TEMPLATE,
                CODEX_MESSAGES,
                (
                    "system=instructions\n\nbase;user=question;assistant=answer;"
                    "user=update\n\nfollow-up;"
                ),
            ),
            (
                "facebook/opt-125m",
                ALTERNATING_TURNS_TEMPLATE,
                BEFORE_ASSISTANT_MESSAGES,
                "user=question\n\nupdate;assistant=answer;user=follow-up;",
            ),
        ],
        ids=[
            "qwen3-codex",
            "qwen3-caller-system",
            "qwen3.5-codex",
            "alternating-turns-codex",
            "alternating-turns-before-assistant",
        ],
    )
    def test_developer_messages_follow_template_system_support(
        self, model_config, model, chat_template, messages, expected
    ):
        tokenizer = get_tokenizer(model)
        result = _render_messages(model_config, tokenizer, messages, chat_template)
        assert result == expected

    def test_caller_system_message_left_to_template(
        self, model_config, qwen3_5_tokenizer
    ):
        with pytest.raises(
            VLLMValidationError, match="System message must be at the beginning."
        ):
            _render_messages(model_config, qwen3_5_tokenizer, CALLER_SYSTEM_MESSAGES)

    @pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"])
    def test_developer_update_keeps_prompt_prefix(self, model_config, model):
        tokenizer = get_tokenizer(model)
        turn = _render_messages(
            model_config, tokenizer, CODEX_MESSAGES, add_generation_prompt=False
        )
        next_turn = _render_messages(
            model_config,
            tokenizer,
            [
                *CODEX_MESSAGES,
                {"role": "assistant", "content": "answer 2"},
                {"role": "developer", "content": "update 2"},
                {"role": "user", "content": "follow-up 2"},
            ],
        )
        assert next_turn.startswith(turn)

    def test_developer_media_stays_in_place(self, model_config, qwen3_5_tokenizer):
        # A developer message after the leading run joins the user turn next
        # to it, so its image placeholder stays in media order. Content parts
        # are as `parse_chat_messages` makes them for an `openai` template.
        conversation: list[ConversationMessage] = [
            {"role": "developer", "content": [{"type": "text", "text": "base"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "question"}, {"type": "image"}],
            },
            {
                "role": "developer",
                "content": [{"type": "text", "text": "update"}, {"type": "image"}],
            },
        ]
        result = safe_apply_chat_template(
            model_config,
            qwen3_5_tokenizer,
            conversation,
            chat_template=qwen3_5_tokenizer.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert result == (
            "<|im_start|>system\nbase<|im_end|>\n"
            "<|im_start|>user\nquestion<|vision_start|><|image_pad|><|vision_end|>"
            "update<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    @pytest.mark.parametrize(
        ("template_kwargs", "expected"),
        [
            ({}, "<system>base<user>question<assistant>answer<user>update\n\nnext"),
            (
                {"allow_later_system": True},
                "<system>base<user>question<assistant>answer<system>update<user>next",
            ),
        ],
    )
    def test_probe_renders_with_template_kwargs(
        self, model_config, tokenizer, template_kwargs, expected
    ):
        # The template renders a later system message only when a template
        # variable allows it, so the probe must see the request's variables.
        chat_template = (
            "{%- for m in messages %}"
            "{%- if m.role == 'system' and not loop.first and not allow_later_system %}"
            "{{ raise_exception('System message must be at the beginning.') }}"
            "{%- endif %}<{{ m.role }}>{{ m.content }}{%- endfor %}"
        )
        conversation: list[ConversationMessage] = [
            {"role": "developer", "content": "base"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "developer", "content": "update"},
            {"role": "user", "content": "next"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        assert result == expected


SYSTEM_FIRST_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{% if not loop.first %}"
    "{{ raise_exception('System message must be at the beginning.') }}"
    "{% endif %}"
    "{{'<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{{'<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% else %}"
    "{{ raise_exception('Unexpected message role: ' + message['role']) }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


EFFORT_VALIDATING_TEMPLATE = (
    "{% if reasoning_effort is defined and "
    "reasoning_effort not in ['xhigh', 'medium', 'low'] %}"
    "{{ raise_exception('Unexpected reasoning effort ' + reasoning_effort + "
    "'. Supported types are xhigh (default), medium, and low.') }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
)

EFFORT_BREAKING_TEMPLATE = (
    "{% if reasoning_effort is defined %}"
    "{{ raise_exception('Template broke for an unrelated reason') }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
)


class TestApplyChatTemplateEffortTolerant:
    """Chat templates that reject unsupported reasoning_effort values should
    surface a 400-style client error instead of crashing the request."""

    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            "facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
        )

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("facebook/opt-125m")

    def test_unsupported_effort_raises_bad_request(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            VLLMValidationError,
            match="Unexpected reasoning effort high",
        ) as excinfo:
            safe_apply_chat_template(
                model_config,
                tokenizer,
                conversation,
                chat_template=EFFORT_VALIDATING_TEMPLATE,
                tokenize=False,
                reasoning_effort="high",
            )
        # The client sees exactly the template's own reason, with no wrapper
        # noise. It tells them which values are supported.
        assert str(excinfo.value) == (
            "Unexpected reasoning effort high. Supported types are xhigh "
            "(default), medium, and low."
        )

    def test_supported_effort_accepted(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [{"role": "user", "content": "Hello"}]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            chat_template=EFFORT_VALIDATING_TEMPLATE,
            tokenize=False,
            reasoning_effort="medium",
        )
        assert result == "user: Hello\n"

    def test_non_effort_template_error_is_bad_request(self, model_config, tokenizer):
        conversation: list[ConversationMessage] = [{"role": "user", "content": "Hello"}]
        with pytest.raises(VLLMValidationError, match="unrelated reason"):
            safe_apply_chat_template(
                model_config,
                tokenizer,
                conversation,
                chat_template=EFFORT_BREAKING_TEMPLATE,
                tokenize=False,
                reasoning_effort="high",
            )


def test_template_error_reason_prefers_template_error():
    # Upstream wrappers must not hide the template's own reason from clients.
    reason = jinja2.TemplateError("Unexpected reasoning effort high")
    wrapper = ValueError(f"An error occurred while rendering the template: {reason}")
    wrapper.__cause__ = reason
    assert _template_error_reason(wrapper) == str(reason)
    assert _template_error_reason(reason) == str(reason)


def test_template_error_reason_falls_back_to_message():
    err = ValueError("plain failure")
    assert _template_error_reason(err) == "plain failure"


GRANITE_20B_FC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


class TestGranite20bFcSystemMessage:
    """`examples/tool_chat_template_granite_20b_fc.jinja` regression tests.

    The template lifts the system message out of the conversation and renders
    it as a `SYSTEM:` preamble, so its message loop has no branch for the
    system role. Iterating the raw `messages` reached that missing branch and
    raised, which `safe_apply_chat_template` surfaces as a 400, rejecting the
    ordinary OpenAI request shape.
    """

    @pytest.fixture
    def chat_template(self):
        template = load_chat_template(
            EXAMPLES_DIR / "tool_chat_template_granite_20b_fc.jinja"
        )
        assert isinstance(template, str)
        return template

    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            "facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
        )

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer("facebook/opt-125m")

    def test_system_message_renders_once_as_the_preamble(
        self, model_config, tokenizer, chat_template
    ):
        conversation: list[ConversationMessage] = [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "What is the weather in Paris?"},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            tools=GRANITE_20B_FC_TOOLS,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "SYSTEM: You are a terse assistant." in result
        # Once as the preamble, and never again from the message loop.
        assert result.count("You are a terse assistant.") == 1
        assert "USER: What is the weather in Paris?" in result
        assert result.rstrip().endswith("ASSISTANT:")

    def test_generation_prompt_survives_an_empty_message_loop(
        self, model_config, tokenizer, chat_template
    ):
        """The assistant cue is emitted outside the loop, so it still renders
        when the system message was the only one."""
        conversation: list[ConversationMessage] = [
            {"role": "system", "content": "You are a terse assistant."},
        ]
        result = safe_apply_chat_template(
            model_config,
            tokenizer,
            conversation,
            tools=GRANITE_20B_FC_TOOLS,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert result.rstrip().endswith("ASSISTANT:")
