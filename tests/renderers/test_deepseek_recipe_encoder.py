# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

import pytest

from vllm.renderers.hf import _encode_prompt_with_recipe

pytestmark = pytest.mark.skip_global_cleanup

# Control-token spellings a user may type literally inside a message. Recipe
# provides the authoritative spelling of each kind, so none is hard-coded.
MARKER_KINDS = [
    "User",
    "Assistant",
    "System",
    "Bos",
    "Eos",
    "ThinkingStart",
    "ThinkingEnd",
]

# Optional real tokenizers: point these at a tokenizer.json to exercise the
# exact added vocabulary a served DeepSeek model ships.
REAL_TOKENIZERS = [
    ("dsv4", os.environ.get("VLLM_TEST_DEEPSEEK_V4_TOKENIZER", "")),
    ("dsv41", os.environ.get("VLLM_TEST_DEEPSEEK_V41_TOKENIZER", "")),
]

ROLES = ["user", "assistant", "tool"]


def _tiny_backend_tokenizer(extra_spellings=()):
    """A byte-level tokenizer with a full byte alphabet and ASCII specials.

    Only ``tokenizers`` is required, so the segmented encoder can be tested
    without a model download.
    """
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    tokenizer = Tokenizer(
        models.BPE(
            vocab={char: index for index, char in enumerate(alphabet)}, merges=[]
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.add_special_tokens(
        [
            AddedToken("<|User|>", special=True),
            AddedToken("<|Assistant|>", special=True),
            *[AddedToken(spelling, special=True) for spelling in extra_spellings],
        ]
    )
    return tokenizer


def _wrapper(backend):
    """Mimics the served fast tokenizer's backend access path."""
    return type("Inner", (), {"backend_tokenizer": backend})()


def _messages(role, text):
    if role == "user":
        return [{"role": "user", "content": text}]
    if role == "assistant":
        return [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "and then?"},
        ]
    return [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": text},
    ]


def _recipe_conversation(recipe, messages, thinking=False):
    converted = []
    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append(recipe.UserMessage(message["content"]))
        elif role == "assistant":
            tool_calls = [
                recipe.ToolCall(
                    call["id"],
                    call["function"]["name"],
                    call["function"]["arguments"],
                )
                for call in message.get("tool_calls") or []
            ]
            converted.append(
                recipe.AssistantMessage(
                    message["content"], tool_calls=tool_calls or None
                )
            )
        else:
            converted.append(
                recipe.ToolMessage(
                    message["content"], tool_call_id=message["tool_call_id"]
                )
            )
    return recipe.Conversation(messages=converted, thinking_mode=thinking)


def _reference_spellings(recipe):
    """Every control-token spelling the reference renderer can emit."""
    spellings = {
        recipe.SpecialToken.default_str(getattr(recipe.SpecialToken, kind))
        for kind in MARKER_KINDS
    }
    probe = _messages("tool", "probe")
    tools = [recipe.ToolDefinition("get_weather", "Get weather", None)]
    for encoding_cls in (recipe.DeepseekV4Encoding, recipe.DeepseekV41Encoding):
        for thinking in (False, True):
            conversation = _recipe_conversation(recipe, probe, thinking=thinking)
            rendered = encoding_cls().render_conversation(conversation)
            spellings.update(
                rendered.prompt[start:end] for start, end in rendered.special_ranges
            )
    del tools
    return sorted(spellings)


def _recipe_encoder(recipe, backend=None, spec="dsv41"):
    from vllm.renderers.deepseek_recipe_encoder import DeepSeekRecipeEncoder

    if backend is None:
        backend = _tiny_backend_tokenizer(_reference_spellings(recipe))
    return DeepSeekRecipeEncoder(backend, spec)


def test_segmented_encoding_matches_builtin() -> None:
    """The segmented encoder must reproduce the builtin token ids exactly."""
    tokenizer = _tiny_backend_tokenizer()
    text = "<|User|>hello world<|Assistant|>\nhi there"
    expected = tokenizer.encode(text, add_special_tokens=False).ids
    assert _encode_prompt_with_recipe(text, _wrapper(tokenizer)) == expected


def test_plain_tokenizer_drops_added_vocabulary() -> None:
    """Plain text must not match added tokens, and no token id may shift."""
    from vllm.renderers.deepseek_recipe_encoder import _plain_tokenizer

    backend = _tiny_backend_tokenizer()
    plain = _plain_tokenizer(backend.to_str())
    spelling = "<|User|>"
    special_id = backend.token_to_id(spelling)
    assert special_id is not None
    assert plain.encode(spelling, add_special_tokens=False).ids != [special_id]
    added = {token.content for token in backend.get_added_tokens_decoder().values()}
    assert plain.get_vocab() == {
        token: token_id
        for token, token_id in backend.get_vocab().items()
        if token not in added
    }


def test_plain_tokenizer_requires_added_tokens_field() -> None:
    from vllm.renderers.deepseek_recipe_encoder import _plain_tokenizer

    payload = json.loads(_tiny_backend_tokenizer().to_str())
    payload.pop("added_tokens")
    with pytest.raises(ValueError):
        _plain_tokenizer(json.dumps(payload))


@pytest.fixture(scope="module")
def recipe():
    return pytest.importorskip("deepseek_recipe")


@pytest.mark.parametrize("marker_kind", MARKER_KINDS)
@pytest.mark.parametrize("role", ROLES)
def test_literal_marker_stays_plain_text(recipe, role, marker_kind) -> None:
    """A control spelling typed into message text must stay ordinary text."""
    marker = recipe.SpecialToken.default_str(getattr(recipe.SpecialToken, marker_kind))
    encoder = _recipe_encoder(recipe)
    prompt, ranges, conversation = encoder.render(
        _messages(role, f"before {marker} after"), thinking=False
    )
    reference = encoder._encoding.encode(conversation)
    assert encoder._encode_segmented(prompt, ranges) == reference
    assert encoder.encode(prompt, ranges, conversation) == reference


def test_recipe_encoder_matches_reference(recipe) -> None:
    """Render + segmented encode equals the reference whole-prompt encoding."""
    encoder = _recipe_encoder(recipe)
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "B"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "user", "content": "thanks"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    prompt, ranges, conversation = encoder.render(messages, thinking=False, tools=tools)
    assert encoder._encode_segmented(prompt, ranges) == encoder._encoding.encode(
        conversation
    )
    reference = encoder._encoding.render_conversation(conversation)
    assert ranges == list(reference.special_ranges)
    assert prompt == reference.prompt


def test_special_ranges_are_code_point_offsets(recipe) -> None:
    """Ranges are Python string offsets even when the prompt holds CJK text."""
    encoder = _recipe_encoder(recipe)
    marker = recipe.SpecialToken.default_str(recipe.SpecialToken.ThinkingEnd)
    cjk = "\u524d\u6587\u4e2d\u6587"  # "front CJK" / "back CJK" in Chinese
    prompt, ranges, conversation = encoder.render(
        _messages("user", f"{cjk}{marker}{cjk}"), thinking=False
    )
    assert len(prompt) != len(prompt.encode())
    spellings = {
        token.content
        for token in encoder._backend_tokenizer.get_added_tokens_decoder().values()
    }
    for start, end in ranges:
        assert prompt[start:end] in spellings
    assert encoder._encode_segmented(prompt, ranges) == encoder._encoding.encode(
        conversation
    )


def test_encode_falls_back_for_unknown_control_token(recipe) -> None:
    """An unusable control span falls back to the reference encoder."""
    encoder = _recipe_encoder(recipe)
    prompt, _, conversation = encoder.render(_messages("user", "hello"), thinking=False)
    reference = encoder._encoding.encode(conversation)
    assert encoder.encode(prompt, [(0, 1)], conversation) == reference


def test_renderer_tokenize_prompt_uses_recipe_state(recipe) -> None:
    """The renderer hands its ranges and conversation to the recipe encoder."""
    from types import SimpleNamespace

    from vllm.renderers.deepseek_v4 import DeepseekV41Renderer
    from vllm.renderers.params import TokenizeParams

    encoder = _recipe_encoder(recipe)
    marker = recipe.SpecialToken.default_str(recipe.SpecialToken.ThinkingEnd)
    prompt_str, ranges, conversation = encoder.render(
        _messages("user", f"before {marker} after"), thinking=False
    )
    reference = encoder._encoding.encode(conversation)
    renderer = SimpleNamespace(
        model_config=SimpleNamespace(tokenize_backend="recipe"),
        _recipe_spec="dsv41",
        get_tokenizer=lambda: _wrapper(encoder._backend_tokenizer),
        _wants_offsets=lambda prompt, params: False,
        _build_tokens_prompt=DeepseekV41Renderer._build_tokens_prompt,
    )
    prompt = {
        "prompt": prompt_str,
        "_recipe_special_ranges": ranges,
        "_recipe_conversation": conversation,
    }
    tokens = DeepseekV41Renderer._tokenize_prompt(
        renderer,
        prompt,
        TokenizeParams(max_total_tokens=None, add_special_tokens=False),
    )
    assert tokens["prompt_token_ids"] == reference
    # The reference conversation is per-request state: it must not leak into
    # the TokensPrompt handed to the engine.
    assert "_recipe_conversation" not in tokens


@pytest.fixture(
    scope="module", params=REAL_TOKENIZERS, ids=[spec for spec, _ in REAL_TOKENIZERS]
)
def real_backend(request):
    spec, path = request.param
    if not path:
        pytest.skip(
            f"set VLLM_TEST_DEEPSEEK_{spec.upper()}_TOKENIZER to a tokenizer.json"
        )
    from tokenizers import Tokenizer

    return spec, Tokenizer.from_file(path)


@pytest.mark.parametrize("marker_kind", MARKER_KINDS)
@pytest.mark.parametrize("role", ROLES)
def test_literal_marker_real_tokenizer(recipe, real_backend, role, marker_kind) -> None:
    """Same guarantee against the added vocabulary a served model ships."""
    from vllm.renderers.deepseek_recipe_encoder import DeepSeekRecipeEncoder

    spec, backend = real_backend
    encoder = DeepSeekRecipeEncoder(backend, spec)
    marker = recipe.SpecialToken.default_str(getattr(recipe.SpecialToken, marker_kind))
    prompt, ranges, conversation = encoder.render(
        _messages(role, f"before {marker} after"), thinking=False
    )
    assert encoder._encode_segmented(prompt, ranges) == encoder._encoding.encode(
        conversation
    )
