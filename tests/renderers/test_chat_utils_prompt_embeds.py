# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline unit tests for `prompt_embeds` chat-completion content parts."""

from __future__ import annotations

import inspect
import io
from typing import Final
from unittest import mock

import pybase64 as base64
import pytest
import regex as re
import torch
from transformers import AutoTokenizer

from vllm.entrypoints.chat_utils import (
    _ENABLE_PROMPT_EMBEDS_ERROR,
    _PROMPT_EMBEDS_MISSING_DATA_ERROR,
    _RESERVED_PLACEHOLDER_IN_TEXT_ERROR,
    MM_PARSER_MAP,
    MODALITY_PLACEHOLDERS_MAP,
    PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.hf import (
    _PROMPT_EMBEDS_PLACEHOLDER_SPAN_MISMATCH_ERROR,
    _build_mixed_prompt_embeds,
    _build_prompt_embeds_positions,
    _build_prompt_embeds_updates,
    _ensure_prompt_embeds_placeholder_token,
    _expand_prompt_embeds_placeholders,
)

# Cover distinct tokenizer families:
#   GPT2TokenizerFast  (BPE, OpenAI-style)
#   Qwen2TokenizerFast (SentencePiece BPE variant)
#   BertTokenizerFast  (WordPiece)
TOKENIZER_IDS: Final[list[str]] = [
    "gpt2",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "bert-base-uncased",
]


@pytest.fixture(params=TOKENIZER_IDS, ids=TOKENIZER_IDS)
def tokenizer(request):
    """A fresh tokenizer instance per tokenizer family."""
    return AutoTokenizer.from_pretrained(request.param)


# Minimal chat template that works with any tokenizer.  Iterates
# `message.content` as either a string or a list of dicts (openai format).
_SIMPLE_CHAT_TEMPLATE: Final[str] = (
    "{% for m in messages %}"
    "{% if m['content'] is string %}{{m['content']}}"
    "{% else %}{% for p in m['content'] %}{{p['text']}}{% endfor %}"
    "{% endif %}\n{% endfor %}"
)


async def _maybe_await(fn, *args, **kwargs):
    """Call *fn* and `await` the result if it's a coroutine."""
    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result):
        result = await result
    return result


# Parametrize over sync / async parse paths so every end-to-end test
# exercises both.
_PARSE_FUNCTIONS = [parse_chat_messages, parse_chat_messages_async]


@pytest.fixture(params=_PARSE_FUNCTIONS, ids=["sync", "async"])
def parse_fn(request):
    """Either the sync or async `parse_chat_messages` callable."""
    return request.param


def _encode_tensor(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_MOCK_HIDDEN_SIZE: Final[int] = 8
_MOCK_DTYPE: Final[torch.dtype] = torch.float32


def _make_mock_model_config(*, enable_prompt_embeds: bool = True) -> mock.MagicMock:
    mc = mock.MagicMock()
    mc.enable_prompt_embeds = enable_prompt_embeds
    mc.multimodal_config = None
    mc.allowed_local_media_path = None
    mc.allowed_media_domains = None
    # Test text-only code path in `MultiModalItemTracker.resolve_items`.
    mc.is_multimodal_model = False
    # `safe_load_prompt_embeds` pins each tensor to the model's hidden_size
    # and dtype, so the mock must return concrete values.
    mc.get_hidden_size.return_value = _MOCK_HIDDEN_SIZE
    mc.dtype = _MOCK_DTYPE
    return mc


def test_prompt_embeds_keys_registered():
    assert "prompt_embeds" in MODALITY_PLACEHOLDERS_MAP
    assert MODALITY_PLACEHOLDERS_MAP["prompt_embeds"] == "<##PROMPT_EMBEDS##>"
    assert "prompt_embeds" in MM_PARSER_MAP


def test_ensure_placeholder_token_is_single_token_and_idempotent(tokenizer):
    """Ensure the placeholder token is a single token and that multiple calls to
    "ensure" are idempotent, across all tokenizer families."""
    tid1 = _ensure_prompt_embeds_placeholder_token(tokenizer)
    tid2 = _ensure_prompt_embeds_placeholder_token(tokenizer)
    assert tid1 == tid2

    ids = tokenizer.encode(PROMPT_EMBEDS_PLACEHOLDER_TOKEN, add_special_tokens=False)
    assert ids == [tid1]

    # Repeating it in a string N times must produce exactly that many tokens.
    N = 5
    ids_rep = tokenizer.encode(
        PROMPT_EMBEDS_PLACEHOLDER_TOKEN * N, add_special_tokens=False
    )
    assert ids_rep == [tid1] * N


def test_parse_chat_messages_openai_format():
    NUM_TOKENS = 3
    t = torch.randn(NUM_TOKENS, _MOCK_HIDDEN_SIZE, dtype=_MOCK_DTYPE)
    b64 = _encode_tensor(t)
    mc = _make_mock_model_config()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "prompt_embeds", "data": b64},
                {"type": "text", "text": " world"},
            ],
        }
    ]
    conv, mm_data, _ = parse_chat_messages(
        messages,
        mc,
        content_format="openai",
    )
    # The middle content part is rewritten to a single placeholder-token
    # sentinel.
    texts = [p["text"] for p in conv[0]["content"]]
    assert texts == [
        "Hello ",
        PROMPT_EMBEDS_PLACEHOLDER_TOKEN,
        " world",
    ]
    assert mm_data is not None and "prompt_embeds" in mm_data
    assert torch.equal(mm_data["prompt_embeds"][0], t)


# Each layout entry is one content part:
#   ("text", "A")  -> {"type": "text", "text": "A"}
#   ("embed", N)   -> {"type": "prompt_embeds", "data": <base64 of (N, H) tensor>}
@pytest.mark.parametrize(
    "layout",
    [
        # Case: Single embed only.
        [("embed", 2)],
        # Case: Embed at the start of the message.
        [("embed", 3), ("text", "B")],
        # Case: Embed at the end of the message.
        [("text", "A"), ("embed", 1)],
        # Case: Embed sandwiched between text spans.
        [("text", "A"), ("embed", 2), ("text", "B")],
        # Case: Multiple embeds with text in between.
        [("text", "A"), ("embed", 2), ("text", "B"), ("embed", 3)],
        # Case: Adjacent embeds with no separating text.
        [("embed", 1), ("embed", 2)],
        # Case: Multiple text spans before a trailing embed.
        [("text", "A"), ("text", "B"), ("embed", 1)],
        # Case: Long-ish run mixing both kinds.
        [
            ("text", "head"),
            ("embed", 4),
            ("text", "mid"),
            ("embed", 1),
            ("embed", 2),
            ("text", "tail"),
        ],
    ],
    ids=[
        "single-embed",
        "embed-then-text",
        "text-then-embed",
        "text-embed-text",
        "text-embed-text-embed",
        "adjacent-embeds",
        "text-text-embed",
        "long-mixed-run",
    ],
)
@pytest.mark.parametrize(
    "interleave_mm_strings",
    # `None`: text-only path where `multimodal_config` is absent.
    # `False`: non-interleave multimodal path (the common default).
    # `True`: sentinel-substitution interleave path.
    # All three must preserve the request ordering of prompt_embeds
    # relative to surrounding text because prompt_embeds are spliced at the
    # token offset during rendering.
    [None, False, True],
    ids=["text-only", "interleave-off", "interleave-on"],
)
def test_parse_chat_messages_string_format_preserves_position(
    layout, interleave_mm_strings
):
    mc = _make_mock_model_config()
    if interleave_mm_strings is not None:
        mm_cfg = mock.MagicMock()
        mm_cfg.interleave_mm_strings = interleave_mm_strings
        mc.multimodal_config = mm_cfg

    content: list[dict] = []
    expected_parts: list[str] = []
    expected_embeds: list[torch.Tensor] = []
    for kind, value in layout:
        if kind == "text":
            content.append({"type": "text", "text": value})
            expected_parts.append(value)
        else:  # prompt embeds
            num_tokens = value
            t = torch.randn(num_tokens, _MOCK_HIDDEN_SIZE, dtype=_MOCK_DTYPE)
            expected_embeds.append(t)
            content.append({"type": "prompt_embeds", "data": _encode_tensor(t)})
            # Parser emits ONE sentinel per part.
            expected_parts.append(PROMPT_EMBEDS_PLACEHOLDER_TOKEN)

    messages = [{"role": "user", "content": content}]
    conv, mm_data, _ = parse_chat_messages(
        messages,
        mc,
        content_format="string",
    )

    assert conv[0]["content"] == "\n".join(expected_parts)
    assert mm_data is not None and "prompt_embeds" in mm_data
    assert len(mm_data["prompt_embeds"]) == len(expected_embeds)
    for got, want in zip(mm_data["prompt_embeds"], expected_embeds, strict=True):
        assert torch.equal(got, want)


def test_parse_chat_messages_requires_flag():
    t = torch.randn(2, 4)
    b64 = _encode_tensor(t)
    mc = _make_mock_model_config(enable_prompt_embeds=False)

    messages = [
        {
            "role": "user",
            "content": [{"type": "prompt_embeds", "data": b64}],
        }
    ]
    with pytest.raises(ValueError, match=_ENABLE_PROMPT_EMBEDS_ERROR):
        parse_chat_messages(
            messages,
            mc,
            content_format="openai",
        )


def test_parse_chat_messages_rejects_missing_data():
    # `data` is marked `Required` on `ChatCompletionContentPartPromptEmbedsParam`;
    # malformed requests without `data` must surface a clear validation error
    # rather than being silently dropped.
    mc = _make_mock_model_config()
    messages = [
        {
            "role": "user",
            "content": [{"type": "prompt_embeds"}],  # no `data`
        }
    ]
    with pytest.raises(ValueError, match=_PROMPT_EMBEDS_MISSING_DATA_ERROR):
        parse_chat_messages(
            messages,
            mc,
            content_format="openai",
        )


# Reserved placeholder guard: when `enable_prompt_embeds=True` the tokenizer is
# mutated to make `<prompt_embeds>` a single unsplittable token. Any user text
# containing that literal sequence would tokenize to the same sentinel ID and
# be mistaken for a splice point, so we reject it at parse time.
_PLACEHOLDER_ERROR_PATTERN: Final[str] = re.sub(
    r"\\{[^}]*\\}", ".*", re.escape(_RESERVED_PLACEHOLDER_IN_TEXT_ERROR)
)


@pytest.mark.parametrize(
    "content",
    [
        # Case: Top-level string content (wrapped as a single text part).
        f"hello {PROMPT_EMBEDS_PLACEHOLDER_TOKEN} world",
        # Case: List with a typed text part containing the placeholder.
        [{"type": "text", "text": f"leading {PROMPT_EMBEDS_PLACEHOLDER_TOKEN}"}],
        # Case: List with a plain-string part (no wrapping dict).
        [f"raw string {PROMPT_EMBEDS_PLACEHOLDER_TOKEN}"],
    ],
    ids=["top-level-string", "typed-text-part", "plain-string-part"],
)
def test_parse_chat_messages_rejects_placeholder_in_user_text(content):
    mc = _make_mock_model_config()  # enable_prompt_embeds=True by default
    messages = [{"role": "user", "content": content}]
    with pytest.raises(ValueError, match=_PLACEHOLDER_ERROR_PATTERN):
        parse_chat_messages(messages, mc, content_format="openai")


def test_parse_chat_messages_allows_placeholder_in_text_when_feature_disabled():
    # When `enable_prompt_embeds=False` the tokenizer is never mutated, so the
    # literal `<prompt_embeds>` is just ordinary text and must pass through.
    mc = _make_mock_model_config(enable_prompt_embeds=False)
    messages = [
        {
            "role": "user",
            "content": f"benign mention of {PROMPT_EMBEDS_PLACEHOLDER_TOKEN} here",
        }
    ]
    conv, mm_data, _ = parse_chat_messages(messages, mc, content_format="openai")
    assert mm_data is None or "prompt_embeds" not in mm_data
    # Text reaches the rendered conversation unchanged.
    texts = [p["text"] for p in conv[0]["content"]]
    assert PROMPT_EMBEDS_PLACEHOLDER_TOKEN in "".join(texts)


# Token-stream spec: ints are regular token IDs, tuples `(N,)` expand to
# a placeholder span of length N (creates corresponding `(N, H)` tensor).
# `expected` lists the `(start_idx, length)` pairs that
# `_build_prompt_embeds_positions` should return.
@pytest.mark.parametrize(
    "stream, expected",
    [
        # Case: Single run in the middle.
        ([10, 20, (3,), 30], [(2, 3)]),
        # Case: Single run at the start.
        ([(2,), 10, 20], [(0, 2)]),
        # Case: Single run at the end.
        ([10, 20, (4,)], [(2, 4)]),
        # Case: Two runs with tokens between.
        ([1, (2,), 2, 3, (3,), 4], [(1, 2), (5, 3)]),
        # Case: Adjacent runs (no separating tokens).
        ([(1,), (2,)], [(0, 1), (1, 2)]),
        # Case: Three runs.
        ([5, (2,), 6, (1,), 7, (3,), 8], [(1, 2), (4, 1), (6, 3)]),
    ],
    ids=[
        "single-middle",
        "single-start",
        "single-end",
        "two-runs-separated",
        "two-runs-adjacent",
        "three-runs",
    ],
)
def test_build_positions(tokenizer, stream, expected):
    H = 4
    tid = _ensure_prompt_embeds_placeholder_token(tokenizer)
    tensors: list[torch.Tensor] = []
    token_ids: list[int] = []
    for item in stream:
        if isinstance(item, tuple):
            length = item[0]
            tensors.append(torch.randn(length, H))
            token_ids.extend([tid] * length)
        else:
            token_ids.append(item)
    mm_updates = _build_prompt_embeds_updates(tensors, tid)
    positions = _build_prompt_embeds_positions(token_ids, len(tensors), mm_updates)
    assert positions == expected


def test_build_positions_length_mismatch(tokenizer):
    N1, H1 = 2, 4
    N2, H2 = 3, 4
    tid = _ensure_prompt_embeds_placeholder_token(tokenizer)
    # 2 tensors expected but only a single placeholder run in the token
    # stream (simulating dropping the second one).
    tensors = [torch.randn(N1, H1), torch.randn(N2, H2)]
    token_ids = [1, tid, tid, 2, 3]
    mm_updates = _build_prompt_embeds_updates(tensors, tid)
    # The error constant is a `str.format` template, escape it and turn
    # the `{field}` placeholders into `.*` so it matches any substitution.
    pattern = re.sub(
        r"\\{[^}]*\\}", ".*", re.escape(_PROMPT_EMBEDS_PLACEHOLDER_SPAN_MISMATCH_ERROR)
    )
    with pytest.raises(ValueError, match=pattern):
        _build_prompt_embeds_positions(token_ids, len(tensors), mm_updates)


# ints  = regular token IDs (any value)
# (N,)  = embed span of length N
@pytest.mark.parametrize(
    "stream",
    [
        [10, 20, (3,), 30],
        [(2,), 10, 20],
        [10, 20, (4,)],
        [1, (2,), 2, 3, (3,), 4],
        [(1,), (2,)],
        [5, (2,), 6, (1,), 7, (3,), 8],
    ],
    ids=[
        "single-middle",
        "single-start",
        "single-end",
        "two-spans-separated",
        "two-spans-adjacent",
        "three-spans",
    ],
)
def test_build_mixed_prompt_embeds(stream):
    H = 8
    _PLACEHOLDER = 0  # sentinel for embed positions in token_ids

    tensors: list[torch.Tensor] = []
    token_ids: list[int] = []
    positions: list[tuple[int, int]] = []
    cursor = 0
    for item in stream:
        if isinstance(item, tuple):
            length = item[0]
            tensors.append(torch.randn(length, H))
            positions.append((cursor, length))
            token_ids.extend([_PLACEHOLDER] * length)
            cursor += length
        else:
            token_ids.append(item)
            cursor += 1

    embeds, mask = _build_mixed_prompt_embeds(token_ids, tensors, positions)

    assert embeds.shape == (len(token_ids), H)
    assert len(mask) == len(token_ids)

    # Mask: False exactly at embed positions, True everywhere else.
    expected_mask = torch.ones(len(token_ids), dtype=torch.bool)
    for start, length in positions:
        expected_mask[start : start + length] = False
    assert mask == expected_mask.tolist()

    # Embed rows match input tensors at the right positions.
    for tensor, (start, length) in zip(tensors, positions):
        assert torch.equal(embeds[start : start + length], tensor)

    # Non-embed positions remain zero-filled.
    assert torch.all(embeds[expected_mask] == 0)


# End-to-end tests: each runs both sync and async parse paths via the
# `parse_fn` fixture.


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "system"])
async def test_end_to_end_expand_and_build(tokenizer, parse_fn, role):
    """Full renderer pipeline: parse -> chat template -> expand -> locate
    -> build mixed prompt, across tokenizers, roles, and sync/async."""
    tokenizer.chat_template = _SIMPLE_CHAT_TEMPLATE
    tid = _ensure_prompt_embeds_placeholder_token(tokenizer)

    LEN_A, LEN_B = 3, 2
    t_a = torch.randn(LEN_A, _MOCK_HIDDEN_SIZE, dtype=_MOCK_DTYPE)
    t_b = torch.randn(LEN_B, _MOCK_HIDDEN_SIZE, dtype=_MOCK_DTYPE)
    NUM_TENSORS = 2

    mc = _make_mock_model_config()

    messages = [
        {
            "role": role,
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "prompt_embeds", "data": _encode_tensor(t_a)},
                {"type": "text", "text": " world "},
                {"type": "prompt_embeds", "data": _encode_tensor(t_b)},
                {"type": "text", "text": "!"},
            ],
        }
    ]

    conv, mm_data, _ = await _maybe_await(
        parse_fn, messages, mc, content_format="openai"
    )
    tensors = list(mm_data["prompt_embeds"])
    assert len(tensors) == NUM_TENSORS

    # Tokenize: each prompt_embeds part becomes 1 placeholder token.
    # `return_dict=False` to get a flat `list[int]` on transformers v5
    # (where the default flipped to True and yields a `BatchEncoding` dict).
    token_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
    assert sum(t == tid for t in token_ids) == NUM_TENSORS

    # Expand, locate, and build.
    mm_updates = _build_prompt_embeds_updates(tensors, tid)
    expanded = _expand_prompt_embeds_placeholders(token_ids, mm_updates)
    assert len(expanded) == len(token_ids) + LEN_A + LEN_B - NUM_TENSORS

    positions = _build_prompt_embeds_positions(expanded, len(tensors), mm_updates)
    assert positions[0][1] == LEN_A
    assert positions[1][1] == LEN_B

    embeds, mask = _build_mixed_prompt_embeds(expanded, tensors, positions)
    assert embeds.shape == (len(expanded), _MOCK_HIDDEN_SIZE)
    assert mask.count(False) == LEN_A + LEN_B
    assert torch.equal(embeds[positions[0][0] : positions[0][0] + LEN_A], t_a)
    assert torch.equal(embeds[positions[1][0] : positions[1][0] + LEN_B], t_b)


@pytest.mark.asyncio
async def test_end_to_end_multi_message_conversation(tokenizer, parse_fn):
    """Full pipeline with prompt_embeds spread across system + user messages,
    verifying ordering and positioning in the final token stream."""
    tokenizer.chat_template = _SIMPLE_CHAT_TEMPLATE
    tid = _ensure_prompt_embeds_placeholder_token(tokenizer)

    LEN_SYS, LEN_USR = 4, 3
    t_sys = torch.randn(LEN_SYS, _MOCK_HIDDEN_SIZE, dtype=_MOCK_DTYPE)
    t_usr = torch.randn(LEN_USR, _MOCK_HIDDEN_SIZE, dtype=_MOCK_DTYPE)
    NUM_TENSORS = 2  # t_sys and t_usr.

    mc = _make_mock_model_config()

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are helpful."},
                {"type": "prompt_embeds", "data": _encode_tensor(t_sys)},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "prompt_embeds", "data": _encode_tensor(t_usr)},
                {"type": "text", "text": "Summarize."},
            ],
        },
    ]

    conv, mm_data, _ = await _maybe_await(
        parse_fn, messages, mc, content_format="openai"
    )
    tensors = list(mm_data["prompt_embeds"])
    assert len(tensors) == NUM_TENSORS

    # Tokenize, expand, locate, and build.
    # `return_dict=False` to get a flat `list[int]` on transformers v5
    # (where the default flipped to True and yields a `BatchEncoding` dict).
    token_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
    mm_updates = _build_prompt_embeds_updates(tensors, tid)
    expanded = _expand_prompt_embeds_placeholders(token_ids, mm_updates)
    positions = _build_prompt_embeds_positions(expanded, len(tensors), mm_updates)

    assert positions[0][1] == LEN_SYS
    assert positions[1][1] == LEN_USR
    # System embed must appear before user embed in the token stream.
    assert positions[0][0] < positions[1][0]

    embeds, mask = _build_mixed_prompt_embeds(expanded, tensors, positions)
    assert embeds.shape == (len(expanded), _MOCK_HIDDEN_SIZE)
    assert mask.count(False) == LEN_SYS + LEN_USR
    assert torch.equal(embeds[positions[0][0] : positions[0][0] + LEN_SYS], t_sys)
    assert torch.equal(embeds[positions[1][0] : positions[1][0] + LEN_USR], t_usr)
