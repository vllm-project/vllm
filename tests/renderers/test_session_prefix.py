# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.inputs.engine import TokensInput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import TokenizeParams
from vllm.renderers.session_prefix import (
    SessionTokenCache,
    TokenSnapshot,
    _marker_tables,
)
from vllm.tokenizers import get_tokenizer
from vllm.utils.session_prefix import SessionPrefixStore, lcp

MODEL = os.environ.get("VLLM_TEST_QWEN3_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture
def renderer(request, caplog_vllm):
    model, revision = (
        request.param
        if isinstance(getattr(request, "param", None), tuple)
        else (MODEL, "c1899de289a04d12100db370d81485cdf75e47ca")
    )
    truncation_side = (
        request.param if isinstance(getattr(request, "param", None), str) else None
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=model,
            tokenizer_revision=revision,
            dtype="float16",
            max_model_len=1024,
            renderer_num_workers=2,
            session_token_cache_mb=8,
        )
    )
    result = HfRenderer(
        config,
        get_tokenizer(
            model,
            revision=config.model_config.tokenizer_revision,
            **({"truncation_side": truncation_side} if truncation_side else {}),
        ),
    )
    yield result
    result.shutdown()


def test_marker_tables_keep_every_safe_boundary():
    """Unsafe markers are scanned but do not disable the other boundaries."""
    specs = [
        # content, normalized, single_word, rstrip
        ("<end_of_turn>", False, False, False),
        ("[INST]", False, False, False),  # no leading "<" is needed
        ("\n", False, False, False),  # contained in "\n\n"
        ("\n\n", False, False, False),
        ("ab", False, False, False),  # "b" ends one and starts the other
        ("bc", False, False, False),
        ("<w>", False, True, False),  # needs word context
        ("<r>", False, False, True),  # right-stripping
        ("<n>", True, False, False),  # matched after normalization
    ]
    markers, boundaries, rstrip = _marker_tables(
        [
            {"id": i, "content": c, "normalized": n, "single_word": w, "rstrip": r}
            for i, (c, n, w, r) in enumerate(specs)
        ]
    )
    assert set(markers) == {c for c, n, *_ in specs if not n}
    assert boundaries == {"<end_of_turn>", "[INST]", "<r>"}
    assert rstrip == {"<r>"}


def test_right_stripping_boundary_requires_non_space_suffix(renderer):
    cache = renderer.session_token_cache
    if not cache.enabled:
        assert renderer.session_prefix_context("rstrip", None) is None
        return  # The pinned-version CI exercises lookup.
    params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
    prefix = "<|im_start|>user\nhi<|im_end|>"
    old = prefix + "\nX"
    snapshot = TokenSnapshot(
        old, renderer.tokenize_prompt({"prompt": old}, params)["prompt_token_ids"]
    )
    for suffix, rstrip, boundary in (
        (" Y", set(), len(prefix)),
        ("Y", {"<|im_end|>"}, len(prefix)),
        (" Y", {"<|im_end|>"}, len("<|im_start|>")),
    ):
        cache.rstrip_markers = rstrip
        new = prefix + suffix
        found = cache._boundary(snapshot, new, lcp(new, old))
        assert found is not None and found[0] == boundary


@pytest.mark.parametrize("edit_at", ["head", "tail"])
def test_snapshot_markers_are_checked_from_the_nearer_end(renderer, edit_at):
    """A token-side marker mismatch falls back to full encoding."""
    cache = renderer.session_token_cache
    if not cache.enabled:
        assert renderer.session_prefix_context("checked", None) is None
        return  # The pinned-version CI exercises lookup.
    params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
    old = ("<|im_start|>user\n" + "one two " * 20 + "<|im_end|>\n") * 4
    old += "<|im_start|>assistant\n"
    # The edit selects a boundary after the first or the last <|im_start|>.
    text = old[:30] + "X" + old[31:] if edit_at == "head" else old[:-3] + "X"
    ids = renderer.tokenize_prompt({"prompt": old}, params)["prompt_token_ids"]
    start = cache.marker_ids["<|im_start|>"]
    position = ids.index(start)
    if edit_at == "tail":
        position = len(ids) - 1 - ids[::-1].index(start)
    expected = renderer.tokenize_prompt({"prompt": text}, params)["prompt_token_ids"]
    for forged, route in ((True, "index_declined"), (False, "partial")):
        snapshot_ids = ids[:position] + ids[position + 1 :] if forged else ids
        reservation = cache.store.reserve((None, "checked"), 1)
        assert reservation is not None
        assert reservation.commit(TokenSnapshot(old, snapshot_ids), 1)
        cache.stats.clear()
        context = cache.context("checked", None)
        actual = renderer.tokenize_prompt(
            {"prompt": text}, replace(params, session_prefix_context=context)
        )
        assert context is not None
        context.finish(publish=False)
        assert actual["prompt_token_ids"] == expected
        assert cache.stats[route] == 1


@pytest.mark.parametrize(
    "method",
    [
        "_encode_plus",
        "_convert_encoding",
        "_get_padding_truncation_strategies",
        "set_truncation_and_padding",
        "_switch_to_input_mode",
    ],
)
def test_custom_encoding_methods_preserve_native_renderer_path(renderer, method):
    """An identical backend cannot certify Python encoding overrides."""
    from transformers import TokenizersBackend

    from vllm.tokenizers.hf import get_cached_tokenizer

    def overridden(self, *args, **kwargs):
        base = getattr(TokenizersBackend, method, None)
        if base is None:
            return None
        result = base(self, *args, **kwargs)
        if method == "_encode_plus":
            text = kwargs.get("text", args[0] if args else "")
            result["input_ids"].append(len(text) % 100 + 1000)
        return result

    custom_type = type("CustomEncoding", (TokenizersBackend,), {method: overridden})
    tokenizer = get_cached_tokenizer(
        custom_type(
            tokenizer_object=copy.deepcopy(renderer.get_tokenizer().backend_tokenizer)
        )
    )
    custom = HfRenderer(VllmConfig(model_config=renderer.model_config), tokenizer)
    try:
        cache = custom.session_token_cache
        assert cache is not None
        assert not cache.enabled
        params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
        prefix = "<|im_start|>user\nbase<|im_end|>"
        for text in (prefix, prefix + "<|im_start|>assistant\nnext"):
            expected = tokenizer(text, **params.get_encode_kwargs())["input_ids"]
            context = cache.context("custom", None)
            assert context is None
            actual = cast(
                TokensPrompt,
                custom.tokenize_prompt(
                    {"prompt": text}, replace(params, session_prefix_context=context)
                ),
            )
            assert actual["prompt_token_ids"] == expected
        assert cache.store.bytes_used == 0
    finally:
        custom.shutdown()


def test_warning_callback_override_preserves_native_ids(renderer):
    """The warning callback receives mutable IDs before encoding returns."""
    from transformers import TokenizersBackend

    from vllm.tokenizers.hf import get_cached_tokenizer

    class CustomWarningTokenizer(TokenizersBackend):
        def _eventual_warn_about_too_long_sequence(self, ids, *args, **kwargs):
            super()._eventual_warn_about_too_long_sequence(ids, *args, **kwargs)
            ids.append(1000 + len(ids) % 100)

    tokenizer = get_cached_tokenizer(
        CustomWarningTokenizer(
            tokenizer_object=copy.deepcopy(renderer.get_tokenizer().backend_tokenizer)
        )
    )
    custom = HfRenderer(VllmConfig(model_config=renderer.model_config), tokenizer)
    try:
        cache = custom.session_token_cache
        assert cache is not None
        params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
        prefix = "<|im_start|>user\nbase<|im_end|>"
        for text in (prefix, prefix + "<|im_start|>assistant\nnext"):
            expected = tokenizer(text, **params.get_encode_kwargs())["input_ids"]
            context = cache.context("custom-warning", None)
            actual = cast(
                TokensPrompt,
                custom.tokenize_prompt(
                    {"prompt": text}, replace(params, session_prefix_context=context)
                ),
            )
            if context is not None:
                context.finish(publish=True)
            assert actual["prompt_token_ids"] == expected
        assert not cache.enabled
        assert cache.store.bytes_used == 0
    finally:
        custom.shutdown()


def test_missing_native_helper_uses_full_computation(tmp_path):
    """Use a real package without the optional extension, not a mocked import."""
    import vllm

    package = tmp_path / "vllm"
    package.mkdir()
    for path in Path(vllm.__file__).parent.iterdir():
        if path.name != "__pycache__" and not path.name.startswith("_session_prefix"):
            (package / path.name).symlink_to(path, target_is_directory=path.is_dir())
    script = """
import importlib.util
import sys
from dataclasses import replace
from vllm.config import ModelConfig, VllmConfig
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import TokenizeParams
from vllm.tokenizers import get_tokenizer

assert importlib.util.find_spec('vllm._session_prefix') is None
config = VllmConfig(model_config=ModelConfig(
    model=sys.argv[1], dtype='float16', max_model_len=1024,
    tokenizer_revision='c1899de289a04d12100db370d81485cdf75e47ca',
    session_token_cache_mb=8))
tokenizer = get_tokenizer(sys.argv[1], revision=config.model_config.tokenizer_revision)
renderer = HfRenderer(config, tokenizer)
try:
    assert renderer.session_token_cache is not None
    assert renderer.session_token_cache.enabled
    assert renderer.session_prefix_context('s', None) is not None
    text = '<|im_start|>user\\nhello<|im_end|>'
    params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
    expected = tokenizer(text, **params.get_encode_kwargs())['input_ids']
    for _ in range(2):
        context = renderer.session_prefix_context('s', None)
        actual = renderer.tokenize_prompt(
            {'prompt': text}, replace(params, session_prefix_context=context)
        )
        assert actual['prompt_token_ids'] == expected
        context.finish(publish=True)
    assert renderer.session_token_cache.stats['exact'] > 0
finally:
    renderer.shutdown()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script, MODEL],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "renderer",
    [
        (MODEL, "c1899de289a04d12100db370d81485cdf75e47ca"),
        (
            os.environ.get("VLLM_TEST_QWEN25_TOKENIZER", "Qwen/Qwen2.5-7B-Instruct"),
            "a09a35458c702b33eeacc393d103063234e8bc28",
        ),
        (
            os.environ.get(
                "VLLM_TEST_LLAMA32_TOKENIZER", "unsloth/Llama-3.2-1B-Instruct"
            ),
            "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
        ),
        (
            os.environ.get("VLLM_TEST_GLM45_TOKENIZER", "zai-org/GLM-4.5"),
            "cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
        ),
        (
            os.environ.get(
                "VLLM_TEST_DEEPSEEK31_TOKENIZER", "deepseek-ai/DeepSeek-V3.1"
            ),
            "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
        ),
    ],
    indirect=True,
    ids=["qwen3", "qwen25", "llama32", "glm45", "deepseek31"],
)
async def test_real_renderer_chat_growth_matches_native_and_hits(renderer, caplog_vllm):
    cache = renderer.session_token_cache
    supported = (
        importlib.metadata.version("transformers"),
        importlib.metadata.version("tokenizers"),
    ) in {
        (transformers_version, tokenizers_version)
        for transformers_version in (
            "5.16.1",
            "5.17.0",
            "5.18.0",
            "5.19.0",
            "5.20.0",
            "5.20.1",
            "5.20.2",
        )
        for tokenizers_version in ("0.23.2", "0.23.3")
    }
    assert cache.enabled == supported
    if not supported:
        assert any(
            "Session token cache disabled:" in record.getMessage()
            for record in caplog_vllm.get_records("setup")
        )
    online = OnlineRenderer(
        renderer.model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs={"enable_thinking": False},
    )
    messages = [{"role": "user", "content": "中文 e\u0301 👩🏽‍💻\n```\nx=1\n```"}]
    for i in range(4):
        request = ChatCompletionRequest(
            model=renderer.model_config.model, messages=messages, max_tokens=1
        )
        _, expected = await online.render_chat(request)
        _, actual = await online.render_chat(request, session_id="s")
        assert (
            cast(TokensInput, actual[0])["prompt_token_ids"]
            == cast(TokensInput, expected[0])["prompt_token_ids"]
        )
        cast(TokensInput, actual[0])["prompt_token_ids"][0] = 0
        _, again = await online.render_chat(request, session_id="s")
        assert (
            cast(TokensInput, again[0])["prompt_token_ids"]
            == cast(TokensInput, expected[0])["prompt_token_ids"]
        )
        messages = messages + [
            {"role": "assistant", "content": "记录好了。"},
            {"role": "user", "content": f"继续 {i}"},
        ]
    if supported:
        assert cache.stats["exact"] >= 4
        assert cache.stats["partial"] >= 3
        assert 0 < cache.store.bytes_used <= cache.store.capacity
    else:
        assert not cache.stats
        assert cache.store.bytes_used == 0


@pytest.mark.asyncio
async def test_reasoning_effort_mapping_survives_session_reuse(renderer):
    """Keep mapped template inputs on both cache misses and history hits."""
    template = (
        "{{ reasoning_effort }}\n"
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
        "{% endfor %}<|im_start|>assistant\n"
    )
    mapped = OnlineRenderer(
        renderer.model_config,
        renderer,
        request_logger=None,
        chat_template=template,
        chat_template_content_format="string",
        reasoning_effort_map={"none": "high"},
    )
    native = OnlineRenderer(
        renderer.model_config,
        renderer,
        request_logger=None,
        chat_template=template,
        chat_template_content_format="string",
    )
    messages = [{"role": "user", "content": "Hello 中文."}]
    for turn in range(2):
        request = ChatCompletionRequest(
            model=renderer.model_config.model,
            messages=messages,
            reasoning_effort="none",
            max_tokens=1,
        )
        _, expected = await native.render_chat(
            request.model_copy(update={"reasoning_effort": "high"})
        )
        _, unmapped = await native.render_chat(request)
        expected_ids = cast(TokensInput, expected[0])["prompt_token_ids"]
        assert expected_ids != cast(TokensInput, unmapped[0])["prompt_token_ids"]
        for _ in range(2):
            _, actual = await mapped.render_chat(request, session_id="mapped-effort")
            assert cast(TokensInput, actual[0])["prompt_token_ids"] == expected_ids
        messages += [
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": f"Continue {turn}."},
        ]
    if renderer.session_token_cache.enabled:
        assert renderer.session_token_cache.stats["exact"] >= 2
        assert renderer.session_token_cache.stats["partial"] >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "renderer",
    [
        (
            os.environ.get(
                "VLLM_TEST_DEEPSEEK31_TOKENIZER", "deepseek-ai/DeepSeek-V3.1"
            ),
            "c0781d039fb7a1ba2abc4add0bdc293e92d2b8db",
        ),
    ],
    indirect=True,
)
async def test_deepseek_documented_tools_preserve_native_ids_and_reuse(
    renderer, tmp_path
):
    from vllm.renderers.chat_utils import load_chat_template

    cache = renderer.session_token_cache
    supported = importlib.metadata.version("transformers") in {
        "5.16.1",
        "5.17.0",
        "5.18.0",
        "5.19.0",
        "5.20.0",
        "5.20.1",
        "5.20.2",
    } and importlib.metadata.version("tokenizers") in {"0.23.2", "0.23.3"}
    assert cache.enabled == supported
    template = (
        Path(__file__).resolve().parents[2]
        / "examples/tool_chat_template_deepseekv31.jinja"
    )
    online = OnlineRenderer(
        renderer.model_config,
        renderer,
        request_logger=None,
        chat_template=load_chat_template(template),
        chat_template_content_format="auto",
        default_chat_template_kwargs={"thinking": False},
        enable_auto_tools=True,
        tool_parser="deepseek_v31",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Read a file or search its Unicode content.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
        for name in ("read_file", "search")
    ]

    def tool_call(identifier, name, text):
        return {
            "id": identifier,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"text": text}, ensure_ascii=False),
            },
        }

    extensions: list[list[dict[str, Any]]] = [
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    tool_call("x", "read_file", "x.py"),
                    tool_call("y", "read_file", "y.py"),
                ],
            },
            {"role": "tool", "tool_call_id": "x", "content": "中文 👩🏽‍💻"},
            {"role": "tool", "tool_call_id": "y", "content": "e\u0301"},
        ],
        [
            {"role": "assistant", "content": "Read both."},
            {"role": "user", "content": "Search e\u0301."},
        ],
        [
            {
                "role": "assistant",
                "content": "Searching.",
                "tool_calls": [tool_call("z", "search", "e\u0301")],
            },
            {"role": "tool", "tool_call_id": "z", "content": "Found 🙂"},
        ],
        [
            {"role": "assistant", "content": "Found."},
            {"role": "user", "content": "Continue."},
        ],
    ]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Read x.py and y.py. 中文 e\u0301 👩🏽‍💻"}
    ]
    for turn in range(5):
        request = ChatCompletionRequest(
            model=renderer.model_config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1,
        )
        _, expected = await online.render_chat(request.model_copy(deep=True))
        native_ids = cast(TokensInput, expected[0])["prompt_token_ids"]
        if turn == 0:
            rendered = renderer.get_tokenizer().decode(native_ids)
            assert "read_file" in rendered and "search" in rendered
        for _repeat in range(2):
            _, actual = await online.render_chat(
                request.model_copy(deep=True), session_id="deepseek-tools"
            )
            assert cast(TokensInput, actual[0])["prompt_token_ids"] == native_ids
            with (tmp_path / "deepseek-tools.jsonl").open(
                "a", encoding="utf-8"
            ) as output:
                output.write(
                    json.dumps(
                        {
                            "turn": turn,
                            "native_ids": native_ids,
                            "actual_ids": cast(TokensInput, actual[0])[
                                "prompt_token_ids"
                            ],
                            "stats": dict(cache.stats),
                        }
                    )
                    + "\n"
                )
        if turn < len(extensions):
            messages = messages + extensions[turn]
    if supported:
        assert cache.stats["exact"] >= 5 and cache.stats["partial"] >= 4
    else:
        assert not cache.stats and cache.store.bytes_used == 0


def test_length_error_and_unsupported_offsets_do_not_publish(renderer):
    cache = renderer.session_token_cache
    params = TokenizeParams(max_total_tokens=1, add_special_tokens=False)
    context = cache.context("s", None)
    with pytest.raises(VLLMValidationError):
        renderer.tokenize_prompt(
            {"prompt": "one two three four"},
            replace(params, session_prefix_context=context),
        )
    if context is not None:
        context.finish(publish=False)
    assert cache.store.bytes_used == 0
    context = cache.context("s", None)
    result = renderer.tokenize_prompt(
        {"prompt": "你好"},
        TokenizeParams(
            max_total_tokens=100,
            add_special_tokens=False,
            return_token_offsets=True,
            session_prefix_context=context,
        ),
    )
    if context is not None:
        context.finish(publish=True)
    assert result["prompt_token_offsets"]
    assert cache.store.bytes_used == 0


def test_publication_reserves_measured_size_without_evicting(renderer):
    """A full store keeps other sessions when the new snapshot fits."""
    cache = renderer.session_token_cache
    if not cache.enabled:
        assert renderer.session_prefix_context("a", None) is None
        return  # The pinned-version CI exercises publication.
    params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
    text = "<|im_start|>user\n" + "one two three four " * 40 + "<|im_end|>"

    def publish(session):
        context = cache.context(session, None)
        renderer.tokenize_prompt(
            {"prompt": text}, replace(params, session_prefix_context=context)
        )
        context.finish(publish=True)

    publish("a")
    size = cache.store.bytes_used
    cache.store.capacity = 2 * size
    publish("b")
    assert cache.store.bytes_used == 2 * size
    for session in ("a", "b"):
        with cache.store.borrow((None, session)) as snapshot:
            assert snapshot is not None


@pytest.mark.parametrize("renderer", ["left", "right"], indirect=True)
@pytest.mark.parametrize("route", ["miss", "exact", "partial"])
def test_length_validation_does_not_repeat_encoding(renderer, route):
    """Oversized requests retain native errors without a second encoding."""
    from transformers import TokenizersBackend

    cache = renderer.session_token_cache
    if not cache.enabled:
        assert renderer.session_prefix_context("length", None) is None
        return  # The pinned-version CI exercises all reuse routes.
    prefix = "<|im_start|>user\n" + "one two three four " * 8 + "<|im_end|>"
    text = prefix + "<|im_start|>assistant\nextra words"
    wide = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
    size = len(renderer.tokenize_prompt({"prompt": text}, wide)["prompt_token_ids"])
    for maximum in (size, size - 1, size - 5):
        session = f"length-{route}-{maximum}"
        key = (None, session)
        if route != "miss":
            context = cache.context(session, None)
            renderer.tokenize_prompt(
                {"prompt": text if route == "exact" else prefix},
                replace(wide, session_prefix_context=context),
            )
            assert context is not None
            context.finish(publish=True)
        params = TokenizeParams(max_total_tokens=maximum, add_special_tokens=False)
        if maximum == size:
            expected = renderer.tokenize_prompt({"prompt": text}, params)
        else:
            with pytest.raises(VLLMValidationError) as native_error:
                renderer.tokenize_prompt({"prompt": text}, params)
        before = cache.store.bytes_used
        with cache.store.borrow(key) as old:
            saved = old
        calls = 0
        previous = sys.getprofile()

        def observe(frame, event, arg):
            nonlocal calls
            if (
                event == "call"
                and frame.f_code is TokenizersBackend._encode_plus.__code__
            ):
                calls += 1

        context = cache.context(session, None)
        assert context is not None
        sys.setprofile(observe)
        try:
            if maximum == size:
                actual = renderer.tokenize_prompt(
                    {"prompt": text}, replace(params, session_prefix_context=context)
                )
                assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
                context.finish(publish=True)
            else:
                with pytest.raises(VLLMValidationError) as reused_error:
                    renderer.tokenize_prompt(
                        {"prompt": text},
                        replace(params, session_prefix_context=context),
                    )
                assert str(reused_error.value) == str(native_error.value)
                assert reused_error.value.parameter == native_error.value.parameter
                assert reused_error.value.value == native_error.value.value
                assert context.pending is None
                assert cache.store.bytes_used == before
                with cache.store.borrow(key) as after:
                    assert after is saved
        finally:
            sys.setprofile(previous)
            context.finish(publish=False)
        assert calls == (0 if route == "exact" else 1)


def test_text_edits_and_marker_edges_match_full_tokenization(renderer):
    cache = renderer.session_token_cache
    params = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
    prefix = "<|im_start|>user\n中文 e\u0301 👩🏽‍💻\n```python\nx=1\n```<|im_end|>"
    texts = [
        prefix,
        prefix,
        prefix + "<|im_start|>assistant\n",
        prefix + "<|im_start|>assistant\n\n  ",
        prefix + "<|im_start|>assistant\n\u0301🙂",
        prefix + "<|im_start|>assistant\n" + "a" * 512,
        prefix + "<|im_start|>assistant\n改写尾部",
        "改写开头" + prefix,
        prefix[:24],
        "<|im_start|>user\n<|im_",
        "<|im_start|>user\n<|im_end|>",
        "<|im_start|>user\n<|im_end|>\u0301",
        "",
    ]
    for text in texts:
        prompt = {"prompt": text}
        expected = renderer.tokenize_prompt(prompt, params)
        context = cache.context("edits", "salt")
        actual = renderer.tokenize_prompt(
            prompt, replace(params, session_prefix_context=context)
        )
        if context is not None:
            context.finish(publish=True)
        assert actual["prompt_token_ids"] == expected["prompt_token_ids"]


@pytest.mark.asyncio
async def test_same_session_concurrent_renderer_calls_are_exact(renderer):
    online = OnlineRenderer(
        renderer.model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    requests = [
        ChatCompletionRequest(
            model=MODEL,
            messages=[{"role": "user", "content": "base 中文" + str(i)}],
            max_tokens=1,
        )
        for i in range(8)
    ]
    expected = await asyncio.gather(*(online.render_chat(r) for r in requests))
    actual = await asyncio.gather(
        *(online.render_chat(r, session_id="s") for r in requests)
    )
    for (_, a), (_, b) in zip(expected, actual):
        assert (
            cast(TokensInput, a[0])["prompt_token_ids"]
            == cast(TokensInput, b[0])["prompt_token_ids"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "gate_event"),
    [
        ("cancel", "call"),
        ("cancel", "return"),
        ("clear", "return"),
        ("close", "return"),
    ],
)
async def test_inflight_renderer_publication_lifecycle(renderer, action, gate_event):
    """Observe real worker execution without replacing tokenizer/render methods."""
    cache = renderer.session_token_cache
    if not cache.enabled:
        assert cache.context("lifecycle", None) is None
        return  # The dedicated pinned-version CI exercises the positive path.
    entered, release, finished, closed = (threading.Event() for _ in range(4))
    previous = threading.getprofile()
    gated = False

    def observe(frame, event, arg):
        nonlocal gated
        if (
            frame.f_code is SessionPrefixStore.clear.__code__
            and event == "return"
            and frame.f_locals["self"] is cache.store
            and cache.store.closed
        ):
            closed.set()
        if frame.f_code is not SessionTokenCache.encode.__code__:
            return
        if frame.f_locals["params"].session_prefix_context is None:
            return
        if event == gate_event and not gated:
            gated = True
            entered.set()
            assert release.wait(15), "worker release timed out"
        if event == "return":
            finished.set()

    online = OnlineRenderer(
        renderer.model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "生命周期 中文 e\u0301 🙂"}],
        max_tokens=1,
    )
    task = closing = None
    threading.setprofile(observe)
    try:
        _, expected = await online.render_chat(request)
        task = asyncio.create_task(online.render_chat(request, session_id="lifecycle"))
        assert await asyncio.to_thread(entered.wait, 15), "encode gate not reached"
        if gate_event == "return":
            assert cache.store.bytes_used > 0  # A real pending snapshot exists.
        if action == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif action == "clear":
            cache.store.clear()
        else:
            closing = asyncio.create_task(asyncio.to_thread(renderer.shutdown))
            assert await asyncio.to_thread(closed.wait, 15), "close not reached"
        release.set()
        assert await asyncio.to_thread(finished.wait, 15), "worker did not finish"
        if action != "cancel":
            _, actual = await task
            assert (
                cast(TokensInput, actual[0])["prompt_token_ids"]
                == cast(TokensInput, expected[0])["prompt_token_ids"]
            )
        if closing is not None:
            await closing
        assert cache.store.bytes_used == 0
        with cache.store.borrow((None, "lifecycle")) as snapshot:
            assert snapshot is None
        if action != "close":
            _, actual = await online.render_chat(request, session_id="lifecycle")
            assert (
                cast(TokensInput, actual[0])["prompt_token_ids"]
                == cast(TokensInput, expected[0])["prompt_token_ids"]
            )
            with cache.store.borrow((None, "lifecycle")) as snapshot:
                assert snapshot is not None  # Same key can be published again.
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await closing
        threading.setprofile(previous)
