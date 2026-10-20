# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare cache results with independent full HF encoding and guard fallbacks."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import pytest
from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import TokenizersBackend

from vllm.tokenizers.segment_cache import SegmentTokenizerCache

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.renderers.chat_utils import ConversationMessage


@pytest.fixture(params=[None, "NFC"])
def tokenizer(request):
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if request.param == "NFC":
        backend.normalizer = normalizers.NFC()
    backend.train_from_iterator(
        ["hello world\n def f(x): return x", "你好 é 👩‍💻 JSON {} "],
        trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    backend.add_special_tokens(["<|start|>", "<|end|>"])
    return TokenizersBackend(tokenizer_object=backend)


@pytest.mark.parametrize("specials", [True, False])
@pytest.mark.parametrize(
    "body",
    [
        "hello world",
        "你好 é 👩‍💻",
        "  \n\t  ",
        'def f(x):\n return {"x": x}',
        "<|end|><|start|>",
        "<|end|> ",
        "",
        "long input " * 2048,
    ],
)
def test_growing_history_matches_full_encoding(tokenizer, specials, body):
    cache = SegmentTokenizerCache(tokenizer, 0.001)
    assert cache.enabled, cache.disabled_reason
    text = ""
    for turn in range(4):
        text += f"<|start|>{body}{turn}<|end|>"
        assert cache.encode(text, add_special_tokens=specials) == tokenizer.encode(
            text, add_special_tokens=specials
        )
    assert cache.hits > 0
    assert 0 < cache.bytes_used <= cache.capacity


@pytest.mark.parametrize(
    "options", [{"return_offsets_mapping": True}, {"max_length": 1}]
)
def test_chat_tokenizer_options_keep_regular_semantics(tokenizer, options):
    from types import SimpleNamespace

    from vllm.renderers.hf import (
        VLLMValidationError,
        _apply_chat_template_with_cache,
        safe_apply_chat_template,
    )

    cache = SegmentTokenizerCache(tokenizer, 0.001)
    kwargs = dict(
        chat_template=(
            "{% for m in messages %}<|start|>{{ m.content }}<|end|>{% endfor %}"
        ),
        tokenizer_kwargs=options,
    )
    model_config = cast("ModelConfig", SimpleNamespace())
    messages: list[ConversationMessage] = [{"role": "user", "content": "hello world"}]
    if "max_length" in options:
        # HF rejects a duplicate tokenizer option; the cache must not hide it.
        with pytest.raises(VLLMValidationError) as regular_error:
            safe_apply_chat_template(model_config, tokenizer, messages, **kwargs)
        with pytest.raises(VLLMValidationError) as cached_error:
            _apply_chat_template_with_cache(
                cache, model_config, tokenizer, messages, **kwargs
            )
        assert isinstance(regular_error.value.__cause__, TypeError)
        assert isinstance(cached_error.value.__cause__, TypeError)
        assert str(cached_error.value) == str(regular_error.value)
    else:
        regular = safe_apply_chat_template(model_config, tokenizer, messages, **kwargs)
        assert (
            _apply_chat_template_with_cache(
                cache, model_config, tokenizer, messages, **kwargs
            )
            == regular
        )
    assert cache.lookups == 0


def test_concurrent_and_returned_list_mutation_preserve_ids(tokenizer):
    cache = SegmentTokenizerCache(tokenizer, 0.001)
    texts = [f"hello<|end|>{i}你好<|end|>tail" for i in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(cache.encode, texts * 3))
    assert actual == [tokenizer.encode(t) for t in texts * 3]
    assert actual[0] is not None
    actual[0].clear()
    assert cache.encode(texts[0]) == tokenizer.encode(texts[0])


def test_tiny_budget_evicts_and_oversize_entries_do_not_break_encoding(tokenizer):
    cache = SegmentTokenizerCache(tokenizer, 1500 / (1 << 30))
    for i in range(12):
        text = f"entry {i}<|end|>" + "long " * 400
        assert cache.encode(text) == tokenizer.encode(text)
        assert cache.bytes_used <= cache.capacity


@pytest.mark.parametrize(
    "option",
    [
        {"return_offsets_mapping": True},
        {"padding": True},
        {"return_tensors": "pt"},
        {"truncation": True, "max_length": 1},
        {"truncation": True},
    ],
)
def test_unsupported_options_preserve_regular_path(tokenizer, option):
    cache = SegmentTokenizerCache(tokenizer, 0.001)
    assert cache.encode("hello<|end|>world", **option) is None


@pytest.mark.parametrize("flag", ["lstrip", "rstrip", "single_word"])
def test_context_sensitive_added_tokens_disable_cache(tokenizer, flag):
    tokenizer.add_tokens([AddedToken("marker", **{flag: True})])
    assert not SegmentTokenizerCache(tokenizer, 0.001).enabled


def test_prefix_space_and_metaspace_disable_cache(tokenizer):
    for pre in [
        pre_tokenizers.ByteLevel(add_prefix_space=True),
        pre_tokenizers.Metaspace(),
    ]:
        tokenizer.backend_tokenizer.pre_tokenizer = pre
        assert not SegmentTokenizerCache(tokenizer, 0.001).enabled


def test_bos_insertion_falls_back_only_when_specials_requested(tokenizer):
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<|start|> $A",
        special_tokens=[("<|start|>", tokenizer.convert_tokens_to_ids("<|start|>"))],
    )
    cache = SegmentTokenizerCache(tokenizer, 0.001)
    text = "hello<|end|>world"
    assert cache.enabled
    assert cache.encode(text, add_special_tokens=True) is None
    assert cache.encode(text, add_special_tokens=False) == tokenizer.encode(
        text, add_special_tokens=False
    )


@pytest.mark.parametrize("capacity", [-1, float("nan"), float("inf")])
def test_invalid_budget_rejected(tokenizer, capacity):
    with pytest.raises(ValueError):
        SegmentTokenizerCache(tokenizer, capacity)


def test_disabled_budget_does_not_encode(tokenizer):
    cache = SegmentTokenizerCache(tokenizer, 0)
    assert cache.encode("hello") is None
    assert cache.lookups == 0


def test_other_normalizers_and_normalized_added_tokens_fall_back(tokenizer):
    tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    assert not SegmentTokenizerCache(tokenizer, 0.001).enabled
    tokenizer.backend_tokenizer.normalizer = normalizers.NFC()
    tokenizer.add_tokens([AddedToken("é", normalized=True)])
    assert not SegmentTokenizerCache(tokenizer, 0.001).enabled


def test_stochastic_bpe_dropout_falls_back(tokenizer):
    tokenizer.backend_tokenizer.model.dropout = 0.1
    assert not SegmentTokenizerCache(tokenizer, 0.001).enabled


@pytest.fixture
def renderer(tokenizer):
    from types import SimpleNamespace

    from vllm.renderers.hf import HfRenderer

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            renderer_num_workers=2,
            tokenizer_cache_gb=0.001,
            enable_prompt_embeds=False,
            is_multimodal_model=False,
            supports_multimodal_inputs=False,
        ),
        parallel_config=SimpleNamespace(_api_process_rank=0),
    )
    renderer = HfRenderer(config, tokenizer)
    try:
        yield renderer
    finally:
        renderer.shutdown()


@pytest.mark.parametrize("offsets", [False, True])
def test_renderer_cache_preserves_offsets_and_truncation(tokenizer, renderer, offsets):
    from vllm.renderers.params import TokenizeParams

    text = "hello<|end|>world<|end|>你好"
    for maximum in (1000, 2):
        params = TokenizeParams(
            max_total_tokens=maximum,
            return_token_offsets=offsets,
            add_special_tokens=False,
        )
        kwargs = params.get_encode_kwargs()
        reference = tokenizer(text, **kwargs, return_offsets_mapping=offsets)
        actual = renderer._tokenize_prompt({"prompt": text}, params)
        assert actual["prompt_token_ids"] == reference["input_ids"]
        if offsets:
            assert actual["prompt_token_offsets"] == reference["offset_mapping"]
    assert renderer._tokenizer_cache is not None
    assert renderer._tokenizer_cache.enabled


def test_renderer_async_chat_uses_cache_with_timeout(tokenizer, renderer):
    messages: list[ConversationMessage] = [{"role": "user", "content": "hello 你好"}]
    template = "{% for m in messages %}<|start|>{{ m.content }}<|end|>{% endfor %}"
    expected = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=True, return_dict=False
    )
    cache = renderer._tokenizer_cache
    assert cache is not None and cache.enabled

    async def render():
        return await renderer._render_with_timeout(
            renderer._apply_chat_template_async(
                cache,
                renderer.model_config,
                tokenizer,
                messages,
                chat_template=template,
            )
        )

    for _ in range(2):
        assert asyncio.run(render()) == expected
    assert cache.hits > 0


def test_custom_encoding_override_preserves_full_call(tokenizer, renderer):
    from types import SimpleNamespace

    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.params import TokenizeParams

    class CustomTokenizer(TokenizersBackend):
        def _encode_plus(self, text, *args, **kwargs):
            output = super()._encode_plus(text, *args, **kwargs)
            if text.startswith("custom-prefix"):
                output["input_ids"].append(self.convert_tokens_to_ids("<|start|>"))
            return output

    custom = CustomTokenizer(tokenizer_object=tokenizer.backend_tokenizer)
    config = SimpleNamespace(
        model_config=renderer.model_config,
        parallel_config=SimpleNamespace(_api_process_rank=0),
    )
    custom_renderer = HfRenderer(config, custom)
    try:
        text = "custom-prefix<|end|>tail"
        params = TokenizeParams(max_total_tokens=1000, add_special_tokens=False)
        expected = custom(text, **params.get_encode_kwargs())["input_ids"]
        actual = custom_renderer._tokenize_prompt({"prompt": text}, params)
        assert actual["prompt_token_ids"] == expected
        assert custom_renderer._tokenizer_cache is None
    finally:
        custom_renderer.shutdown()


@pytest.mark.parametrize("generation_prompt", [False, True])
def test_chat_cache_matches_regular_template_for_each_request(
    tokenizer, generation_prompt
):
    from types import SimpleNamespace

    from vllm.renderers.hf import (
        _apply_chat_template_with_cache,
        safe_apply_chat_template,
    )

    cache = SegmentTokenizerCache(tokenizer, 0.001)
    model_config = cast("ModelConfig", SimpleNamespace())
    messages: list[ConversationMessage] = [{"role": "user", "content": "hello 你好"}]
    for template in (
        "{% for m in messages %}<|start|>{{ m.content }}<|end|>{% endfor %}",
        "changed {% for m in messages %}{{ m.content }}<|end|>{% endfor %}",
    ):
        kwargs: dict[str, Any] = dict(
            chat_template=template, add_generation_prompt=generation_prompt
        )
        regular = safe_apply_chat_template(model_config, tokenizer, messages, **kwargs)
        for _ in range(2):
            actual = _apply_chat_template_with_cache(
                cache, model_config, tokenizer, messages, **kwargs
            )
            assert actual == regular
    assert cache.hits > 0


def test_cache_context_does_not_consume_template_variables(tokenizer):
    from types import SimpleNamespace

    from vllm.renderers.hf import (
        _apply_chat_template_with_cache,
        safe_apply_chat_template,
    )

    cache = SegmentTokenizerCache(tokenizer, 0.001)
    model_config = cast("ModelConfig", SimpleNamespace())
    messages: list[ConversationMessage] = [{"role": "user", "content": "hello"}]
    kwargs: dict[str, Any] = dict(
        chat_template="<|start|>{{ tokenizer_cache }}<|end|>",
        tokenizer_cache="hello world",
    )
    regular = safe_apply_chat_template(model_config, tokenizer, messages, **kwargs)
    assert regular == tokenizer.encode(
        "<|start|>hello world<|end|>", add_special_tokens=False
    )
    assert (
        _apply_chat_template_with_cache(
            cache, model_config, tokenizer, messages, **kwargs
        )
        == regular
    )


def test_cached_chat_token_ids_requires_text_and_matching_ids(tokenizer):
    from vllm.tokenizers.segment_cache import cached_chat_token_ids

    cache = SegmentTokenizerCache(tokenizer, 0.001)
    assert cache.enabled, cache.disabled_reason
    text = "<|start|>hello<|end|>"
    expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def render(*, tokenize=True):
        return expected if tokenize else text

    assert cached_chat_token_ids(None, render, {}) is None
    assert cached_chat_token_ids(cache, render, {"tokenize": False}) is None
    assert (
        cached_chat_token_ids(cache, render, {"return_assistant_tokens_mask": True})
        is None
    )
    assert cached_chat_token_ids(cache, render, {}) == expected
    assert cached_chat_token_ids(cache, render, {}) == expected
    assert cache.hits > 0


def test_cached_chat_token_ids_falls_back_on_non_text_or_mismatch(tokenizer):
    from vllm.tokenizers.segment_cache import cached_chat_token_ids

    text = "<|start|>hello<|end|>"
    expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def render_ids_only(*, tokenize=True):
        return expected

    ids_only_cache = SegmentTokenizerCache(tokenizer, 0.001)
    assert cached_chat_token_ids(ids_only_cache, render_ids_only, {}) is None

    def render_mismatch(*, tokenize=True):
        return expected + [0] if tokenize else text

    mismatch_cache = SegmentTokenizerCache(tokenizer, 0.001)
    assert cached_chat_token_ids(mismatch_cache, render_mismatch, {}) is None
    assert mismatch_cache._chat_ok is False


def test_template_override_tokenizer_still_gets_a_cache(tokenizer):
    from types import SimpleNamespace

    from vllm.renderers.hf import HfRenderer

    class TemplateOverrideTokenizer(TokenizersBackend):
        def apply_chat_template(self, messages, **kwargs):
            text = "".join(f"<|start|>{m['content']}<|end|>" for m in messages)
            if kwargs.get("tokenize", True):
                return self.encode(text, add_special_tokens=False)
            return text

    custom = TemplateOverrideTokenizer(tokenizer_object=tokenizer.backend_tokenizer)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            renderer_num_workers=1,
            tokenizer_cache_gb=0.001,
            enable_prompt_embeds=False,
            is_multimodal_model=False,
            supports_multimodal_inputs=False,
        ),
        parallel_config=SimpleNamespace(_api_process_rank=0),
    )
    renderer = HfRenderer(config, custom)
    try:
        cache = renderer._tokenizer_cache
        assert cache is not None, "a template-only override must keep the cache"
        assert cache.enabled, cache.disabled_reason
        messages: list[ConversationMessage] = [{"role": "user", "content": "hello"}]
        ids = renderer._cached_chat_token_ids(
            lambda **kw: custom.apply_chat_template(messages, **kw), {}
        )
        assert ids == custom.apply_chat_template(messages, tokenize=True)
    finally:
        renderer.shutdown()
