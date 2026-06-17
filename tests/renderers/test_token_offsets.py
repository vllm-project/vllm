# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.renderers.params import TokenizeParams


def test_tokenize_params_default_return_token_offsets_false():
    """Default value of return_token_offsets must be False so existing
    callers see zero behavior change."""
    params = TokenizeParams(max_total_tokens=None)
    assert params.return_token_offsets is False


def test_tokenize_params_return_token_offsets_constructible_true():
    """The new field must be constructible via kwarg."""
    params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
    assert params.return_token_offsets is True


def test_tokens_prompt_supports_offsets_field():
    """TokensPrompt accepts the new prompt_token_offsets field as a
    NotRequired TypedDict member. TypedDict has no runtime validation,
    so the assertion is structural: the field shows up in __annotations__."""
    from vllm.inputs.llm import TokensPrompt

    assert "prompt_token_offsets" in TokensPrompt.__annotations__


def test_tokens_input_supports_offsets_field():
    from vllm.inputs.engine import TokensInput

    assert "prompt_token_offsets" in TokensInput.__annotations__


@pytest.fixture
def fast_tokenizer():
    """gpt2 has a Fast tokenizer available; use it to test the offsets
    happy path."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)


def _make_base_renderer_with(tokenizer):
    """Build a minimal BaseRenderer subclass that exposes the tokenizer
    so we can call `_tokenize_prompt` directly. BaseRenderer is
    abstract because of `render_messages`; we just need a stub."""
    from vllm.renderers.base import BaseRenderer

    class _StubRenderer(BaseRenderer):
        def __init__(self, tok):
            # Bypass BaseRenderer.__init__ — we don't need a VllmConfig.
            from vllm.utils.async_utils import make_async

            self.tokenizer = tok
            self._executor = None
            # Mirror BaseRenderer.__init__: the async path offloads the sync
            # ``_tokenize_prompt`` to a thread pool.
            self._async_tokenize_prompt = make_async(self._tokenize_prompt)
            self.mm_processor = None

        def get_tokenizer(self):
            return self.tokenizer

        def _can_produce_offsets(self):
            # Mirror HfRenderer: offsets only for fast tokenizers.
            return self.tokenizer is not None and self.tokenizer.is_fast

        def render_messages(self, messages, params):  # pragma: no cover
            raise NotImplementedError

    return _StubRenderer(tokenizer)


class TestTokenizePromptOffsets:
    def test_fast_tokenizer_with_flag_returns_offsets(self, fast_tokenizer):
        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello, world."}

        result = renderer._tokenize_prompt(prompt, params)

        assert "prompt_token_ids" in result
        assert "prompt_token_offsets" in result
        offsets = result["prompt_token_offsets"]
        assert offsets is not None
        # Length must match the token sequence.
        assert len(offsets) == len(result["prompt_token_ids"])
        # Each offset is (int, int) with start <= end <= len(text).
        text_len = len("Hello, world.")
        for s, e in offsets:
            assert isinstance(s, int) and isinstance(e, int)
            assert 0 <= s <= e <= text_len

    def test_base_renderer_without_override_yields_no_offsets(self, fast_tokenizer):
        """A renderer that does not override `_can_produce_offsets` never
        emits offsets, even with a fast tokenizer and the flag set. This
        locks in the base-default-False / subclass-override design."""
        from vllm.renderers.base import BaseRenderer
        from vllm.renderers.params import TokenizeParams

        class _BareRenderer(BaseRenderer):
            def __init__(self, tok):
                self.tokenizer = tok
                self._executor = None
                self.mm_processor = None

            def get_tokenizer(self):
                return self.tokenizer

            def render_messages(self, messages, params):  # pragma: no cover
                raise NotImplementedError

        renderer = _BareRenderer(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello, world."}

        result = renderer._tokenize_prompt(prompt, params)

        assert "prompt_token_offsets" not in result

    def test_default_flag_no_offsets(self, fast_tokenizer):
        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None)  # flag defaults False
        prompt = {"prompt": "Hello, world."}

        result = renderer._tokenize_prompt(prompt, params)

        # Field must be absent (not set to None) so JSON serialization
        # of TokensInput stays minimal for existing consumers.
        assert "prompt_token_offsets" not in result

    def test_slow_tokenizer_with_flag_no_offsets(self, fast_tokenizer):
        """Force is_fast=False to simulate a Slow tokenizer. The flag is
        True but offsets must not be returned because the tokenizer
        cannot produce them."""
        from unittest.mock import PropertyMock, patch

        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello, world."}

        with patch.object(
            type(fast_tokenizer),
            "is_fast",
            new_callable=PropertyMock,
            return_value=False,
        ):
            result = renderer._tokenize_prompt(prompt, params)

        assert "prompt_token_offsets" not in result

    def test_multimodal_data_with_flag_no_offsets(self, fast_tokenizer):
        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello.", "multi_modal_data": {"image": ["fake"]}}

        result = renderer._tokenize_prompt(prompt, params)

        assert "prompt_token_offsets" not in result

    def test_multimodal_uuids_with_flag_no_offsets(self, fast_tokenizer):
        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello.", "multi_modal_uuids": {"image": ["uuid-1"]}}

        result = renderer._tokenize_prompt(prompt, params)

        assert "prompt_token_offsets" not in result

    @pytest.mark.asyncio
    async def test_async_tokenize_prompt_returns_offsets(self, fast_tokenizer):
        """The async path must produce the same shape of result as the
        sync path."""
        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello, world."}

        result = await renderer._tokenize_prompt_async(prompt, params)

        assert "prompt_token_offsets" in result
        offsets = result["prompt_token_offsets"]
        assert offsets is not None
        assert len(offsets) == len(result["prompt_token_ids"])

    @pytest.mark.asyncio
    async def test_async_tokenize_prompt_default_no_offsets(self, fast_tokenizer):
        from vllm.renderers.params import TokenizeParams

        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None)  # flag default False
        prompt = {"prompt": "Hello, world."}

        result = await renderer._tokenize_prompt_async(prompt, params)

        assert "prompt_token_offsets" not in result


class TestProcessTokensForwardsOffsets:
    """End-to-end regression tests for the chain
    `_tokenize_prompt` -> `_process_tokens` -> `TokensInput`.

    The previously-passing endpoint tests mocked the preprocessing
    layer and returned hand-crafted engine inputs, which masked a bug
    where `_process_tokens` rebuilt the engine input from scratch and
    forwarded only `prompt` and `cache_salt`, dropping
    `prompt_token_offsets` before it reached `create_tokenize`.
    """

    def test_process_tokens_forwards_offsets_to_engine_input(self, fast_tokenizer):
        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello, world."}

        tokens_prompt = renderer._tokenize_prompt(prompt, params)
        # Sanity: the TokensPrompt must already carry offsets, otherwise
        # this regression test is checking the wrong layer.
        assert "prompt_token_offsets" in tokens_prompt
        expected_offsets = tokens_prompt["prompt_token_offsets"]

        engine_input = renderer._process_tokens(tokens_prompt)

        assert "prompt_token_offsets" in engine_input
        assert engine_input["prompt_token_offsets"] == expected_offsets

    @pytest.mark.asyncio
    async def test_process_tokens_async_forwards_offsets_to_engine_input(
        self, fast_tokenizer
    ):
        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None, return_token_offsets=True)
        prompt = {"prompt": "Hello, world."}

        tokens_prompt = await renderer._tokenize_prompt_async(prompt, params)
        assert "prompt_token_offsets" in tokens_prompt
        expected_offsets = tokens_prompt["prompt_token_offsets"]

        engine_input = await renderer._process_tokens_async(tokens_prompt)

        assert "prompt_token_offsets" in engine_input
        assert engine_input["prompt_token_offsets"] == expected_offsets

    def test_process_tokens_no_offsets_when_flag_off(self, fast_tokenizer):
        """Negative case: when the flag is off, `_tokenize_prompt` does
        not set the field on the TokensPrompt, and `_process_tokens`
        must not add it to the engine input either."""
        renderer = _make_base_renderer_with(fast_tokenizer)
        params = TokenizeParams(max_total_tokens=None)  # flag defaults False
        prompt = {"prompt": "Hello, world."}

        tokens_prompt = renderer._tokenize_prompt(prompt, params)
        assert "prompt_token_offsets" not in tokens_prompt

        engine_input = renderer._process_tokens(tokens_prompt)

        assert "prompt_token_offsets" not in engine_input
