# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for streaming derender.

Tests are split into two layers:

1. Unit tests (no server): covers ``_detokenize_delta`` correctness
   (chunked == one-shot) and ``derender_completion_stream`` /
   ``derender_chat_stream`` logic via a real tokenizer on a tiny model.
   The parser path is covered both with a deterministic stub parser and
   with the real ``HarmonyParser`` (skipped without ``openai_harmony``).

2. Integration tests (require a running render server): covers the full
   HTTP round-trip through the streaming endpoint.  Marked with
   ``@pytest.mark.asyncio`` and gated by the ``server`` / ``client``
   fixtures from the sibling ``test_derender.py``.
"""

import json
from collections.abc import Callable

import pytest
import pytest_asyncio

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    DerenderStreamState,
    GenerateResponse,
    GenerateResponseChoice,
    GenerateResponseStreamChoice,
    GenerateStreamResponse,
)
from vllm.parser import Parser
from vllm.utils import random_uuid

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


class _FakeParser(Parser):
    """Deterministic `Parser` stub for unit testing the replay/merge/pin
    mechanics of `OnlineDerenderer._derender_chat_stream_parsed` in
    isolation from any real reasoning/tool parser's markup grammar.

    Keys purely off `delta_token_ids[0]` and ignores `delta_text`:

    - `TOOL_START` opens a new tool call at index 0 (id + name).
    - `TOOL_ARG` appends one `"a"` to that tool call's arguments.
    - `REASON` emits one `"r"` of reasoning (suppressed when the request
      has `include_reasoning=False`, mirroring the real parsers).
    - `CONTENT` emits one `"c"` of content.
    - An empty `delta_token_ids` with `finished=True` (the finish only
      flush call) emits a `"FLUSH"` content sentinel so tests can confirm
      it happened.
    - Anything else emits nothing.
    """

    TOOL_START = 9
    TOOL_ARG = 10
    REASON = 11
    CONTENT = 12

    def parse_delta(
        self,
        delta_text,
        delta_token_ids,
        request,
        prompt_token_ids=None,
        *,
        finished,
    ):
        if not delta_token_ids:
            return DeltaMessage(content="FLUSH") if finished else None

        tok = delta_token_ids[0]
        if tok == self.TOOL_START:
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        id=f"call-{random_uuid()}",
                        type="function",
                        function=DeltaFunctionCall(name="get_weather", arguments=""),
                        index=0,
                    )
                ]
            )
        if tok == self.TOOL_ARG:
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(index=0, function=DeltaFunctionCall(arguments="a"))
                ]
            )
        if tok == self.REASON:
            if not request.include_reasoning:
                return None
            return DeltaMessage(reasoning="r")
        if tok == self.CONTENT:
            return DeltaMessage(content="c")
        return None


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_stream_chunk(
    token_ids: list[int],
    index: int = 0,
    finish_reason: str | None = None,
    request_id: str = "test-req",
    usage: dict | None = None,
) -> GenerateStreamResponse:
    """Build a GenerateStreamResponse SSE chunk."""
    from vllm.entrypoints.openai.engine.protocol import UsageInfo

    return GenerateStreamResponse(
        request_id=request_id,
        choices=[
            GenerateResponseStreamChoice(
                index=index,
                token_ids=token_ids,
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(**usage) if usage else None,
    )


def _make_usage_chunk(
    completion_tokens: int,
    prompt_tokens: int = 0,
    request_id: str = "test-req",
) -> GenerateStreamResponse:
    """Build a usage only final SSE chunk (empty choices)."""
    from vllm.entrypoints.openai.engine.protocol import UsageInfo

    return GenerateStreamResponse(
        request_id=request_id,
        choices=[],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests — no running server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tiny tokenizer used across unit tests."""
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(MODEL_NAME)


@pytest.fixture(scope="module")
def derenderer(tokenizer):
    """Construct a minimal OnlineDerenderer backed by a stub renderer."""
    from unittest.mock import MagicMock

    from vllm.renderers.online_derenderer import OnlineDerenderer

    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer

    model_config = MagicMock()
    model_config.hf_config.model_type = "llama"
    model_config.model = MODEL_NAME

    return OnlineDerenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
        trust_request_chat_template=False,
        enable_auto_tools=False,
        tool_parser=None,
        reasoning_parser=None,
    )


@pytest.fixture(scope="module")
def parsed_derenderer(tokenizer):
    """OnlineDerenderer with `_FakeParser` wired in as `self.parser`.

    The parser configured path runs `_derender_chat_stream_parsed` via
    `make_async(..., executor=renderer._executor)`, so unlike
    `derenderer` above (whose streaming paths never touch the executor),
    `renderer._executor` must be a real `ThreadPoolExecutor` here.
    `loop.run_in_executor` cannot submit work to a `MagicMock`.
    """
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import MagicMock

    from vllm.renderers.online_derenderer import OnlineDerenderer

    renderer = MagicMock()
    renderer.get_tokenizer.return_value = tokenizer
    renderer._executor = ThreadPoolExecutor(max_workers=2)

    model_config = MagicMock()
    model_config.hf_config.model_type = "llama"
    model_config.model = MODEL_NAME

    dr = OnlineDerenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
    )
    dr.parser = _FakeParser
    return dr


class TestDetokenizeDelta:
    """_detokenize_delta: chunked decode must equal one shot decode."""

    def _one_shot(self, tokenizer, token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def _chunked(self, derenderer, tokenizer, chunks: list[list[int]]) -> str:
        state = DerenderStreamState()
        parts: list[str] = []
        for delta in chunks:
            text, state = derenderer._detokenize_delta(
                tokenizer, delta, state, skip_special_tokens=True
            )
            parts.append(text)
        return "".join(parts)

    def test_single_chunk(self, derenderer, tokenizer):
        """All tokens in one chunk == one shot decode."""
        token_ids = tokenizer.encode("Hello world")[:8]
        assert self._chunked(derenderer, tokenizer, [token_ids]) == self._one_shot(
            tokenizer, token_ids
        )

    def test_two_equal_chunks(self, derenderer, tokenizer):
        """Split in half and reassemble == one shot."""
        token_ids = tokenizer.encode("Hello world from streaming derender")[:12]
        mid = len(token_ids) // 2
        chunks = [token_ids[:mid], token_ids[mid:]]
        assert self._chunked(derenderer, tokenizer, chunks) == self._one_shot(
            tokenizer, token_ids
        )

    def test_single_token_per_chunk(self, derenderer, tokenizer):
        """One token per chunk (most granular streaming) == one shot."""
        token_ids = tokenizer.encode("incremental detokenization test")[:10]
        chunks = [[t] for t in token_ids]
        assert self._chunked(derenderer, tokenizer, chunks) == self._one_shot(
            tokenizer, token_ids
        )

    def test_empty_delta_passthrough(self, derenderer, tokenizer):
        """An empty delta (usage only chunk) emits empty string and preserves state."""
        token_ids = tokenizer.encode("Hello")[:4]
        _, state = derenderer._detokenize_delta(
            tokenizer, token_ids, DerenderStreamState(), skip_special_tokens=True
        )
        text, new_state = derenderer._detokenize_delta(
            tokenizer, [], state, skip_special_tokens=True
        )
        assert text == ""
        assert new_state.prev_tokens == state.prev_tokens
        assert new_state.prefix_offset == state.prefix_offset
        assert new_state.read_offset == state.read_offset

    def test_multibyte_char_split_across_chunks(self, derenderer, tokenizer):
        """A CJK/emoji char straddling chunk boundaries == one shot.

        Regression test for held back trailing incomplete UTF-8 byte
        sequences being dropped when the rebuild window marks them as
        already read (see #46159).
        """
        token_ids = tokenizer.encode("Hello ✅ world 日本語")[:16]
        chunks = [[t] for t in token_ids]
        assert self._chunked(derenderer, tokenizer, chunks) == self._one_shot(
            tokenizer, token_ids
        )

    def test_state_carries_across_calls(self, derenderer, tokenizer):
        """Decode state threads across calls. Text still matches one shot."""
        t1 = tokenizer.encode("Hello")[:2]
        t2 = tokenizer.encode(" world")[:2]
        state = DerenderStreamState()
        text1, state = derenderer._detokenize_delta(tokenizer, t1, state)
        text2, state = derenderer._detokenize_delta(tokenizer, t2, state)
        assert text1 + text2 == self._one_shot(tokenizer, t1 + t2)
        # Offsets are rebased to the carried tail each chunk
        assert state.prefix_offset == 0

    def test_state_window_stays_bounded(self, derenderer, tokenizer):
        """prev_tokens must not grow with the number of chunks (bounded transport).

        Guards that the carried decode window is a small constant
        tail, so cumulative ``stream_state`` transport is O(n) and not O(n^2).
        """
        token_ids = tokenizer.encode(
            "a reasonably long ascii stream of tokens used to exercise the "
            "window bound across many single token chunks so the carried "
            "state cannot grow linearly with the generation length"
        )
        assert len(token_ids) > 32
        state = DerenderStreamState()
        max_window = 0
        for tok in token_ids:
            _, state = derenderer._detokenize_delta(tokenizer, [tok], state)
            max_window = max(max_window, len(state.prev_tokens))
        # Bounded by a small constant independent of len(token_ids)
        assert max_window <= 32

    def test_n_independent_streams_same_result(self, derenderer, tokenizer):
        """N parallel streams with the same token sequence give the same text."""
        token_ids = tokenizer.encode("parallel streams")[:8]
        mid = len(token_ids) // 2

        results = []
        for _ in range(3):
            state = DerenderStreamState()
            text, state = derenderer._detokenize_delta(
                tokenizer, token_ids[:mid], state
            )
            text2, _ = derenderer._detokenize_delta(tokenizer, token_ids[mid:], state)
            results.append(text + text2)

        assert len(set(results)) == 1, "All independent streams must produce same text"
        assert results[0] == self._one_shot(tokenizer, token_ids)


class TestDerenderCompletionStream:
    """derender_completion_stream: streaming output parity with one shot."""

    @pytest.mark.asyncio
    async def test_chunked_equals_oneshot(self, derenderer, tokenizer):
        """Sum of streaming text chunks == one shot tokenizer.decode."""
        token_ids = tokenizer.encode("streaming completion test")[:10]
        mid = len(token_ids) // 2

        state = DerenderStreamState()
        chunk1, state = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[:mid]),
            state=state,
        )
        chunk2, _ = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[mid:], finish_reason="stop"),
            state=state,
        )

        streamed_text = chunk1.choices[0].text + chunk2.choices[0].text
        one_shot = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert streamed_text == one_shot

    @pytest.mark.asyncio
    async def test_usage_chunk_passthrough(self, derenderer, tokenizer):
        """Usage only final chunk (empty choices) is passed through correctly."""
        usage_chunk = _make_usage_chunk(completion_tokens=10, prompt_tokens=5)
        chunk, state = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=usage_chunk,
        )
        assert chunk.choices == []
        assert chunk.usage is not None
        assert chunk.usage.completion_tokens == 10
        assert chunk.usage.prompt_tokens == 5

    @pytest.mark.asyncio
    async def test_prompt_tokens_in_usage(self, derenderer, tokenizer):
        """prompt_tokens is correctly forwarded into usage on a usage chunk."""
        token_ids = tokenizer.encode("hello")[:3]
        usage_chunk = _make_usage_chunk(
            completion_tokens=len(token_ids), prompt_tokens=7
        )
        chunk, _ = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=usage_chunk,
            prompt_tokens=7,
        )
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 7

    @pytest.mark.asyncio
    async def test_none_state_initialises_correctly(self, derenderer, tokenizer):
        """Passing state=None (first call) initialises an empty DerenderStreamState."""
        token_ids = tokenizer.encode("hello")[:4]
        chunk, state = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids),
            state=None,
        )
        assert isinstance(state, DerenderStreamState)
        assert chunk.choices[0].text == tokenizer.decode(
            token_ids, skip_special_tokens=True
        )

    @pytest.mark.asyncio
    async def test_skip_special_tokens_threaded(self, derenderer, tokenizer):
        """completion_request.skip_special_tokens is honored (not hardcoded True)."""
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest

        eos = tokenizer.eos_token_id
        if eos is None:
            pytest.skip("tokenizer has no eos token to exercise special stripping")
        token_ids = tokenizer.encode("hi")[:2] + [eos]

        async def _text(skip: bool) -> str:
            req = CompletionRequest(
                model=MODEL_NAME, prompt="x", skip_special_tokens=skip
            )
            chunk, _ = await derenderer.derender_completion_stream(
                model=MODEL_NAME,
                generate_chunk=_make_stream_chunk(token_ids),
                completion_request=req,
            )
            return chunk.choices[0].text

        # skip=False must retain the special token; skip=True must strip it.
        assert await _text(False) != await _text(True)

    @pytest.mark.asyncio
    async def test_finish_reason_forwarded(self, derenderer, tokenizer):
        """finish_reason from the generate chunk reaches the derendered choice."""
        token_ids = tokenizer.encode("done")[:2]
        chunk, _ = await derenderer.derender_completion_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids, finish_reason="length"),
        )
        assert chunk.choices[0].finish_reason == "length"


class TestDerenderChatStream:
    """derender_chat_stream: plain detok branch (no parser)."""

    @pytest.mark.asyncio
    async def test_role_on_first_chunk_only(self, derenderer, tokenizer):
        """role='assistant' appears in the first chunk, not subsequent ones."""
        token_ids = tokenizer.encode("hello world")[:6]
        mid = len(token_ids) // 2

        state = DerenderStreamState()
        chunk1, state = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[:mid]),
            state=state,
        )
        chunk2, _ = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[mid:], finish_reason="stop"),
            state=state,
        )

        assert chunk1.choices[0].delta.role == "assistant"
        assert chunk2.choices[0].delta.role is None

    @pytest.mark.asyncio
    async def test_chunked_equals_oneshot(self, derenderer, tokenizer):
        """Sum of streaming content deltas == one shot decode."""
        token_ids = tokenizer.encode("streaming chat derender text")[:10]
        mid = len(token_ids) // 2

        state = DerenderStreamState()
        chunk1, state = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[:mid]),
            state=state,
        )
        chunk2, _ = await derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(token_ids[mid:]),
            state=state,
        )

        streamed = (chunk1.choices[0].delta.content or "") + (
            chunk2.choices[0].delta.content or ""
        )
        one_shot = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert streamed == one_shot

    @pytest.mark.asyncio
    async def test_parser_configured_missing_chat_request_raises(
        self, parsed_derenderer
    ):
        """A parser configured model must never fall through to plain detok
        even when `chat_request` is omitted. If allowed this would leak raw
        reasoning/tool markup into `delta.content`. `ServingDerender`
        pre-checks this too (400 before touching the tokenizer). This pins
        the `OnlineDerenderer` level backstop."""
        with pytest.raises(ValueError, match="chat_request"):
            await parsed_derenderer.derender_chat_stream(
                model=MODEL_NAME,
                generate_chunk=_make_stream_chunk([_FakeParser.CONTENT]),
                state=None,
                chat_request=None,
            )


def _chat_request(**kwargs):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    kwargs.setdefault("model", MODEL_NAME)
    return ChatCompletionRequest(**kwargs)


class TestDerenderChatStreamParsed:
    """derender_chat_stream: parser branch (replay + parse_delta) exercised
    against the deterministic `_FakeParser` so these pin OnlineDerenderer's
    own replay/merge/pin/finish_reason logic independent of any real
    parser's markup grammar (covered separately by the parser_server backed
    integration tests below)."""

    @pytest.mark.asyncio
    async def test_dispatches_and_emits_content(self, parsed_derenderer):
        chunk, state = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(
                [_FakeParser.CONTENT], finish_reason="stop"
            ),
            chat_request=_chat_request(),
        )
        assert chunk.choices[0].delta.content == "c"
        assert chunk.choices[0].delta.role == "assistant"
        assert chunk.choices[0].finish_reason == "stop"
        assert state.output_token_ids == [_FakeParser.CONTENT]

    @pytest.mark.asyncio
    async def test_role_sent_once(self, parsed_derenderer):
        chat_request = _chat_request()
        chunk1, state = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([_FakeParser.CONTENT]),
            chat_request=chat_request,
        )
        chunk2, _ = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(
                [_FakeParser.CONTENT], finish_reason="stop"
            ),
            state=state,
            chat_request=chat_request,
        )
        assert chunk1.choices[0].delta.role == "assistant"
        assert chunk2.choices[0].delta.role is None

    @pytest.mark.asyncio
    async def test_finish_only_chunk_flushes(self, parsed_derenderer):
        """A finish only chunk (no new tokens) still calls `parse_delta`
        once with `finished=True` to flush buffered state."""
        chat_request = _chat_request()
        _, state = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([_FakeParser.CONTENT]),
            chat_request=chat_request,
        )
        chunk2, _ = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([], finish_reason="stop"),
            state=state,
            chat_request=chat_request,
        )
        assert chunk2.choices[0].delta.content == "FLUSH"
        assert chunk2.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chunking_invariance(self, parsed_derenderer):
        """1 token per chunk vs a single whole chunk assemble identically
        which is the property the per-token replay granularity is meant to buy."""
        token_ids = [
            _FakeParser.REASON,
            _FakeParser.REASON,
            _FakeParser.TOOL_START,
            _FakeParser.TOOL_ARG,
            _FakeParser.TOOL_ARG,
            _FakeParser.CONTENT,
        ]
        chat_request = _chat_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
            include_reasoning=True,
        )

        async def _assemble(chunks: list[list[int]]) -> dict:
            state = None
            content = ""
            reasoning = ""
            tool_args = ""
            for i, tids in enumerate(chunks):
                finish = "stop" if i == len(chunks) - 1 else None
                chunk, state = await parsed_derenderer.derender_chat_stream(
                    model=MODEL_NAME,
                    generate_chunk=_make_stream_chunk(tids, finish_reason=finish),
                    state=state,
                    chat_request=chat_request,
                )
                delta = chunk.choices[0].delta
                content += delta.content or ""
                reasoning += delta.reasoning or ""
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        tool_args += tc.function.arguments
            return {"content": content, "reasoning": reasoning, "tool_args": tool_args}

        whole = await _assemble([token_ids])
        one_at_a_time = await _assemble([[t] for t in token_ids])

        assert whole == one_at_a_time
        assert whole == {"content": "c", "reasoning": "rr", "tool_args": "aa"}

    @pytest.mark.asyncio
    async def test_tool_call_id_pinned_across_chunks(self, parsed_derenderer):
        """Once an index's ID is recorded in `last_tool_call_ids`, a later
        id bearing delta for that same index is pinned to the recorded
        value rather than a freshly (re-)generated one."""
        chat_request = _chat_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
        )
        chunk1, state = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([_FakeParser.TOOL_START]),
            chat_request=chat_request,
        )
        first_id = chunk1.choices[0].delta.tool_calls[0].id
        assert first_id is not None
        assert state.last_tool_call_ids == [first_id]

        chunk2, state = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(
                [_FakeParser.TOOL_START], finish_reason="stop"
            ),
            state=state,
            chat_request=chat_request,
        )
        assert chunk2.choices[0].delta.tool_calls[0].id == first_id
        assert state.last_tool_call_ids == [first_id]

    @pytest.mark.asyncio
    async def test_finish_reason_rewritten_to_tool_calls(self, parsed_derenderer):
        chat_request = _chat_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
        )
        chunk, _ = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(
                [_FakeParser.TOOL_START, _FakeParser.TOOL_ARG], finish_reason="stop"
            ),
            chat_request=chat_request,
        )
        assert chunk.choices[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_finish_reason_stop_for_named_tool_choice(self, parsed_derenderer):
        # ChatCompletionRequest's `check_tool_usage` is a mode="before"
        # validator, so it sees the raw value and only accepts "auto",
        # "required" or a dict, never an already built
        # ChatCompletionNamedToolChoiceParam.
        chat_request = _chat_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        chunk, _ = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(
                [_FakeParser.TOOL_START, _FakeParser.TOOL_ARG], finish_reason="stop"
            ),
            chat_request=chat_request,
        )
        assert chunk.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_include_reasoning_false_suppresses_reasoning(
        self, parsed_derenderer
    ):
        chat_request = _chat_request(include_reasoning=False)
        chunk, _ = await parsed_derenderer.derender_chat_stream(
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk(
                [_FakeParser.REASON, _FakeParser.CONTENT], finish_reason="stop"
            ),
            chat_request=chat_request,
        )
        delta = chunk.choices[0].delta
        assert delta.reasoning is None
        assert delta.content == "c"


# ---------------------------------------------------------------------------
# Harmony / GPT-OSS replay — unit, no server
# ---------------------------------------------------------------------------

HARMONY_MODEL = "openai/gpt-oss-20b"
HARMONY_REASONING = "The user wants 2 plus 3."
HARMONY_ANSWER = "The answer is 5."
HARMONY_PROMPT = "<|start|>user<|message|>Add 2 and 3.<|end|><|start|>assistant"
HARMONY_OUTPUT = (
    f"<|channel|>analysis<|message|>{HARMONY_REASONING}<|end|>"
    f"<|start|>assistant<|channel|>final<|message|>{HARMONY_ANSWER}<|return|>"
)


@pytest.fixture(scope="module")
def harmony_encode():
    """Encoder for canned GPT-OSS harmony token sequences."""
    pytest.importorskip("openai_harmony")

    # Pre-caches the o200k_base BPE file that openai-harmony's Rust backend
    # downloads on first use, same as the sibling suite's E2E harmony tests.
    from tests.entrypoints.scale_out.derender.test_derender import (
        _ensure_harmony_vocab,
    )
    from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    _ensure_harmony_vocab()
    encoding = get_encoding()

    def _encode(harmony_str: str) -> list[int]:
        return encoding.encode(harmony_str, allowed_special="all")

    return _encode


@pytest.fixture(scope="module")
def harmony_tokenizer():
    pytest.importorskip("openai_harmony")
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(HARMONY_MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def harmony_derenderer(harmony_tokenizer):
    """OnlineDerenderer whose parser resolves to the real `HarmonyParser`.

    Same shape as `parsed_derenderer` (mocked renderer, real executor) but
    with a real tokenizer and parser, so the replay path is exercised
    against Harmony's actual channel grammar rather than a stub.
    """
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import MagicMock

    from vllm.parser.harmony import HarmonyParser
    from vllm.renderers.online_derenderer import OnlineDerenderer

    renderer = MagicMock()
    renderer.get_tokenizer.return_value = harmony_tokenizer
    renderer._executor = ThreadPoolExecutor(max_workers=2)

    model_config = MagicMock()
    model_config.model = HARMONY_MODEL
    model_config.hf_config.model_type = "gpt_oss"
    model_config.hf_text_config.model_type = "gpt_oss"
    model_config.hf_overrides = None

    dr = OnlineDerenderer(
        model_config=model_config,
        renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
        enable_auto_tools=True,
        tool_parser="openai",
        reasoning_parser="openai_gptoss",
    )
    assert dr.use_harmony
    assert dr.parser is HarmonyParser
    return dr


async def _stream_harmony_deltas(
    derenderer,
    chat_request,
    output_ids: list[int],
    prompt_ids: list[int],
    chunk_size: int,
) -> list[DeltaMessage]:
    """Drive `output_ids` through the parser path in fixed size chunks.

    Returns the per chunk `DeltaMessage`s in order so callers can assert on
    intermediate emissions, not just the assembled result.
    """
    deltas: list[DeltaMessage] = []
    state = None
    for start in range(0, len(output_ids), chunk_size):
        tids = output_ids[start : start + chunk_size]
        is_last = start + chunk_size >= len(output_ids)
        chunk, state = await derenderer.derender_chat_stream(
            model=HARMONY_MODEL,
            generate_chunk=_make_stream_chunk(
                tids, finish_reason="stop" if is_last else None
            ),
            state=state,
            chat_request=chat_request,
            prompt_token_ids=prompt_ids,
        )
        deltas.append(chunk.choices[0].delta)
    return deltas


class TestDerenderChatStreamHarmony:
    """derender_chat_stream: replay + `parse_delta` against real HarmonyParser.

    Skipped where `openai_harmony` is not installed. `prompt_token_ids` is
    threaded through for call shape fidelity only, since HarmonyParser
    derives its state from the output tokens alone.
    """

    @pytest.mark.asyncio
    async def test_reasoning_never_leaks_as_content_midstream(
        self, harmony_derenderer, harmony_encode
    ):
        """Regression guard for the RFC's original replay + diff design.

        `HarmonyParser.parse()` always flushes to EOS. Mid-stream that
        raises `HarmonyError` and the recovery branch re-emits the in
        flight message on the `final` channel, so partial analysis surfaces
        as content. Replay + `parse_delta` must never do that. Driven one
        token per chunk, the finest granularity a client can produce.
        """
        output_ids = harmony_encode(HARMONY_OUTPUT)
        prompt_ids = harmony_encode(HARMONY_PROMPT)
        chat_request = _chat_request(model=HARMONY_MODEL, include_reasoning=True)

        deltas = await _stream_harmony_deltas(
            harmony_derenderer, chat_request, output_ids, prompt_ids, chunk_size=1
        )

        reasoning = ""
        content = ""
        for i, delta in enumerate(deltas):
            reasoning += delta.reasoning or ""
            content += delta.content or ""
            # Any analysis text emitted as content breaks the prefix
            # property, since the two channels share no prefix here.
            assert HARMONY_ANSWER.startswith(content), (
                f"content after chunk {i} is not a prefix of the final "
                f"channel text: {content!r}"
            )
            assert HARMONY_REASONING.startswith(reasoning)
            assert not delta.tool_calls

        assert reasoning == HARMONY_REASONING
        assert content == HARMONY_ANSWER

    @pytest.mark.asyncio
    async def test_stream_matches_batch(self, harmony_derenderer, harmony_encode):
        """Streamed assembly equals one shot `/derender` over the same IDs."""
        output_ids = harmony_encode(HARMONY_OUTPUT)
        prompt_ids = harmony_encode(HARMONY_PROMPT)
        chat_request = _chat_request(model=HARMONY_MODEL, include_reasoning=True)

        deltas = await _stream_harmony_deltas(
            harmony_derenderer, chat_request, output_ids, prompt_ids, chunk_size=3
        )
        reasoning = "".join(d.reasoning or "" for d in deltas)
        content = "".join(d.content or "" for d in deltas)

        batch_choices = await harmony_derenderer.derender_chat(
            GenerateResponse(
                request_id="test-harmony-batch",
                choices=[
                    GenerateResponseChoice(
                        index=0, token_ids=output_ids, finish_reason="stop"
                    )
                ],
            ),
            chat_request,
        )
        message = batch_choices[0].message

        assert reasoning == message.reasoning
        assert content == message.content


class TestDerenderStreamStateValidation:
    """DerenderStreamState rejects malformed caller supplied offsets/lengths."""

    def test_negative_prefix_offset_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DerenderStreamState(prefix_offset=-1)

    def test_negative_read_offset_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DerenderStreamState(read_offset=-1)

    def test_prev_tokens_over_cap_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DerenderStreamState(prev_tokens=["a"] * 1025)

    def test_prev_tokens_at_cap_accepted(self):
        state = DerenderStreamState(prev_tokens=["a"] * 1024)
        assert len(state.prev_tokens) == 1024


class TestServingDerenderStreamErrorHandling:
    """Malformed stream_state must surface as 400 and not an unhandled 500."""

    def _make_serving(self, side_effect: Exception):
        from unittest.mock import AsyncMock, MagicMock

        from vllm.entrypoints.scale_out.derender.serving import ServingDerender

        models = MagicMock()
        models.is_base_model.return_value = True
        models.model_config = MagicMock()
        models.model_config.max_model_len = 100_000

        online_derenderer = MagicMock()
        online_derenderer.parser = None
        online_derenderer.derender_completion_stream = AsyncMock(
            side_effect=side_effect
        )
        online_derenderer.derender_chat_stream = AsyncMock(side_effect=side_effect)

        return ServingDerender(models=models, online_derenderer=online_derenderer)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [KeyError("bad byte"), IndexError("oob")])
    async def test_completion_stream_bad_state_returns_400(self, exc):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderCompletionStreamRequest,
        )

        serving = self._make_serving(exc)
        request = DerenderCompletionStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2]),
            stream_state=DerenderStreamState(),
        )
        result = await serving.derender_completion_stream_response(request)
        assert isinstance(result, ErrorResponse)
        assert result.error.code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exc", [KeyError("bad byte"), IndexError("oob")])
    async def test_chat_stream_bad_state_returns_400(self, exc):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(exc)
        request = DerenderChatStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2]),
            stream_state=DerenderStreamState(),
        )
        result = await serving.derender_chat_stream_response(request)
        assert isinstance(result, ErrorResponse)
        assert result.error.code == 400


class TestServingDerenderStreamValidation:
    """If checks cannot run in `derender_chat_stream_response`: all must
    reject with 400 before ever calling into `online_derenderer` (i.e.
    before touching the tokenizer)."""

    def _make_serving(self, *, parser_configured: bool, max_model_len: int = 100_000):
        from unittest.mock import AsyncMock, MagicMock

        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionStreamResponse,
        )
        from vllm.entrypoints.scale_out.derender.serving import ServingDerender

        models = MagicMock()
        models.is_base_model.return_value = True
        models.model_config = MagicMock()
        models.model_config.max_model_len = max_model_len

        online_derenderer = MagicMock()
        online_derenderer.parser = MagicMock() if parser_configured else None
        online_derenderer.derender_chat_stream = AsyncMock(
            return_value=(
                ChatCompletionStreamResponse(id="t", model=MODEL_NAME, choices=[]),
                DerenderStreamState(),
            )
        )
        return ServingDerender(models=models, online_derenderer=online_derenderer)

    @pytest.mark.asyncio
    async def test_missing_chat_request_with_parser_rejected(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(parser_configured=True)
        request = DerenderChatStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2]),
        )
        result = await serving.derender_chat_stream_response(request)
        assert isinstance(result, ErrorResponse)
        assert result.error.code == 400
        assert "chat_request" in result.error.message
        serving.online_derenderer.derender_chat_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_chat_request_without_parser_ok(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(parser_configured=False)
        request = DerenderChatStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2]),
        )
        result = await serving.derender_chat_stream_response(request)
        assert not isinstance(result, ErrorResponse)

    @pytest.mark.asyncio
    async def test_missing_prompt_token_ids_with_parser_rejected(self):
        """A parser configured model must reject a missing prompt_token_ids
        the same way it rejects a missing chat_request. Without it,
        parse_delta cannot tell whether the prompt left reasoning open and
        would misclassify reasoning content as plain content."""
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(parser_configured=True)
        request = DerenderChatStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2]),
            chat_request=_chat_request(),
        )
        result = await serving.derender_chat_stream_response(request)
        assert isinstance(result, ErrorResponse)
        assert result.error.code == 400
        assert "prompt_token_ids" in result.error.message
        serving.online_derenderer.derender_chat_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_prompt_token_ids_present_with_parser_ok(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(parser_configured=True)
        request = DerenderChatStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2]),
            chat_request=_chat_request(),
            prompt_token_ids=[1, 2, 3],
        )
        result = await serving.derender_chat_stream_response(request)
        assert not isinstance(result, ErrorResponse)

    @pytest.mark.asyncio
    async def test_oversized_output_token_ids_rejected(self):
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(parser_configured=False, max_model_len=4)
        request = DerenderChatStreamRequest(
            stream=True,
            model=MODEL_NAME,
            generate_chunk=_make_stream_chunk([1, 2, 3]),
            stream_state=DerenderStreamState(output_token_ids=[1, 2]),
        )
        result = await serving.derender_chat_stream_response(request)
        assert isinstance(result, ErrorResponse)
        assert result.error.code == 400
        assert "max_model_len" in result.error.message
        serving.online_derenderer.derender_chat_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_too_many_choices_rejected(self):
        """Each streamed chunk contains at most one choice (a single
        DerenderStreamState is threaded through every choice). The check
        could never fire since derender_chat_stream itself rejects anything
        above 1 first."""
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
            DerenderChatStreamRequest,
        )

        serving = self._make_serving(parser_configured=False)
        two_choices = GenerateStreamResponse(
            request_id="t",
            choices=[
                GenerateResponseStreamChoice(index=i, token_ids=[1]) for i in range(2)
            ],
        )
        request = DerenderChatStreamRequest(
            stream=True, model=MODEL_NAME, generate_chunk=two_choices
        )
        result = await serving.derender_chat_stream_response(request)
        assert isinstance(result, ErrorResponse)
        assert result.error.code == 400
        assert "at most one choice" in result.error.message
        serving.online_derenderer.derender_chat_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests — require a live render server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    from tests.utils import RemoteLaunchRenderServer

    with RemoteLaunchRenderServer(MODEL_NAME, []) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    import httpx

    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


async def _render_chat(client) -> dict:
    """Render a minimal chat request and return the GenerateRequest dict."""

    resp = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_streaming_completion_derender_roundtrip(client):
    """Streaming completions derender: chunked text == non streaming text."""
    gen_req = await _render_chat(client)
    token_ids: list[int] = gen_req["token_ids"][:8]
    mid = len(token_ids) // 2
    chunk1_ids, chunk2_ids = token_ids[:mid], token_ids[mid:]

    # Non streaming baseline.
    non_stream_resp = await client.post(
        "/v1/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_responses": [
                {
                    "request_id": "test-ns",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": token_ids,
                            "finish_reason": "stop",
                        }
                    ],
                }
            ],
        },
    )
    assert non_stream_resp.status_code == 200
    expected_text = non_stream_resp.json()["choices"][0]["text"]

    # Streaming call 1.
    r1 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk1_ids, "finish_reason": None}
                ],
            },
            "stream_state": None,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    text1 = d1["chunk"]["choices"][0]["text"]
    state1 = d1["stream_state"]

    # Streaming call 2 (final chunk).
    r2 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk2_ids, "finish_reason": "stop"}
                ],
            },
            "stream_state": state1,
        },
    )
    assert r2.status_code == 200
    text2 = r2.json()["chunk"]["choices"][0]["text"]

    assert text1 + text2 == expected_text


@pytest.mark.asyncio
async def test_streaming_chat_derender_roundtrip(client):
    """Streaming chat derender (plain detok): chunked text == non streaming text."""
    gen_req = await _render_chat(client)
    token_ids: list[int] = gen_req["token_ids"][:8]
    mid = len(token_ids) // 2
    chunk1_ids, chunk2_ids = token_ids[:mid], token_ids[mid:]

    # Non streaming baseline.
    ns = await client.post(
        "/v1/chat/completions/derender",
        json={
            "model": MODEL_NAME,
            "generate_response": {
                "request_id": "test-ns",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": token_ids,
                        "finish_reason": "stop",
                    }
                ],
            },
        },
    )
    assert ns.status_code == 200
    expected_content = ns.json()["choices"][0]["message"]["content"]

    # Streaming call 1.
    r1 = await client.post(
        "/v1/chat/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk1_ids, "finish_reason": None}
                ],
            },
            "stream_state": None,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    text1 = d1["chunk"]["choices"][0]["delta"].get("content") or ""
    state1 = d1["stream_state"]
    # role=assistant on the first chunk
    assert d1["chunk"]["choices"][0]["delta"].get("role") == "assistant"

    # Streaming call 2.
    r2 = await client.post(
        "/v1/chat/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "test-s",
                "choices": [
                    {"index": 0, "token_ids": chunk2_ids, "finish_reason": "stop"}
                ],
            },
            "stream_state": state1,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    text2 = d2["chunk"]["choices"][0]["delta"].get("content") or ""
    # role must NOT be repeated on subsequent chunks
    assert d2["chunk"]["choices"][0]["delta"].get("role") is None

    assert text1 + text2 == expected_content


@pytest.mark.asyncio
async def test_streaming_derender_invalid_body_returns_400(client):
    """Missing required field in streaming request returns 400."""
    r = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            # missing required 'model' and 'generate_chunk'
        },
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_streaming_derender_non_object_body_returns_400(client):
    """A non object JSON body (e.g. a list) returns 400, not a 500."""
    r = await client.post(
        "/v1/completions/derender",
        json=[1, 2, 3],
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_streaming_usage_chunk(client):
    """Usage only final chunk is forwarded with correct token counts."""
    gen_req = await _render_chat(client)
    token_ids: list[int] = gen_req["token_ids"][:6]
    state: dict = {}

    # Send content chunk first.
    r1 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "usage-test",
                "choices": [
                    {"index": 0, "token_ids": token_ids, "finish_reason": "stop"}
                ],
            },
            "stream_state": None,
        },
    )
    assert r1.status_code == 200
    state = r1.json()["stream_state"]

    # Send usage only final chunk.
    r2 = await client.post(
        "/v1/completions/derender",
        json={
            "stream": True,
            "model": MODEL_NAME,
            "generate_chunk": {
                "request_id": "usage-test",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": len(token_ids),
                    "total_tokens": 10 + len(token_ids),
                },
            },
            "stream_state": state,
            "prompt_tokens": 10,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["chunk"]["choices"] == []
    assert d2["chunk"]["usage"]["prompt_tokens"] == 10
    assert d2["chunk"]["usage"]["completion_tokens"] == len(token_ids)


# ---------------------------------------------------------------------------
# Integration tests for parser configured (reasoning + tool calls) require a
# live render server. Mirrors the parser_server / parser_tokenizer pattern
# from test_derender.py.
# ---------------------------------------------------------------------------

PARSER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

_PARSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


@pytest.fixture(scope="module")
def parser_server():
    from tests.utils import RemoteLaunchRenderServer

    args = [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "deepseek_r1",
    ]
    with RemoteLaunchRenderServer(PARSER_MODEL, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def parser_client(parser_server):
    import httpx

    async with httpx.AsyncClient(
        base_url=parser_server.url_for(""), timeout=60.0
    ) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def parser_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(PARSER_MODEL)


def _require_parser_markers(tokenizer, text: str, *markers: str) -> list[int]:
    """Encode text and skip the test if any marker is lost in roundtrip."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    for m in markers:
        if m not in decoded:
            pytest.skip(f"Marker {m!r} lost in encode->decode roundtrip")
    return ids


async def _render_parser_chat(client, messages: list[dict]) -> dict:
    resp = await client.post(
        "/v1/chat/completions/render",
        json={"model": PARSER_MODEL, "messages": messages},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _stream_chat_derender(
    client,
    output_ids: list[int],
    chunk_sizes: list[int],
    chat_request: dict,
    prompt_tokens: int,
    prompt_token_ids: list[int],
    on_chunk: Callable[[list[dict]], None] | None = None,
) -> dict:
    """Feed `output_ids` through the streaming chat derender endpoint in
    the given `chunk_sizes`, threading `stream_state` across calls and
    return the assembled message.

    If `on_chunk` is given, it is called after every chunk with a
    snapshot (deep copy) of the `tool_calls` accumulator so far, letting
    callers assert properties of the intermediate deltas (e.g. monotonic
    argument growth) rather than only the final assembled result.
    """
    state = None
    content = ""
    reasoning = ""
    tool_calls: list[dict] = []
    finish_reason = None

    pos = 0
    for i, size in enumerate(chunk_sizes):
        tids = output_ids[pos : pos + size]
        pos += size
        is_last = i == len(chunk_sizes) - 1
        resp = await client.post(
            "/v1/chat/completions/derender",
            json={
                "stream": True,
                "model": PARSER_MODEL,
                "generate_chunk": {
                    "request_id": "stream-test",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": tids,
                            "finish_reason": "stop" if is_last else None,
                        }
                    ],
                },
                "stream_state": state,
                "prompt_tokens": prompt_tokens,
                "prompt_token_ids": prompt_token_ids,
                "chat_request": chat_request,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        state = data["stream_state"]
        delta = data["chunk"]["choices"][0]["delta"]
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning") or ""
        for tc in delta.get("tool_calls") or []:
            idx = tc["index"]
            while len(tool_calls) <= idx:
                tool_calls.append({"id": None, "name": None, "arguments": ""})
            if tc.get("id"):
                tool_calls[idx]["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                tool_calls[idx]["name"] = fn["name"]
            if fn.get("arguments"):
                tool_calls[idx]["arguments"] += fn["arguments"]
        if is_last:
            finish_reason = data["chunk"]["choices"][0]["finish_reason"]
        if on_chunk is not None:
            on_chunk([dict(tc) for tc in tool_calls])

    return {
        "content": content or None,
        "reasoning": reasoning or None,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }


@pytest.mark.asyncio
async def test_stream_parsed_matches_batch_reasoning(parser_client, parser_tokenizer):
    """Streamed == batch: assembled reasoning/content equal the non
    streaming `/derender` result over the same token IDs, for every
    chunking of the same output (chunking invariance)."""
    messages = [{"role": "user", "content": "What is 2+3?"}]
    gen_req = await _render_parser_chat(parser_client, messages)

    reasoning_text = "The user wants 2 plus 3. That is 5."
    answer_text = "The answer is 5."
    output_text = f"<think>{reasoning_text}</think>{answer_text}"
    output_ids = _require_parser_markers(parser_tokenizer, output_text, "</think>")

    chat_request = {
        "model": PARSER_MODEL,
        "messages": messages,
        "include_reasoning": True,
    }

    batch_resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": {
                "request_id": "batch",
                "choices": [
                    {"index": 0, "token_ids": output_ids, "finish_reason": "stop"}
                ],
            },
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": chat_request,
        },
    )
    assert batch_resp.status_code == 200, batch_resp.text
    batch_msg = batch_resp.json()["choices"][0]["message"]

    n = len(output_ids)
    for chunk_sizes in ([n], [1] * n, [3] * (n // 3) + [n - 3 * (n // 3)]):
        chunk_sizes = [c for c in chunk_sizes if c > 0]
        streamed = await _stream_chat_derender(
            parser_client,
            output_ids,
            chunk_sizes,
            chat_request,
            len(gen_req["token_ids"]),
            gen_req["token_ids"],
        )
        assert streamed["content"] == batch_msg["content"]
        assert streamed["reasoning"] == batch_msg.get("reasoning")


@pytest.mark.asyncio
async def test_stream_parsed_matches_batch_tool_call(parser_client, parser_tokenizer):
    """Tool call name+id+arguments parity between streamed and batch, plus
    the finish_reason -> "tool_calls" rewrite for auto tool choice."""
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    gen_req = await _render_parser_chat(parser_client, messages)

    output_text = (
        "<think>Let me check.</think>"
        '<tool_call>\n{"name": "get_weather", '
        '"arguments": {"city": "Paris"}}\n</tool_call>'
    )
    output_ids = _require_parser_markers(
        parser_tokenizer, output_text, "</think>", "<tool_call>", "</tool_call>"
    )
    chat_request = {
        "model": PARSER_MODEL,
        "messages": messages,
        "tools": _PARSER_TOOLS,
        "tool_choice": "auto",
    }

    batch_resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "model": PARSER_MODEL,
            "generate_response": {
                "request_id": "batch",
                "choices": [
                    {"index": 0, "token_ids": output_ids, "finish_reason": "stop"}
                ],
            },
            "prompt_tokens": len(gen_req["token_ids"]),
            "chat_request": chat_request,
        },
    )
    assert batch_resp.status_code == 200, batch_resp.text
    batch_choice = batch_resp.json()["choices"][0]
    if not batch_choice["message"]["tool_calls"]:
        pytest.skip("Model did not emit a tool call")

    for chunk_sizes in ([len(output_ids)], [1] * len(output_ids)):
        # Snapshot of tool_calls[0]["arguments"] after every chunk, to check
        # it only ever grows by appending. It never retracts or duplicates
        # already streamed text. That's the exact failure mode the
        # replay + parse_delta design (over diff-based streaming) exists to
        # avoid. Parity with the batch result alone wouldn't catch a
        # transient mid-stream regression.
        arg_snapshots: list[str] = []

        def _record(
            tool_calls: list[dict], _snapshots: list[str] = arg_snapshots
        ) -> None:
            if tool_calls and tool_calls[0]["arguments"]:
                _snapshots.append(tool_calls[0]["arguments"])

        streamed = await _stream_chat_derender(
            parser_client,
            output_ids,
            chunk_sizes,
            chat_request,
            len(gen_req["token_ids"]),
            gen_req["token_ids"],
            on_chunk=_record,
        )
        assert streamed["tool_calls"]
        assert streamed["tool_calls"][0]["name"] == "get_weather"
        assert streamed["tool_calls"][0]["id"] is not None
        assert json.loads(streamed["tool_calls"][0]["arguments"]) == json.loads(
            batch_choice["message"]["tool_calls"][0]["function"]["arguments"]
        )
        assert streamed["finish_reason"] == "tool_calls"

        assert arg_snapshots, "expected at least one tool-call argument delta"
        for prev, curr in zip(arg_snapshots, arg_snapshots[1:]):
            assert curr.startswith(prev), (
                f"tool-call arguments regressed: {prev!r} -> {curr!r}"
            )
        assert arg_snapshots[-1] == streamed["tool_calls"][0]["arguments"]


@pytest.mark.asyncio
async def test_stream_parsed_cjk_across_chunk_boundaries(
    parser_client, parser_tokenizer
):
    """CJK/emoji tokens split across chunk boundaries with a parser active
    across the reasoning -> content transition (parser path regression
    guard analogous to #46159 in the plain detok path)."""
    messages = [{"role": "user", "content": "Reply in Chinese"}]
    gen_req = await _render_parser_chat(parser_client, messages)

    reasoning_text = "思考中"
    answer_text = "你好世界 😀"
    output_text = f"<think>{reasoning_text}</think>{answer_text}"
    output_ids = _require_parser_markers(parser_tokenizer, output_text, "</think>")

    chat_request = {
        "model": PARSER_MODEL,
        "messages": messages,
        "include_reasoning": True,
    }
    streamed = await _stream_chat_derender(
        parser_client,
        output_ids,
        [1] * len(output_ids),
        chat_request,
        len(gen_req["token_ids"]),
        gen_req["token_ids"],
    )
    assert "�" not in (streamed["content"] or "")
    assert "�" not in (streamed["reasoning"] or "")
    assert answer_text in (streamed["content"] or "")


@pytest.mark.asyncio
async def test_stream_parsed_include_reasoning_false(parser_client, parser_tokenizer):
    """include_reasoning=False emits no reasoning deltas on the streaming
    parser path."""
    messages = [{"role": "user", "content": "What is 2+3?"}]
    gen_req = await _render_parser_chat(parser_client, messages)

    output_text = "<think>reasoning here</think>The answer is 5."
    output_ids = _require_parser_markers(parser_tokenizer, output_text, "</think>")

    chat_request = {
        "model": PARSER_MODEL,
        "messages": messages,
        "include_reasoning": False,
    }
    streamed = await _stream_chat_derender(
        parser_client,
        output_ids,
        [1] * len(output_ids),
        chat_request,
        len(gen_req["token_ids"]),
        gen_req["token_ids"],
    )
    assert streamed["reasoning"] is None


@pytest.mark.asyncio
async def test_stream_parsed_missing_chat_request_rejected(parser_client):
    """Parser configured + no chat_request on the streaming endpoint -> 400
    (the live-server counterpart to the mocked
    `TestServingDerenderStreamValidation` checks above)."""
    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "stream": True,
            "model": PARSER_MODEL,
            "generate_chunk": {
                "request_id": "reject-test",
                "choices": [{"index": 0, "token_ids": [1, 2], "finish_reason": None}],
            },
        },
    )
    assert resp.status_code == 400
    assert "chat_request" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_stream_parsed_missing_prompt_token_ids_rejected(parser_client):
    """Parser configured + chat_request given but no prompt_token_ids on the
    streaming endpoint -> 400 (the live-server counterpart to the mocked
    `test_missing_prompt_token_ids_with_parser_rejected` above). Without
    prompt_token_ids, parse_delta cannot tell whether the prompt left
    reasoning open and would silently misclassify reasoning as content."""
    messages = [{"role": "user", "content": "Hello"}]
    resp = await parser_client.post(
        "/v1/chat/completions/derender",
        json={
            "stream": True,
            "model": PARSER_MODEL,
            "generate_chunk": {
                "request_id": "reject-test",
                "choices": [{"index": 0, "token_ids": [1, 2], "finish_reason": None}],
            },
            "chat_request": {
                "model": PARSER_MODEL,
                "messages": messages,
                "include_reasoning": True,
            },
        },
    )
    assert resp.status_code == 400
    assert "prompt_token_ids" in resp.json()["error"]["message"]
