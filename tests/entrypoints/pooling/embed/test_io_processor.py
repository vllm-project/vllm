# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EmbedIOProcessor."""

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedContent,
    CohereEmbedInput,
    CohereEmbedRequest,
    EmbeddingChatRequest,
)
from vllm.entrypoints.pooling.typing import PoolingServeContext


class TestResolveTruncation:
    """Unit tests for EmbedIOProcessor._resolve_cohere_truncation."""

    @staticmethod
    def _make_request(**kwargs) -> CohereEmbedRequest:
        defaults = {
            "model": "test",
            "input_type": "search_document",
            "texts": ["hello"],
        }
        return CohereEmbedRequest(**(defaults | kwargs))

    def test_truncate_end_default(self):
        req = self._make_request()
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens == -1
        assert side is None

    def test_truncate_end_explicit(self):
        req = self._make_request(truncate="END")
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens == -1
        assert side is None

    def test_truncate_end_with_max_tokens(self):
        req = self._make_request(truncate="END", max_tokens=128)
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens == 128
        assert side is None

    def test_truncate_none(self):
        req = self._make_request(truncate="NONE")
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens is None
        assert side is None

    def test_truncate_none_with_max_tokens(self):
        """truncate=NONE should NOT set truncate_prompt_tokens; the
        max_tokens limit is enforced separately via _check_max_tokens."""
        req = self._make_request(truncate="NONE", max_tokens=10)
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens is None
        assert side is None

    def test_truncate_start(self):
        req = self._make_request(truncate="START")
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens == -1
        assert side == "left"

    def test_truncate_start_with_max_tokens(self):
        req = self._make_request(truncate="START", max_tokens=64)
        tokens, side = EmbedIOProcessor._resolve_cohere_truncation(req)
        assert tokens == 64
        assert side == "left"


class TestApplyStPrompt:
    """Unit tests for EmbedIOProcessor._apply_task_instruction."""

    @staticmethod
    def _make_handler(task_instructions: dict[str, str] | None):
        handler = object.__new__(EmbedIOProcessor)
        handler.task_instructions = task_instructions
        return handler

    def test_no_prompts_configured(self):
        handler = self._make_handler(None)
        texts = ["hello", "world"]
        assert handler._apply_task_instruction(texts, "query") is texts

    def test_matching_input_type(self):
        handler = self._make_handler({"query": "search_query: "})
        result = handler._apply_task_instruction(["hello"], "query")
        assert result == ["search_query: hello"]

    def test_non_matching_input_type(self):
        handler = self._make_handler({"query": "search_query: "})
        texts = ["hello"]
        assert handler._apply_task_instruction(texts, "document") is texts

    def test_multiple_texts(self):
        handler = self._make_handler(
            {"query": "Represent this sentence for searching: "}
        )
        result = handler._apply_task_instruction(["a", "b", "c"], "query")
        assert result == [
            "Represent this sentence for searching: a",
            "Represent this sentence for searching: b",
            "Represent this sentence for searching: c",
        ]

    def test_empty_prefix_returns_unchanged(self):
        handler = self._make_handler({"passage": ""})
        texts = ["hello"]
        assert handler._apply_task_instruction(texts, "passage") is texts


class TestLoadTaskInstructions:
    """Unit tests for EmbedIOProcessor._load_task_instructions."""

    def test_no_attribute(self):
        class FakeConfig:
            pass

        assert EmbedIOProcessor._load_task_instructions(FakeConfig()) is None

    def test_with_task_instructions(self):
        class FakeConfig:
            task_instructions = {
                "retrieval.query": "Represent the query: ",
                "retrieval.passage": "",
            }

        result = EmbedIOProcessor._load_task_instructions(FakeConfig())
        assert result == {
            "retrieval.query": "Represent the query: ",
            "retrieval.passage": "",
        }

    def test_empty_dict(self):
        class FakeConfig:
            task_instructions = {}

        assert EmbedIOProcessor._load_task_instructions(FakeConfig()) is None

    def test_non_dict(self):
        class FakeConfig:
            task_instructions = "not a dict"

        assert EmbedIOProcessor._load_task_instructions(FakeConfig()) is None


class TestCheckMaxTokens:
    """Unit tests for EmbedIOProcessor._check_cohere_max_tokens."""

    @staticmethod
    def _fake_output(n_tokens: int):
        class _Out:
            def __init__(self, n: int):
                self.prompt_token_ids = list(range(n))

        return _Out(n_tokens)

    def test_none_check_is_noop(self):
        outs = [self._fake_output(100)]
        EmbedIOProcessor._check_cohere_max_tokens(outs, None)

    def test_within_limit(self):
        outs = [self._fake_output(5), self._fake_output(3)]
        EmbedIOProcessor._check_cohere_max_tokens(outs, 5)

    def test_exceeds_limit(self):
        outs = [self._fake_output(3), self._fake_output(10)]
        with pytest.raises(ValueError, match="exceeds max_tokens=5"):
            EmbedIOProcessor._check_cohere_max_tokens(outs, 5)

    def test_exact_limit(self):
        outs = [self._fake_output(5)]
        EmbedIOProcessor._check_cohere_max_tokens(outs, 5)


class TestValidateInputType:
    """Unit tests for EmbedIOProcessor._validate_input_type."""

    @staticmethod
    def _make_handler(task_instructions: dict[str, str] | None):
        handler = object.__new__(EmbedIOProcessor)
        handler.task_instructions = task_instructions
        return handler

    def test_none_input_type_always_accepted(self):
        handler = self._make_handler(None)
        handler._validate_input_type(None)
        handler_with = self._make_handler({"query": "q: "})
        handler_with._validate_input_type(None)

    def test_no_prompts_rejects(self):
        handler = self._make_handler(None)
        with pytest.raises(ValueError, match="does not define any input_type"):
            handler._validate_input_type("anything")

    def test_known_type_accepted(self):
        handler = self._make_handler({"query": "q: ", "document": "d: "})
        handler._validate_input_type("query")
        handler._validate_input_type("document")

    def test_unknown_type_rejected(self):
        handler = self._make_handler({"query": "q: ", "document": "d: "})
        with pytest.raises(ValueError, match="Unsupported input_type 'other'"):
            handler._validate_input_type("other")

    def test_error_lists_supported(self):
        handler = self._make_handler({"a": "", "b": ""})
        with pytest.raises(ValueError, match="Supported values: a, b"):
            handler._validate_input_type("z")


class TestPreProcessCohereOnline:
    """Unit tests for EmbedIOProcessor._pre_process_cohere_online."""

    @staticmethod
    def _make_context(**request_kwargs) -> PoolingServeContext[CohereEmbedRequest]:
        return PoolingServeContext(
            request=CohereEmbedRequest(model="test", **request_kwargs),
            pooling_params=PoolingParams(),
            model_name="test",
            request_id="embd-test",
        )

    @staticmethod
    def _make_handler():
        handler = object.__new__(EmbedIOProcessor)
        handler._validate_input_type = lambda _input_type: None
        return handler

    def test_text_only_without_task_prefix_uses_completion_path(self):
        handler = self._make_handler()
        ctx = self._make_context(texts=["hello"])
        calls: list[tuple[str, object]] = []

        def preprocess_cmpl_online(request, prompt_input, prompt_embeds):
            calls.append(("completion", prompt_input))
            return ["completion"]

        handler._get_task_instruction_prefix = lambda _input_type: None
        handler._has_chat_template = lambda: False
        handler._preprocess_cmpl_online = preprocess_cmpl_online
        handler._batch_render_chat = lambda *_args, **_kwargs: (
            pytest.fail("text-only request should not require chat rendering")
        )

        handler._pre_process_cohere_online(ctx)

        assert ctx.engine_inputs == ["completion"]
        assert calls == [("completion", ["hello"])]

    def test_text_only_falls_back_to_prefixed_completion_without_template(self):
        handler = self._make_handler()
        ctx = self._make_context(texts=["hello"], input_type="query")
        calls: list[tuple[str, object]] = []

        def preprocess_cmpl(request, prompt_input, prompt_embeds):
            calls.append(("completion", prompt_input))
            return ["fallback"]

        handler._get_task_instruction_prefix = lambda _input_type: "query: "
        handler._has_chat_template = lambda: False
        handler._batch_render_chat = lambda *_args, **_kwargs: (
            pytest.fail("chat rendering should be skipped without a template")
        )
        handler._preprocess_cmpl_online = preprocess_cmpl

        handler._pre_process_cohere_online(ctx)

        assert ctx.engine_inputs == ["fallback"]
        assert calls == [("completion", ["query: hello"])]

    def test_text_only_with_template_uses_chat_path(self):
        handler = self._make_handler()
        ctx = self._make_context(texts=["hello"], input_type="query")
        calls: list[tuple[str, object]] = []

        def batch_render_chat(
            request,
            all_messages,
            truncate_prompt_tokens,
            truncation_side,
        ):
            calls.append(
                (
                    "chat",
                    {
                        "request": request,
                        "all_messages": all_messages,
                        "truncate_prompt_tokens": truncate_prompt_tokens,
                        "truncation_side": truncation_side,
                    },
                )
            )
            return ["chat"]

        handler._get_task_instruction_prefix = lambda _input_type: "query: "
        handler._has_chat_template = lambda: True
        handler._batch_render_chat = batch_render_chat
        handler._preprocess_cmpl_online = lambda *_args, **_kwargs: (
            pytest.fail("completion path should be skipped when a template exists")
        )

        handler._pre_process_cohere_online(ctx)

        assert ctx.engine_inputs == ["chat"]
        assert calls == [
            (
                "chat",
                {
                    "request": ctx.request,
                    "all_messages": [
                        handler._mixed_input_to_messages(
                            CohereEmbedInput(
                                content=[CohereEmbedContent(type="text", text="hello")]
                            ),
                            task_prefix="query: ",
                        )
                    ],
                    "truncate_prompt_tokens": -1,
                    "truncation_side": None,
                },
            )
        ]


class TestPreProcessChatOnline:
    """Unit tests for EmbedIOProcessor chat request batch splitting.

    Each message in an EmbeddingChatRequest should produce a separate
    engine input (and therefore a separate embedding in the response).
    """

    @staticmethod
    def _make_context(
        messages, **request_kwargs
    ) -> PoolingServeContext[EmbeddingChatRequest]:
        return PoolingServeContext(
            request=EmbeddingChatRequest(
                model="test",
                messages=messages,
                **request_kwargs,
            ),
            model_name="test",
            request_id="embd-test",
        )

    @staticmethod
    def _make_handler(*, enable_chunked_processing=False):
        handler = object.__new__(EmbedIOProcessor)
        handler.enable_chunked_processing = enable_chunked_processing
        handler.trust_request_chat_template = False
        return handler

    def test_multiple_messages_produce_multiple_engine_inputs(self):
        """3 messages should produce 3 engine_inputs, not 1."""
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        handler = self._make_handler()
        ctx = self._make_context(messages)

        captured_conversations: list[list] = []

        def fake_pre_process_chat_online(ctx_arg):
            # Simulate the expected behaviour: split messages and render
            all_messages = [[msg] for msg in ctx_arg.request.messages]
            captured_conversations.extend(all_messages)
            ctx_arg.engine_inputs = [
                f"engine_input_{i}" for i in range(len(all_messages))
            ]

        handler._pre_process_chat_online = fake_pre_process_chat_online

        handler.pre_process_online(ctx)

        assert len(ctx.engine_inputs) == 3
        assert ctx.engine_inputs == [
            "engine_input_0",
            "engine_input_1",
            "engine_input_2",
        ]
        # Each conversation should be a single-message list
        assert captured_conversations == [
            [messages[0]],
            [messages[1]],
            [messages[2]],
        ]

    def test_single_message_produces_one_engine_input(self):
        """A single message should produce exactly 1 engine_input."""
        messages = [{"role": "user", "content": "hello"}]
        handler = self._make_handler()
        ctx = self._make_context(messages)

        calls: list[list] = []

        def fake_pre_process_chat_online(ctx_arg):
            all_messages = [[msg] for msg in ctx_arg.request.messages]
            calls.append(all_messages)
            ctx_arg.engine_inputs = ["engine_input_0"]

        handler._pre_process_chat_online = fake_pre_process_chat_online

        handler.pre_process_online(ctx)

        assert len(ctx.engine_inputs) == 1
        assert ctx.engine_inputs == ["engine_input_0"]
        assert calls == [[[messages[0]]]]

    def test_untrusted_chat_template_is_rejected(self):
        """A request-level chat_template must be rejected when
        trust_request_chat_template is False."""
        messages = [{"role": "user", "content": "hello"}]
        handler = self._make_handler()
        ctx = self._make_context(
            messages,
            chat_template=("{% for msg in messages %}{{ msg.content }}{% endfor %}"),
        )

        # _pre_process_chat_online calls _validate_chat_template, which
        # is inherited from the base class and checks the flag.
        with pytest.raises(ValueError, match="trust-request-chat-template"):
            handler.pre_process_online(ctx)

    def test_prompt_extras_are_passed_through(self):
        """mm_processor_kwargs and cache_salt should reach render_chat."""
        messages = [{"role": "user", "content": "hello"}]
        handler = self._make_handler()
        ctx = self._make_context(
            messages,
            mm_processor_kwargs={"crop_size": 224},
            cache_salt="random-salt",
        )

        captured_extras: dict = {}

        class FakeRenderer:
            tokenizer = None

            def render_chat(
                self, conversations, chat_params, tok_params, *, prompt_extras=None
            ):
                captured_extras.update(prompt_extras or {})
                return ([[]], ["engine_input_0"])

        class FakeModelConfig:
            max_model_len = 512
            pooler_config = None
            encoder_config = None
            multimodal_config = None

        handler.renderer = FakeRenderer()
        handler.model_config = FakeModelConfig()
        handler.chat_template = None
        handler.chat_template_content_format = "auto"

        handler._pre_process_chat_online(ctx)

        assert captured_extras == {
            "mm_processor_kwargs": {"crop_size": 224},
            "cache_salt": "random-salt",
        }

    def test_chunked_processing_runs_after_chat_preprocessing(self):
        """enable_chunked_processing should still trigger
        _pre_process_chunked after the chat path."""
        messages = [{"role": "user", "content": "hello"}]
        handler = self._make_handler(enable_chunked_processing=True)
        ctx = self._make_context(messages)

        chunked_calls: list[bool] = []

        def fake_pre_process_chat_online(ctx_arg):
            ctx_arg.engine_inputs = ["engine_input_0"]

        def fake_pre_process_chunked(ctx_arg):
            chunked_calls.append(True)

        handler._pre_process_chat_online = fake_pre_process_chat_online
        handler._pre_process_chunked = fake_pre_process_chunked

        handler.pre_process_online(ctx)

        assert chunked_calls == [True]
