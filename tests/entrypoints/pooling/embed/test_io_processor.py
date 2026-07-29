# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EmbedIOProcessor."""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from vllm import PoolingParams
from vllm.entrypoints.pooling.embed.io_processor import EmbedIOProcessor
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    EmbeddingBatchChatInputRequest,
    EmbeddingBatchChatRequest,
    EmbeddingChatInputRequest,
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
)
from vllm.entrypoints.pooling.typing import (
    PoolingEngineInput,
    PoolingServeContext,
)
from vllm.outputs import PoolingOutput, PoolingRequestOutput


class TestEmbeddingRequestParsing:
    """Unit tests for OpenAI embedding request parsing."""

    def test_input_messages_parses_as_chat_request(self):
        request = TypeAdapter(EmbeddingRequest).validate_python(
            {
                "model": "test",
                "input": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"instruction": "Represent the query: "},
            }
        )

        assert isinstance(request, EmbeddingChatInputRequest)
        assert request.input == [{"role": "user", "content": "hello"}]
        assert request.messages == [{"role": "user", "content": "hello"}]
        assert request.chat_template_kwargs == {"instruction": "Represent the query: "}

    def test_batched_input_messages_parses_as_batch_chat_input_request(self):
        request = TypeAdapter(EmbeddingRequest).validate_python(
            {
                "model": "test",
                "input": [
                    [{"role": "user", "content": "hello"}],
                    [{"role": "user", "content": "goodbye"}],
                ],
                "chat_template_kwargs": {"instruction": "Represent the query: "},
            }
        )

        assert isinstance(request, EmbeddingBatchChatInputRequest)
        assert request.input == [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "goodbye"}],
        ]
        assert request.messages == [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "goodbye"}],
        ]
        assert request.chat_template_kwargs == {"instruction": "Represent the query: "}

    def test_token_ids_still_parse_as_completion_request(self):
        request = TypeAdapter(EmbeddingRequest).validate_python(
            {
                "model": "test",
                "input": [[1, 2, 3], [4, 5]],
            }
        )

        assert isinstance(request, EmbeddingCompletionRequest)
        assert request.input == [[1, 2, 3], [4, 5]]

    def test_messages_still_parses_as_chat_request(self):
        request = TypeAdapter(EmbeddingRequest).validate_python(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"instruction": "Represent the query: "},
            }
        )

        assert isinstance(request, EmbeddingChatRequest)
        assert request.messages == [{"role": "user", "content": "hello"}]
        assert request.chat_template_kwargs == {"instruction": "Represent the query: "}

    def test_batched_messages_parses_as_batch_chat_request(self):
        request = TypeAdapter(EmbeddingRequest).validate_python(
            {
                "model": "test",
                "messages": [
                    [{"role": "user", "content": "hello"}],
                    [{"role": "user", "content": "goodbye"}],
                ],
                "chat_template_kwargs": {"instruction": "Represent the query: "},
            }
        )

        assert isinstance(request, EmbeddingBatchChatRequest)
        assert request.messages == [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "goodbye"}],
        ]
        assert request.chat_template_kwargs == {"instruction": "Represent the query: "}


class TestCohereEmbedRequestParsing:
    """Unit tests for Cohere embed request parsing."""

    @pytest.mark.parametrize(
        "request_body",
        [
            {"model": "test"},
            {"model": "test", "texts": ["hello"], "images": ["image-uri"]},
            {
                "model": "test",
                "texts": ["hello"],
                "inputs": [
                    {"content": [{"type": "text", "text": "hello"}]},
                ],
            },
            {
                "model": "test",
                "images": ["image-uri"],
                "inputs": [
                    {"content": [{"type": "text", "text": "hello"}]},
                ],
            },
            {"model": "test", "texts": []},
            {"model": "test", "images": []},
            {"model": "test", "inputs": []},
        ],
    )
    def test_rejects_invalid_input_field_combinations(self, request_body):
        with pytest.raises(
            ValidationError,
            match="Exactly one of texts, images, or inputs must be provided",
        ):
            CohereEmbedRequest(**request_body)

    @pytest.mark.parametrize(
        "request_body",
        [
            {"model": "test", "texts": ["hello"]},
            {"model": "test", "images": ["image-uri"]},
            {
                "model": "test",
                "inputs": [
                    {"content": [{"type": "text", "text": "hello"}]},
                ],
            },
            {
                "model": "test",
                "inputs": [
                    {
                        "content": [
                            {"type": "image_url", "image_url": {"url": "image-uri"}}
                        ]
                    },
                ],
            },
        ],
    )
    def test_accepts_exactly_one_non_empty_input_field(self, request_body):
        request = CohereEmbedRequest(**request_body)

        assert request.model == "test"

    @pytest.mark.parametrize(
        ("content", "error"),
        [
            (
                {"type": "text"},
                "CohereEmbedContent with type='text' requires text",
            ),
            (
                {"type": "image_url"},
                "CohereEmbedContent with type='image_url' requires image_url.url",
            ),
            (
                {"type": "image_url", "image_url": {}},
                "CohereEmbedContent with type='image_url' requires image_url.url",
            ),
            (
                {"type": "image_url", "image_url": {"url": ""}},
                "CohereEmbedContent with type='image_url' requires image_url.url",
            ),
        ],
    )
    def test_rejects_invalid_mixed_content_payloads(self, content, error):
        with pytest.raises(ValidationError, match=error):
            CohereEmbedRequest(
                model="test",
                inputs=[
                    {
                        "content": [content],
                    },
                ],
            )


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


class TestChunkedEmbeddingProcessing:
    """Unit tests for chunked embedding aggregation."""

    class _FakeModelConfig:
        max_model_len = 3

    @classmethod
    def _make_handler(cls):
        handler = object.__new__(EmbedIOProcessor)
        handler.model_config = cls._FakeModelConfig()
        handler.enable_chunked_processing = True
        return handler

    @staticmethod
    def _make_context() -> PoolingServeContext[EmbeddingCompletionRequest]:
        request = TypeAdapter(EmbeddingRequest).validate_python(
            {
                "model": "test",
                "input": [[0, 1, 2, 3, 4], [10, 11]],
            }
        )
        assert isinstance(request, EmbeddingCompletionRequest)
        pooling_params = PoolingParams()
        return PoolingServeContext(
            request=request,
            pooling_params=pooling_params,
            model_name="test",
            request_id="embd-client-prompt-999-chunk-888",
            engine_inputs=[
                PoolingEngineInput(
                    prompts={"prompt_token_ids": [0, 1, 2, 3, 4]},
                    params=pooling_params,
                    lora_requests=None,
                    priorities=0,
                ),
                PoolingEngineInput(
                    prompts={"prompt_token_ids": [10, 11]},
                    params=pooling_params,
                    lora_requests=None,
                    priorities=0,
                ),
            ],
            lora_request=None,
            priorities=0,
            prompt_extras=None,
        )

    @staticmethod
    def _make_output(
        request_id: str,
        prompt_token_ids: list[int],
        embedding: list[float],
    ) -> PoolingRequestOutput:
        return PoolingRequestOutput(
            request_id=request_id,
            outputs=PoolingOutput(data=torch.tensor(embedding)),
            prompt_token_ids=prompt_token_ids,
            num_cached_tokens=0,
            finished=True,
        )

    def test_aggregation_uses_metadata_not_request_id_parsing(self):
        handler = self._make_handler()
        ctx = self._make_context()

        handler.maybe_pre_process_chunked(ctx)

        assert ctx.prompt_request_ids == [
            "embd-client-prompt-999-chunk-888-prompt-0-chunk-0",
            "embd-client-prompt-999-chunk-888-prompt-0-chunk-1",
            "embd-client-prompt-999-chunk-888-prompt-1-chunk-0",
        ]
        assert ctx.chunked_embedding_metadata is not None
        assert [
            (item.prompt_index, item.chunk_index)
            for item in ctx.chunked_embedding_metadata
        ] == [(0, 0), (0, 1), (1, 0)]

        ctx.final_res_batch = [
            self._make_output(ctx.prompt_request_ids[0], [0, 1, 2], [1.0, 1.0]),
            self._make_output(ctx.prompt_request_ids[1], [3, 4], [4.0, 7.0]),
            self._make_output(ctx.prompt_request_ids[2], [10, 11], [9.0, 9.0]),
        ]

        handler._post_process_chunked(ctx)

        assert len(ctx.final_res_batch) == 2
        assert ctx.final_res_batch[0].request_id == (
            "embd-client-prompt-999-chunk-888-prompt-0"
        )
        assert ctx.final_res_batch[0].prompt_token_ids == [0, 1, 2, 3, 4]
        assert torch.allclose(
            ctx.final_res_batch[0].outputs.data,
            torch.tensor([2.2, 3.4]),
        )
        assert ctx.final_res_batch[1].request_id == (
            "embd-client-prompt-999-chunk-888-prompt-1"
        )
        assert ctx.final_res_batch[1].prompt_token_ids == [10, 11]
        assert torch.allclose(
            ctx.final_res_batch[1].outputs.data,
            torch.tensor([9.0, 9.0]),
        )
