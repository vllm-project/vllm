# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartTextParam,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.score.protocol import (
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
    ScoreRequest,
    ScoreResponse,
    ScoreResponseData,
)
from vllm.entrypoints.pooling.score.utils import (
    ScoreContentPartParam,
    ScoreMultiModalParam,
    _cosine_similarity,
    _validate_score_input_lens,
    compress_token_type_ids,
    get_score_prompt,
)
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.async_utils import make_async, merge_async_iterators

logger = init_logger(__name__)


class ServingScores(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        score_template: str | None = None,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )
        self.score_template = score_template

        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)

    def _convert_documents_to_multimodal(
        self,
        documents: list[str] | list[dict[str, Any]] | ScoreMultiModalParam,
    ) -> list[str] | list[ScoreContentPartParam] | list[list[ScoreContentPartParam]]:
        """
        Convert documents in various formats to a standardized list format.

        Args:
            documents: Can be:
                - list[str]: Simple text documents
                - list[dict[str, Any]]: Documents with
                  {"text": "...", "image": "..."} format
                - ScoreMultiModalParam: Already in multimodal format

        Returns:
            list[str] or list[ScoreContentPartParam] or
            list[list[ScoreContentPartParam]]: Standardized document
            list. Returns list[list[ScoreContentPartParam]] when
            documents contain multiple parts (text + image) to
            preserve document boundaries.
        """
        if isinstance(documents, dict) and "content" in documents:
            # Already in ScoreMultiModalParam format
            return documents["content"]  # type: ignore[return-value]

        if not isinstance(documents, list):
            return documents  # type: ignore[return-value]

        # Check if it's a list of dicts with text/image keys
        if documents and isinstance(documents[0], dict):
            converted_docs: list[list[ScoreContentPartParam]] = []
            has_multi_part_docs = False

            for doc in documents:
                if not isinstance(doc, dict):
                    raise ValueError(
                        f"Expected dict in documents list, got {type(doc)}"
                    )

                doc_parts: list[ScoreContentPartParam] = []

                # Handle text field
                if "text" in doc:
                    from vllm.entrypoints.chat_utils import (
                        ChatCompletionContentPartTextParam,
                    )

                    doc_parts.append(
                        ChatCompletionContentPartTextParam(
                            type="text", text=doc["text"]
                        )
                    )

                # Handle image field
                if "image" in doc:
                    from vllm.entrypoints.chat_utils import (
                        ChatCompletionContentPartImageParam,
                    )

                    image_url = doc["image"]
                    doc_parts.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url={"url": image_url},
                        )
                    )

                # Track if any document has multiple parts
                if len(doc_parts) > 1:
                    has_multi_part_docs = True

                # Keep document as a single unit (list of parts)
                converted_docs.append(doc_parts)

            # If we have multi-part documents, return nested structure
            # Otherwise, flatten for backward compatibility with single-part documents
            if has_multi_part_docs:
                return converted_docs
            else:
                # Flatten single-part documents for backward compatibility
                return [part for doc in converted_docs for part in doc]

        # It's a simple list of strings
        return documents  # type: ignore[return-value]

    async def _embedding_score(
        self,
        tokenizer: TokenizerLike,
        data_1: list[str]
        | list[ScoreContentPartParam]
        | list[list[ScoreContentPartParam]],
        data_2: list[str]
        | list[ScoreContentPartParam]
        | list[list[ScoreContentPartParam]],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        tokenization_kwargs: dict[str, Any] | None = None,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        # Flatten nested documents if present (for multimodal inputs)
        # Embedding scoring typically works with text, so we extract
        # text from multimodal docs
        def extract_text(
            item: str | ScoreContentPartParam | list[ScoreContentPartParam],
        ) -> str:
            if isinstance(item, str):
                return item
            elif isinstance(item, list):
                # Multi-part document: extract text parts
                for part in item:
                    if isinstance(part, ChatCompletionContentPartTextParam):
                        return part.text
                # If no text part, return empty string
                return ""
            elif isinstance(item, ChatCompletionContentPartTextParam):
                return item.text
            else:
                # Other types (image, video) don't have text
                return ""

        # Convert to text for embedding scoring
        text_data_1 = [extract_text(item) for item in data_1]
        text_data_2 = [extract_text(item) for item in data_2]

        input_texts = text_data_1 + text_data_2

        engine_prompts: list[TokensPrompt] = []
        tokenize_async = make_async(
            tokenizer.__call__, executor=self._tokenizer_executor
        )

        tokenization_kwargs = tokenization_kwargs or {}
        tokenized_prompts = await asyncio.gather(
            *(tokenize_async(t, **tokenization_kwargs) for t in input_texts)
        )

        for tok_result, input_text in zip(tokenized_prompts, input_texts):
            text_token_prompt = self._validate_input(
                request, tok_result["input_ids"], input_text
            )

            engine_prompts.append(
                TokensPrompt(prompt_token_ids=text_token_prompt["prompt_token_ids"])
            )

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []
        pooling_params = request.to_pooling_params()

        try:
            pooling_params.verify("embed", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                input_texts[i],
                params=pooling_params,
                lora_request=lora_request,
            )

            generators.append(
                self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )
            )

        result_generator = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: list[PoolingRequestOutput] = []

        embeddings: list[PoolingRequestOutput | None] = [None] * len(engine_prompts)

        async for i, res in result_generator:
            embeddings[i] = res

        emb_data_1: list[PoolingRequestOutput] = []
        emb_data_2: list[PoolingRequestOutput] = []

        for i in range(0, len(data_1)):
            assert (emb := embeddings[i]) is not None
            emb_data_1.append(emb)

        for i in range(len(data_1), len(embeddings)):
            assert (emb := embeddings[i]) is not None
            emb_data_2.append(emb)

        if len(emb_data_1) == 1:
            emb_data_1 = emb_data_1 * len(emb_data_2)

        final_res_batch = _cosine_similarity(
            tokenizer=tokenizer, embed_1=emb_data_1, embed_2=emb_data_2
        )

        return final_res_batch

    def _preprocess_score(
        self,
        request: RerankRequest | ScoreRequest,
        tokenizer: TokenizerLike,
        tokenization_kwargs: dict[str, Any],
        data_1: str | ScoreContentPartParam,
        data_2: str | ScoreContentPartParam,
    ) -> tuple[str, TokensPrompt]:
        model_config = self.model_config

        full_prompt, engine_prompt = get_score_prompt(
            model_config=model_config,
            data_1=data_1,
            data_2=data_2,
            tokenizer=tokenizer,
            tokenization_kwargs=tokenization_kwargs,
            score_template=self.score_template,
        )
        self._validate_input(request, engine_prompt["prompt_token_ids"], full_prompt)
        if request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        return full_prompt, engine_prompt

    async def _cross_encoding_score(
        self,
        tokenizer: TokenizerLike,
        data_1: list[str]
        | list[ScoreContentPartParam]
        | list[list[ScoreContentPartParam]],
        data_2: list[str]
        | list[ScoreContentPartParam]
        | list[list[ScoreContentPartParam]],
        request: RerankRequest | ScoreRequest,
        request_id: str,
        tokenization_kwargs: dict[str, Any] | None = None,
        lora_request: LoRARequest | None | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        request_prompts: list[str] = []
        engine_prompts: list[TokensPrompt] = []

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)

        if isinstance(tokenizer, MistralTokenizer):
            raise ValueError("MistralTokenizer not supported for cross-encoding")

        tokenization_kwargs = tokenization_kwargs or {}

        # Flatten nested documents for scoring
        # A nested document [text_part, image_part] becomes a single item for scoring
        def flatten_if_nested(
            item: str | ScoreContentPartParam | list[ScoreContentPartParam],
        ) -> str | ScoreContentPartParam:
            if isinstance(item, list):
                # Multi-part document: merge into a single representation
                # For now, we just return the first part as the representative
                # This maintains the document as a single unit in the scoring pipeline
                # TODO: Future enhancement could create a composite representation
                if len(item) == 1:
                    return item[0]
                # For multi-part, we need to handle this specially
                # For cross-encoding, we'll use the first text part if available
                for part in item:
                    if isinstance(part, ChatCompletionContentPartTextParam):
                        return part
                # If no text part, return first part
                return item[0]
            return item

        flattened_data_1 = [flatten_if_nested(item) for item in data_1]
        flattened_data_2 = [flatten_if_nested(item) for item in data_2]

        input_pairs = [(t1, t2) for t1, t2 in zip(flattened_data_1, flattened_data_2)]

        preprocess_async = make_async(
            self._preprocess_score, executor=self._tokenizer_executor
        )

        preprocessed_prompts = await asyncio.gather(
            *(
                preprocess_async(
                    request=request,
                    tokenizer=tokenizer,
                    tokenization_kwargs=tokenization_kwargs,
                    data_1=t1,
                    data_2=t2,
                )
                for t1, t2 in input_pairs
            )
        )

        for full_prompt, engine_prompt in preprocessed_prompts:
            request_prompts.append(full_prompt)
            engine_prompts.append(engine_prompt)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[PoolingRequestOutput, None]] = []

        default_pooling_params = request.to_pooling_params()

        try:
            default_pooling_params.verify("score", self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                request_prompts[i],
                params=default_pooling_params,
                lora_request=lora_request,
            )

            if token_type_ids := engine_prompt.pop("token_type_ids", None):
                pooling_params = default_pooling_params.clone()
                compressed = compress_token_type_ids(token_type_ids)
                pooling_params.extra_kwargs = {"compressed_token_type_ids": compressed}
            else:
                pooling_params = default_pooling_params

            generator = self.engine_client.encode(
                engine_prompt,
                pooling_params,
                request_id_item,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
            )

            generators.append(generator)

        result_generator = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: list[PoolingRequestOutput | None] = [None] * len(
            engine_prompts
        )

        async for i, res in result_generator:
            final_res_batch[i] = res

        return [out for out in final_res_batch if out is not None]

    async def _run_scoring(
        self,
        data_1: list[str] | str | dict[str, Any] | ScoreMultiModalParam,
        data_2: list[str] | str | list[dict[str, Any]] | ScoreMultiModalParam,
        request: ScoreRequest | RerankRequest,
        request_id: str,
        raw_request: Request | None = None,
    ) -> list[PoolingRequestOutput] | ErrorResponse:
        lora_request = self._maybe_get_adapters(request)
        tokenizer = self.renderer.get_tokenizer()

        truncate_prompt_tokens = getattr(request, "truncate_prompt_tokens", None)

        tokenization_kwargs: dict[str, Any] = {}
        _validate_truncation_size(
            self.max_model_len, truncate_prompt_tokens, tokenization_kwargs
        )

        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        if not self.model_config.is_multimodal_model and (
            isinstance(data_1, dict) or isinstance(data_2, dict)
        ):
            raise ValueError(
                f"MultiModalParam is not supported for {self.model_config.architecture}"  # noqa: E501
            )

        # Process data_1 (query)
        if isinstance(data_1, str):
            data_1 = [data_1]
        elif isinstance(data_1, dict):
            if "content" in data_1:
                # ScoreMultiModalParam format
                data_1 = data_1.get("content")  # type: ignore[assignment]
            elif "text" in data_1 or "image" in data_1:
                # Single dict with {"text": "...", "image": "..."} format
                data_1 = self._convert_documents_to_multimodal([data_1])  # type: ignore[assignment]

        # Process data_2 (documents)
        if isinstance(data_2, str):
            data_2 = [data_2]
        elif isinstance(data_2, dict) and "content" in data_2:
            data_2 = data_2.get("content")  # type: ignore[assignment]
        elif isinstance(data_2, list) and data_2 and isinstance(data_2[0], dict):
            # Handle list[dict[str, Any]] format with {"text": "...", "image": "..."}
            data_2 = self._convert_documents_to_multimodal(data_2)  # type: ignore[assignment]

        _validate_score_input_lens(data_1, data_2)  # type: ignore[arg-type]

        if self.model_config.is_cross_encoder:
            return await self._cross_encoding_score(
                tokenizer=tokenizer,
                data_1=data_1,  # type: ignore[arg-type]
                data_2=data_2,  # type: ignore[arg-type]
                request=request,
                request_id=request_id,
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )

        else:
            return await self._embedding_score(
                tokenizer=tokenizer,
                data_1=data_1,  # type: ignore[arg-type]
                data_2=data_2,  # type: ignore[arg-type]
                request=request,
                request_id=request_id,
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )

    async def create_score(
        self,
        request: ScoreRequest,
        raw_request: Request | None = None,
    ) -> ScoreResponse | ErrorResponse:
        """
        Score API similar to Sentence Transformers cross encoder

        See https://sbert.net/docs/package_reference/cross_encoder
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"score-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        try:
            final_res_batch = await self._run_scoring(
                request.data_1,
                request.data_2,
                request,
                request_id,
                raw_request,
            )
            if isinstance(final_res_batch, ErrorResponse):
                return final_res_batch

            return self.request_output_to_score_response(
                final_res_batch,
                request_id,
                created_time,
                self.models.model_name(),
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def do_rerank(
        self, request: RerankRequest, raw_request: Request | None = None
    ) -> RerankResponse | ErrorResponse:
        """
        Rerank API based on JinaAI's rerank API; implements the same
        API interface. Designed for compatibility with off-the-shelf
        tooling, since this is a common standard for reranking APIs

        See example client implementations at
        https://github.com/infiniflow/ragflow/blob/main/rag/llm/rerank_model.py
        numerous clients use this standard.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"rerank-{self._base_request_id(raw_request)}"
        documents = request.documents
        top_n = (
            request.top_n
            if request.top_n > 0
            else (
                len(documents)
                if isinstance(documents, list)
                else len(documents["content"])
            )
        )

        try:
            final_res_batch = await self._run_scoring(
                request.query,
                documents,
                request,
                request_id,
                raw_request,
            )
            if isinstance(final_res_batch, ErrorResponse):
                return final_res_batch

            return self.request_output_to_rerank_response(
                final_res_batch,
                request_id,
                self.models.model_name(),
                documents,
                top_n,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def request_output_to_score_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> ScoreResponse:
        items: list[ScoreResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            classify_res = ScoringRequestOutput.from_base(final_res)

            item = ScoreResponseData(
                index=idx,
                score=classify_res.outputs.score,
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return ScoreResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )

    def request_output_to_rerank_response(
        self,
        final_res_batch: list[PoolingRequestOutput],
        request_id: str,
        model_name: str,
        documents: list[str] | list[dict[str, Any]] | ScoreMultiModalParam,
        top_n: int,
    ) -> RerankResponse:
        """
        Convert the output of do_rank to a RerankResponse
        """
        results: list[RerankResult] = []
        num_prompt_tokens = 0

        # Handle nested document structure (multi-part documents)
        # We need to track if documents were converted to nested structure
        converted_docs = self._convert_documents_to_multimodal(documents)

        for idx, final_res in enumerate(final_res_batch):
            classify_res = ScoringRequestOutput.from_base(final_res)

            # Determine document representation
            if isinstance(documents, dict) and "content" in documents:
                # ScoreMultiModalParam format
                document = RerankDocument(multi_modal=documents["content"][idx])
            elif isinstance(documents, list):
                if (
                    documents
                    and isinstance(documents[0], dict)
                    and ("text" in documents[0] or "image" in documents[0])
                ):
                    # list[dict[str, Any]] format with
                    # {"text": "...", "image": "..."}
                    doc_dict = documents[idx]
                    if isinstance(doc_dict, dict):
                        text_content = doc_dict.get("text", "")
                    else:
                        text_content = ""

                    # Build multi_modal parts list
                    doc_parts: list[ScoreContentPartParam] = []
                    if text_content:
                        from vllm.entrypoints.chat_utils import (
                            ChatCompletionContentPartTextParam,
                        )

                        doc_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text", text=text_content
                            )
                        )
                    if "image" in doc_dict:
                        from vllm.entrypoints.chat_utils import (
                            ChatCompletionContentPartImageParam,
                        )

                        if isinstance(doc_dict, dict):
                            doc_parts.append(
                                ChatCompletionContentPartImageParam(
                                    type="image_url",
                                    image_url={"url": doc_dict["image"]},
                                )
                            )

                    # Store both text and multi_modal representation
                    if len(doc_parts) > 1:
                        # Multi-part document (text + image)
                        document = RerankDocument(
                            text=text_content, multi_modal=doc_parts
                        )
                    elif len(doc_parts) == 1:
                        # Single part document
                        if text_content:
                            document = RerankDocument(text=text_content)
                        else:
                            document = RerankDocument(multi_modal=doc_parts[0])
                    else:
                        raise ValueError(f"Document at index {idx} has no content")
                else:
                    # Simple list of strings
                    document = RerankDocument(text=documents[idx])
            else:
                raise ValueError(f"Unsupported documents format: {type(documents)}")

            result = RerankResult(
                index=idx,
                document=document,
                relevance_score=classify_res.outputs.score,
            )
            results.append(result)
            prompt_token_ids = final_res.prompt_token_ids
            num_prompt_tokens += len(prompt_token_ids)

        # sort by relevance, then return the top n if set
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if top_n < len(documents):
            results = results[:top_n]

        return RerankResponse(
            id=request_id,
            model=model_name,
            results=results,
            usage=RerankUsage(
                total_tokens=num_prompt_tokens, prompt_tokens=num_prompt_tokens
            ),
        )
