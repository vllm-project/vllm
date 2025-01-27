import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse, RerankDocument,
                                              RerankRequest, RerankResponse,
                                              RerankResult, RerankUsage)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import make_async, merge_async_iterators

logger = init_logger(__name__)


class JinaAIServingRerank(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

    async def do_rerank(
        self,
        request: RerankRequest,
        raw_request: Optional[Request] = None
    ) -> Union[RerankResponse, ErrorResponse]:
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

        model_name = request.model
        request_id = f"rerank-{self._base_request_id(raw_request)}"
        truncate_prompt_tokens = request.truncate_prompt_tokens
        query = request.query
        documents = request.documents
        request_prompts = []
        engine_prompts = []
        top_n = request.top_n if request.top_n > 0 else len(documents)

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for scoring models")

            if isinstance(tokenizer, MistralTokenizer):
                raise ValueError(
                    "MistralTokenizer not supported for cross-encoding")

            if not self.model_config.is_cross_encoder:
                raise ValueError("Model is not cross encoder.")

            if truncate_prompt_tokens is not None and \
                    truncate_prompt_tokens > self.max_model_len:
                raise ValueError(
                    f"truncate_prompt_tokens value ({truncate_prompt_tokens}) "
                    f"is greater than max_model_len ({self.max_model_len})."
                    f" Please, select a smaller truncation size.")
            for doc in documents:
                request_prompt = f"{query}{tokenizer.sep_token}{doc}"
                tokenization_kwargs: Dict[str, Any] = {}
                if truncate_prompt_tokens is not None:
                    tokenization_kwargs["truncation"] = True
                    tokenization_kwargs["max_length"] = truncate_prompt_tokens

                tokenize_async = make_async(tokenizer.__call__,
                                            executor=self._tokenizer_executor)
                prompt_inputs = await tokenize_async(text=query,
                                                     text_pair=doc,
                                                     **tokenization_kwargs)

                input_ids = prompt_inputs["input_ids"]
                text_token_prompt = \
                    self._validate_input(request, input_ids, request_prompt)
                engine_prompt = TokensPrompt(
                    prompt_token_ids=text_token_prompt["prompt_token_ids"],
                    token_type_ids=prompt_inputs.get("token_type_ids"))

                request_prompts.append(request_prompt)
                engine_prompts.append(engine_prompt)

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []

        try:
            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
        result_generator = merge_async_iterators(*generators)

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: List[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts

        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(List[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_rerank_response(
                final_res_batch_checked, request_id, model_name, documents,
                top_n)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def request_output_to_rerank_response(
            self, final_res_batch: List[PoolingRequestOutput], request_id: str,
            model_name: str, documents: List[str],
            top_n: int) -> RerankResponse:
        """
        Convert the output of do_rank to a RerankResponse
        """
        results: List[RerankResult] = []
        num_prompt_tokens = 0
        for idx, final_res in enumerate(final_res_batch):
            classify_res = ScoringRequestOutput.from_base(final_res)

            result = RerankResult(
                index=idx,
                document=RerankDocument(text=documents[idx]),
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
            usage=RerankUsage(total_tokens=num_prompt_tokens))
