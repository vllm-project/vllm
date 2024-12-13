import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (ErrorResponse, ScoreRequest,
                                              ScoreResponse, ScoreResponseData,
                                              UsageInfo)
from vllm.entrypoints.openai.serving_engine import BaseModelPath, OpenAIServing
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, ScoringRequestOutput
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import make_async, merge_async_iterators

logger = init_logger(__name__)


def request_output_to_score_response(
        final_res_batch: List[PoolingRequestOutput], request_id: str,
        created_time: int, model_name: str) -> ScoreResponse:
    data: List[ScoreResponseData] = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        classify_res = ScoringRequestOutput.from_base(final_res)

        score_data = ScoreResponseData(index=idx,
                                       score=classify_res.outputs.score)
        data.append(score_data)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return ScoreResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        data=data,
        usage=usage,
    )


def make_pairs(text_1: Union[List[str], str], text_2: Union[List[str],
                                                            str]) -> List:
    if isinstance(text_1, (str, dict)):
        # Convert a single prompt to a list.
        text_1 = [text_1]
    text_1 = [t for t in text_1]

    if isinstance(text_2, (str, dict)):
        # Convert a single prompt to a list.
        text_2 = [text_2]
    text_2 = [t for t in text_2]
    if len(text_1) > 1 and len(text_1) != len(text_2):
        raise ValueError("Input lengths must be either 1:1, 1:N or N:N")
    if len(text_1) == 0:
        raise ValueError("At least one text element must be given")
    if len(text_2) == 0:
        raise ValueError("At least one text_pair element must be given")

    if len(text_1) == 1:
        text_1 = text_1 * len(text_2)

    return [(t1, t2) for t1, t2 in zip(text_1, text_2)]


class OpenAIServingScores(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        *,
        request_logger: Optional[RequestLogger],
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=None,
                         prompt_adapters=None,
                         request_logger=request_logger)

    async def create_score(
        self,
        request: ScoreRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[ScoreResponse, ErrorResponse]:
        """
        Score API similar to Sentence Transformers cross encoder

        See https://sbert.net/docs/package_reference/cross_encoder
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        model_name = request.model
        request_id = f"score-{self._base_request_id(raw_request)}"
        created_time = int(time.monotonic())
        truncate_prompt_tokens = request.truncate_prompt_tokens

        request_prompts = []
        engine_prompts = []

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

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []

        input_pairs = make_pairs(request.text_1, request.text_2)

        for q, t in input_pairs:
            request_prompt = f"{q}{tokenizer.sep_token}{t}"

            tokenization_kwargs: Dict[str, Any] = {}
            if truncate_prompt_tokens is not None:
                tokenization_kwargs["truncation"] = True
                tokenization_kwargs["max_length"] = truncate_prompt_tokens

            tokenize_async = make_async(tokenizer.__call__,
                                        executor=self._tokenizer_executor)
            prompt_inputs = await tokenize_async(text=q,
                                                 text_pair=t,
                                                 **tokenization_kwargs)
            engine_prompt = TokensPrompt(
                prompt_token_ids=prompt_inputs["input_ids"],
                token_type_ids=prompt_inputs.get("token_type_ids"))

            request_prompts.append(request_prompt)
            engine_prompts.append(engine_prompt)

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

        result_generator = merge_async_iterators(
            *generators,
            is_cancelled=raw_request.is_disconnected if raw_request else None,
        )

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

            response = request_output_to_score_response(
                final_res_batch_checked, request_id, created_time, model_name)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response
