# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import asyncio
import time
from collections.abc import AsyncGenerator
from collections.abc import Sequence as GenericSequence

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing, clamp_prompt_logprobs
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.collection_utils import as_list

logger = init_logger(__name__)


class ServingTokens(OpenAIServing):
    """Provides Tokens IN <> Tokens OUT functionality to vLLM API."""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        force_no_detokenize: bool = False,
        return_tokens_as_token_ids: bool = False,
        log_error_stack: bool = False,
        enable_prompt_tokens_details: bool = False,
        enable_log_outputs: bool = False,
    ):
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
            log_error_stack=log_error_stack,
        )
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_log_outputs = enable_log_outputs
        self.force_no_detokenize = force_no_detokenize
        if force_no_detokenize:
            logger.info(
                "Tokens-only mode is enabled, skipping detokenization "
                "step for incoming requests."
            )

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        lora_request = None
        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        request_id = (
            f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # TODO(NickLucche): Change to EngineCoreRequest once Renderer work is
        # completed
        engine_prompts = await self._preprocess_completion(
            request,
            prompt_input=request.token_ids,
            prompt_embeds=None,
        )
        assert len(engine_prompts) == 1
        engine_prompt = engine_prompts[0]

        # Schedule the request and get the result generator.
        result_generator: AsyncGenerator[RequestOutput, None] | None = None
        try:
            sampling_params = request.sampling_params
            if self.force_no_detokenize:
                sampling_params.detokenize = False

            self._log_inputs(
                request_id,
                TokensPrompt(prompt_token_ids=request.token_ids),
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            tok_params = request.build_tok_params(self.model_config)
            tokenization_kwargs = tok_params.get_encode_kwargs()

            result_generator = self.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=request.priority,
            )

        except ValueError as e:
            return self.create_error_response(str(e))

        # TODO(NickLucche): Implement streaming response

        try:
            assert result_generator is not None
            return await self.serve_tokens_full_generator(
                request, result_generator, request_id, model_name, request_metadata
            )
        except ValueError as e:
            return self.create_error_response(str(e))

    async def serve_tokens_full_generator(
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None
        sampling_params: SamplingParams = request.sampling_params

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[GenerateResponseChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            # This is top_logprobs in completions API
            if sampling_params.logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_tokens_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=sampling_params.logprobs,
                )
            else:
                logprobs = None

            choice_data = GenerateResponseChoice(
                index=output.index,
                logprobs=logprobs,
                finish_reason=output.finish_reason if output.finish_reason else "stop",
                token_ids=as_list(output.token_ids),
            )

            choices.append(choice_data)
            num_generated_tokens += len(output.token_ids)

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            # This info is not available at the /coordinator level
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )

        request_metadata.final_usage_info = usage

        response = GenerateResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                # Get the corresponding output token IDs
                output_token_ids = None
                if choice.index < len(final_res.outputs):
                    output_token_ids = final_res.outputs[choice.index].token_ids

                if output_token_ids:
                    # Log token_ids only.
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs="",
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _create_tokens_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        num_output_top_logprobs: int | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        for i, token_id in enumerate(token_ids):
            token = f"token_id:{token_id}"
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        logprob=max(step_token.logprob, -9999.0),
                        top_logprobs=[
                            ChatCompletionLogProb(
                                token=token,
                                logprob=max(p[1].logprob, -9999.0),
                            )
                            for i, p in enumerate(step_top_logprobs.items())
                            if num_output_top_logprobs is not None
                            and i < max(num_output_top_logprobs, 1)
                        ],
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)
