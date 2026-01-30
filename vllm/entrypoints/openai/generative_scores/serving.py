# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving class for the Generative Scores API.

This module implements the OpenAIServingGenerativeScores class which handles
requests to compute the probability of specified token IDs appearing as the
next token after a given query+item prompt.
"""

import asyncio
import math
import time
from collections.abc import AsyncGenerator, Mapping

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.generative_scores.protocol import (
    GenerativeScoreItemResult,
    GenerativeScoreRequest,
    GenerativeScoreResponse,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.inputs.data import TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import merge_async_iterators

logger = init_logger(__name__)


class OpenAIServingGenerativeScores(OpenAIServing):
    """Serving class for the Generative Scores API.

    This class handles computing the probability of specified token IDs 
    appearing as the next token after concatenating query and item prompts.

    The key operation is:
    1. For each item, build a prompt: query + item (or item + query if item_first)
    2. Run a forward pass to get the next token distribution
    3. Extract probabilities for the specified label_token_ids
    4. Normalize either over the full vocab (apply_softmax=False) or 
       over just the label_token_ids (apply_softmax=True)
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        log_error_stack: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            log_error_stack=log_error_stack,
        )

    async def create_generative_score(
        self,
        request: GenerativeScoreRequest,
        raw_request: Request | None = None,
    ) -> GenerativeScoreResponse | ErrorResponse:
        """Create generative scores for the given request.

        Args:
            request: The GenerativeScoreRequest containing query, items, and 
                label_token_ids.
            raw_request: The raw FastAPI request object.

        Returns:
            GenerativeScoreResponse with probabilities for each item, or
            ErrorResponse if an error occurred.
        """
        # Check model
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Check if engine is alive
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Get tokenizer
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            return self.create_error_response(
                "Tokenizer not available. Cannot process generative score request."
            )

        # Validate label_token_ids
        vocab_size = self.model_config.get_vocab_size()
        for token_id in request.label_token_ids:
            if token_id < 0 or token_id >= vocab_size:
                return self.create_error_response(
                    f"label_token_id {token_id} is out of vocabulary range "
                    f"[0, {vocab_size}). Please provide valid token IDs."
                )

        if len(request.label_token_ids) == 0:
            return self.create_error_response(
                "label_token_ids must contain at least one token ID."
            )

        # Validate items
        if len(request.items) == 0:
            return self.create_error_response(
                "items must contain at least one item."
            )

        try:
            lora_request = self._maybe_get_adapters(request)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        request_id = f"genscore-{self._base_request_id(raw_request, request.request_id)}"
        created_time = int(time.time())

        # Build prompts for each item
        try:
            engine_prompts, prompt_token_counts = await self._build_prompts(
                request, tokenizer
            )
        except (ValueError, TypeError) as e:
            logger.exception("Error building prompts")
            return self.create_error_response(e)

        # Create sampling params for scoring
        # We use max_tokens=1 with logprobs=-1 to get full vocab logprobs
        # for the next token distribution
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=request.temperature if request.temperature else 0.0,
            top_k=request.top_k if request.top_k is not None else 0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            logprobs=-1,  # Get all vocab logprobs
            n=1,
        )

        # Get trace headers
        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        # Schedule requests for all prompts
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_prompt in enumerate(engine_prompts):
            request_id_item = f"{request_id}-{i}"

            self._log_inputs(
                request_id_item,
                engine_prompt,
                params=sampling_params,
                lora_request=lora_request,
            )

            generator = self.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id_item,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
            )
            generators.append(generator)

        # Collect results
        result_generator = merge_async_iterators(*generators)
        results: list[RequestOutput | None] = [None] * len(engine_prompts)

        try:
            async for i, res in result_generator:
                results[i] = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except Exception as e:
            logger.exception("Error during generation")
            return self.create_error_response(e)

        # Process results to extract label token probabilities
        item_results: list[GenerativeScoreItemResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, result in enumerate(results):
            if result is None:
                return self.create_error_response(
                    f"Failed to generate result for item {i}"
                )

            # Check for errors
            if result.outputs and result.outputs[0].finish_reason == "error":
                return self.create_error_response(
                    f"Generation error for item {i}"
                )

            # Get logprobs from the generated token
            if not result.outputs or len(result.outputs) == 0:
                return self.create_error_response(
                    f"No output generated for item {i}"
                )

            output = result.outputs[0]
            if output.logprobs is None or len(output.logprobs) == 0:
                return self.create_error_response(
                    f"No logprobs available for item {i}. "
                    "This might indicate an issue with logprobs configuration."
                )

            # The logprobs dict maps token_id -> Logprob object
            # For logprobs=-1, this contains all vocab tokens
            logprobs_dict = output.logprobs[0]

            # Extract logprobs for label tokens
            label_logprobs: dict[int, float] = {}
            missing_tokens = []
            for token_id in request.label_token_ids:
                if token_id in logprobs_dict:
                    label_logprobs[token_id] = logprobs_dict[token_id].logprob
                else:
                    missing_tokens.append(token_id)

            if missing_tokens:
                return self.create_error_response(
                    f"Token IDs {missing_tokens} not found in logprobs for item {i}. "
                    "This might indicate the tokens are outside the model's vocabulary."
                )

            # Compute probabilities based on apply_softmax setting
            token_probs = self._compute_probabilities(
                label_logprobs,
                apply_softmax=request.apply_softmax,
            )

            item_results.append(
                GenerativeScoreItemResult(
                    index=i,
                    token_probs={str(k): v for k, v in token_probs.items()},
                )
            )

            # Update token counts
            total_prompt_tokens += prompt_token_counts[i]
            total_completion_tokens += len(output.token_ids)

        # Build response
        model_name = self.models.model_name(lora_request)
        response = GenerativeScoreResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            results=item_results,
            usage=UsageInfo(
                prompt_tokens=total_prompt_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                completion_tokens=total_completion_tokens,
            ),
        )

        return response

    async def _build_prompts(
        self,
        request: GenerativeScoreRequest,
        tokenizer,
    ) -> tuple[list[TokensPrompt], list[int]]:
        """Build prompts by concatenating query and items.

        Args:
            request: The request containing query, items, and settings.
            tokenizer: The tokenizer to use.

        Returns:
            Tuple of (list of TokensPrompt, list of prompt token counts).
        """
        # Tokenize query if it's a string
        if isinstance(request.query, str):
            async_tokenizer = self._get_async_tokenizer(tokenizer)
            query_result = await async_tokenizer(
                request.query,
                add_special_tokens=request.add_special_tokens,
            )
            query_token_ids = query_result.input_ids
        else:
            query_token_ids = request.query

        engine_prompts: list[TokensPrompt] = []
        prompt_token_counts: list[int] = []

        for item in request.items:
            # Tokenize item if it's a string
            if isinstance(item, str):
                async_tokenizer = self._get_async_tokenizer(tokenizer)
                # Don't add special tokens for items to avoid duplicate BOS/EOS
                item_result = await async_tokenizer(
                    item,
                    add_special_tokens=False,
                )
                item_token_ids = item_result.input_ids
            else:
                item_token_ids = item

            # Concatenate based on item_first setting
            if request.item_first:
                prompt_token_ids = item_token_ids + query_token_ids
            else:
                prompt_token_ids = query_token_ids + item_token_ids

            engine_prompts.append(
                TokensPrompt(prompt_token_ids=prompt_token_ids)
            )
            prompt_token_counts.append(len(prompt_token_ids))

        return engine_prompts, prompt_token_counts

    def _compute_probabilities(
        self,
        label_logprobs: dict[int, float],
        apply_softmax: bool,
    ) -> dict[int, float]:
        """Compute probabilities from logprobs.

        Args:
            label_logprobs: Dictionary mapping token_id to logprob.
            apply_softmax: If True, normalize over only the label tokens.
                If False, return true model probabilities (exp(logprob)).

        Returns:
            Dictionary mapping token_id to probability.
        """
        if apply_softmax:
            # Normalize over only the label tokens (subset softmax)
            # softmax(gathered_logits) over the subset
            logprobs_list = list(label_logprobs.values())
            max_logprob = max(logprobs_list)

            # Compute exp(logprob - max) for numerical stability
            exp_values = {
                token_id: math.exp(logprob - max_logprob)
                for token_id, logprob in label_logprobs.items()
            }
            sum_exp = sum(exp_values.values())

            return {
                token_id: exp_val / sum_exp
                for token_id, exp_val in exp_values.items()
            }
        else:
            # Return true model probabilities
            # Since logprobs are already log(softmax(logits)), 
            # we just need to exp() them
            return {
                token_id: math.exp(logprob)
                for token_id, logprob in label_logprobs.items()
            }

    async def _get_trace_headers(
        self,
        headers: Mapping[str, str],
    ) -> Mapping[str, str] | None:
        """Extract trace headers from request headers."""
        from vllm.tracing import (
            contains_trace_headers,
            extract_trace_headers,
            log_tracing_disabled_warning,
        )

        if not contains_trace_headers(headers):
            return None

        if not await self.engine_client.is_tracing_enabled():
            log_tracing_disabled_warning()
            return None

        return extract_trace_headers(headers)

    def _base_request_id(
        self,
        raw_request: Request | None,
        request_id: str | None,
    ) -> str:
        """Get base request ID from raw request or generate one."""
        if request_id:
            return request_id
        if raw_request:
            return getattr(raw_request.state, "request_id", None) or \
                   str(id(raw_request))
        from vllm.utils import random_uuid
        return random_uuid()

    def _log_inputs(
        self,
        request_id: str,
        prompt: TokensPrompt,
        params: SamplingParams,
        lora_request: LoRARequest | None,
    ) -> None:
        """Log request inputs."""
        if self.request_logger is None:
            return

        self.request_logger.log_inputs(
            request_id=request_id,
            prompt=str(prompt.get("prompt_token_ids", [])[:10]) + "...",
            prompt_token_ids=None,
            prompt_embeds=None,
            params=params,
            lora_request=lora_request,
        )
