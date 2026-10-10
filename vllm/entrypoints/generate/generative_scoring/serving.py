# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generative Scoring implementation for generative models.

This module implements generative scoring functionality that computes the
probability of specified token IDs appearing as the next token after a
given query+item prompt. This works on any generative model that produces
logits (task="generate").
"""

import asyncio
import math
import time
from typing import Any, Literal

from fastapi import Request
from pydantic import Field, model_validator

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.base.protocol import validate_cache_salt
from vllm.entrypoints.generate.label_reads import next_token_label_reads
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import (
    ErrorResponse,
    OpenAIBaseModel,
    UsageInfo,
)
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput, tokens_input
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils import random_uuid

logger = init_logger(__name__)


# ============================================================================
# Protocol definitions
# ============================================================================


class GenerativeScoringRequest(OpenAIBaseModel):
    """Request for computing generative scoring.

    Attributes:
        model: The model to use for scoring. Optional, follows existing patterns.
        query: The query text or pre-tokenized query token IDs.
        items: The item text(s) or pre-tokenized item token IDs.
        label_token_ids: List of token IDs to compute probabilities for.
        apply_softmax: Whether to normalize probabilities using softmax over only
            the label_token_ids (True) or return true model probabilities over
            the full vocab for those ids (False).
        item_first: If True, prepend items to query. Otherwise append items to query.
        add_special_tokens: Whether to add special tokens when tokenizing.
        cache_salt: Optional salt for prefix caching.

    """

    model: str | None = None
    query: str | list[int] = Field(
        ...,
        description="The query text or pre-tokenized query token IDs.",
    )
    items: list[str] | list[list[int]] = Field(
        ...,
        description="List of item texts or pre-tokenized item token IDs.",
    )
    label_token_ids: list[int] = Field(
        ...,
        description="List of token IDs to compute probabilities for.",
    )
    apply_softmax: bool = Field(
        default=True,
        description=(
            "If True, normalize probabilities using softmax over only the "
            "label_token_ids. If False, return the true model probabilities "
            "over the full vocab for those ids."
        ),
    )
    item_first: bool = Field(
        default=False,
        description="If True, prepend items to query. Otherwise append items to query.",
    )
    add_special_tokens: bool = Field(
        default=True,
        description="Whether to add special tokens when tokenizing.",
    )
    cache_salt: str | None = Field(
        default=None,
        description=(
            "If specified, the prefix cache will be salted with the provided "
            "string to prevent an attacker to guess prompts in multi-user "
            "environments. The salt should be random, protected from "
            "access by 3rd parties, and long enough to be "
            "unpredictable (e.g., 43 characters base64-encoded, corresponding "
            "to 256 bit)."
        ),
    )
    priority: int = Field(
        default=0,
        ge=-(2**63),
        le=2**63 - 1,
        description=(
            "The priority of the request (lower means earlier handling; default: 0)."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description="The request_id related to this request.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_cache_salt_support(cls, data: Any) -> Any:
        if isinstance(data, dict):
            validate_cache_salt(data.get("cache_salt"))
        return data


class GenerativeScoringItemResult(OpenAIBaseModel):
    """Result for a single item in the generative scoring response.

    Attributes:
        index: The index of this item in the input items list.
        object: Type of object, always "score".
        score: The probability score for the first label token.

    """

    index: int
    object: Literal["score"] = "score"
    score: float


class GenerativeScoringResponse(OpenAIBaseModel):
    """Response from the generative scoring computation.

    Attributes:
        id: Unique identifier for this response.
        object: Type of object, always "list".
        created: Unix timestamp of when the response was created.
        model: The model used for scoring.
        data: List of scoring results, one per input item.
        usage: Token usage information.

    """

    id: str = Field(default="")
    object: Literal["list"] = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: list[GenerativeScoringItemResult]
    usage: UsageInfo


# ============================================================================
# Serving class
# ============================================================================


class ServingGenerativeScoring(BaseServing):
    """Serving class for generative scoring computation.

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
    ) -> None:
        super().__init__(
            models=models,
            model_config=engine_client.model_config,
            request_logger=request_logger,
        )
        self.engine_client = engine_client
        self.renderer = engine_client.renderer

    async def create_generative_scoring(
        self,
        request: GenerativeScoringRequest,
        raw_request: Request | None = None,
    ) -> GenerativeScoringResponse | ErrorResponse:
        """Create generative scoring for the given request.

        Args:
            request: The GenerativeScoringRequest containing query, items, and
                label_token_ids.
            raw_request: The raw FastAPI request object.

        Returns:
            GenerativeScoringResponse with probabilities for each item, or
            ErrorResponse if an error occurred.

        """
        spec_config = self.engine_client.vllm_config.speculative_config
        if (
            spec_config is not None
            and spec_config.rejection_sample_method == "synthetic"
        ):
            return self.create_error_response(
                "Generative scoring is unsupported with synthetic acceptance"
            )
        # Check model
        error_check_ret = await self._check_model(request)  # type: ignore[arg-type]
        if error_check_ret is not None:
            return error_check_ret

        # Check if engine is alive
        if self.engine_client.errored:
            raise self.engine_client.dead_error
        self.engine_client.check_admission(len(request.items))

        # Get tokenizer
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            return self.create_error_response(
                "Tokenizer not available. Cannot process generative scoring request."
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
            return self.create_error_response("items must contain at least one item.")

        # Note: Mixed item types (string and token list) are validated by
        # Pydantic at request parsing time, so we don't need to check here.

        try:
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"generative-scoring-{base_id}"
        created_time = int(time.time())

        # Build prompts for each item
        try:
            engine_inputs, prompt_token_counts = await self._build_prompts(
                request, tokenizer, self.model_config.max_model_len
            )
        except (ValueError, TypeError) as e:
            logger.exception("Error building prompts")
            return self.create_error_response(e)

        # Create sampling params for scoring
        # We use max_tokens=1 with logprob_token_ids to efficiently get
        # logprobs for only the specified label tokens (not full vocab)
        # Note: temperature/top_k/top_p don't affect logprobs - they only
        # affect the sampling distribution. Logprobs are computed from raw
        # logits via log_softmax before any sampling transformations.
        sampling_params = SamplingParams(
            max_tokens=1,
            logprobs=len(request.label_token_ids),
            logprob_token_ids=request.label_token_ids,
            n=1,
            watermarking=False,
        )

        # Get trace headers
        trace_headers = (
            None
            if raw_request is None
            else await self._get_trace_headers(raw_request.headers)
        )

        for i, engine_input in enumerate(engine_inputs):
            self._log_inputs(
                f"{request_id}-{i}",
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

        try:
            reads = await next_token_label_reads(
                self.engine_client,
                engine_inputs,
                [sampling_params] * len(engine_inputs),
                request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(e)
        except Exception as e:
            logger.exception("Error during generation")
            return self.create_error_response(e)

        # Process results to extract label token probabilities
        item_results: list[GenerativeScoringItemResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, read in enumerate(reads):
            label_logprobs = dict(zip(request.label_token_ids, read.logprobs))

            # Compute probabilities based on apply_softmax setting
            token_probs = self._compute_probabilities(
                label_logprobs,
                apply_softmax=request.apply_softmax,
            )

            # Use the first label token's probability as the score
            first_label_id = request.label_token_ids[0]
            score = token_probs[first_label_id]

            item_results.append(
                GenerativeScoringItemResult(
                    index=i,
                    score=score,
                )
            )

            # Update token counts
            total_prompt_tokens += prompt_token_counts[i]
            total_completion_tokens += len(read.result.outputs[0].token_ids)

        # Build response
        model_name = self.models.model_name(lora_request)
        response = GenerativeScoringResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=item_results,
            usage=UsageInfo(
                prompt_tokens=total_prompt_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                completion_tokens=total_completion_tokens,
            ),
        )

        return response

    async def _build_prompts(
        self,
        request: GenerativeScoringRequest,
        tokenizer: TokenizerLike,
        max_model_len: int,
    ) -> tuple[list[EngineInput], list[int]]:
        """Build prompts by concatenating query and items.

        Uses the Renderer's tokenizer to tokenize text inputs, then
        creates EngineInput via tokens_input() for engine consumption.

        Args:
            request: The request containing query, items, and settings.
            tokenizer: The tokenizer to use.
            max_model_len: Maximum model context length for truncation.

        Returns:
            Tuple of (list of EngineInput, list of prompt token counts).

        """
        # Tokenize query if it's a string
        if isinstance(request.query, str):
            query_token_ids = tokenizer.encode(
                request.query,
                add_special_tokens=request.add_special_tokens,
            )
        else:
            query_token_ids = request.query

        engine_inputs: list[EngineInput] = []
        prompt_token_counts: list[int] = []

        for item in request.items:
            # Tokenize item if it's a string
            if isinstance(item, str):
                # Don't add special tokens for items to avoid duplicate BOS/EOS
                item_token_ids = tokenizer.encode(
                    item,
                    add_special_tokens=False,
                )
            else:
                item_token_ids = item

            # Concatenate based on item_first setting
            if request.item_first:
                prompt_token_ids = item_token_ids + query_token_ids
            else:
                prompt_token_ids = query_token_ids + item_token_ids

            # Truncate to max_model_len - 1 to leave room for 1 output token
            max_prompt_len = max_model_len - 1
            if len(prompt_token_ids) > max_prompt_len:
                prompt_token_ids = prompt_token_ids[:max_prompt_len]

            engine_inputs.append(
                tokens_input(prompt_token_ids, cache_salt=request.cache_salt)
            )
            prompt_token_counts.append(len(prompt_token_ids))

        return engine_inputs, prompt_token_counts

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
                token_id: exp_val / sum_exp for token_id, exp_val in exp_values.items()
            }
        else:
            # Return true model probabilities
            # Since logprobs are already log(softmax(logits)),
            # we just need to exp() them
            return {
                token_id: math.exp(logprob)
                for token_id, logprob in label_logprobs.items()
            }
