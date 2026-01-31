# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol definitions for the Generative Scores API.

This module defines the request and response models for the /v1/generative-scores
endpoint, which computes the probability of specified token IDs appearing as the
next token after a given query+item prompt.
"""

import time
from typing import Literal

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
from vllm.utils import random_uuid


class GenerativeScoreRequest(OpenAIBaseModel):
    """Request for computing generative scores.

    This endpoint scores the probability of specified token IDs appearing after 
    the given query and item are appended together. For example:

        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223]  # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"

        The API would then return the probabilities of the model producing "Yes" 
        and "No" as the next token.

    Attributes:
        model: The model to use for scoring. Optional, follows existing patterns.
        query: The query text or pre-tokenized query token IDs.
        items: The item text(s) or pre-tokenized item token IDs.
        label_token_ids: List of token IDs to compute probabilities for.
        apply_softmax: Whether to normalize probabilities using softmax over only
            the label_token_ids (True) or return true model probabilities over
            the full vocab for those ids (False).
        item_first: If True, prepend items to query. Otherwise append items to query.
        temperature: Temperature for logits. Default 0.0 for scoring (greedy).
        top_k: Top-k filtering. Default 0 (disabled) for scoring.
        top_p: Top-p filtering. Default 1.0 (disabled) for scoring.
        add_special_tokens: Whether to add special tokens when tokenizing.
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
    temperature: float | None = Field(
        default=0.0,
        description="Temperature for logits. Default 0.0 for scoring.",
    )
    top_k: int | None = Field(
        default=0,
        description="Top-k filtering. Default 0 (disabled) for scoring.",
    )
    top_p: float | None = Field(
        default=1.0,
        description="Top-p filtering. Default 1.0 (disabled) for scoring.",
    )
    add_special_tokens: bool = Field(
        default=True,
        description="Whether to add special tokens when tokenizing.",
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0)."
        ),
    )
    request_id: str = Field(
        default_factory=random_uuid,
        description="The request_id related to this request.",
    )


class GenerativeScoreItemResult(OpenAIBaseModel):
    """Result for a single item in the generative scores response.

    Attributes:
        index: The index of this item in the input items list.
        token_probs: Dictionary mapping token IDs (as strings) to their probabilities.
    """

    index: int
    token_probs: dict[str, float] = Field(
        description="Mapping of token ID (as string) to probability."
    )


class GenerativeScoreResponse(OpenAIBaseModel):
    """Response from the generative scores endpoint.

    Attributes:
        id: Unique identifier for this response.
        object: Type of object, always "generative_score".
        created: Unix timestamp of when the response was created.
        model: The model used for scoring.
        results: List of scoring results, one per input item.
        usage: Token usage information.
    """

    id: str = Field(default="")
    object: Literal["generative_score"] = "generative_score"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    results: list[GenerativeScoreItemResult]
    usage: UsageInfo
