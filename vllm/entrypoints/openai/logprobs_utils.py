# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility functions for converting vLLM logprobs to OpenAI formats."""

from collections.abc import Sequence
from typing import Optional

from openai.types.responses.response_output_text import (Logprob,
                                                         LogprobTopLogprob)
from openai.types.responses import response_text_delta_event

from vllm.logprobs import Logprob as SampleLogprob
from vllm.logprobs import SampleLogprobs
from vllm.transformers_utils.tokenizer import AnyTokenizer


def _topk_logprobs(logprobs: dict[int, SampleLogprob], top_logprobs: int,
                   tokenizer: AnyTokenizer) -> list[LogprobTopLogprob]:
    """Returns the top-k logprobs from the logprobs dictionary."""
    out = []
    for i, (token_id, _logprob) in enumerate(logprobs.items()):
        if i >= top_logprobs:
            break
        text = _logprob.decoded_token if _logprob.decoded_token \
            is not None else tokenizer.decode([token_id])
        out.append(
            LogprobTopLogprob(
                token=text,
                logprob=max(_logprob.logprob, -9999.0),
                bytes=list(text.encode("utf-8", errors="replace")),
            ))
    return out


def create_stream_response_logprobs(
        token_ids: Sequence[int],
        logprobs: Optional[SampleLogprobs],
        tokenizer: AnyTokenizer,
        top_logprobs: Optional[int] = None
) -> list[response_text_delta_event.Logprob]:
    """Create streaming response logprobs for OpenAI Responses API."""
    lgs = create_response_logprobs(token_ids=token_ids,
                                   logprobs=logprobs,
                                   tokenizer=tokenizer,
                                   top_logprobs=top_logprobs)
    return [
        response_text_delta_event.Logprob(
            token=lg.token,
            logprob=lg.logprob,
            top_logprobs=[
                response_text_delta_event.LogprobTopLogprob(
                    token=tl.token, logprob=tl.logprob)
                for tl in lg.top_logprobs
            ]) for lg in lgs
    ]


def create_response_logprobs(
        token_ids: Sequence[int],
        logprobs: Optional[SampleLogprobs],
        tokenizer: AnyTokenizer,
        top_logprobs: Optional[int] = None) -> list[Logprob]:
    assert logprobs is not None, "logprobs must be provided"
    assert len(token_ids) == len(logprobs), (
        "token_ids and logprobs.token_ids must have the same length")
    out = []
    for i, token_id in enumerate(token_ids):
        logprob = logprobs[i]
        token_logprob = logprob[token_id]
        text = token_logprob.decoded_token if token_logprob.decoded_token \
            is not None else tokenizer.decode([token_id])
        out.append(
            Logprob(
                token=text,
                logprob=max(token_logprob.logprob, -9999.0),
                bytes=list(text.encode("utf-8", errors="replace")),
                top_logprobs=_topk_logprobs(logprob,
                                            top_logprobs=top_logprobs,
                                            tokenizer=tokenizer)
                if top_logprobs else [],
            ))
    return out
