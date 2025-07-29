# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import concurrent.futures
import os
from enum import Enum
from json import dumps as json_dumps
from typing import Optional, Union

from regex import escape as regex_escape
from transformers import PreTrainedTokenizerBase

from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    JSONLogitsProcessor, RegexLogitsProcessor)
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import GuidedDecodingParams


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"


global_thread_pool = None  # used for generating logits processor fsm

# It's not yet clear that using more provides a benefit, and it could
# potentially starve other processes on the machine. We'll cap this for now and
# adjust later if testing proves it to help overcome a bottleneck.
_MAX_THREADPOOL_WORKERS = 16


async def get_outlines_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase,
    reasoner: Optional[ReasoningParser]
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    """
    global global_thread_pool
    guide, mode = _get_guide_and_mode(guided_params)
    if not guide or not mode:
        return None

    if global_thread_pool is None:
        max_workers = os.cpu_count() or 2
        if max_workers > _MAX_THREADPOOL_WORKERS:
            max_workers = _MAX_THREADPOOL_WORKERS
        global_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(global_thread_pool,
                                      _get_logits_processor, guide, tokenizer,
                                      mode, guided_params.whitespace_pattern,
                                      reasoner)


def get_local_outlines_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase,
    reasoner: Optional[ReasoningParser]
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    """
    guide, mode = _get_guide_and_mode(guided_params)
    if not guide or not mode:
        return None

    return _get_logits_processor(guide, tokenizer, mode,
                                 guided_params.whitespace_pattern, reasoner)


def _get_guide_and_mode(
    guided_params: GuidedDecodingParams
) -> Union[tuple[str, GuidedDecodingMode], tuple[None, None]]:
    if guided_params.json:
        if isinstance(guided_params.json, dict):
            # turn dict into hashable string
            json = json_dumps(guided_params.json)
        else:
            json = guided_params.json
        return json, GuidedDecodingMode.JSON
    elif guided_params.regex:
        return guided_params.regex, GuidedDecodingMode.REGEX
    elif guided_params.choice:
        # choice just uses regex
        choices = [
            regex_escape(str(choice)) for choice in guided_params.choice
        ]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex, GuidedDecodingMode.CHOICE
    elif guided_params.grammar:
        raise ValueError(
            "The `outlines` guided decoding backend no longer supports grammar "
            "guided generation. Please use either the `xgrammar` or `guidance` "
            "backend")
    else:
        return None, None


def _get_logits_processor(
    guide: str,
    tokenizer: PreTrainedTokenizerBase,
    mode: GuidedDecodingMode,
    whitespace_pattern: Union[str, None],
    reasoner: Optional[ReasoningParser],
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor]:
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, tokenizer, whitespace_pattern,
                                   reasoner)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, tokenizer, reasoner)
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")
