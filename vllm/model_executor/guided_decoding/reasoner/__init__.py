# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from transformers import PreTrainedTokenizer

from vllm.logger import init_logger
from vllm.model_executor.guided_decoding.reasoner.deepseek_reasoner import (  # noqa: E501
    DeepSeekReasoner)
from vllm.model_executor.guided_decoding.reasoner.reasoner import Reasoner

logger = init_logger(__name__)


def get_reasoner(tokenizer: PreTrainedTokenizer,
                 reasoning_backend: str | None) -> Reasoner | None:
    if reasoning_backend is None:
        # No reasoning backend specified
        return None
    elif reasoning_backend == "deepseek_r1":
        return DeepSeekReasoner.from_tokenizer(tokenizer)
    elif reasoning_backend == "granite":
        logger.warning(
            "Granite reasoner not yet implemented for structured outputs")
        return None
    else:
        # Raise a warning for unknown reasoning backend and return None
        # We cannot raise an error here because some reasoning models
        # may not have a corresponding Reasoner class.
        logger.warning("Unknown reasoning backend %s for structured outputs ",
                       reasoning_backend)
        return None


__all__ = ["Reasoner", "get_reasoner"]
