# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from transformers import PreTrainedTokenizer

from vllm.model_executor.guided_decoding.reasoner.deepseek_reasoner import (  # noqa: E501
    DeepSeekReasoner)
from vllm.model_executor.guided_decoding.reasoner.reasoner import Reasoner


def get_reasoner(tokenizer: PreTrainedTokenizer,
                 reasoning_backend: str | None) -> Reasoner | None:
    if reasoning_backend is None:
        # No reasoning backend specified
        return None
    elif reasoning_backend == "deepseek_r1":
        return DeepSeekReasoner.from_tokenizer(tokenizer)
    else:
        raise ValueError(f"Unknown reasoning backend '{reasoning_backend}'")


__all__ = ["Reasoner", "get_reasoner"]
