# SPDX-License-Identifier: Apache-2.0
from transformers import PreTrainedTokenizer

from vllm.model_executor.guided_decoding.reasoner.reasoner import (
    Reasoner, ReasonerConfig)


def get_reasoner(reasoning_backend: str,
                 tokenizer: PreTrainedTokenizer) -> Reasoner:
    if reasoning_backend == "deepseek_r1":
        from vllm.model_executor.guided_decoding.reasoner.deepseek_reasoner import (  # noqa
            DeepSeekReasoner)
        return DeepSeekReasoner(tokenizer)

    raise ValueError(f"Unknown reasoner '{reasoning_backend}'. "
                     "Must be one of 'deepseek'")


__all__ = ["get_reasoner", "ReasonerConfig", "Reasoner"]
