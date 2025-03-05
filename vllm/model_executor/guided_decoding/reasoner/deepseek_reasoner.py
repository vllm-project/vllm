# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from vllm.model_executor.guided_decoding.reasoner.reasoner import Reasoner


@dataclass
class DeepSeekReasoner(Reasoner):
    """
    Reasoner for DeepSeek R series models.
    """
    start_token_id: int
    end_token_id: int

    start_token: str = "<think>"
    end_token: str = "</think>"

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizer) -> Reasoner:
        return cls(start_token_id=tokenizer.encode(
            "<think>", add_special_tokens=False)[0],
                   end_token_id=tokenizer.encode("</think>",
                                                 add_special_tokens=False)[0])

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids
