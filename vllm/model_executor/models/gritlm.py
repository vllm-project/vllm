# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Set

import numpy as np
import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import (
    DispatchPooler,
    Pooler,
    PoolerHead,
    PoolerNormalize,
    PoolingParamsUpdate,
    get_prompt_lens,
    get_prompt_token_ids,
)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.tasks import PoolingTask
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .interfaces_base import default_pooling_type

logger = init_logger(__name__)


class GritLMMeanPool(nn.Module):
    """As `MeanPool`, but only includes non-instruction tokens."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.model_config = model_config

        tokenizer = cached_tokenizer_from_config(self.model_config)

        # Collect the tokens needed for pattern matching.
        # "▁<" is different from "_<". The former uses "▁" to indicate that
        # the next token is the start of a word.
        # "<0x0A>" is the newline token (i.e. "\n")."
        self.token_ids = {
            tok: tokenizer.convert_tokens_to_ids([tok])[0]
            for tok in ["<s>", "▁<", "<", "|", "embed", ">", "<0x0A>", "user"]
        }

        def tokens_to_ids(tokens: list[str]) -> np.ndarray:
            return np.array([self.token_ids[token] for token in tokens])

        self.user_pattern_ids = tokens_to_ids(["▁<", "|", "user", "|", ">", "<0x0A>"])
        self.embed_newline_pattern_ids = tokens_to_ids(
            ["<0x0A>", "<", "|", "embed", "|", ">", "<0x0A>"]
        )
        self.embed_pattern_ids = tokens_to_ids(["▁<", "|", "embed", "|", ">", "<0x0A>"])

    def _find_array(
        self,
        arr: np.ndarray,
        target: np.ndarray,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> int:
        """
        Find the first occurrence of `target` in `arr` starting from
        `start_idx`.

        Args:
            arr: The array to search within.
            target: The consecutive subsequence to find.
            start_idx: The starting index to search from (inclusive).
            end_idx: The ending index to search from (exclusive).

        Returns:
            The index of the first occurrence of `target` in `arr`.
        """
        if start_idx < 0:
            raise ValueError("`start_idx` must be non-negative")
        if len(arr) == 0 or len(target) == 0:
            raise ValueError("Empty `arr` or `target` not allowed")

        arr_len = len(arr)
        target_len = len(target)

        if end_idx is None:
            end_idx = arr_len

        for i in range(start_idx, min(end_idx, arr_len - target_len + 1)):
            if (arr[i : i + target_len] == target).all():
                return i

        return -1

    def _get_instruction_len(self, prompt_token_ids: np.ndarray) -> int:
        """
        Get the length of the instruction in the prompt.

        We do a pattern matching to find the instruction in the prompt,
        and then return the length of the instruction.

        The pattern matching is done using integers instead of strings
        because the prompt is given as a list of token IDs.
        """
        instruction_len = 0

        # Return no instruction in case of missing BOS token.
        if prompt_token_ids[0] != self.token_ids["<s>"]:
            logger.warning(
                "BOS token not found in prompt, "
                "thus using empty string for instruction. "
                "GritLM requires BOS token in prompt."
            )
            return instruction_len

        # If user pattern is found in the prompt, that means there should be
        # a newline token before the embed pattern.
        embed_pattern_ids = self.embed_pattern_ids
        if (
            self._find_array(
                prompt_token_ids, self.user_pattern_ids, start_idx=1, end_idx=2
            )
            == 1
        ):
            embed_pattern_ids = self.embed_newline_pattern_ids

        # Find the embed pattern in the prompt.
        found_embed_pattern_idx = self._find_array(
            prompt_token_ids, embed_pattern_ids, start_idx=1
        )

        if found_embed_pattern_idx != -1:
            instruction_len = found_embed_pattern_idx + len(embed_pattern_ids)
        else:
            logger.warning(
                "Query instruction not found in prompt, "
                "thus using BOS token as instruction instead. "
                "GritLM requires query instruction in prompt."
            )
            instruction_len = 1

        return instruction_len

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> list[torch.Tensor] | torch.Tensor:
        prompt_lens = get_prompt_lens(hidden_states, pooling_metadata)
        instr_lens = torch.tensor(
            [
                self._get_instruction_len(token_ids.cpu().numpy())
                for token_ids in get_prompt_token_ids(pooling_metadata)
            ],
            device="cpu",
        )

        offset = 0
        pooled_data = list[torch.Tensor]()
        for prompt_len, instr_len in zip(prompt_lens, instr_lens):
            pooled_data.append(
                hidden_states[offset + instr_len : offset + prompt_len].mean(
                    dim=0, dtype=torch.float32
                )
            )
            offset += prompt_len

        return pooled_data


class GritLMPooler(Pooler):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.pooling = GritLMMeanPool(model_config)
        self.head = PoolerHead(PoolerNormalize())

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooling.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data


@default_pooling_type("MEAN")
class GritLM(LlamaForCausalLM):
    """This class implements the embedding model for parasail-ai/GritLM-7B-vllm.

    The class inherits from LlamaForCausalLM and provides a custom pooling
    layer.

    The main difference between the pooling layer in GritLM and the one in
    LlamaForCausalLM is that GritLM ignores the query instruction in the prompt
    when pooling the hidden states.

    Embedding prompts should be in the following format:
    - With instruction: "<|user|>\nINSTRUCTION\n<|embed|>\nPROMPT".
    - Without instruction: "<|embed|>\nPROMPT".

    Generation prompts should be in the following format:
    - "<|user|>\nPROMPT\n<|assistant|>\n"
    """

    is_pooling_model = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        **kwargs,
    ) -> None:
        if vllm_config.model_config.runner_type == "pooling":
            hf_config = vllm_config.model_config.hf_config
            hf_config.is_causal = False

            vllm_config.cache_config.sliding_window = None

            hf_config.sliding_window = None

        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config is not None:
            self.pooler = DispatchPooler(
                {
                    "token_embed": Pooler.for_token_embed(pooler_config),
                    "embed": GritLMPooler(vllm_config.model_config),
                }
            )
