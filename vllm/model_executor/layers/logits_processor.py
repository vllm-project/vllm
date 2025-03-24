# SPDX-License-Identifier: Apache-2.0
"""A layer that compute logits from hidden_stats."""
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform

_logits_processor_threadpool: Optional[ThreadPoolExecutor] = None
if envs.VLLM_LOGITS_PROCESSOR_THREADS is not None:
    _logits_processor_threadpool = ThreadPoolExecutor(
        envs.VLLM_LOGITS_PROCESSOR_THREADS)


class LogitsProcessor(nn.Module):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(hidden_states,
                                                     sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            if sampling_metadata is not None and \
                sampling_metadata.seq_groups is not None:
                logits = _apply_logits_processors(logits, sampling_metadata)

        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head,
                                            hidden_states,
                                            bias=embedding_bias)

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    # NOTE(kzawora): The if guard is needed for Gaudi - in some scenarios
    # (warmup, profile_run) we might not have selected_token_indices,
    # so we skip pruning.
    if sampling_metadata.selected_token_indices is not None:
        return hidden_states.index_select(
            0, sampling_metadata.selected_token_indices)
    else:
        return hidden_states


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    found_logits_processors = False
    logits_processed = 0
    logits_row_ids_and_logits_row_futures = []
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True

            for seq_id, logits_row_idx in zip(seq_ids,
                                              seq_group.sample_indices):
                logits_row = logits[logits_row_idx]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

                if _logits_processor_threadpool is not None:
                    logits_row_ids_and_logits_row_futures.append(
                        (logits_row_idx,
                         _logits_processor_threadpool.submit(
                             _apply_logits_processors_single_seq, logits_row,
                             logits_processors, past_tokens_ids,
                             prompt_tokens_ids)))
                else:
                    logits[logits_row_idx] = \
                        _apply_logits_processors_single_seq(
                            logits_row, logits_processors, past_tokens_ids,
                            prompt_tokens_ids)

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)

    for logits_row_idx, future in logits_row_ids_and_logits_row_futures:
        logits[logits_row_idx] = future.result()

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]
    return logits


def _apply_logits_processors_single_seq(logits_row, logits_processors,
                                        past_tokens_ids,
                                        prompt_tokens_ids) -> torch.Tensor:
    for logits_processor in logits_processors:
        parameters = inspect.signature(logits_processor).parameters
        if len(parameters) == 3:
            logits_row = logits_processor(prompt_tokens_ids, past_tokens_ids,
                                          logits_row)
        else:
            logits_row = logits_processor(past_tokens_ids, logits_row)
    return logits_row
