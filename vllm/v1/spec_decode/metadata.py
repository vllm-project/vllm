# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

if typing.TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )


@dataclass
class ProposeInput:
    """
    Unified input dataclass for speculative decoding proposers.

    This dataclass encapsulates all possible inputs needed by different
    speculative decoding methods (ngram, ngram-gpu, suffix, medusa, eagle,
    draft model, extract_hidden_states). Each proposer will use only the
    fields it needs.

    Attributes:
        sampled_token_ids: Sampled token IDs from target model.
            - For ngram: list[list[int]] (CPU-side, valid tokens per req)
        num_tokens_no_spec: Number of tokens without spec decode per request.
        token_ids_cpu: Token IDs on CPU [batch_size, max_model_len].

    """

    # Common inputs
    sampled_token_ids: torch.Tensor | list[list[int]] | None = None
    input_batch: "InputBatch" | None = None

    # Input batch state (for ngram)
    num_tokens_no_spec: np.ndarray | torch.Tensor | None = None
    token_ids_cpu: np.ndarray | torch.Tensor | None = None
    # Medusa inputs
    hidden_states: torch.Tensor | None = None
    # Extract hidden states inputs
    target_hidden_states: list[torch.Tensor] | None = None
    common_hidden_states: torch.Tensor | None = None
    # Ngram-gpu inputs
    token_ids_gpu: torch.Tensor | None = None
    valid_sampled_token_ids_gpu: torch.Tensor | None = None
    valid_sampled_tokens_count: torch.Tensor | None = None


class SpecDecodeProposer(ABC):
    """
    Abstract base class for all speculative decoding proposers.

    This interface provides a unified way to prepare inputs and propose
    draft tokens, eliminating the need for if-else chains in the model runner.
    """

    @abstractmethod
    def prepare_inputs(
        self,
        sampled_token_ids: torch.Tensor | list[list[int]],
        input_batch: "InputBatch",
        **kwargs,
    ) -> ProposeInput:
        """
        Prepare inputs for draft token proposal.

        This method handles all the method-specific input preparation logic,
        returning a unified SpecDecodeInput container.

        Args:
            sampled_token_ids: Sampled token IDs from target model.
            input_batch: Input batch with request states
            **kwargs: Additional method-specific arguments

        Returns:
            ProposeInput containing prepared inputs
        """
        pass

    @abstractmethod
    def propose(self, inputs: ProposeInput) -> torch.Tensor | list[list[int]]:
        """
        Propose draft tokens using prepared inputs.

        Args:
            inputs: Prepared inputs from prepare_inputs()

        Returns:
            Draft token IDs (tensor or list depending on method)
        """
        pass
