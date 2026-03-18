# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import InputBatch


@dataclass
class SpecDecodeInput:
    """
    Unified input container for speculative decoding proposers.

    This dataclass encapsulates all possible inputs needed by different
    speculative decoding methods (ngram, ngram-gpu, suffix, medusa, eagle,
    draft model, extract_hidden_states). Each proposer will use only the
    fields it needs.

    Attributes:
        # Common inputs
        sampled_token_ids: Sampled token IDs from target model.
            - For ngram/suffix: list[list[int]] (CPU-side, valid tokens per req)
            - For eagle/draft_model: torch.Tensor (GPU-side, padded batch)
            - For extract_hidden_states: torch.Tensor (GPU-side)
        sampling_metadata: Sampling metadata for the current batch.
        common_attn_metadata: Common attention metadata.

        # Hidden states from target model
        hidden_states: Full hidden states from target model [num_tokens, hidden_size].
        sample_hidden_states: Hidden states at sample positions.
        aux_hidden_states: Auxiliary hidden states from target model (for eagle3).

        # Target model inputs (for eagle/draft_model)
        target_token_ids: Target token IDs [num_tokens].
        target_positions: Target positions [num_tokens] or [3, num_tokens] for M-RoPE.
        target_hidden_states: Target hidden states for drafting.
            - For eagle/draft_model: torch.Tensor [num_tokens, hidden_size]
            - For extract_hidden_states: list[torch.Tensor] (one per aux layer)

        # Draft model inputs (for eagle/draft_model)
        next_token_ids: Next token IDs for draft model [batch_size].
        token_indices_to_sample: Indices of tokens to sample.
        mm_embed_inputs: Multi-modal embeddings (embeds, is_mm_mask).
        num_rejected_tokens_gpu: Count of rejected tokens per request (GPU tensor).

        # Input batch state (for ngram/suffix)
        input_batch: Input batch containing request states.
        num_tokens_no_spec: Number of tokens without spec decode per request.
        token_ids_cpu: Token IDs on CPU [batch_size, max_model_len].

        # Slot mappings for KV cache
        slot_mappings: Slot mappings for attention layers.
    """

    # Common inputs
    sampled_token_ids: torch.Tensor | list[list[int]] | None = None
    sampling_metadata: "SamplingMetadata | None" = None
    common_attn_metadata: "CommonAttentionMetadata | None" = None

    # Hidden states from target model
    hidden_states: torch.Tensor | None = None
    sample_hidden_states: torch.Tensor | None = None
    aux_hidden_states: list[torch.Tensor] | None = None

    # Target model inputs
    target_token_ids: torch.Tensor | None = None
    target_positions: torch.Tensor | None = None
    target_hidden_states: torch.Tensor | list[torch.Tensor] | None = None

    # Draft model inputs
    next_token_ids: torch.Tensor | None = None
    token_indices_to_sample: torch.Tensor | None = None
    mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
    num_rejected_tokens_gpu: torch.Tensor | None = None
    valid_sampled_tokens_count: torch.Tensor | None = None

    # Input batch state (for ngram/suffix)
    input_batch: "InputBatch | None" = None
    # Can be numpy array (CPU ngram) or torch tensor (GPU ngram)
    num_tokens_no_spec: np.ndarray | torch.Tensor | None = None
    token_ids_cpu: np.ndarray | torch.Tensor | None = None

    # Slot mappings
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None


@dataclass
class SpecDecodePrepareOutput:
    """
    Output from the prepare_inputs() method.

    Contains the prepared SpecDecodeInput and any additional state needed
    for post-processing after propose().
    """

    # Prepared inputs for propose()
    inputs: SpecDecodeInput

    # Optional: post-processing callback or state
    # For methods that need additional processing after propose()
    post_process_fn: callable | None = None  # noqa: F821


class SpecDecodeProposer(ABC):
    """
    Abstract base class for all speculative decoding proposers.

    This interface provides a unified way to prepare inputs and propose
    draft tokens, eliminating the need for if-else chains in the model runner.
    """

    @abstractmethod
    def prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: "SamplingMetadata",
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: "SpecDecodeMetadata | None",
        common_attn_metadata: "CommonAttentionMetadata",
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
        input_batch: "InputBatch",
    ) -> SpecDecodePrepareOutput:
        """
        Prepare inputs for draft token proposal.

        This method handles all the method-specific input preparation logic,
        returning a unified SpecDecodeInput container.

        Args:
            scheduler_output: Scheduler output with scheduling information
            sampled_token_ids: Sampled token IDs from target model
            sampling_metadata: Sampling metadata
            hidden_states: Full hidden states from target model
            sample_hidden_states: Hidden states at sample positions
            aux_hidden_states: Auxiliary hidden states (for eagle3, etc.)
            spec_decode_metadata: Speculative decoding metadata (if any)
            common_attn_metadata: Common attention metadata
            slot_mappings: Slot mappings for KV cache
            input_batch: Input batch with request states

        Returns:
            SpecDecodePrepareOutput containing prepared inputs
        """
        pass

    @abstractmethod
    def propose(self, inputs: SpecDecodeInput) -> torch.Tensor | list[list[int]]:
        """
        Propose draft tokens using prepared inputs.

        Args:
            inputs: Prepared inputs from prepare_inputs()

        Returns:
            Draft token IDs (tensor or list depending on method)
        """
        pass


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
