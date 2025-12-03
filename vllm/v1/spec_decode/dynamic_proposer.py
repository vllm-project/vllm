# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)

# Constants for dynamic k adjustment
MAX_SPEC_TOKENS = 10
MIN_SPEC_TOKENS = 2
ACCEPTANCE_HISTORY_LEN = 10
ACCEPTANCE_RATE_HYSTERESIS = 0.05
MIN_HISTORY_FOR_ADJUSTMENT = 3


class SequenceState:
    """Data class to store the speculative decoding state for each sequence."""

    def __init__(self, initial_spec_tokens: int):
        self.num_spec_tokens = initial_spec_tokens
        self.acceptance_rate_history: deque[float] = deque(
            maxlen=ACCEPTANCE_HISTORY_LEN
        )


class DynamicProposer(EagleProposer):
    """
    A proposer that dynamically adjusts the number of speculative tokens (k)
    for each sequence based on its historical acceptance rate.
    """

    num_speculative_tokens: int

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)

        self.seq_states: dict[str, SequenceState] = {}
        self.last_proposed_k_per_seq: dict[str, int] = {}
        self._initial_spec_tokens = max(
            MIN_SPEC_TOKENS, min(self.num_speculative_tokens, MAX_SPEC_TOKENS)
        )

        self.acceptance_rate_threshold = (
            vllm_config.speculative_config.acceptance_rate_threshold
        )

        logger.info("DynamicProposer initialized for adaptive k.")

        # If the method is eagle_dynamic and the draft model is eagle3,
        # we treat it as eagle3 to enable eagle3-specific logic
        # (e.g. hidden state combination).
        if (
            self.method == "eagle_dynamic"
            and "eagle3" in self.draft_model_config.model.lower()
        ):
            self.method = "eagle3"
            self.eagle3_use_aux_hidden_state = (
                self._get_eagle3_use_aux_hidden_state_from_config()
            )
            logger.info(
                "DynamicProposer detected Eagle3 draft model. "
                "Enabling Eagle3 specific logic."
            )

    def update_sequence_states(
        self,
        req_ids: Sequence[str | None],
        num_accepted_tokens: Sequence[int],
    ) -> None:
        """
        Updates sequence states with acceptance information from the previous step.
        """
        if not self.last_proposed_k_per_seq:
            return

        for req_id, num_accepted in zip(req_ids, num_accepted_tokens):
            if req_id is None:
                continue
            num_proposed = self.last_proposed_k_per_seq.get(req_id)
            if num_proposed is None or num_proposed <= 0:
                continue

            acc = max(int(num_accepted), 0)
            acceptance_rate = (acc / num_proposed) if num_proposed > 0 else 0.0
            state = self._get_or_create_state(req_id)
            state.acceptance_rate_history.append(acceptance_rate)

        self.last_proposed_k_per_seq.clear()

    def cleanup_finished_seqs(
        self,
        req_ids_in_batch: Sequence[str | None],
    ) -> None:
        """Cleans up the state for sequences that are actually finished."""
        if self.runner is None:
            return

        # Get all requests still considered active by the engine/scheduler
        active_req_ids = set(self.runner.requests.keys())

        # Only delete state for requests that are truly finished
        finished_req_ids = set(self.seq_states.keys()) - active_req_ids
        for req_id in finished_req_ids:
            del self.seq_states[req_id]
            self.last_proposed_k_per_seq.pop(req_id, None)

    def _get_or_create_state(self, req_id: str) -> SequenceState:
        """Retrieves or creates the state for a given sequence."""
        state = self.seq_states.get(req_id)
        if state is None:
            state = SequenceState(self._initial_spec_tokens)
            self.seq_states[req_id] = state
        return state

    def _adjust_and_get_spec_tokens_for_batch(
        self,
        req_ids: Sequence[str | None],
    ) -> list[int]:
        """
        Calculates the number of speculative tokens for each sequence based on
        its average acceptance rate.
        """
        spec_tokens_for_batch: list[int] = []
        for req_id in req_ids:
            if req_id is None:
                spec_tokens_for_batch.append(MIN_SPEC_TOKENS)
                continue

            state = self._get_or_create_state(req_id)
            history = state.acceptance_rate_history
            if len(history) < MIN_HISTORY_FOR_ADJUSTMENT:
                spec_tokens_for_batch.append(state.num_spec_tokens)
                continue

            avg_acceptance_rate = float(np.mean(history))
            upper_bound = self.acceptance_rate_threshold + ACCEPTANCE_RATE_HYSTERESIS
            lower_bound = self.acceptance_rate_threshold - ACCEPTANCE_RATE_HYSTERESIS

            new_k = state.num_spec_tokens
            if avg_acceptance_rate >= upper_bound:
                new_k = min(state.num_spec_tokens + 1, MAX_SPEC_TOKENS)
            elif avg_acceptance_rate <= lower_bound:
                new_k = max(state.num_spec_tokens - 1, MIN_SPEC_TOKENS)

            if new_k != state.num_spec_tokens:
                logger.debug(
                    "Accept rate %.2f -> req %s: k %d -> %d",
                    avg_acceptance_rate,
                    req_id,
                    state.num_spec_tokens,
                    new_k,
                )
                state.num_spec_tokens = new_k

            spec_tokens_for_batch.append(state.num_spec_tokens)

        return spec_tokens_for_batch

    @torch.inference_mode()
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> list[list[int]]:
        if self.runner is None:
            raise RuntimeError("DynamicProposer requires GPUModelRunner")

        batch_size = next_token_ids.shape[0]
        req_ids = self.runner.input_batch.req_ids[:batch_size]

        # 1. Update states with acceptance results from the previous step.
        accepted_tokens = self.runner.input_batch.num_accepted_tokens_cpu[
            :batch_size
        ].tolist()
        self.update_sequence_states(req_ids, accepted_tokens)
        self.cleanup_finished_seqs(req_ids)

        # 2. Determine and record k for each sequence for the current step.
        per_sequence_k = self._adjust_and_get_spec_tokens_for_batch(req_ids)
        self.last_proposed_k_per_seq = {
            req_id: k
            for req_id, k in zip(req_ids, per_sequence_k)
            if req_id is not None
        }

        # Safeguard against potential length mismatch, though unlikely.
        if len(per_sequence_k) != batch_size:
            fixed_k = [0] * batch_size
            for i in range(min(len(per_sequence_k), batch_size)):
                fixed_k[i] = int(per_sequence_k[i])
            per_sequence_k = fixed_k

        max_k_in_batch = max(per_sequence_k) if per_sequence_k else 0
        if max_k_in_batch == 0:
            # If no drafts are proposed in this step, return empty lists.
            return [[] for _ in range(batch_size)]

        # 3. Get draft tokens from the parent (EagleProposer), requesting up to max_k.
        original_num_tokens = self.num_speculative_tokens
        self.num_speculative_tokens = max_k_in_batch
        try:
            full_draft_token_ids = super().propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=last_token_indices,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
            )
        finally:
            self.num_speculative_tokens = original_num_tokens

        if full_draft_token_ids.numel() == 0:
            return [[] for _ in range(batch_size)]

        # 4. Convert to ragged list directly (no rectangular padding needed).
        #    This aligns with the n-gram proposer pattern where each sequence
        #    can have a different number of draft tokens.
        draft_lists = full_draft_token_ids.tolist()
        ragged_drafts = [draft_lists[i][: per_sequence_k[i]] for i in range(batch_size)]

        return ragged_drafts
