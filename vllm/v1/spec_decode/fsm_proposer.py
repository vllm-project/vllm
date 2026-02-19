# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FSM Proposer for Speculative Decoding.

This module implements speculative decoding using Finite State Machines (FSMs).
The FSM constrains generation to follow specific patterns, allowing the proposer
to draft tokens that are guaranteed to be valid according to the FSM structure.

How FSM Speculative Decoding Works:
    1. FSM defines valid token sequences as a graph of states and transitions
    2. Proposer tracks current FSM state for each request
    3. When FSM has deterministic path (single valid next token), propose it
    4. When FSM has multiple paths or wildcards, cannot propose (non-deterministic)
    5. Target model validates proposals and updates FSM state

Key Concepts:
    - Deterministic path: Only one valid token from current state -> can propose
    - Non-deterministic: Multiple valid tokens -> cannot propose, let model choose
    - Wildcard (-1): Any token allowed -> cannot propose specific token
    - State tracking: Each request maintains its own FSM state independently
"""

import logging
import time

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.custom_fsm import CustomFSM

logger = logging.getLogger(__name__)


class FSMProposer:
    """
    Proposer that uses FSM to generate draft tokens with per-request state tracking.

    The proposer can only draft tokens when the FSM has a deterministic path
    (single valid next token). When multiple paths exist or wildcards are present,
    the proposer returns no drafts and lets the target model choose.

    Attributes:
        fsm: The finite state machine defining valid token sequences
        k: Number of speculative tokens to propose
        max_model_len: Maximum sequence length
        req_states: Tracks current FSM state for each request (req_id -> state_id)
    """

    def __init__(self, vllm_config: VllmConfig, fsm: CustomFSM):
        assert vllm_config.speculative_config is not None

        self.fsm = fsm
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Track FSM state per request: req_id -> current_state_id
        # None means request hasn't started or FSM path ended
        self.req_states: dict[str, int | None] = {}

    def cleanup_finished_requests(self, finished_req_ids: set[str]) -> None:
        """
        Remove FSM states for finished requests.

        Args:
            finished_req_ids: Set of request IDs that have completed
        """
        for req_id in finished_req_ids:
            self.req_states.pop(req_id, None)
        logger.info("FSM cleanup: remaining: %d", len(self.req_states))

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """
        Propose draft tokens using FSM deterministic paths.

        For each request:
        1. Get current FSM state (or initialize to start state)
        2. Check if current state has deterministic path (single valid token)
        3. If deterministic, propose up to k tokens following the path
        4. If non-deterministic or wildcard, return empty (no proposals)

        Args:
            sampled_token_ids: Most recent sampled tokens per request
            req_ids: Request IDs corresponding to sampled tokens
            num_tokens_no_spec: Number of tokens generated without speculation
            token_ids_cpu: All generated tokens for each request
            slot_mappings: Unused, for interface compatibility

        Returns:
            draft_token_ids: list[list[int]] - Draft token sequences per request.
                            Empty list means no drafts for that request.
        """
        start = time.perf_counter()
        draft_token_ids: list[list[int]] = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = req_ids[i]
            """
            if req_id in spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue
            """

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            # Initialize or update FSM state for this request
            if req_id not in self.req_states:
                # First time: initialize from sampled tokens
                self.req_states[req_id] = self._get_state_from_prefix(sampled_ids)
            else:
                # Update state with sampled tokens
                current_state = self.req_states[req_id]
                for token in sampled_ids:
                    current_state = self.fsm.get_next_state(current_state, token)
                self.req_states[req_id] = current_state

            # Get draft tokens from current state
            drafts = self._get_draft_tokens_from_state(self.req_states[req_id])
            draft_token_ids.append(drafts)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("FSM propose: %.2fms, drafts: %s", elapsed, draft_token_ids)
        return draft_token_ids

    def _get_state_from_prefix(self, prefix: list[int]) -> int | None:
        """Navigate FSM to get state after processing prefix."""
        state: int | None = self.fsm.start_state
        for token in prefix:
            state = self.fsm.get_next_state(state, token)
        return state

    def _get_draft_tokens_from_state(self, state: int | None) -> list[int]:
        """Get deterministic draft tokens from current FSM state."""
        drafts = []
        current_state = state
        for _ in range(self.k):
            tokens = self.fsm.get_tokens_from_state(current_state)
            if len(tokens) != 1:
                break
            token = tokens[0]
            drafts.append(token)
            current_state = self.fsm.get_next_state(current_state, token)
        return drafts

    def load_model(self, *args, **kwargs):
        # No model to load
        pass
