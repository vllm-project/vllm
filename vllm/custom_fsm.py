# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Custom FSM for constrained decoding.

This module provides a Finite State Machine (FSM) implementation for constraining
LLM generation to follow specific patterns or structures.

FSM Graph Structure:
    The FSM is represented as a nested dictionary:
    ``graph[current_state][token_id] = next_state``

    Example:
        {
            0: {100: 1, 200: 2},  # State 0: token 100->state 1, 200->state 2
            1: {300: 3},           # From state 1, token 300 -> state 3
            2: {-1: 4},            # State 2: wildcard (-1) -> state 4
        }

Special Token Values:
    - Regular token IDs (>= 0): Only that specific token is allowed
    - -1 (wildcard): Any token is allowed (freeform generation)
    - EOS token: Should be used to terminate generation paths

Usage Patterns:
    1. Deterministic path: Each state has one outgoing transition
    2. Non-deterministic branching: State has multiple outgoing transitions
    3. Wildcard sections: Use -1 to allow freeform text
    4. Termination: Always end paths with EOS token to stop generation cleanly
"""

import json
from typing import Any

from tqdm import tqdm

from vllm.logger import init_logger

logger = init_logger(__name__)


class CustomFSM:
    """
    Finite State Machine for constrained text generation.

    The FSM constrains LLM output by only allowing specific token sequences.
    Each state defines which tokens are valid next, creating a graph of
    allowed generation paths.

    Attributes:
        graph: FSM structure as dict[state_id, dict[token_id, next_state_id]]
        start_state: Initial state (always 0)
        _state_counter: Counter for generating unique state IDs
    """

    def __init__(self):
        """Initialize empty FSM with start state 0."""
        self.graph: dict[int, dict[int, int]] = {}
        self.start_state = 0
        self._state_counter = 0

    def _new_state(self) -> int:
        """Generate a new unique state ID."""
        self._state_counter += 1
        return self._state_counter

    @classmethod
    def from_prebuilt(cls, file_path: str) -> "CustomFSM":
        """
        Load FSM from a prebuilt JSON file.

        Args:
            file_path: Path to JSON file containing FSM graph

        Returns:
            CustomFSM instance with loaded graph
        """
        logger.info("Loading FSM from %s...", file_path)
        with open(file_path) as f:
            graph_data = json.load(f)

        fsm = cls()
        fsm.graph = {
            int(k): {int(tk): int(v) for tk, v in edges.items()}
            for k, edges in tqdm(graph_data.items(), desc="Processing states")
        }
        return fsm

    def save(self, output_file: str):
        """
        Save FSM graph to JSON file.

        Args:
            output_file: Path where FSM graph will be saved
        """
        with open(output_file, "w") as f:
            json.dump(self.graph, f)

    def _add_transition(self, from_state: int, token: int, to_state: int):
        """
        Add a transition from_state --token--> to_state.

        Args:
            from_state: Source state ID
            token: Token ID that triggers transition (-1 for wildcard)
            to_state: Destination state ID
        """
        if from_state not in self.graph:
            self.graph[from_state] = {}
        self.graph[from_state][token] = to_state

    def _build_freeform_section(self, length: int) -> list[int]:
        """
        Create N freeform states as linked list.
        Uses -1 token_id to indicate all tokens allowed.

        Returns:
            List of freeform states in order [state0, state1, ..., stateN-1]
        """
        states = [self._new_state() for _ in range(length)]

        for i in range(length - 1):
            self._add_transition(states[i], -1, states[i + 1])

        return states

    def build_fsm(self, inps: Any, tokenizer: Any):
        """
        Build FSM graph structure.

        This method should be implemented to construct the FSM graph based on
        your specific requirements. Use helper methods:
        - _build_freeform_section(length): Create freeform states
        - _add_transition(from_state, token, to_state): Add transitions
        - _new_state(): Create new states

        Example structure:
        1. Build fixed text sections using tries
        2. Insert freeform sections where needed
        3. Connect components together
        """
        raise NotImplementedError(
            "_build_graph must be implemented in subclass or instance"
        )

    def get_next_tokens(self, prefix: list[int]) -> list[int]:
        """Get all valid next tokens given a prefix sequence."""
        current_state = self.start_state

        for token in prefix:
            if current_state not in self.graph:
                return []
            if token in self.graph[current_state]:
                current_state = self.graph[current_state][token]
            elif -1 in self.graph[current_state]:
                current_state = self.graph[current_state][-1]
            else:
                return []

        if current_state in self.graph:
            # If wildcard, return empty list (all tokens allowed)
            if -1 in self.graph[current_state]:
                return []
            return list(self.graph[current_state].keys())
        return []

    def get_next_state(self, current_state: int | None, token: int) -> int | None:
        """Get next state given current state and token.

        Returns:
            Next state ID, or None if transition is invalid
        """
        if current_state is None or current_state not in self.graph:
            return None

        # Check for specific token first (allows exit from freeform)
        if token in self.graph[current_state]:
            return self.graph[current_state][token]

        # Fall back to wildcard transition
        if -1 in self.graph[current_state]:
            return self.graph[current_state][-1]

        return None

    def get_tokens_from_state(self, state: int | None) -> list[int]:
        """Get all valid next tokens from a given state.

        Args:
            state: Current FSM state ID

        Returns:
            List of valid token IDs, or [] if all tokens allowed (wildcard -1)
        """
        if state is None or state not in self.graph:
            return []

        # If wildcard exists, return empty list to indicate all tokens allowed
        if -1 in self.graph[state]:
            return []

        return list(self.graph[state].keys())
