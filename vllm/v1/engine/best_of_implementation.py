# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Implementation of best-of sampling for vLLM v1.

Best-of sampling generates multiple sequences and returns the best ones
based on cumulative log probability scores.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request


@dataclass
class SequenceData:
    """Tracks data for a single sequence in best-of sampling."""
    sequence_id: int  # Index within the request (0 to best_of-1)
    token_ids: List[int]
    cumulative_logprob: float
    is_finished: bool
    finish_reason: Optional[str] = None


class BestOfTracker:
    """
    Manages multiple sequences for best-of sampling per request.
    
    This class tracks multiple sequences being generated for each request
    and handles the selection of the best sequences based on their
    cumulative log probabilities.
    """
    
    def __init__(self):
        # Map from request_id to list of sequences
        self.request_sequences: Dict[str, List[SequenceData]] = {}
        # Map from request_id to original n value (before best_of modification)
        self.original_n_values: Dict[str, int] = {}
        # Map from (request_id, sequence_id) to cumulative logprob
        self.cumulative_logprobs: Dict[Tuple[str, int], float] = {}
    
    def init_request(self, request_id: str, best_of: int, n: int) -> None:
        """Initialize tracking for a new request with best_of sampling."""
        self.request_sequences[request_id] = [
            SequenceData(
                sequence_id=i,
                token_ids=[],
                cumulative_logprob=0.0,
                is_finished=False
            )
            for i in range(best_of)
        ]
        self.original_n_values[request_id] = n
        for i in range(best_of):
            self.cumulative_logprobs[(request_id, i)] = 0.0
    
    def update_sequence(
        self,
        request_id: str,
        sequence_id: int,
        new_token_id: int,
        token_logprob: float,
        is_finished: bool = False,
        finish_reason: Optional[str] = None
    ) -> None:
        """Update a sequence with a new token and its log probability."""
        if request_id not in self.request_sequences:
            return
        
        seq = self.request_sequences[request_id][sequence_id]
        seq.token_ids.append(new_token_id)
        seq.cumulative_logprob += token_logprob
        seq.is_finished = is_finished
        if finish_reason:
            seq.finish_reason = finish_reason
        
        # Update cumulative logprob tracker
        self.cumulative_logprobs[(request_id, sequence_id)] = seq.cumulative_logprob
    
    def get_best_sequences(self, request_id: str) -> List[SequenceData]:
        """
        Get the best n sequences for a request based on cumulative log probability.
        
        Returns:
            List of the top n sequences sorted by cumulative log probability.
        """
        if request_id not in self.request_sequences:
            return []
        
        sequences = self.request_sequences[request_id]
        n = self.original_n_values.get(request_id, 1)
        
        # Sort by cumulative log probability (higher is better)
        sorted_sequences = sorted(
            sequences,
            key=lambda seq: seq.cumulative_logprob,
            reverse=True
        )
        
        # Return top n sequences
        return sorted_sequences[:n]
    
    def cleanup_request(self, request_id: str) -> None:
        """Clean up tracking data for a completed request."""
        self.request_sequences.pop(request_id, None)
        self.original_n_values.pop(request_id, None)
        # Clean up cumulative logprobs
        keys_to_remove = [
            key for key in self.cumulative_logprobs
            if key[0] == request_id
        ]
        for key in keys_to_remove:
            del self.cumulative_logprobs[key]
    
    def is_best_of_request(self, request_id: str) -> bool:
        """Check if a request is using best_of sampling."""
        return request_id in self.request_sequences
    
    def get_sequence_count(self, request_id: str) -> int:
        """Get the number of sequences being generated for a request."""
        if request_id not in self.request_sequences:
            return 1
        return len(self.request_sequences[request_id])


def process_sampling_params_for_best_of(params: SamplingParams) -> Tuple[int, Optional[int]]:
    """
    Process sampling parameters for best_of sampling.
    
    Returns:
        Tuple of (effective_n, original_n) where:
        - effective_n is the number of sequences to generate (best_of value)
        - original_n is the original n value (number of sequences to return)
    """
    if params.best_of is None or params.best_of <= 1:
        return params.n, None
    
    if params.best_of < params.n:
        raise ValueError(
            f"best_of must be greater than or equal to n, "
            f"got n={params.n} and best_of={params.best_of}."
        )
    
    # Return best_of as the effective n, and the original n
    return params.best_of, params.n


def select_best_outputs(
    all_outputs: List[Tuple[List[int], float, Optional[str]]],
    n: int
) -> List[Tuple[List[int], float, Optional[str]]]:
    """
    Select the best n outputs based on cumulative log probability.
    
    Args:
        all_outputs: List of (token_ids, cumulative_logprob, finish_reason) tuples
        n: Number of best outputs to select
    
    Returns:
        List of the top n outputs sorted by cumulative log probability
    """
    # Sort by cumulative log probability (higher is better)
    sorted_outputs = sorted(
        all_outputs,
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_outputs[:n]


# Integration points for v1 engine:

def modify_request_for_best_of(request: Request) -> None:
    """
    Modify a request to handle best_of sampling.
    
    This should be called during request initialization if best_of > 1.
    """
    if request.sampling_params is None:
        return
    
    params = request.sampling_params
    if params.best_of is None or params.best_of <= 1:
        return
    
    # Store original n value
    if not hasattr(params, '_real_n'):
        params._real_n = params.n  # type: ignore
        params.n = params.best_of


def should_use_best_of(params: Optional[SamplingParams]) -> bool:
    """Check if a request should use best_of sampling."""
    if params is None:
        return False
    return params.best_of is not None and params.best_of > 1