# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PerRequestCascadeState:
    """Per-request state for Cascade utility-driven adaptive k.

    Cascade operates in two phases:
    - Test: Probe k=1..K_max to measure acceptance rates and costs
    - Set: Lock K* (utility-maximizing k) for stable throughput

    utility(k) = tokens_accepted / verification_cost(k)
    
    This addresses the MoE problem where verification data movement scales
    with k, making static k suboptimal. Cascade paper (MLSys 2026):
    https://arxiv.org/abs/2506.20675
    """
    req_id: str
    phase: str  # "test" or "set"
    test_k_values: List[int] = field(default_factory=list)
    acceptance_history: Dict[int, List[int]] = field(default_factory=dict)
    current_k: int = 1
    optimal_k: int = 1
    utility_scores: Dict[int, List[float]] = field(default_factory=dict)
    step_count: int = 0
    num_disabled_steps: int = 0  # Track steps where spec decode was disabled
    
    def calculate_utility(
        self, 
        k: int, 
        num_accepted: int,
        alpha: float = 0.1,
    ) -> float:
        """Calculate utility = tokens_accepted / verification_cost(k)
        
        Args:
            k: Number of draft tokens
            num_accepted: Number of tokens that passed verification
            alpha: MoE expert activation overhead multiplier
        
        Returns:
            Utility score (higher is better)
        """
        # Cost model: verification_cost(k) = 1 + alpha*k
        # For dense models: alpha ≈ 0, cost ≈ 1 (constant)
        # For MoE: alpha represents expert routing overhead per draft
        cost = 1.0 + alpha * k
        utility = num_accepted / cost if cost > 0 else 0.0
        return utility
    
    def get_mean_utility(self, k: int) -> float:
        """Get mean utility for a given k across all observations."""
        if k not in self.utility_scores or not self.utility_scores[k]:
            return 0.0
        return float(np.mean(self.utility_scores[k]))
    
    def should_transition_to_set_phase(
        self, 
        min_test_samples: int = 5,
    ) -> bool:
        """Check if all k values have been tested sufficiently."""
        if not self.acceptance_history:
            return False
        return all(
            len(history) >= min_test_samples
            for history in self.acceptance_history.values()
        )
    
    def find_optimal_k(self) -> int:
        """Find k with highest mean utility."""
        best_k = 1
        best_utility = 0.0
        for k, utilities in self.utility_scores.items():
            if utilities:
                mean_util = float(np.mean(utilities))
                if mean_util > best_utility:
                    best_utility = mean_util
                    best_k = k
        
        logger.info(
            f"Cascade req {self.req_id}: K*={best_k} with utility={best_utility:.3f}"
        )
        return best_k


class CascadeController:
    """Manages Cascade state for all requests in a batch.
    
    Implements the Cascade algorithm:
    1. Test phase: cycle through k=1..K_max, measure acceptance and cost
    2. Set phase: lock K*, disable if utility(K*) < 1.0
    
    Expected improvements (from paper on H100):
    - Worst-case slowdown: 54% → 5%
    - Throughput: +7-14% on MoE models
    """
    
    def __init__(
        self,
        k_max: int = 5,
        test_phase_steps: int = 5,
        set_phase_min_steps: int = 10,
        moe_overhead_alpha: float = 0.1,
    ):
        """Initialize Cascade controller.
        
        Args:
            k_max: Maximum number of speculative tokens to test
            test_phase_steps: Minimum steps to test each k value
            set_phase_min_steps: Minimum steps in set phase before re-evaluation
            moe_overhead_alpha: MoE expert activation cost multiplier
        """
        self.k_max = k_max
        self.test_phase_steps = test_phase_steps
        self.set_phase_min_steps = set_phase_min_steps
        self.moe_overhead_alpha = moe_overhead_alpha
        
        # Per-request state: req_id -> PerRequestCascadeState
        self.per_request_state: Dict[str, PerRequestCascadeState] = {}
    
    def get_or_create_state(self, req_id: str) -> PerRequestCascadeState:
        """Get existing or create new cascade state for a request."""
        if req_id not in self.per_request_state:
            state = PerRequestCascadeState(
                req_id=req_id,
                phase="test",
                acceptance_history={k: [] for k in range(1, self.k_max + 1)},
                utility_scores={k: [] for k in range(1, self.k_max + 1)},
                current_k=1,
                optimal_k=self.k_max,
            )
            self.per_request_state[req_id] = state
        return self.per_request_state[req_id]
    
    def get_k_per_request(self, req_ids: List[str]) -> List[int]:
        """Get current k value for each request in batch."""
        k_values = []
        for req_id in req_ids:
            state = self.get_or_create_state(req_id)
            k_values.append(state.current_k)
        return k_values
    
    def update_after_verification(
        self,
        req_ids: List[str],
        num_accepted_per_req: List[int],
        k_per_req: List[int],
    ) -> None:
        """Update cascade state after token verification.
        
        Implements phase transitions and utility tracking.
        """
        for req_id, num_accepted, k in zip(req_ids, num_accepted_per_req, k_per_req):
            if k == 0:
                # Speculation disabled for this request
                continue
                
            state = self.get_or_create_state(req_id)
            state.step_count += 1
            
            # Record acceptance
            state.acceptance_history[k].append(num_accepted)
            
            # Calculate and record utility
            utility = state.calculate_utility(
                k, num_accepted, alpha=self.moe_overhead_alpha
            )
            state.utility_scores[k].append(utility)
            
            # Phase transitions
            if state.phase == "test":
                if state.should_transition_to_set_phase(self.test_phase_steps):
                    state.phase = "set"
                    state.optimal_k = state.find_optimal_k()
                    state.current_k = state.optimal_k
                    logger.info(
                        f"Cascade req {req_id}: transitioned to SET phase with K*={state.optimal_k}"
                    )
                else:
                    # Cycle to next k value in test phase
                    state.current_k = (state.current_k % self.k_max) + 1
            else:  # set phase
                # Periodically re-evaluate utility
                if state.step_count % self.set_phase_min_steps == 0:
                    mean_util = state.get_mean_utility(state.current_k)
                    if mean_util < 1.0:
                        logger.warning(
                            f"Cascade req {req_id}: disabling spec decode "
                            f"(utility={mean_util:.3f} < 1.0)"
                        )
                        state.current_k = 0  # Disable spec decoding
                        state.num_disabled_steps += 1
    
    def cleanup_request(self, req_id: str) -> None:
        """Remove state for completed request."""
        if req_id in self.per_request_state:
            state = self.per_request_state[req_id]
            if state.num_disabled_steps > 0:
                logger.info(
                    f"Cascade req {req_id}: disabled for {state.num_disabled_steps} steps"
                )
            del self.per_request_state[req_id]
    
    def clear_all(self) -> None:
        """Clear all request states."""
        self.per_request_state.clear()
