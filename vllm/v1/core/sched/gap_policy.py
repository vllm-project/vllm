# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gap Policy for KV Cache Recomputation

This module provides abstractions for deciding where to insert recomputation gaps
within prefix-cached tokens. Gap policies are independent of where cached tokens
came from (local prefix cache, external connector, or both).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class GapPolicy(ABC):
    """
    Decides where to insert recomputation gaps within prefix-cached tokens.
    
    Gap policies are independent of where cached tokens came from (local prefix
    cache, external connector, or both). They operate on the unified view of
    all computed tokens.
    """
    
    @abstractmethod
    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        """
        Return gap intervals within [0, num_computed_tokens) to recompute.
        
        Args:
            request: The request object containing prompt tokens and metadata
            num_computed_tokens: Total cached tokens (local + external)
            num_external_tokens: Number of tokens from external connector
            
        Returns:
            List of (start, end) tuples representing half-open intervals [start, end)
            that should be recomputed. Intervals must be:
            - Within bounds: 0 <= start < end <= num_computed_tokens
            - Non-overlapping and strictly increasing
            - Empty list means no gaps (use all cached tokens)
        """
        pass


class NoGapPolicy(GapPolicy):
    """Default policy: no gaps, use all cached tokens."""
    
    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        """Return empty list - no gaps."""
        return []


class SpanAwareGapPolicy(GapPolicy):
    """
    Creates gaps at span boundaries marked by specific token IDs.
    
    This policy identifies span start positions (e.g., token_id=10) and creates
    gaps of configurable length at those positions. Useful for segmented prefill
    scenarios where certain token boundaries need recomputation.
    
    The gap logic is migrated from SegmentedPrefillExampleConnector._choose_gaps()
    to work uniformly across all cached tokens (local + external).
    """
    
    DEFAULT_GAP_LENGTH = 32
    DEFAULT_SPAN_MARKER_TOKEN_ID = 10
    DEFAULT_MIN_EXTERNAL_TOKENS = 32  # 2 blocks * 16 tokens/block
    
    def __init__(
        self,
        gap_length: int = DEFAULT_GAP_LENGTH,
        span_marker_token_id: int = DEFAULT_SPAN_MARKER_TOKEN_ID,
        min_external_tokens: int = DEFAULT_MIN_EXTERNAL_TOKENS,
        block_size: int = 16,
    ):
        """
        Initialize SpanAwareGapPolicy.
        
        Args:
            gap_length: Length of each gap in tokens (0 disables gaps)
            span_marker_token_id: Token ID that marks span boundaries
            min_external_tokens: Minimum external tokens required to create gaps
            block_size: Block size for alignment (used in debug output)
        """
        self.gap_length = gap_length
        self.span_marker_token_id = span_marker_token_id
        self.min_external_tokens = min_external_tokens
        self.block_size = block_size
        
        logger.info(
            "SpanAwareGapPolicy initialized: gap_length=%d, "
            "span_marker_token_id=%d, min_external_tokens=%d",
            gap_length, span_marker_token_id, min_external_tokens
        )
    
    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        """
        Create gaps at span boundaries within external token range.
        
        Logic migrated from SegmentedPrefillExampleConnector._choose_gaps()
        """
        # Disable gaps if gap_length is 0 or insufficient external tokens
        if self.gap_length <= 0:
            return []
        
        # Calculate external token range
        external_start = num_computed_tokens - num_external_tokens
        external_end = num_computed_tokens
        
        logger.debug(
            "Choosing gaps: external_start=%d, external_end=%d, "
            "num_computed_tokens=%d, num_external_tokens=%d",
            external_start, external_end, num_computed_tokens, num_external_tokens
        )
        
        if external_end - external_start < self.min_external_tokens:
            logger.debug(
                "Insufficient external tokens (%d < %d), no gaps created",
                external_end - external_start, self.min_external_tokens
            )
            return []
        
        # Find all span start positions in external range
        span_starts = []
        if request.prompt_token_ids:
            for i, token_id in enumerate(request.prompt_token_ids):
                if (token_id == self.span_marker_token_id and 
                    external_start <= i < external_end):
                    span_starts.append(i)
        
        if not span_starts:
            logger.debug("No span markers found in external range, no gaps created")
            return []
        
        logger.debug("Found span starts at positions: %s", span_starts)
        
        # Create gaps for each span
        gaps = []
        for idx, gap_start in enumerate(span_starts):
            # Find end of this span (next span start or external_end)
            next_span_start = (
                span_starts[idx + 1] if idx + 1 < len(span_starts) 
                else external_end
            )
            
            span_length = next_span_start - gap_start
            logger.debug(
                "Span at %d: length=%d, next_span at %d",
                gap_start, span_length, next_span_start
            )
            
            # Gap length is min(gap_length, span_length)
            gap_end = min(
                gap_start + self.gap_length,
                next_span_start,
                external_end
            )
            
            if gap_end > gap_start:
                logger.debug("Adding gap: (%d, %d)", gap_start, gap_end)
                gaps.append((gap_start, gap_end))
        
        logger.info(
            "Created %d gaps for request %s: %s",
            len(gaps), request.request_id, gaps
        )
        
        # Print visual representation for debugging
        self._print_gaps_representation(
            gaps, num_external_tokens, num_computed_tokens
        )
        
        return gaps
    
    def _print_gaps_representation(
        self,
        gaps: list[tuple[int, int]],
        num_external_tokens: int,
        num_computed_tokens: int,
    ) -> None:
        """Print a human-readable representation of the tokens and gaps for debugging."""
        total_tokens = num_computed_tokens
        block_size = self.block_size
        representation = []
        
        for block_start in range(0, total_tokens, block_size):
            block_end = min(block_start + block_size, total_tokens)
            block_chars = []
            
            for i in range(block_start, block_end):
                if i < num_computed_tokens - num_external_tokens:
                    block_chars.append("C")  # Computed token (local)
                else:
                    # Check if in gap
                    in_gap = any(start <= i < end for start, end in gaps)
                    block_chars.append("-" if in_gap else "E")  # Gap or External token
            
            # Determine the character for this block
            unique_chars = set(block_chars)
            # Print 'X' if mixed token types in block
            char = unique_chars.pop() if len(unique_chars) == 1 else "X"
            representation.append(char)
        
        logger.debug("Cache status per block (C=computed, E=external, -=gap, X=mixed):")
        logger.debug("".join(representation))
        logger.debug("Gaps: %s", gaps)
        logger.debug(
            "Total tokens: %d, computed tokens: %d, external tokens: %d",
            total_tokens, num_computed_tokens, num_external_tokens
        )


class GapPolicyFactory:
    """Factory for creating GapPolicy instances from configuration."""
    
    _POLICIES = {
        "none": NoGapPolicy,
        "span_aware": SpanAwareGapPolicy,
    }
    
    @classmethod
    def create_policy(
        cls,
        policy_name: Optional[str] = None,
        policy_config: Optional[dict] = None,
    ) -> Optional[GapPolicy]:
        """
        Create a GapPolicy instance from configuration.
        
        Args:
            policy_name: Name of the policy ("none", "span_aware", or None)
            policy_config: Configuration dict for the policy
            
        Returns:
            GapPolicy instance or None if policy_name is None
        """
        if policy_name is None:
            return None
        
        policy_name_lower = policy_name.lower()
        if policy_name_lower not in cls._POLICIES:
            logger.warning(
                "Unknown gap policy '%s'. Available: %s. Using NoGapPolicy.",
                policy_name, list(cls._POLICIES.keys())
            )
            policy_name_lower = "none"
        
        policy_class = cls._POLICIES[policy_name_lower]
        policy_config = policy_config or {}
        
        try:
            return policy_class(**policy_config)
        except TypeError as e:
            logger.error(
                "Failed to create %s policy with config %s: %s. Using NoGapPolicy.",
                policy_name, policy_config, e
            )
            return NoGapPolicy()
    
    @classmethod
    def register_policy(cls, name: str, policy_class: type[GapPolicy]) -> None:
        """
        Register a custom gap policy.
        
        Args:
            name: Name to register the policy under
            policy_class: GapPolicy subclass to register
        """
        if not issubclass(policy_class, GapPolicy):
            raise ValueError(f"{policy_class} must be a subclass of GapPolicy")
        
        cls._POLICIES[name.lower()] = policy_class
        logger.info("Registered gap policy: %s -> %s", name, policy_class.__name__)

