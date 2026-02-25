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
    
    def __init__(
        self,
        gap_length: int = DEFAULT_GAP_LENGTH,
        span_marker_token_id: int = DEFAULT_SPAN_MARKER_TOKEN_ID,
        block_size: int = 16,
    ):
        """
        Initialize SpanAwareGapPolicy.
        
        Args:
            gap_length: Length of each gap in tokens (0 disables gaps)
            span_marker_token_id: Token ID that marks span boundaries
            block_size: Block size for alignment (used in debug output)
        """
        self.gap_length = gap_length
        self.span_marker_token_id = span_marker_token_id
        self.block_size = block_size
        
        logger.info(
            "SpanAwareGapPolicy initialized: gap_length=%d, "
            "span_marker_token_id=%d",
            gap_length, span_marker_token_id
        )
    
    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        """
        Create gaps at span boundaries within ALL computed tokens (local + external).
        
        This allows gaps to be created across the entire cached token range,
        not just in externally-loaded tokens.
        """
        # Disable gaps if gap_length is 0
        if self.gap_length <= 0:
            return []
        
        if num_computed_tokens == 0:
            return []
        
        logger.debug(
            "Choosing gaps: num_computed_tokens=%d, num_external_tokens=%d",
            num_computed_tokens, num_external_tokens
        )
        
        # Find all span start positions in the ENTIRE computed token range
        span_starts = []
        if request.prompt_token_ids:
            for i, token_id in enumerate(request.prompt_token_ids):
                if (token_id == self.span_marker_token_id and
                    i < num_computed_tokens):
                    span_starts.append(i)
        
        if not span_starts:
            logger.debug(
                "No span markers found in computed range [0, %d), no gaps created",
                num_computed_tokens
            )
            return []
        
        logger.debug(
            "Found %d span markers at positions: %s",
            len(span_starts), span_starts
        )
        
        # Create gaps for each span
        gaps = []
        for idx, gap_start in enumerate(span_starts):
            # Find end of this span (next span start or end of computed tokens)
            next_span_start = (
                span_starts[idx + 1] if idx + 1 < len(span_starts)
                else num_computed_tokens
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
                num_computed_tokens
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
        
        # Calculate local token boundary
        num_local_tokens = num_computed_tokens - num_external_tokens
        
        for block_start in range(0, total_tokens, block_size):
            block_end = min(block_start + block_size, total_tokens)
            block_chars = []
            
            for i in range(block_start, block_end):
                # Check if token is in a gap
                in_gap = any(start <= i < end for start, end in gaps)
                
                if in_gap:
                    block_chars.append("-")  # Gap (will be recomputed)
                elif i < num_local_tokens:
                    block_chars.append("L")  # Local cached token
                else:
                    block_chars.append("E")  # External cached token
            
            # Determine the character for this block
            unique_chars = set(block_chars)
            # Print 'X' if mixed token types in block
            char = unique_chars.pop() if len(unique_chars) == 1 else "X"
            representation.append(char)
        
        logger.debug("Cache status per block (L=local, E=external, -=gap, X=mixed):")
        logger.debug("".join(representation))
        logger.debug("Gaps: %s", gaps)
        logger.debug(
            "Total tokens: %d (local: %d, external: %d)",
            total_tokens, num_local_tokens, num_external_tokens
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

