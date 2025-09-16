# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Hashable
from typing import Optional

import numpy as np

from vllm.config import VllmConfig
from vllm.v1.spec_decode.suffix_decode.suffix_cache import SuffixCache


class SuffixProposer:
    """Proposer for suffix-decoding based speculative decoding."""

    def __init__(self, vllm_config: VllmConfig):
        self.spec_config = vllm_config.speculative_config
        self.vllm_config = vllm_config

        # Initialize suffix cache with configuration parameters
        self._suffix_cache = SuffixCache(
            max_tree_depth=self.spec_config.suffix_cache_max_depth,
            max_cached_requests=self.spec_config.suffix_cache_max_requests)

        self.max_spec_tokens = self.spec_config.num_speculative_tokens
        self.max_spec_factor = self.spec_config.suffix_max_spec_factor
        self.max_spec_offset = self.spec_config.suffix_max_spec_offset
        self.min_token_prob = self.spec_config.suffix_min_token_prob

        # Track active requests
        self._active_requests: set[Hashable] = set()

    def start_request(self, req_id: Hashable, prompt_token_ids: list[int]):
        """Start tracking a new request."""
        if req_id not in self._active_requests:
            self._suffix_cache.start_request(req_id, prompt_token_ids)
            self._active_requests.add(req_id)

    def stop_request(self, req_id: Hashable):
        """Stop tracking a request."""
        if req_id in self._active_requests:
            self._suffix_cache.stop_request(req_id)
            self._active_requests.remove(req_id)

    def update_response(self, req_id: Hashable, token_ids: list[int]):
        """Update the cached response for a request."""
        if req_id in self._active_requests:
            self._suffix_cache.add_active_response(req_id, token_ids)

    def propose(self,
                context_token_ids: np.ndarray,
                req_id: Optional[Hashable] = None) -> Optional[np.ndarray]:
        """Propose speculative tokens based on suffix matching."""
        if req_id is None or req_id not in self._active_requests:
            # If no request ID or not an active request, return empty proposal
            return None

        # Convert numpy array to list for pattern matching
        pattern = context_token_ids.tolist()

        # Get speculation result from suffix cache
        result = self._suffix_cache.speculate(
            req_id=req_id,
            pattern=pattern,
            max_spec_tokens=self.max_spec_tokens,
            max_spec_factor=self.max_spec_factor,
            max_spec_offset=self.max_spec_offset,
            min_token_prob=self.min_token_prob,
            use_tree_spec=False  # TODO: Add configuration for tree speculation
        )

        if result.token_ids:
            return np.array(result.token_ids, dtype=np.int32)
        else:
            return None

    def load_model(self, *args, **kwargs):
        # No model to load for suffix-decode based speculative decoding
        pass
