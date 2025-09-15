# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Hashable, KeysView, Sequence
from dataclasses import dataclass, field
from typing import Optional, Union, List, NamedTuple
import torch


class Candidate(NamedTuple):
    """Result of suffix tree speculation from C++ implementation."""
    token_ids: List[int]
    parents: List[int]
    probs: List[float]
    score: float
    match_len: int


class SuffixTree:
    """Python wrapper for the C++ SuffixTree implementation using TORCH_LIBRARY."""
    
    def __init__(self, max_depth: int):
        """Initialize a new suffix tree.
        
        Args:
            max_depth: Maximum depth of the suffix tree.
        """
        self._handle = torch.ops._suffix_cache.suffix_tree_create(max_depth)
        self._destroyed = False
    
    def __del__(self):
        """Clean up the C++ suffix tree object."""
        if hasattr(self, '_handle') and not self._destroyed:
            torch.ops._suffix_cache.suffix_tree_destroy(self._handle)
            self._destroyed = True
    
    def num_seqs(self) -> int:
        """Get the number of sequences in the suffix tree."""
        return int(torch.ops._suffix_cache.suffix_tree_num_seqs(self._handle))
    
    def append(self, seq_id: int, token: int) -> None:
        """Append a new element to the sequence with id seq_id.
        
        Args:
            seq_id: ID of the sequence to append to.
            token: Token to append.
        """
        torch.ops._suffix_cache.suffix_tree_append(self._handle, seq_id, token)
    
    def extend(self, seq_id: int, tokens: List[int]) -> None:
        """Append multiple new elements to the sequence with id seq_id.
        
        Args:
            seq_id: ID of the sequence to extend.
            tokens: List of tokens to append.
        """
        tokens_tensor = torch.tensor(tokens, dtype=torch.int64)
        torch.ops._suffix_cache.suffix_tree_extend(self._handle, seq_id, tokens_tensor)
    
    def remove(self, seq_id: int) -> None:
        """Remove the sequence with id seq_id.
        
        Args:
            seq_id: ID of the sequence to remove.
        """
        torch.ops._suffix_cache.suffix_tree_remove(self._handle, seq_id)
    
    def speculate(self, 
                  pattern: List[int],
                  max_spec_tokens: int,
                  max_spec_factor: float = 1.0,
                  max_spec_offset: float = 0.0,
                  min_token_prob: float = 0.1,
                  use_tree_spec: bool = False) -> Candidate:
        """Given a pattern, speculate the next tokens using the suffix tree.
        
        Args:
            pattern: The pattern to match.
            max_spec_tokens: Maximum number of tokens to speculate.
            max_spec_factor: Maximum speculation factor.
            max_spec_offset: Maximum speculation offset.
            min_token_prob: Minimum token probability threshold.
            use_tree_spec: Whether to use tree-based speculation.
            
        Returns:
            Candidate object containing speculation results.
        """
        pattern_tensor = torch.tensor(pattern, dtype=torch.int64)
        
        token_ids, parents, probs, score, match_len = torch.ops._suffix_cache.suffix_tree_speculate(
            self._handle, 
            pattern_tensor,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec
        )
        
        return Candidate(
            token_ids=token_ids.tolist(),
            parents=parents.tolist(),
            probs=probs.tolist(),
            score=float(score),
            match_len=int(match_len)
        )
    
    def check_integrity(self) -> str:
        """Check the integrity of the suffix tree.
        
        Returns:
            Empty string if ok, otherwise an error message.
        """
        return torch.ops._suffix_cache.suffix_tree_check_integrity(self._handle)
    
    def estimate_memory(self) -> int:
        """Estimate memory usage of the suffix tree.
        
        Note: This walks the entire tree so can be slow.
        
        Returns:
            Estimated memory usage in bytes.
        """
        return int(torch.ops._suffix_cache.suffix_tree_estimate_memory(self._handle))


@dataclass
class SuffixSpecResult:
    """
    A dataclass representing the result of a speculation using SuffixDecoding.

    Attributes:
        token_ids (List[int]): List of token IDs in the speculation result.
        parents (List[int]): List of parent indices for each token used to
            encode the tree structure. The parent token of token_ids[i] is
            token_ids[parents[i]].
        probs (List[float]): List of estimated probabilities for each token.
        score (float): The overall score of the suffix match computed as the
            sum of the estimated probabilities of each speculated token.
        match_len (int): The length of the pattern match that yielded this
            speculation result.
    """
    token_ids: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    score: float = 0.0
    match_len: int = 0

    @staticmethod
    def from_candidate(candidate: Candidate) -> SuffixSpecResult:
        return SuffixSpecResult(
            token_ids=candidate.token_ids,
            parents=candidate.parents,
            probs=candidate.probs,
            score=candidate.score,
            match_len=candidate.match_len,
        )


class SuffixCache:

    def __init__(self,
                 max_tree_depth: int = 64,
                 max_cached_requests: int = 1000):
        """
        Initialize the SuffixCache.

        Args:
            max_tree_depth (int): The maximum depth of the suffix trees.
            max_cached_requests (int, optional): The maximum number of cached
                requests. Cache eviction is used when the limit is reached. If
                `None`, there is no limit on the number of cached requests.
        """
        self._max_tree_depth = max_tree_depth
        self._max_cached_requests = max_cached_requests

        # Global suffix tree caches previous responses in a single tree.
        self._global_tree = SuffixTree(max_tree_depth)

        # Local suffix trees cache prompts for each active request separately.
        self._local_trees: dict[Hashable, SuffixTree] = {}

        # Maps between Python request ID and int32_t sequence ID. Tracks all
        # request IDs that are in the global tree or one of the local trees.
        self._req_to_seq_id: dict[Hashable, int] = {}
        self._seq_to_req_id: dict[int, Hashable] = {}

        # Unused sequence ID to assign to a new request ID.
        self._next_seq_id = 0

    @property
    def max_tree_depth(self) -> int:
        return self._max_tree_depth

    @property
    def max_cached_requests(self) -> int:
        return self._max_cached_requests

    @property
    def active_requests(self) -> KeysView:
        """
        Returns a view of the currently active request IDs. Active requests are
        those that have been started via `start_request` and not yet stopped
        via `stop_request`. The prompts of active requests are stored so they
        can be used during speculation for the same request.
        """
        return self._local_trees.keys()

    @property
    def cached_requests(self) -> KeysView:
        """
        Returns a view of all request IDs that have their responses cached in
        the global suffix tree. The response for the cached request can be used
        during speculation for other requests, until the response is evicted.
        """
        return self._req_to_seq_id.keys()

    def start_request(self, req_id: Hashable, prompt_token_ids: Sequence[int]):
        """
        This method should be called when starting to process a new request. It
        will store the prompt for the request, allowing future speculations for
        the same request to use the prompt context. The prompt will be stored
        until `stop_request` is called.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (Sequence[int]): A sequence of token IDs
                representing the prompt of the request.

        Raises:
            ValueError: If a request with the same `req_id` is already active
                or cached.
        """
        if req_id in self._req_to_seq_id:
            raise ValueError(f"Request '{req_id}' is already active or cached")
        seq_id = self._generate_seq_id(req_id)
        self._local_trees[req_id] = SuffixTree(self._max_tree_depth)
        self._local_trees[req_id].extend(seq_id, prompt_token_ids)

    def stop_request(self, req_id: Hashable):
        """
        This method should be called when a request is completed. It will evict
        the prompt for the request, freeing up memory.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")
        del self._local_trees[req_id]

    def add_active_response(
        self,
        req_id: Hashable,
        token_ids: Union[int, Sequence[int]],
    ):
        """
        Update the cached response for a given request by appending token(s) to
        its end. Once the response is updated, the new tokens can be used for
        future speculations for all requests.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Union[int, Sequence[int]]): Either a single token ID
                (int) or a sequence of token IDs to be appended to the response
                for the given request.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")
        seq_id = self._req_to_seq_id[req_id]
        if isinstance(token_ids, int):
            self._global_tree.append(seq_id, token_ids)
            self._local_trees[req_id].append(seq_id, token_ids)
        else:
            self._global_tree.extend(seq_id, token_ids)
            self._local_trees[req_id].extend(seq_id, token_ids)

    def insert_new_response(
        self,
        req_id: Hashable,
        token_ids: Union[int, Sequence[int]],
    ):
        """
        Insert a complete response to the global cache for a request that is
        not active and is not already cached.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (Sequence[int]): A sequence of token IDs to be inserted
                as the response for the given request.

        Raises:
            ValueError: If a request with the same `req_id` is already active
                or cached.
        """
        if req_id in self._req_to_seq_id:
            raise ValueError(f"Request '{req_id}' is already active or cached")
        seq_id = self._generate_seq_id(req_id)
        self._global_tree.extend(seq_id, token_ids)

    def evict_request(self, req_id: Hashable):
        """
        Evicts the given request's prompt and response from the cache. If the
        request is active, it becomes inactive. The `req_id` can then be reused
        after eviction.

        Args:
            req_id (Hashable): The unique identifier for the request that
                should be evicted.

        Raises:
            ValueError: If no response exists for the given request identifier.
        """
        if req_id not in self._req_to_seq_id:
            raise ValueError(f"Request '{req_id}' is not active or cached")
        if req_id in self._local_trees:
            del self._local_trees[req_id]
        seq_id = self._req_to_seq_id.pop(req_id)
        self._seq_to_req_id.pop(seq_id)
        self._global_tree.remove(seq_id)

    def speculate(
        self,
        req_id: Hashable,
        pattern: Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> SuffixSpecResult:
        """
        Speculates and returns the most likely continuation of a given token
        pattern using the request's prompt and the global cache of previous
        responses. This method can only be called for active requests (i.e.
        after calling `start_request` and before calling `stop_request`).

        Args:
            req_id (Hashable): The unique identifier for the request.
            pattern (Sequence[int]): The sequence of token IDs to match and
                continue from.
            max_spec_tokens (int): Maximum number of tokens to speculate. If 0,
                uses the cache's max_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched pattern length.
            min_token_prob (float): Minimum estimated probability threshold for
                candidate tokens.
            use_tree_spec (bool): If True, uses tree-based speculation.
        
        Returns:
            The speculation result containing the most likely continuation
            tokens, their probabilities, and overall score.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._local_trees:
            raise ValueError(f"Request '{req_id}' is not active")

        if max_spec_tokens is None:
            max_spec_tokens = self._max_tree_depth

        if len(pattern) > self._max_tree_depth:
            pattern = pattern[-self._max_tree_depth:]

        candidate = self._local_trees[req_id].speculate(
            pattern, max_spec_tokens, max_spec_factor, max_spec_offset,
            min_token_prob, use_tree_spec)
        result = SuffixSpecResult.from_candidate(candidate)

        candidate = self._global_tree.speculate(pattern, max_spec_tokens,
                                                max_spec_factor,
                                                max_spec_offset,
                                                min_token_prob, use_tree_spec)
        if candidate.score > result.score:
            result = SuffixSpecResult.from_candidate(candidate)

        return result

    def _generate_seq_id(self, req_id: Hashable) -> int:
        # Find the next available seq_id not used by an active request.
        while True:
            seq_id = self._next_seq_id
            # Increment to the next non-negative int32_t value.
            self._next_seq_id = (self._next_seq_id + 1) & 0x7FFFFFFF
            if (seq_id not in self._seq_to_req_id
                    or self._seq_to_req_id[seq_id] not in self._local_trees):
                break
        # Check if the seq_id is used by an inactive but cached request.
        if seq_id in self._seq_to_req_id:
            # This seq_id is already used, should be a very rare case that
            # only happens when the seq_id has wrapped around and collided.
            # We evict the old cached request to free up the seq_id.
            del self._req_to_seq_id[self._seq_to_req_id[seq_id]]
            del self._seq_to_req_id[seq_id]
            self._global_tree.remove(seq_id)
        # Allocate the seq_id to the new req_id.
        self._req_to_seq_id[req_id] = seq_id
        self._seq_to_req_id[seq_id] = req_id
        self._maybe_evict_requests(seq_id)
        return seq_id

    def _maybe_evict_requests(self, new_seq_id: int):
        if self._max_cached_requests is None:
            return
        while len(self._req_to_seq_id) > self._max_cached_requests:
            # Evict the first request that is not active. Should be FIFO order
            # in python 3.7+ as dict preserves insertion order. We also want to
            # avoid evicting the request that was just added (new_seq_id).
            for req_id, seq_id in self._req_to_seq_id.items():
                if seq_id != new_seq_id and req_id not in self._local_trees:
                    self.evict_request(req_id)
                    break
            else:
                # All previously cached requests are active, cannot evict any.
                break
