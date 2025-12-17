# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accumulator for KV cache tensors during decode phase."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global singleton manager
_accumulator_manager: KVAccumulatorManager | None = None


@dataclass
class RequestKVAccumulator:
    """Accumulates KV tensors for a single request across decode steps."""

    req_id: str
    prompt_token_ids: list[int]
    decode_token_ids: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0

    # layer_idx -> (list of K tensors, list of V tensors)
    # Each tensor is [1, num_kv_heads, head_size] (single decode token)
    layer_data: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]] = field(
        default_factory=dict
    )

    def add_kv(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """
        Add K and V tensors for a decode step.

        Args:
            layer_idx: Layer index
            key: Key tensor [1, num_kv_heads, head_size]
            value: Value tensor [1, num_kv_heads, head_size]
        """
        if layer_idx not in self.layer_data:
            self.layer_data[layer_idx] = ([], [])

        k_list, v_list = self.layer_data[layer_idx]
        k_list.append(key)
        v_list.append(value)

    def add_decode_token(self, token_id: int):
        """Add a decoded token ID."""
        self.decode_token_ids.append(token_id)

    @property
    def num_decode_tokens(self) -> int:
        """Number of decode tokens accumulated."""
        return len(self.decode_token_ids)

    def get_accumulated_kv(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Get accumulated K and V tensors for a layer.

        Returns:
            Tuple of (K, V) tensors with shape [S, num_kv_heads, head_size]
            where S is the number of decode tokens, or None if no data.
        """
        if layer_idx not in self.layer_data:
            return None

        k_list, v_list = self.layer_data[layer_idx]
        if not k_list:
            return None

        # Concatenate along sequence dimension
        K = torch.cat(k_list, dim=0)  # [S, num_kv_heads, head_size]
        V = torch.cat(v_list, dim=0)  # [S, num_kv_heads, head_size]
        return K, V


class KVAccumulatorManager:
    """Manages KV accumulators for multiple requests."""

    def __init__(self):
        self._accumulators: dict[str, RequestKVAccumulator] = {}

    def start_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        num_prompt_tokens: int,
    ):
        """
        Initialize accumulator for a new request.

        Args:
            req_id: Request ID
            prompt_token_ids: Prompt token IDs
            num_prompt_tokens: Number of prompt tokens
        """
        if req_id in self._accumulators:
            logger.warning(f"Request {req_id} already exists, resetting")

        self._accumulators[req_id] = RequestKVAccumulator(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            num_prompt_tokens=num_prompt_tokens,
        )
        logger.debug(
            f"Started KV accumulator for request {req_id} "
            f"with {num_prompt_tokens} prompt tokens"
        )

    def accumulate(
        self,
        req_id: str,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Accumulate K and V tensors for a request.

        Args:
            req_id: Request ID
            layer_idx: Layer index
            key: Key tensor (will be cloned to CPU)
            value: Value tensor (will be cloned to CPU)
        """
        if req_id not in self._accumulators:
            logger.warning(f"Request {req_id} not found, ignoring KV data")
            return

        acc = self._accumulators[req_id]
        acc.add_kv(layer_idx, key, value)

    def add_decode_token(self, req_id: str, token_id: int):
        """
        Add a decoded token ID to the accumulator.

        Args:
            req_id: Request ID
            token_id: Decoded token ID
        """
        if req_id not in self._accumulators:
            return

        self._accumulators[req_id].add_decode_token(token_id)

    def finish_request(self, req_id: str) -> RequestKVAccumulator | None:
        """
        Finish accumulating for a request and return the accumulator.

        Args:
            req_id: Request ID

        Returns:
            The RequestKVAccumulator or None if not found
        """
        acc = self._accumulators.pop(req_id, None)
        if acc is not None:
            logger.debug(
                f"Finished KV accumulator for request {req_id} "
                f"with {acc.num_decode_tokens} decode tokens"
            )
        return acc

    def has_request(self, req_id: str) -> bool:
        """Check if a request is being tracked."""
        return req_id in self._accumulators

    def get_decode_token_count(self, req_id: str) -> int:
        """Get the current decode token count for a request."""
        if req_id not in self._accumulators:
            return 0
        return self._accumulators[req_id].num_decode_tokens

    def clear(self):
        """Clear all accumulators."""
        self._accumulators.clear()


def get_accumulator_manager() -> KVAccumulatorManager:
    """Get the global accumulator manager singleton."""
    global _accumulator_manager
    if _accumulator_manager is None:
        _accumulator_manager = KVAccumulatorManager()
    return _accumulator_manager
