# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable, not trying
    HALF_OPEN = "half_open"  # Testing recovery


class SafeLMCacheConnectorV1(KVConnectorBase_V1):
    """Safe wrapper around LMCacheConnectorV1 with circuit breaker.

    This connector implements a circuit breaker pattern that:
    - Tracks failures per operation type
    - Attempts recovery when service might be back
    - Provides graceful degradation without permanent disabling
    - Uses exponential backoff for recovery attempts

    Circuit States:
    - CLOSED: Normal operation, all calls go through
    - OPEN: Too many failures, calls are short-circuited
    - HALF_OPEN: Testing recovery, limited calls allowed

    Usage:
        --kv-transfer-config '{
            "kv_connector":"SafeLMCacheConnectorV1",
            "kv_role":"kv_both"
        }'
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        # Circuit breaker configuration
        self.failure_threshold = 3  # Failures before opening circuit
        self.recovery_timeout = 30.0  # Seconds before attempting recovery
        self.max_recovery_timeout = 300.0  # Max seconds between recovery
        self.half_open_max_calls = 3  # Max calls in half-open state
        self.success_threshold = 2  # Successes needed to close circuit

        # Circuit breaker state
        self.circuit_state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.current_recovery_timeout = self.recovery_timeout
        self.half_open_calls = 0
        self.half_open_successes = 0

        # Per-operation failure tracking
        self.failure_counts: dict[str, int] = defaultdict(int)
        self.last_operation_failure: dict[str, float] = defaultdict(float)

        # Success tracking for recovery
        self.total_operations = 0
        self.total_successes = 0

        # Initialize base connector
        self.base_connector: Optional[LMCacheConnectorV1]
        try:
            self.base_connector = LMCacheConnectorV1(vllm_config, role)
            logger.info("SafeLMCacheConnectorV1 initialized successfully "
                        "with circuit breaker")
        except Exception as e:
            logger.warning(
                "Failed to initialize LMCache connector: %s. "
                "Starting in OPEN circuit state.", e)
            self.base_connector = None
            self.circuit_state = CircuitState.OPEN
            self.last_failure_time = time.time()

    def _should_attempt_call(self, operation: str) -> bool:
        """Determine if we should attempt operation based on circuit state."""
        current_time = time.time()

        # Global circuit state check
        if self.circuit_state == CircuitState.CLOSED:
            return True
        elif self.circuit_state == CircuitState.OPEN:
            # Check if it's time to attempt recovery
            time_since_failure = current_time - self.last_failure_time
            if time_since_failure >= self.current_recovery_timeout:
                self.circuit_state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.half_open_successes = 0
                logger.info("LMCache circuit breaker entering HALF_OPEN state "
                            "for recovery attempt")
                return True
            return False
        elif self.circuit_state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.half_open_max_calls

        return False

    def _record_success(self, operation: str) -> None:
        """Record a successful operation."""
        self.total_operations += 1
        self.total_successes += 1

        # Reset operation-specific failure count
        if operation in self.failure_counts:
            self.failure_counts[operation] = 0

        # Handle circuit state transitions
        if self.circuit_state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            self.half_open_calls += 1

            # If enough successes in half-open, close the circuit
            if self.half_open_successes >= self.success_threshold:
                self.circuit_state = CircuitState.CLOSED
                # Reset timeout
                self.current_recovery_timeout = self.recovery_timeout
                logger.info(
                    "LMCache circuit breaker CLOSED - service recovered")

    def _record_failure(self, operation: str, error: Exception) -> None:
        """Record a failed operation and update circuit state."""
        current_time = time.time()
        self.total_operations += 1

        # Update failure tracking
        self.failure_counts[operation] += 1
        self.last_operation_failure[operation] = current_time

        # Create error message
        error_msg = str(error) if str(error) else type(error).__name__

        # Handle circuit state transitions
        if self.circuit_state == CircuitState.CLOSED:
            if self.failure_counts[operation] >= self.failure_threshold:
                self.circuit_state = CircuitState.OPEN
                self.last_failure_time = current_time
                logger.warning(
                    "LMCache circuit breaker OPENED due to %s failures: %s",
                    operation, error_msg)
            else:
                logger.warning("LMCache %s failed (%d/%d): %s", operation,
                               self.failure_counts[operation],
                               self.failure_threshold, error_msg)

        elif self.circuit_state == CircuitState.HALF_OPEN:
            # Failed during recovery, go back to open
            self.circuit_state = CircuitState.OPEN
            self.last_failure_time = current_time
            self.half_open_calls += 1

            # Exponential backoff for recovery timeout
            self.current_recovery_timeout = min(
                self.current_recovery_timeout * 2, self.max_recovery_timeout)
            logger.warning(
                "LMCache recovery attempt failed for %s: %s. "
                "Next attempt in %.1fs", operation, error_msg,
                self.current_recovery_timeout)

    def _safe_call(self, operation: str, func):
        """Safely call an LMCache operation with circuit breaker protection."""
        if self.base_connector is None:
            return self._get_fallback_result(operation)

        if not self._should_attempt_call(operation):
            return self._get_fallback_result(operation)

        try:
            result = func()
            self._record_success(operation)
            return result
        except Exception as e:
            self._record_failure(operation, e)
            return self._get_fallback_result(operation)

    def _get_fallback_result(self, operation: str):
        """Get appropriate fallback result for each operation type."""
        fallbacks = {
            'get_num_new_matched_tokens': (0, False),
            'update_state_after_alloc': None,
            'build_connector_meta': KVConnectorMetadata(),
            'request_finished': (False, None),
            'start_load_kv': None,
            'wait_for_layer_load': None,
            'save_kv_layer': None,
            'wait_for_save': None,
            'get_finished': (None, None),
            'register_kv_caches': None,
        }
        result = fallbacks.get(operation)
        if result is None and operation not in fallbacks:
            logger.warning("No fallback defined for operation: %s", operation)
        return result

    # ==============================
    # Worker-side methods with circuit breaker
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading KV cache with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.start_load_kv(forward_context, **kwargs)

        self._safe_call("start_load_kv", _call)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.wait_for_layer_load(layer_name)

        self._safe_call("wait_for_layer_load", _call)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Save KV layer with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.save_kv_layer(layer_name, kv_layer,
                                                     attn_metadata, **kwargs)

        self._safe_call("save_kv_layer", _call)

    def wait_for_save(self):
        """Wait for save completion with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.wait_for_save()

        self._safe_call("wait_for_save", _call)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get finished request IDs with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.get_finished(finished_req_ids)

        return self._safe_call("get_finished", _call)

    # ==============================
    # Scheduler-side methods with circuit breaker
    # ==============================

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """Get number of cache hit tokens with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.get_num_new_matched_tokens(
                request, num_computed_tokens)

        return self._safe_call("get_num_new_matched_tokens", _call)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """Update state after allocation with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.update_state_after_alloc(
                request, blocks, num_external_tokens)

        self._safe_call("update_state_after_alloc", _call)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Build connector metadata with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.build_connector_meta(scheduler_output)

        return self._safe_call("build_connector_meta", _call)

    def request_finished(
            self, request: "Request",
            block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        """Handle request completion with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.request_finished(request, block_ids)

        return self._safe_call("request_finished", _call)

    # ==============================
    # Additional utility methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches with circuit breaker protection."""

        def _call():
            assert self.base_connector is not None
            return self.base_connector.register_kv_caches(kv_caches)

        self._safe_call("register_kv_caches", _call)

    @property
    def circuit_status(self) -> dict:
        """Return detailed circuit breaker status."""
        return {
            'state':
            self.circuit_state.value,
            'total_operations':
            self.total_operations,
            'success_rate':
            (self.total_successes / max(1, self.total_operations)),
            'failure_counts':
            dict(self.failure_counts),
            'current_recovery_timeout':
            self.current_recovery_timeout,
            'time_until_next_attempt':
            max(
                0, self.current_recovery_timeout -
                (time.time() - self.last_failure_time))
        }

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state (for testing/recovery purposes)."""
        self.circuit_state = CircuitState.CLOSED
        self.failure_counts.clear()
        self.current_recovery_timeout = self.recovery_timeout
        self.half_open_calls = 0
        self.half_open_successes = 0
        logger.info(
            "SafeLMCacheConnectorV1 circuit breaker reset to CLOSED state")
