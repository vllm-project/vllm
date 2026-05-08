# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SlowMockKVConnector: a synthetic KV connector for the prefetch microbench.

Simulates a disk-backed KV store: every request takes
`simulated_disk_latency_ms` of wall-clock time before its lookup result is
"ready". Until the latency has elapsed, `get_num_new_matched_tokens` returns
(None, True) -- i.e. "ask again later, async". Once the latency has elapsed,
it returns (matched_tokens, True). This is the failure mode the early
prefetch pass is meant to hide: when the running queue saturates the per-
step token budget, the waiting-queue scheduling loop never runs and the
lookup timer never starts.

The connector is registered via `KVTransferConfig.kv_connector_module_path`
so worker processes can import it without modifying the vLLM package. It
intentionally does **no** real KV transfer: the worker-side methods are
no-ops, and `get_num_new_matched_tokens` reports "matched but already
loaded" so the scheduler does not actually wait for KV blocks.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


class _Empty(KVConnectorMetadata):
    pass


class SlowMockKVConnector(KVConnectorBase_V1):
    """Synthetic disk-like connector with configurable lookup latency.

    Configured via env vars (read at construction time):
      VLLM_SLOWMOCK_LATENCY_MS:  per-request lookup wall-clock latency.
                                 Default 200 ms.
      VLLM_SLOWMOCK_MATCHED_TOK: number of tokens reported as cache-hit
                                 once the lookup is "ready". Default 0
                                 (we just want to show the *latency hide*,
                                 not actually transfer KV).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._latency_s = _env_float("VLLM_SLOWMOCK_LATENCY_MS", 200.0) / 1000.0
        self._matched_tokens = _env_int("VLLM_SLOWMOCK_MATCHED_TOK", 0)
        # request_id -> monotonic time at which the lookup is considered done.
        self._lookup_ready_at: dict[str, float] = {}
        # request_id -> True if the early prefetch hook already kicked off
        # the lookup timer for this request.
        self._prefetched: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Scheduler-side
    # ------------------------------------------------------------------
    def maybe_prefetch_request(self, request: Request) -> bool:
        if request.request_id in self._lookup_ready_at:
            return False
        self._lookup_ready_at[request.request_id] = time.monotonic() + self._latency_s
        self._prefetched[request.request_id] = True
        return True

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # First time we see this request through the regular path: start the
        # timer now (mirrors the LMCache behavior of starting the lookup
        # inside get_num_new_matched_tokens). If maybe_prefetch_request was
        # already called for this request, the timer has been running since
        # then -- which is the whole point of the optimization.
        ready_at = self._lookup_ready_at.get(request.request_id)
        if ready_at is None:
            ready_at = time.monotonic() + self._latency_s
            self._lookup_ready_at[request.request_id] = ready_at

        if time.monotonic() < ready_at:
            # Lookup still in flight -- ask the scheduler to retry later.
            return None, True
        return self._matched_tokens, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        return

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return _Empty()

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._lookup_ready_at.pop(request.request_id, None)
        self._prefetched.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------
    # Worker-side: this connector does no real transfer.
    # ------------------------------------------------------------------
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return
