# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from collections import deque
from collections.abc import Callable
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)

# Minimum time between polls for finished transfers.
_POLL_INTERVAL_S = 0.0004
# Stop polling once no step has run for this long: the engine is idle or paused.
_POLL_ACTIVE_WINDOW_S = 1.0


def _merge_transfer_results(dst: KVConnectorOutput, src: KVConnectorOutput) -> None:
    if src.finished_sending:
        dst.finished_sending = src.finished_sending | (dst.finished_sending or set())
    if src.finished_recving:
        dst.finished_recving = src.finished_recving | (dst.finished_recving or set())
    dst.failed_recving |= src.failed_recving
    dst.invalid_block_ids |= src.invalid_block_ids
    if (src_meta := src.kv_connector_worker_meta) is not None:
        dst_meta = dst.kv_connector_worker_meta
        dst.kv_connector_worker_meta = (
            src_meta if dst_meta is None else dst_meta.aggregate(src_meta)
        )


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None" = None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def finish_forward(self) -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str], defer_finalize: bool = False
    ) -> KVConnectorOutput | None:
        return None

    def finalize_step(self) -> None:
        pass

    def record_mrope_position_delta(self, req_id: str, delta: int) -> None:
        pass

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def next_poll_time(self) -> float | None:
        """When poll() is next due (time.monotonic()), or None if not needed."""
        return None

    def poll(self) -> None:
        pass

    def transfer_results_taker(
        self,
    ) -> Callable[[ModelRunnerOutput], None] | None:
        """For an output whose step's GPU work is pending: a function that
        adds held transfer results to it once that work is done. Thread-safe."""
        return None

    def set_disabled(self, disabled: bool) -> None:
        pass

    def reset_capture_state(self) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._pending_load_kwargs: dict[str, Any] | None = None
        self._disabled = False

        # A KV producer reports each new request's M-RoPE position delta so the
        # scheduler can pass it on to the decode instance.
        kv_transfer_config = vllm_config.kv_transfer_config
        self._reports_mrope_position_deltas = (
            kv_transfer_config is not None and kv_transfer_config.is_kv_producer
        )
        self._mrope_position_deltas: dict[str, int] = {}

        # With polling, transfer results are held until an output takes them
        # after its step's GPU work.
        model_config = vllm_config.model_config
        self._polls = (
            self.kv_connector.supports_transfer_polling
            and vllm_config.parallel_config.pipeline_parallel_size == 1
            and (model_config is None or model_config.runner_type != "pooling")
        )
        self._results_lock = threading.Lock()
        # Outputs that take held results once their GPU work is done are numbered
        # in step order. Results are held with the first one that may take them.
        self._held_results = deque[tuple[int, KVConnectorOutput]]()
        self._last_taker = 0
        self._last_taken = 0
        self._has_sync_loads = False
        # From an output's take until the next post_forward(), which finds the
        # same results, polls would only contend with sending that output.
        self._polls_paused = False
        # Raised by the next post_forward(); polls have no RPC to fail.
        self._poll_error: Exception | None = None
        self._has_pending_transfers = False
        self._model_thread = threading.get_ident()
        self._last_step_time = 0.0
        self._next_poll_time = 0.0

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None" = None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._disabled:
            return

        self._has_pending_transfers = scheduler_output.has_pending_kv_transfers
        self._model_thread = threading.get_ident()
        self._last_step_time = time.monotonic()
        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        self._pending_load_kwargs = self._build_load_kwargs(
            scheduler_output, input_batch, attn_metadata
        )

        self._has_sync_loads = scheduler_output.has_sync_kv_loads
        if self._has_sync_loads:
            # Leave the loads' failures to this step's post_forward().
            self._polls_paused = True
            # Sync loads need to run before this step's forward.
            self._start_load_kv()
        # Otherwise start the async load after forward to keep submission
        # off the critical path.

    @staticmethod
    def _build_load_kwargs(
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None",
        attn_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Keyword args for the connector's start_load_kv()."""
        if input_batch is None:
            # No forward this step.
            return {}
        load_kwargs: dict[str, Any] = dict(
            scheduler_output=scheduler_output,
            request_state_indices=input_batch.idx_mapping,
            request_ids=input_batch.req_ids,
            num_tokens=input_batch.num_tokens,
        )
        if attn_metadata is not None:
            load_kwargs["attn_metadata"] = attn_metadata
        return load_kwargs

    def _start_load_kv(self) -> None:
        load_kwargs = self._pending_load_kwargs
        assert load_kwargs is not None
        self._pending_load_kwargs = None
        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context(), **load_kwargs)
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context(), **load_kwargs)

    def finish_forward(self) -> None:
        if not self._disabled:
            self.kv_connector.finish_forward()

    def reset_capture_state(self) -> None:
        self.kv_connector.reset_capture_state()

    def post_forward(
        self, finished_req_ids: set[str], defer_finalize: bool = False
    ) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        if self._pending_load_kwargs is not None:
            self._start_load_kv()

        self._polls_paused = False
        if self._poll_error is not None:
            error, self._poll_error = self._poll_error, None
            raise error
        if not defer_finalize:
            self.kv_connector.wait_for_save()
        output = self._get_transfer_results(finished_req_ids)
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        if self._polls:
            self._hold_transfer_results(output)
            output = KVConnectorOutput()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        if self._mrope_position_deltas:
            output.mrope_position_deltas = self._mrope_position_deltas
            self._mrope_position_deltas = {}
        if not defer_finalize:
            self.kv_connector.clear_connector_metadata()
        return output

    def record_mrope_position_delta(self, req_id: str, delta: int) -> None:
        if self._reports_mrope_position_deltas and not self._disabled:
            self._mrope_position_deltas[req_id] = delta

    def finalize_step(self) -> None:
        if self._disabled:
            return
        self.kv_connector.wait_for_save()
        self.kv_connector.clear_connector_metadata()

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        self.finish_forward()
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids)
        output = ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
        if self._polls:
            if output is EMPTY_MODEL_RUNNER_OUTPUT:
                output = copy(output)
            self._take_transfer_results(output)
        return output

    def next_poll_time(self) -> float | None:
        if (
            (not self._polls or self._disabled or not self._has_pending_transfers)
            or (self._polls_paused or self._poll_error is not None)
            or time.monotonic() - self._last_step_time > _POLL_ACTIVE_WINDOW_S
        ):
            return None
        return self._next_poll_time

    def poll(self) -> None:
        try:
            with torch.inference_mode():
                results = self._get_transfer_results(set())
                if not results.is_empty():
                    # Metadata may also record these completions; the scheduler
                    # needs both in the same output.
                    results.kv_connector_worker_meta = (
                        self.kv_connector.build_connector_worker_meta()
                    )
        except Exception as e:
            logger.exception("Polling for finished KV transfers failed")
            self._poll_error = e
            return
        finally:
            # From the end of a poll, so that slow polls don't run back to back.
            self._next_poll_time = time.monotonic() + _POLL_INTERVAL_S
        self._hold_transfer_results(results)

    def _get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorOutput:
        results = self.kv_connector.get_transfer_results(finished_req_ids)
        return KVConnectorOutput(
            finished_sending=results.finished_sending or None,
            finished_recving=results.finished_recving or None,
            failed_recving=results.failed_recving,
            invalid_block_ids=self.kv_connector.get_block_ids_with_load_errors(),
        )

    def _hold_transfer_results(self, results: KVConnectorOutput) -> None:
        # Failures are held with their requests' completions, so the scheduler
        # never sees a load finish before learning that it failed.
        if results.is_empty():
            return
        with self._results_lock:
            # Failed sync loads must be reported with this step's output. Later
            # results must not overtake them.
            first_taker = self._held_results[-1][0] if self._held_results else 0
            if self._has_sync_loads:
                first_taker = self._last_taker
            if self._held_results and self._held_results[-1][0] == first_taker:
                _merge_transfer_results(self._held_results[-1][1], results)
            else:
                self._held_results.append((first_taker, results))

    def transfer_results_taker(self) -> Callable[[ModelRunnerOutput], None] | None:
        # Warmup outputs are never collected.
        if not self._polls or self._disabled:
            return None
        with self._results_lock:
            self._last_taker += 1
            return partial(self._take_transfer_results, taker=self._last_taker)

    def _take_transfer_results(
        self, output: ModelRunnerOutput, taker: int | None = None
    ) -> None:
        """Move held results to `output`. Outputs are processed in step order,
        so one without pending GPU work (no `taker`) leaves them to pending ones."""
        if (
            taker is not None and threading.get_ident() == self._model_thread
        ) and self.next_poll_time() is not None:
            # Model thread was blocked, not polling (uniproc or sync scheduling).
            self.poll()
        with self._results_lock:
            if taker is not None:
                self._last_taken = taker
                self._polls_paused = True
            elif self._last_taken < self._last_taker:
                return
            else:
                taker = self._last_taker
            while self._held_results and self._held_results[0][0] <= taker:
                _, held = self._held_results.popleft()
                if output.kv_connector_output is None:
                    output.kv_connector_output = held
                else:
                    _merge_transfer_results(output.kv_connector_output, held)

    def set_disabled(self, disabled: bool) -> None:
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
