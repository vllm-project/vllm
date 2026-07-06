# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TransferQueue trajectory artifact producer."""

from __future__ import annotations

import asyncio
import importlib
import json
import pickle
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.artifact_transfer.artifact_connector.v1.base import (
    ArtifactConnectorBase_V1,
    ArtifactConnectorMetadata,
    ArtifactConnectorOutput,
    ArtifactConnectorRole,
    ArtifactConnectorWorkerMetadata,
    ArtifactHandle,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TransferQueueArtifactConnectorMetadata(ArtifactConnectorMetadata):
    scheduled_request_ids: list[str] = field(default_factory=list)
    export_fields: list[str] = field(default_factory=list)
    partition_id: str = ""
    step_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferQueueArtifactConnectorWorkerMetadata(ArtifactConnectorWorkerMetadata):
    num_published: int = 0
    handles: dict[str, ArtifactHandle] = field(default_factory=dict)

    def aggregate(
        self, other: ArtifactConnectorWorkerMetadata
    ) -> ArtifactConnectorWorkerMetadata:
        assert isinstance(other, TransferQueueArtifactConnectorWorkerMetadata)
        handles = dict(self.handles)
        handles.update(other.handles)
        return TransferQueueArtifactConnectorWorkerMetadata(
            num_published=self.num_published + other.num_published,
            handles=handles,
        )


@dataclass
class _TrajectoryAccumulator:
    run_id: str
    policy_version: str | int
    model_id: str
    group_id: str | None
    sample_index: int
    prompt_token_ids: list[int]
    response_logprobs: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class _QueuedArtifact:
    record: Any
    metrics: dict[str, Any]
    enqueued_at_ns: int


class _AsyncArtifactPublisher:
    def __init__(
        self,
        *,
        initialize_transfer_queue: Any,
        write_debug_metrics: Any,
        maxsize: int,
        publish_mode: str,
        batch_size: int,
        flush_interval_ms: float,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("publish_queue_maxsize must be positive")
        if batch_size <= 0:
            raise ValueError("publish_batch_size must be positive")
        if flush_interval_ms < 0:
            raise ValueError("publish_flush_interval_ms must be non-negative")
        self._initialize_transfer_queue = initialize_transfer_queue
        self._write_debug_metrics = write_debug_metrics
        self._publish_mode = publish_mode
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_ms / 1000
        self._queue: queue.Queue[_QueuedArtifact | None] = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-artifact-tq-publisher",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def publish(self, record: Any, metrics: dict[str, Any]) -> float:
        enqueue_start_ns = time.perf_counter_ns()
        self._queue.put(
            _QueuedArtifact(
                record=record,
                metrics=metrics,
                enqueued_at_ns=time.perf_counter_ns(),
            )
        )
        return (time.perf_counter_ns() - enqueue_start_ns) / 1e6

    def _run(self) -> None:
        if self._publish_mode == "async_batch":
            self._run_batches()
            return
        self._run_singles()

    def _run_singles(self) -> None:
        while True:
            queued = self._queue.get()
            try:
                if queued is None:
                    return
                publish_start_ns = time.perf_counter_ns()
                status = "ok"
                error = None
                try:
                    transfer_queue = self._initialize_transfer_queue()
                    transfer_queue.kv_put(
                        key=queued.record.key,
                        partition_id=queued.record.partition_id,
                        fields=queued.record.fields,
                        tag=queued.record.tag,
                    )
                except Exception as exc:  # noqa: BLE001
                    status = "error"
                    error = str(exc)
                    logger.exception(
                        "Async artifact publish failed for %s",
                        queued.record.key,
                    )
                publish_end_ns = time.perf_counter_ns()
                metrics = dict(queued.metrics)
                metrics.update(
                    {
                        "status": status,
                        "async_publish": True,
                        "async_batch_publish": False,
                        "queue_wait_ms": (publish_start_ns - queued.enqueued_at_ns)
                        / 1e6,
                        "kv_put_ms": (publish_end_ns - publish_start_ns) / 1e6,
                        "total_ms": (publish_end_ns - queued.metrics["total_start_ns"])
                        / 1e6,
                    }
                )
                if error is not None:
                    metrics["error"] = error
                metrics.pop("total_start_ns", None)
                self._write_debug_metrics(metrics)
            finally:
                self._queue.task_done()

    def _run_batches(self) -> None:
        loop = asyncio.new_event_loop()
        pending: _QueuedArtifact | None = None
        try:
            asyncio.set_event_loop(loop)
            while True:
                if pending is None:
                    queued = self._queue.get()
                else:
                    queued = pending
                    pending = None
                if queued is None:
                    self._queue.task_done()
                    return

                batch = [queued]
                saw_stop = False
                deadline = time.monotonic() + self._flush_interval_s
                while len(batch) < self._batch_size:
                    timeout = max(0.0, deadline - time.monotonic())
                    if timeout == 0:
                        break
                    try:
                        next_queued = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if next_queued is None:
                        saw_stop = True
                        break
                    if next_queued.record.partition_id != batch[0].record.partition_id:
                        pending = next_queued
                        break
                    batch.append(next_queued)

                try:
                    self._publish_batch(loop, batch)
                finally:
                    for _ in batch:
                        self._queue.task_done()
                    if saw_stop:
                        self._queue.task_done()
                        return
        finally:
            loop.close()

    def _publish_batch(
        self,
        loop: asyncio.AbstractEventLoop,
        batch: list[_QueuedArtifact],
    ) -> None:
        publish_start_ns = time.perf_counter_ns()
        status = "ok"
        error = None
        try:
            transfer_queue = self._initialize_transfer_queue()
            loop.run_until_complete(
                transfer_queue.async_kv_batch_put(
                    keys=[queued.record.key for queued in batch],
                    partition_id=batch[0].record.partition_id,
                    fields=self._build_batched_fields(
                        [queued.record for queued in batch]
                    ),
                    tags=[queued.record.tag for queued in batch],
                )
            )
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = str(exc)
            logger.exception(
                "Async batch artifact publish failed for partition %s",
                batch[0].record.partition_id,
            )
        publish_end_ns = time.perf_counter_ns()
        publish_ms = (publish_end_ns - publish_start_ns) / 1e6
        publish_ms_per_artifact = publish_ms / len(batch)
        for batch_index, queued in enumerate(batch):
            metrics = dict(queued.metrics)
            metrics.update(
                {
                    "status": status,
                    "async_publish": True,
                    "async_batch_publish": True,
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "queue_wait_ms": (publish_start_ns - queued.enqueued_at_ns) / 1e6,
                    "kv_batch_put_ms": publish_ms,
                    "kv_put_ms": publish_ms,
                    "publish_ms_per_artifact": publish_ms_per_artifact,
                    "total_ms": (publish_end_ns - queued.metrics["total_start_ns"])
                    / 1e6,
                }
            )
            if error is not None:
                metrics["error"] = error
            metrics.pop("total_start_ns", None)
            self._write_debug_metrics(metrics)

    @staticmethod
    def _build_batched_fields(records: list[Any]) -> Any:
        import torch
        from tensordict import TensorDict

        field_names = list(records[0].fields)
        batched_fields = {}
        for record in records:
            if list(record.fields) != field_names:
                raise ValueError("batched artifact records must share fields")
        for field_name in field_names:
            values = [record.fields[field_name] for record in records]
            first_shape = tuple(values[0].shape)
            if all(tuple(value.shape) == first_shape for value in values):
                batched_fields[field_name] = torch.stack(values)
            else:
                try:
                    batched_fields[field_name] = torch.nested.as_nested_tensor(
                        values, layout=torch.jagged
                    )
                except TypeError:
                    batched_fields[field_name] = torch.nested.as_nested_tensor(values)
        return TensorDict(batched_fields, batch_size=[len(records)])

    def shutdown(self, *, drain: bool = True) -> None:
        if self._closed:
            return
        if drain:
            self._queue.join()
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            logger.warning("Timed out waiting for async artifact publisher")


class TransferQueueArtifactConnector(ArtifactConnectorBase_V1):
    """Publish completed rollout trajectories from the scheduler process."""

    def __init__(self, vllm_config: VllmConfig, role: ArtifactConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._step_id = 0
        self._module: Any | None = None
        self._owns_ray = False
        self._tq_initialized = False
        self._trajectories: dict[str, _TrajectoryAccumulator] = {}
        self._worker_handles: dict[str, ArtifactHandle] = {}
        self._async_publisher: _AsyncArtifactPublisher | None = None

    def _extra(self, key: str, default: Any) -> Any:
        return self._artifact_transfer_config.get_from_extra_config(key, default)

    def _default_model_id(self) -> str:
        model_config = self._vllm_config.model_config
        if model_config is not None:
            return str(model_config.served_model_name)
        return str(self._extra("model_id", "unknown"))

    def _initialize_transfer_queue(self) -> Any:
        if self._tq_initialized:
            assert self._module is not None
            return self._module
        try:
            module = importlib.import_module(
                str(self._extra("transfer_queue_module", "transfer_queue"))
            )
        except ImportError as exc:
            raise RuntimeError(
                "TransferQueueArtifactConnector requires transfer_queue"
            ) from exc
        init_config = self._extra("transfer_queue_init_config", None)
        config_path = self._extra("transfer_queue_config_path", None)
        if init_config is not None and config_path is not None:
            raise ValueError("specify only one TransferQueue client config source")
        if config_path is not None:
            with open(str(config_path), "rb") as config_file:
                init_config = pickle.load(config_file)
        if init_config is not None:
            interface = importlib.import_module(f"{module.__name__}.interface")
            init_client = getattr(interface, "_maybe_create_tq_client", None)
            if init_client is None:
                raise RuntimeError(
                    "TransferQueue does not expose a config-based client initializer"
                )
            init_client(init_config)
        else:
            try:
                ray = importlib.import_module("ray")
            except ImportError as exc:
                raise RuntimeError(
                    "TransferQueueArtifactConnector requires ray when no "
                    "transfer_queue_init_config is provided"
                ) from exc
            if not ray.is_initialized():
                ray.init(
                    address=str(self._extra("ray_address", "auto")),
                    **dict(self._extra("ray_init_kwargs", {})),
                )
                self._owns_ray = True
            module.init()
        self._module = module
        self._tq_initialized = True
        return module

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ArtifactConnectorMetadata:
        step_id = self._step_id
        self._step_id += 1
        return TransferQueueArtifactConnectorMetadata(
            scheduled_request_ids=list(scheduler_output.num_scheduled_tokens),
            export_fields=list(self._artifact_transfer_config.export_fields),
            step_id=step_id,
            metadata={
                "engine_id": self._artifact_transfer_config.engine_id,
                "transfer_mode": self._artifact_transfer_config.transfer_mode,
            },
        )

    def on_new_request(self, request: Request) -> None:
        if self.role is not ArtifactConnectorRole.SCHEDULER or not self.is_producer:
            return
        if request.sampling_params is None:
            return
        params = dict(request.artifact_transfer_params or {})
        run_id = str(
            params.get(
                "run_id",
                self._extra("run_id", self._artifact_transfer_config.engine_id),
            )
        )
        policy_version = params.get(
            "policy_version", self._extra("policy_version", "0")
        )
        model_id = str(
            params.get(
                "model_id",
                self._extra("model_id", self._default_model_id()),
            )
        )
        sample_index = int(params.get("sample_index", self._extra("sample_index", 0)))
        group_id_value = params.get("group_id", self._extra("group_id", None))
        group_id = None if group_id_value is None else str(group_id_value)
        if request.prompt_token_ids is None:
            raise ValueError("Trajectory artifact transfer requires prompt_token_ids")
        if request.sampling_params.num_logprobs is None:
            request.sampling_params.num_logprobs = 0
        self._trajectories[request.request_id] = _TrajectoryAccumulator(
            run_id=run_id,
            policy_version=policy_version,
            model_id=model_id,
            group_id=group_id,
            sample_index=sample_index,
            prompt_token_ids=list(request.prompt_token_ids),
        )

    def record_request_output(
        self,
        request: Request,
        token_ids: list[int],
        logprobs: Any | None,
    ) -> None:
        state = self._trajectories.get(request.request_id)
        if state is None or not token_ids:
            return
        if logprobs is None:
            state.error = "sampled-token logprobs were not returned by the model runner"
            return
        selected_token_ids = logprobs.logprob_token_ids[:, 0].tolist()
        selected_logprobs = logprobs.logprobs[:, 0].tolist()
        if selected_token_ids != token_ids:
            state.error = (
                "sampled-token logprob ids do not match accepted output token ids"
            )
            return
        state.response_logprobs.extend(float(value) for value in selected_logprobs)

    def update_connector_output(
        self, connector_output: ArtifactConnectorOutput
    ) -> None:
        self._worker_handles.update(connector_output.handles)
        worker_meta = connector_output.worker_meta
        if isinstance(worker_meta, TransferQueueArtifactConnectorWorkerMetadata):
            self._worker_handles.update(worker_meta.handles)

    def _handle_publish_failure(self, request_id: str, exc: Exception) -> None:
        policy = self._artifact_transfer_config.failure_policy
        if policy == "fail_request":
            raise RuntimeError(
                f"Failed to publish trajectory artifact for {request_id}"
            ) from exc
        if policy == "fallback_to_request_output":
            logger.exception(
                "Trajectory publication failed for %s; returning normal output",
                request_id,
            )
        else:
            logger.warning(
                "Ignoring trajectory publication failure for %s: %s",
                request_id,
                exc,
            )

    def _write_debug_metrics(self, metrics: dict[str, Any]) -> None:
        metrics_path = self._extra("artifact_metrics_path", None)
        if metrics_path is None:
            return
        metrics = dict(metrics)
        metrics["created_at_ns"] = time.time_ns()
        try:
            with open(str(metrics_path), "a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        except Exception:
            logger.exception("Failed to write artifact transfer debug metrics")

    def _publish_mode(self) -> str:
        mode = str(self._extra("publish_mode", "sync"))
        if mode not in ("sync", "async", "async_batch"):
            raise ValueError(f"Unsupported artifact publish_mode: {mode}")
        return mode

    def _get_async_publisher(self) -> _AsyncArtifactPublisher:
        if self._async_publisher is None:
            self._async_publisher = _AsyncArtifactPublisher(
                initialize_transfer_queue=self._initialize_transfer_queue,
                write_debug_metrics=self._write_debug_metrics,
                maxsize=int(self._extra("publish_queue_maxsize", 4096)),
                publish_mode=self._publish_mode(),
                batch_size=int(self._extra("publish_batch_size", 8)),
                flush_interval_ms=float(self._extra("publish_flush_interval_ms", 2.0)),
            )
        return self._async_publisher

    def _should_drain_async_publisher(self) -> bool:
        return bool(self._extra("publish_drain_on_shutdown", True))

    def request_finished(self, request: Request) -> ArtifactHandle | None:
        worker_handle = self._worker_handles.pop(request.request_id, None)
        state = self._trajectories.pop(request.request_id, None)
        if state is None:
            return worker_handle
        total_start_ns = time.perf_counter_ns()
        try:
            from vllm.distributed.artifact_transfer.schema import (
                TrajectoryArtifactV1Alpha1,
            )

            if state.error is not None:
                raise RuntimeError(state.error)
            finish_reason = request.get_finished_reason()
            if finish_reason is not None:
                finish_reason = str(finish_reason)
            build_start_ns = time.perf_counter_ns()
            artifact = TrajectoryArtifactV1Alpha1(
                run_id=state.run_id,
                request_id=request.request_id,
                engine_id=str(self._artifact_transfer_config.engine_id),
                model_id=state.model_id,
                policy_version=state.policy_version,
                sample_index=state.sample_index,
                group_id=state.group_id,
                finish_reason=None if finish_reason is None else str(finish_reason),
                prompt_token_ids=state.prompt_token_ids,
                response_token_ids=list(request.output_token_ids),
                response_logprobs=state.response_logprobs,
            )
            record = artifact.to_transfer_queue_record()
            build_end_ns = time.perf_counter_ns()
            handle = record.to_handle()
            common_metrics = {
                "request_id": request.request_id,
                "artifact_id": record.key,
                "partition_id": record.partition_id,
                "run_id": state.run_id,
                "policy_version": state.policy_version,
                "prompt_token_count": len(state.prompt_token_ids),
                "response_token_count": len(request.output_token_ids),
                "response_logprob_count": len(state.response_logprobs),
                "schema_build_ms": (build_end_ns - build_start_ns) / 1e6,
                "total_start_ns": total_start_ns,
            }
            publish_mode = self._publish_mode()
            if publish_mode in ("async", "async_batch"):
                enqueue_ms = self._get_async_publisher().publish(
                    record,
                    common_metrics,
                )
                enqueue_end_ns = time.perf_counter_ns()
                self._write_debug_metrics(
                    {
                        **common_metrics,
                        "status": "queued",
                        "async_publish": True,
                        "async_batch_publish": publish_mode == "async_batch",
                        "enqueue_ms": enqueue_ms,
                        "queue_depth": self._get_async_publisher().depth,
                        "total_ms": (enqueue_end_ns - total_start_ns) / 1e6,
                    }
                )
                return handle
            init_start_ns = time.perf_counter_ns()
            transfer_queue = self._initialize_transfer_queue()
            init_end_ns = time.perf_counter_ns()
            kv_put_start_ns = time.perf_counter_ns()
            transfer_queue.kv_put(
                key=record.key,
                partition_id=record.partition_id,
                fields=record.fields,
                tag=record.tag,
            )
            kv_put_end_ns = time.perf_counter_ns()
            self._write_debug_metrics(
                {
                    "status": "ok",
                    **common_metrics,
                    "async_publish": False,
                    "tq_init_ms": (init_end_ns - init_start_ns) / 1e6,
                    "kv_put_ms": (kv_put_end_ns - kv_put_start_ns) / 1e6,
                    "total_ms": (kv_put_end_ns - total_start_ns) / 1e6,
                }
            )
            return handle
        except Exception as exc:
            self._write_debug_metrics(
                {
                    "status": "error",
                    "async_publish": False,
                    "request_id": request.request_id,
                    "run_id": state.run_id,
                    "policy_version": state.policy_version,
                    "error": str(exc),
                    "total_ms": (time.perf_counter_ns() - total_start_ns) / 1e6,
                }
            )
            self._handle_publish_failure(request.request_id, exc)
            return None

    def shutdown(self) -> None:
        self._trajectories.clear()
        if self._async_publisher is not None:
            self._async_publisher.shutdown(drain=self._should_drain_async_publisher())
            self._async_publisher = None
        if self._tq_initialized and self._module is not None:
            try:
                self._module.get_client().close()
            except Exception:
                logger.exception("Failed to close the local TransferQueue client")
        if self._owns_ray:
            importlib.import_module("ray").shutdown()
        self._module = None
        self._tq_initialized = False
        self._owns_ray = False
