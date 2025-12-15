# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
import traceback
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from typing import Any, cast

import cloudpickle
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_inner_dp_world_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_loopback_ip,
    get_open_port,
)
from vllm.utils.system_utils import (
    _maybe_force_spawn,
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor, FailureCallback
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque[tuple["FutureWrapper", Callable]],
        aggregate: Callable = lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.aggregate = aggregate
        super().__init__()

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")
        # Drain any futures ahead of us in the queue.
        while not self.done():
            future, get_response = self.futures_queue.pop()
            future.wait_for_response(get_response)
        return super().result()

    def wait_for_response(self, get_response: Callable):
        try:
            response = self.aggregate(get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class MultiprocExecutor(Executor):
    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig, monitor_workers: bool = True):
        self.monitor_workers = monitor_workers
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: FailureCallback | None = None

        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # use the loopback address get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=self.parallel_config.master_addr,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                unready_workers.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    )
                )

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.futures_queue = deque[tuple[FutureWrapper, Callable]]()

        self.output_rank = self._get_output_rank()

    def start_worker_monitor(self, inline=False) -> None:
        workers = self.workers
        self_ref = weakref.ref(self)

        # Monitors worker process liveness. If any die unexpectedly,
        # logs an error, shuts down the executor and invokes the failure
        # callback to inform the engine.
        def monitor_workers():
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, "shutting_down", False):
                return
            _self.is_failed = True
            proc_name = next(h.proc.name for h in workers if h.proc.sentinel == died[0])
            logger.error(
                "Worker proc %s died unexpectedly, shutting down executor.", proc_name
            )
            _self.shutdown()
            callback = _self.failure_callback
            if callback is not None:
                _self.failure_callback = None
                callback()

        if not inline:
            Thread(
                target=monitor_workers, daemon=True, name="MultiprocWorkerMonitor"
            ).start()
            return

        monitor_workers()

    def register_failure_callback(self, callback: FailureCallback):
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            unique_reply_rank=self.output_rank,
            non_block=non_block,
            timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            kv_output_aggregator=self.kv_output_aggregator,
        )

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch", unique_reply_rank=self.output_rank)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        # OPTIMIZATION: Get output only from a single worker (output_rank)
        return self.collective_rpc(
            "take_draft_token_ids", unique_reply_rank=self.output_rank
        )

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator | None = None,
    ) -> Any:
        """Returns single result if unique_reply_rank and/or kv_output_aggregator
        is provided, otherwise list."""
        assert self.rpc_broadcast_mq is not None, (
            "collective_rpc should not be called on follower node"
        )
        if self.is_failed:
            raise RuntimeError("Executor failed.")

        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}

        if kv_output_aggregator is not None:
            output_rank = None
            aggregate: Callable[[Any], Any] = partial(
                kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0
            )
        else:
            output_rank = unique_reply_rank
            aggregate = lambda x: x

        if isinstance(method, str):
            send_method = method
        else:
            send_method = cloudpickle.dumps(method, protocol=pickle.HIGHEST_PROTOCOL)
        self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))

        response_mqs: Sequence[MessageQueue] = self.response_mqs
        if output_rank is not None:
            response_mqs = (response_mqs[output_rank],)

        shutdown_event = self.shutdown_event

        def get_response():
            responses = []
            for mq in response_mqs:
                dequeue_timeout = (
                    None if deadline is None else (deadline - time.monotonic())
                )
                try:
                    status, result = mq.dequeue(
                        timeout=dequeue_timeout, cancel=shutdown_event
                    )
                except TimeoutError as e:
                    raise TimeoutError(f"RPC call to {method} timed out.") from e
                if status != WorkerProc.ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        f"Worker failed with error '{result}', please check the"
                        " stack trace above for the root cause"
                    )
                responses.append(result)
            return responses[0] if output_rank is not None else responses

        if non_block:
            future = FutureWrapper(self.futures_queue, aggregate=aggregate)
            self.futures_queue.appendleft((future, get_response))
            return future

        # First drain any pending futures in the queue.
        while self.futures_queue:
            future, get_fut_response = self.futures_queue.pop()
            future.wait_for_response(get_fut_response)

        return aggregate(get_response())

    @staticmethod
    def _ensure_worker_termination(worker_procs: list[BaseProcess]):
        """Ensure that all worker processes are terminated. Assumes workers have
        received termination requests. Waits for processing, then sends
        termination and kill signals if needed."""

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [proc for proc in worker_procs if proc.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, "shutting_down", False):
            self.shutting_down = True

            # Make sure all the worker processes are terminated first.
            if workers := getattr(self, "workers", None):
                for w in workers:
                    # Close death_writer to signal child processes to exit
                    if w.death_writer is not None:
                        w.death_writer.close()
                        w.death_writer = None
                    w.worker_response_mq = None
                self._ensure_worker_termination([w.proc for w in workers])

            self.shutdown_event.set()

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return

    @cached_property
    def max_concurrent_batches(self) -> int:
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _get_output_rank(self) -> int:
        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        # (the first TP worker of the last PP stage).
        # Example:
        # Assuming TP=8, PP=4, then the world_size=32
        # 0-7, PP rank 0
        # 8-15, PP rank 1
        # 16-23, PP rank 2
        # 24-31, PP rank 3
        # so world_size - tp_size = 32 - 8 = 24 should be PP rank = -1 (i.e. 3)
        return (
            self.world_size
            - self.parallel_config.tensor_parallel_size
            * self.parallel_config.prefill_context_parallel_size
        )


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""

    proc: BaseProcess
    rank: int
    ready_pipe: Connection
    death_writer: Connection | None = None


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    # The worker process writes to this MQ in single-node mode
    worker_response_mq: MessageQueue | None
    # This is only non empty on driver node,
    # the peer worker process i writes to MQ
    # `peer_worker_response_mqs[i]`
    peer_worker_response_mqs: list[MessageQueue | None]
    death_writer: Connection | None = None

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: MessageQueue | None,
        peer_worker_response_mqs: list[MessageQueue | None],
    ) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"
    rpc_broadcast_mq: MessageQueue | None
    worker_response_mq: MessageQueue | None

    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        else:
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                blocking=False,
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(
            vllm_config=vllm_config, rpc_rank=local_rank, global_rank=rank
        )
        # TODO: move `init_worker` to executor level as a collective rpc call
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        is_driver_worker = rank % vllm_config.parallel_config.tensor_parallel_size == 0
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # Initialize device
        self.worker.init_device()

        # Set process title and log prefix
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # Load model
        self._init_message_queues(input_shm_handle, vllm_config)
        self.worker.load_model()

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)

    @staticmethod
    def wait_for_response_handle_ready(
        handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: MessageQueue | None = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        peer_response_handles = handles["peer_response_handles"]
        peer_worker_response_mqs = [
            MessageQueue.create_from_handle(handle, -1)
            if handle.remote_subscribe_addr is not None
            else None
            for handle in peer_response_handles
        ]
        return WorkerProcHandle.from_unready_handle(
            proc_handle,
            worker_response_mq,
            peer_worker_response_mqs=peer_worker_response_mqs,
        )

    @staticmethod
    def wait_for_ready(
        unready_proc_handles: list[UnreadyWorkerProcHandle],
    ) -> list[WorkerProcHandle]:
        e = Exception(
            "WorkerProc initialization failed due to "
            "an exception in a background process. "
            "See stack trace for root cause."
        )

        pipes = {handle.ready_pipe: handle for handle in unready_proc_handles}
        ready_proc_handles: list[WorkerProcHandle | None] = [None] * len(
            unready_proc_handles
        )
        while pipes:
            ready = multiprocessing.connection.wait(pipes.keys())
            for pipe in ready:
                assert isinstance(pipe, Connection)
                try:
                    # Wait until the WorkerProc is ready.
                    unready_proc_handle = pipes.pop(pipe)
                    response: dict[str, Any] = pipe.recv()
                    if response["status"] != "READY":
                        raise e

                    idx = unready_proc_handle.rank % len(ready_proc_handles)
                    ready_proc_handles[idx] = WorkerProc.wait_for_response_handle_ready(
                        response, unready_proc_handle
                    )
                except EOFError:
                    e.__suppress_context__ = True
                    raise e from None

                finally:
                    # Close connection.
                    pipe.close()

        return cast(list[WorkerProcHandle], ready_proc_handles)

    def shutdown(self):
        self.worker.shutdown()
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """Worker initialization and execution loops.
        This runs a background process"""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        shutdown_event = threading.Event()
        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info_once("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(
                target=monitor_parent_death, daemon=True, name="WorkerDeathMonitor"
            )
            death_monitor.start()

        try:
            reader.close()
            worker = WorkerProc(*args, **kwargs)
            assert worker.worker_response_mq is not None

            # Send READY once we know everything is loaded
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            if worker.rpc_broadcast_mq is not None:
                worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("WorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("WorkerProc shutting down.")
            else:
                logger.exception("WorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def enqueue_output(self, output: Any):
        """Prepares output from the worker and enqueues it to the
        worker_response_mq. If the output is an Exception, it is
        converted to a FAILURE response.
        """
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()

        if isinstance(output, Exception):
            result = (WorkerProc.ResponseStatus.FAILURE, str(output))
        else:
            result = (WorkerProc.ResponseStatus.SUCCESS, output)
        if (response_mq := self.worker_response_mq) is not None:
            response_mq.enqueue(result)

    def handle_output(self, output: Any):
        """Handles output from the worker. If async scheduling is enabled,
        it is passed to the async_output_busy_loop thread. Otherwise, it is
        enqueued directly to the worker_response_mq.
        """
        if self.use_async_scheduling:
            self.async_output_queue.put(output)
        else:
            self.enqueue_output(output)

    def async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            output = self.async_output_queue.get()
            self.enqueue_output(output)

    def worker_busy_loop(self, cancel: threading.Event | None = None):
        """Main busy loop for Multiprocessing Workers"""
        assert self.rpc_broadcast_mq is not None
        while True:
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
                cancel=cancel, indefinite=True
            )
            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)

                output = func(*args, **kwargs)
            except Exception as e:
                # Notes have been introduced in python 3.11
                if hasattr(e, "add_note"):
                    e.add_note(traceback.format_exc())
                logger.exception("WorkerProc hit an exception.")
                # exception might not be serializable, so we convert it to
                # string, only for logging purpose.
                if output_rank is None or self.rank == output_rank:
                    self.handle_output(e)
                continue

            if output_rank is None or self.rank == output_rank:
                self.handle_output(output)

    @staticmethod
    def setup_proc_title_and_log_prefix(enable_ep: bool) -> None:
        dp_size = get_dp_group().world_size
        dp_rank = get_dp_group().rank_in_group
        pp_size = get_pp_group().world_size
        pp_rank = get_pp_group().rank_in_group
        pcp_size = get_pcp_group().world_size
        pcp_rank = get_pcp_group().rank_in_group
        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        dcp_size = get_dcp_group().world_size
        dcp_rank = get_dcp_group().rank_in_group
        process_name = "Worker"
        if dp_size > 1:
            process_name += f"_DP{dp_rank}"
        if pp_size > 1:
            process_name += f"_PP{pp_rank}"
        if pcp_size > 1:
            process_name += f"_PCP{pcp_rank}"
        if tp_size > 1:
            process_name += f"_TP{tp_rank}"
        if dcp_size > 1:
            process_name += f"_DCP{dcp_rank}"
        if enable_ep:
            ep_rank = get_ep_group().rank_in_group
            process_name += f"_EP{ep_rank}"
        set_process_title(name=process_name)
        decorate_logs(process_name)


def set_multiprocessing_worker_envs():
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1
    if (
        "OMP_NUM_THREADS" not in os.environ
        and (current_parallelism := torch.get_num_threads()) > default_omp_num_threads
    ):
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism,
            default_omp_num_threads,
        )
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
