# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import signal
import sys
import time
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle

from vllm.config import VllmConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.executor.multiproc_worker_utils import (
    _add_prefix, set_multiprocessing_worker_envs)
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_mp_context,
                        get_open_port)
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000


class MultiprocExecutor(Executor):

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self.workers: List[WorkerProcHandle] = []
        self._finalizer = weakref.finalize(self, shutdown, self.workers)

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        assert self.world_size == tensor_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}). "
            f"Pipeline parallelism is not yet implemented in v1")

        # Set multiprocessing envs that are common to V0 and V1
        set_multiprocessing_worker_envs(self.parallel_config)

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # 127.0.0.1 for communication.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        unready_workers: List[UnreadyWorkerProcHandle] = []
        for rank in range(self.world_size):
            unready_worker = WorkerProc.make_worker_process(
                vllm_config=self.vllm_config,
                local_rank=rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
            )
            unready_workers.append(unready_worker)

        # Workers must be created before wait_for_ready to avoid
        # deadlock, since worker.init_device() does a device sync.
        for unready_worker in unready_workers:
            try:
                # WorkerProc.wait_for_ready waits on the ready_pipe.
                # If any errors encountered, shutdown all the workers.
                # This makes sure we cleanup the WorkerProcs when using
                # the InprocClient.
                worker = WorkerProc.wait_for_ready(unready_worker)
                self.workers.append(worker)
            except Exception:
                shutdown(unready_workers)
                raise

        # Ensure message queues are ready. Will deadlock if re-ordered
        # Must be kept consistent with the WorkerProc
        self.rpc_broadcast_mq.wait_until_ready()
        for w in self.workers:
            w.worker_response_mq.wait_until_ready()

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        start_time = time.monotonic()
        kwargs = kwargs or {}

        # NOTE: If the args are heterogeneous, then we pack them into a list,
        # and unpack them in the method of every worker, because every worker
        # knows their own rank.
        try:
            if isinstance(method, str):
                send_method = method
            else:
                send_method = cloudpickle.dumps(
                    method, protocol=pickle.HIGHEST_PROTOCOL)
            self.rpc_broadcast_mq.enqueue((send_method, args, kwargs))

            responses = [None] * self.world_size
            for w in self.workers:
                dequeue_timeout = timeout - (time.monotonic() - start_time
                                             ) if timeout is not None else None
                status, result = w.worker_response_mq.dequeue(
                    timeout=dequeue_timeout)

                if status != WorkerProc.ResponseStatus.SUCCESS:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        raise RuntimeError("Worker failed")

                responses[w.rank] = result

            return responses
        except TimeoutError as e:
            self.shutdown()
            raise TimeoutError(f"RPC call to {method} timed out.") from e
        except Exception as e:
            self.shutdown()
            raise e

    def shutdown(self):
        """Properly shut down the executor and its workers"""
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            self._finalizer()

        self.rpc_broadcast_mq = None

    def check_health(self) -> None:
        self.collective_rpc("check_health", timeout=10)
        return


@dataclass
class UnreadyWorkerProcHandle:
    """WorkerProcess handle before READY."""
    proc: BaseProcess
    rank: int
    ready_pipe: Tuple[Connection, Connection]


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    rank: int
    worker_response_mq: MessageQueue  # The worker process writes to this MQ

    @classmethod
    def from_unready_handle(
            cls, unready_handle: UnreadyWorkerProcHandle,
            worker_response_mq: MessageQueue) -> "WorkerProcHandle":
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
        )


# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(workers: Union[List[WorkerProcHandle],
                            List[UnreadyWorkerProcHandle]]):
    for w in workers:
        if hasattr(w, "worker_response_mq"):
            w.worker_response_mq = None

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
    active_procs = [w.proc for w in workers if w.proc.is_alive()]
    for p in active_procs:
        p.terminate()
    if not wait_for_termination(active_procs, 4):
        # Send SIGKILL if still running
        active_procs = [p for p in active_procs if p.is_alive()]
        for p in active_procs:
            p.kill()


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        ready_pipe: Connection,
    ):

        try:
            self.rank = rank
            wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
            # TODO: move init_worker to executor level as a collective rpc call
            all_kwargs: List[Dict] = [
                {} for _ in range(vllm_config.parallel_config.world_size)
            ]
            all_kwargs[rank] = {
                "vllm_config": vllm_config,
                "local_rank": local_rank,
                "rank": rank,
                "distributed_init_method": distributed_init_method,
                "is_driver_worker": rank == 0,
            }
            wrapper.init_worker(all_kwargs)
            self.worker = wrapper

            pid = os.getpid()
            _add_prefix(sys.stdout, f"VllmWorker rank={rank}", pid)
            _add_prefix(sys.stderr, f"VllmWorker rank={rank}", pid)

            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank)

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)
            worker_response_mq_handle = self.worker_response_mq.export_handle()

            # Initialize device and loads weights
            self.worker.init_device()
            self.worker.load_model()

            # Send READY once we know everything is loaded
            ready_pipe.send({
                "status": "READY",
                "handle": pickle.dumps(worker_response_mq_handle)
            })

        except Exception as e:
            logger.exception("WorkerProc got Exception at startup:",
                             exc_info=e)
            ready_pipe.send({"status": "FAILED"})

        finally:
            ready_pipe.close()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        pipe_tuple = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": pipe_tuple[1],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.worker_main,
                               kwargs=process_kwargs,
                               daemon=True)
        proc.start()

        return UnreadyWorkerProcHandle(proc, rank, pipe_tuple)

    @staticmethod
    def wait_for_ready(
            unready_proc_handle: UnreadyWorkerProcHandle) -> WorkerProcHandle:

        e = Exception("WorkerProc initialization failed due to "
                      "an exception in a background process. "
                      "See stack trace for root cause.")

        ready_pipe = unready_proc_handle.ready_pipe[0]
        try:
            # Wait until the WorkerProc is ready.
            response = ready_pipe.recv()
            if response["status"] != "READY":
                raise e

            # Extract the message queue handle.
            mq_handle = pickle.loads(response["handle"])
            worker_response_mq = MessageQueue.create_from_handle(mq_handle, 0)
            return WorkerProcHandle.from_unready_handle(
                unready_proc_handle, worker_response_mq)

        except EOFError:
            e.__suppress_context__ = True
            raise e from None

        finally:
            # Close connection.
            unready_proc_handle.ready_pipe[0].close()
            unready_proc_handle.ready_pipe[1].close()

    def shutdown(self):
        self.rpc_broadcast_mq = None
        self.worker_response_mq = None
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

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
        try:
            worker = WorkerProc(*args, **kwargs)

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()

            worker.worker_busy_loop()

        except Exception as e:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            logger.exception("WorkerProc got an Exception:", exc_info=e)

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()
                worker = None

    class ResponseStatus(Enum):
        SUCCESS = auto()
        FAILURE = auto()

    def worker_busy_loop(self):
        """Main busy loop for Multiprocessing Workers"""
        while True:
            method, args, kwargs = self.rpc_broadcast_mq.dequeue()

            try:
                if isinstance(method, str):
                    func = getattr(self.worker, method)
                elif isinstance(method, bytes):
                    func = partial(cloudpickle.loads(method), self.worker)
                output = func(*args, **kwargs)
            except Exception as e:
                logger.exception("WorkerProc got an Exception:", exc_info=e)
                self.worker_response_mq.enqueue(
                    (WorkerProc.ResponseStatus.FAILURE, e))
                continue

            self.worker_response_mq.enqueue(
                (WorkerProc.ResponseStatus.SUCCESS, output))
