# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any, Optional

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    UnreadyWorkerProcHandle,
    WorkerProc,
    WorkerProcHandle,
)

logger = init_logger(__name__)


class MultiprocDistributedExecutor(MultiprocExecutor):
    def get_worker_proc_cls(self) -> type["WorkerProc"]:
        return DistrbutedWorkerProc

    def init_workers(self, unready_workers: list[UnreadyWorkerProcHandle]) -> None:
        self.workers = DistrbutedWorkerProc.wait_for_ready(unready_workers)

    def init_request_rpc_mq(self) -> None:
        if self.parallel_config.distributed_node_rank == 0:
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=self.parallel_config.distributed_master_ip,
            )
            self.scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        else:
            # retrieve through remote sync from remote driver
            self.rpc_broadcast_mq = None
            self.scheduler_output_handle = None

    def init_response_mqs(self) -> None:
        assert isinstance(self.workers[0], DistributedWorkerProcHandle)
        self.response_mqs = [
            mq for mq in self.workers[0].rpc_response_mqs if mq is not None
        ]

    def get_message_queues(
        self, unique_reply_rank: Optional[int] = None
    ) -> list[MessageQueue]:
        message_queues = []
        for rank in range(self.world_size):
            if rank < self.local_world_size:
                local_message_queue = self.workers[rank].worker_response_mq
                message_queues.append(local_message_queue)
            else:
                assert isinstance(self.workers[0], DistributedWorkerProcHandle)
                remote_message_queue = self.workers[0].rpc_response_mqs[rank]
                assert remote_message_queue is not None
                message_queues.append(remote_message_queue)
        if unique_reply_rank is not None:
            message_queues = [message_queues[unique_reply_rank]]
        return message_queues


@dataclass
class DistributedWorkerProcHandle(WorkerProcHandle):
    worker_response_mq: Optional[MessageQueue]  # The worker process writes to this MQ
    rpc_response_mqs: list[Optional[MessageQueue]] = field(default_factory=list)

    @classmethod
    def from_unready_handle(
        cls,
        unready_handle: UnreadyWorkerProcHandle,
        worker_response_mq: Optional[MessageQueue],
        **kwargs,
    ) -> "DistributedWorkerProcHandle":
        assert "rpc_response_mqs" in kwargs
        rpc_response_mqs = kwargs["rpc_response_mqs"]
        return cls(
            proc=unready_handle.proc,
            rank=unready_handle.rank,
            worker_response_mq=worker_response_mq,
            rpc_response_mqs=rpc_response_mqs,
            death_writer=unready_handle.death_writer,
        )


class DistrbutedWorkerProc(WorkerProc):
    """Wrapper that runs one Worker in a separate process."""

    def init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
        # Initialize MessageQueue for receiving SchedulerOutput
        node_size = vllm_config.parallel_config.distributed_node_size
        if node_size == 1:
            super().init_message_queues(input_shm_handle, vllm_config)
            self.rpc_response_handles = []
        else:
            # multi node
            # generate mq broadcaster from world group
            # for cross-node communication
            self.rpc_broadcast_mq = get_world_group().create_mq_broadcaster(
                extra_writer_handler=input_shm_handle,
                # we will wait until ready later
                blocking=False,
            )
            self.worker_response_mq, self.rpc_response_handles = (
                get_world_group().create_single_reader_mq_broadcasters(reader_rank=0)
            )

    @classmethod
    def wait_for_response_handle_ready(
        cls, handles: dict[str, Any], proc_handle: UnreadyWorkerProcHandle
    ) -> WorkerProcHandle:
        response_handle = handles["handle"]
        worker_response_mq: Optional[MessageQueue] = None
        if len(response_handle.local_reader_ranks) > 0:
            worker_response_mq = MessageQueue.create_from_handle(response_handle, 0)
        assert "rpc_response_handles" in handles
        remote_response_handles = handles["rpc_response_handles"]
        remote_response_mqs = [
            MessageQueue.create_from_handle(response, -1)
            if response.remote_subscribe_addr is not None
            else None
            for response in remote_response_handles
        ]
        return DistributedWorkerProcHandle.from_unready_handle(
            proc_handle, worker_response_mq, rpc_response_mqs=remote_response_mqs
        )

    @classmethod
    def get_ready_proc_handles(cls, worker: "WorkerProc") -> dict[str, Any]:
        assert isinstance(worker, DistrbutedWorkerProc), (
            "worker must be DistrbutedWorkerProc"
        )
        return {
            "status": WorkerProc.READY_STR,
            "handle": worker.worker_response_mq.export_handle(),
            "rpc_response_handles": worker.rpc_response_handles,
        }
