# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import multiprocessing
import os
import sys

from vllm.worker_controller.config.vllm import VllmConfig, DummyVllmConfig
from vllm.worker_controller.config.model import ModelConfig, DummyModelConfig
from vllm.worker_controller.config.cache import CacheConfig
from vllm.worker_controller.config.parallel import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

from vllm.worker_controller.executor.proxy_executor import ProxyExecutor
from vllm.worker_controller.entrypoint.api_server import run_server

import uvloop

logger = init_logger(__name__)


def run_api_server(request_queue, response_queue, engine_uuid, vllm_config, port):
    """
    Entry point for the spawned API server process.
    """
    try:
        multiprocessing.set_start_method("forkserver", force=True)
        multiprocessing.set_forkserver_preload(
            ["vllm.worker_controller.entrypoint.api_server"]
        )
    except RuntimeError:
        # Context already set
        pass

    # Create default args
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    # Parse empty args to get defaults
    args = parser.parse_args([])

    # Override with provided config
    args.model = vllm_config.model_config.model
    args.port = port

    # Inject queues and config for RemoteExecutor
    args.request_queue = request_queue
    args.response_queue = response_queue
    args.engine_uuid = engine_uuid
    # Pass the full vllm_config - API server will use this directly
    args.vllm_config = vllm_config

    # Disable frontend multiprocessing in the child process as it is managed by Controller
    args.disable_frontend_multiprocessing = True

    uvloop.run(run_server(args))


class ResourceAllocator:
    def __init__(self, number_of_gpus: int, start_port: int = 8001):
        # rank -> uid or 0 (free)
        self.resources = {i: 0 for i in range(number_of_gpus)}
        self.uuid_to_port = {}  # uid -> port
        self.rank_to_uid = {}  # rank -> uid
        self.start_port = start_port
        self.next_port = start_port

    def assign(self, num: int, uuid: str):
        """Assign `num` free GPUs to a given UUID and allocate a port."""
        if uuid in self.uuid_to_port:
            port = self.uuid_to_port[uuid]
        else:
            port = self.next_port
            self.uuid_to_port[uuid] = port
            self.next_port += 1

        assigned_ranks = []
        for rank, val in self.resources.items():
            if val == 0 and len(assigned_ranks) < num:
                self.resources[rank] = uuid
                self.rank_to_uid[rank] = uuid
                assigned_ranks.append(rank)

        if len(assigned_ranks) < num:
            # Roll back partial assignment
            for rank in assigned_ranks:
                self.resources[rank] = 0
                del self.rank_to_uid[rank]
            raise RuntimeError(
                f"Only {len(assigned_ranks)} free resources, requested {num}"
            )

        return assigned_ranks, port

    def get_ranks_by_uuid(self, uid: str):
        """Return list of ranks assigned to the specified UUID."""
        return [rank for rank, val in self.resources.items() if val == uid]

    def get_port_by_uuid(self, uid: str):
        """Return the port assigned to the specified UID."""
        return self.uuid_to_port.get(uid)

    def release_by_uuid(self, uid: str):
        """Release all ranks and the port assigned to this UID."""
        released_ranks = []
        for rank, val in list(self.resources.items()):
            if val == uid:
                self.resources[rank] = 0
                self.rank_to_uid.pop(rank, None)
                released_ranks.append(rank)

        port = self.uuid_to_port.pop(uid, None)
        return released_ranks, port


class WorkerController:
    def __init__(self, start_port: int = 8001) -> None:
        # Modified Executor will create the empty worker processes and return the pipes
        model_config = DummyModelConfig("dummy", enforce_eager=True)
        cache_config = CacheConfig(gpu_memory_utilization=0.85)
        parallel_config = ParallelConfig(
            pipeline_parallel_size=2,
            worker_cls="vllm.worker_controller.worker.gpu_worker.Worker",
        )

        dummy_vllm_config = DummyVllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
        )
        # type: ignore[call-arg]
        self.executor = ProxyExecutor(vllm_config=dummy_vllm_config)
        self.resource_allocator = ResourceAllocator(
            number_of_gpus=dummy_vllm_config.parallel_config.world_size,
            start_port=start_port,
        )
        # Set the resource allocator in the executor
        self.executor.resource = self.resource_allocator
        self.rpc_broadcast_mq = self.executor.rpc_broadcast_mq

    # Create API server using RemoteProxyExecutor
    def create(self, vllm_config: VllmConfig, engine_uuid: str):
        """
        Create an API server that uses PipeExecutor to communicate
        with the existing workers via pipes.

        Args:
            vllm_config: Configuration for the model/engine
            engine_uuid: Unique identifier for this API server instance
        """
        logger.info(f"WorkerController.create called for {engine_uuid}")

        # 1. Allocate resources
        num_gpus = vllm_config.parallel_config.world_size
        assigned_ranks, port = self.resource_allocator.assign(num_gpus, engine_uuid)
        # Allocate a port for distributed communication (e.g., tcp store)
        # Only needed when world_size > 1 (multi-GPU)
        if num_gpus > 1:
            dist_port = self.resource_allocator.next_port
            self.resource_allocator.next_port += 1
        else:
            dist_port = None

        logger.info(
            f"Assigned ranks {assigned_ranks}, port {port} to engine {engine_uuid}"
        )

        # 2. Create queues for communication
        ctx = multiprocessing.get_context("forkserver")
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()

        # 3. Register engine with ProxyExecutor
        logger.info(f"Adding engine {engine_uuid} to ProxyExecutor")
        self.executor.add_engine(
            engine_uuid, assigned_ranks, request_queue, response_queue, dist_port
        )

        # 4. Spawn API Server process
        logger.info(f"Spawning APIServer process for {engine_uuid}")
        proc = ctx.Process(
            target=run_api_server,
            args=(request_queue, response_queue, engine_uuid, vllm_config, port),
            name=f"APIServer-{engine_uuid}",
        )
        proc.start()
        logger.info(f"APIServer process started with PID {proc.pid}")

        # Store process in executor engines dict (ProxyExecutor stores it too? No, ProxyExecutor stores queues/ranks)
        # We might want to store proc somewhere to manage it (delete it later).
        # ProxyExecutor engines dict has: ranks, queues.
        # We can add proc to it or keep it separate.
        # WorkerController.delete needs to terminate it.
        # ProxyExecutor currently has delete_engine but it doesn't know about the process handle unless we pass it.
        # But we can store it in ProxyExecutor.engines[uuid]['proc'] = proc

        if engine_uuid in self.executor.engines:
            self.executor.engines[engine_uuid]["proc"] = proc

        return proc

    def delete(self, engine_uuid: str):
        """
        Delete an API server and release its resources.

        Args:
            engine_uuid: Unique identifier of the API server to delete
        """
        # Terminate the API server process first
        if engine_uuid in self.executor.engines:
            engine_info = self.executor.engines[engine_uuid]
            proc = engine_info.get("proc")
            if proc and proc.is_alive():
                logger.info(f"Terminating API server process {proc.pid}")
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    logger.warning(f"Force killing API server process {proc.pid}")
                    proc.kill()

        # Delegate to ProxyExecutor which handles worker unloading (if implemented)
        self.executor.delete_engine(engine_uuid)

        # Release resources
        self.resource_allocator.release_by_uuid(engine_uuid)
