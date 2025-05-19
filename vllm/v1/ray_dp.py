# SPDX-License-Identifier: Apache-2.0
import os
import queue
import threading
from typing import Any, Union

import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, logger
from vllm.v1.executor.abstract import Executor


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray Actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        on_head_node: bool,
        input_address: str,
        output_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # TODO(rui): improve shutdown handling

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        parallel_config: ParallelConfig = vllm_config.parallel_config
        assert parallel_config.data_parallel_size > 1 or dp_rank > 0
        # Set data parallel rank for this engine process.
        parallel_config.data_parallel_rank = dp_rank
        parallel_config.data_parallel_rank_local = local_dp_rank

        input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()

        executor_fail_callback = lambda: input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        # Create input socket.
        input_ctx = zmq.Context()
        identity = engine_index.to_bytes(length=2, byteorder="little")
        input_socket = make_zmq_socket(input_ctx,
                                       input_address,
                                       zmq.DEALER,
                                       identity=identity,
                                       bind=False)

        try:
            # Ray sets CUDA_VISIBLE_DEVICES to empty string,
            # we clean this up to be able to properly initialize
            # data parallel groups.
            del os.environ['CUDA_VISIBLE_DEVICES']
            # Set up data parallel environment.
            self._init_data_parallel(vllm_config)

            # Counts forward-passes of the model so that we can synchronize
            # finished with DP peers every N steps.
            self.counter = 0

            # Initialize engine core and model.
            EngineCore.__init__(self, vllm_config, executor_class, log_stats,
                                executor_fail_callback)

            self.step_fn = (self.step if self.batch_queue is None else
                            self.step_with_batch_queue)
            self.engines_running = False

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            self.input_queue = input_queue
            self.output_queue = queue.Queue[Union[EngineCoreOutputs, bytes]]()
            threading.Thread(target=self.process_input_socket,
                             args=(input_socket, ),
                             daemon=True).start()
            input_socket = None
            self.output_thread = threading.Thread(
                target=self.process_output_socket,
                args=(output_address, engine_index),
                daemon=True)
            self.output_thread.start()
        finally:
            if input_socket is not None:
                input_socket.close(linger=0)

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()


class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        input_address: str,
        output_address: str,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        import ray

        from vllm.v1.ray_dp import DPEngineCoreActor

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        # TODO(rui): use proper placement strategy to put engine actors
        # on the desired nodes.
        refs = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index
            actor = ray.remote(DPEngineCoreActor).remote(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                input_address=input_address,
                output_address=output_address,
                on_head_node=True,
                engine_index=global_index,
                dp_rank=global_index,
                local_dp_rank=local_index)
            self.local_engine_actors.append(actor)
            refs.append(actor.wait_for_init.remote())

        dp_size = vllm_config.parallel_config.data_parallel_size
        for index in range(dp_size - local_engine_count):
            local_index = index
            global_index = local_engine_count + index
            actor = ray.remote(DPEngineCoreActor).remote(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                input_address=input_address,
                output_address=output_address,
                on_head_node=False,
                engine_index=global_index,
                dp_rank=global_index,
                local_dp_rank=local_index)
            self.remote_engine_actors.append(actor)
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        for actor in self.local_engine_actors + self.remote_engine_actors:
            actor.run.remote()

    def close(self):
        import ray
        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
