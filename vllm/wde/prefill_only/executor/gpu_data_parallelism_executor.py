import atexit
from queue import Queue
from threading import Thread
from typing import List, Optional

from vllm.logger import init_logger
from vllm.wde.core.config import EngineConfig
from vllm.wde.core.layers.attention import AttentionBackend
from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.worker import WorkerBase, create_worker
from vllm.wde.core.workflow import Workflow
from vllm.wde.prefill_only.executor.gpu_executor import (
    double_buffer_execute_loop, simple_execute_loop)

logger = init_logger(__name__)


class GPUDataParallelismExecutor:
    support_scheduling = ["async_scheduling"]

    def __init__(self, engine_config: EngineConfig, workflow: Workflow,
                 attn_backend: AttentionBackend, executor_in: Queue,
                 executor_out: Queue) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self.output_to_cpu = False

        self.executor_in = executor_in
        self.executor_out = executor_out

        self.workers: Optional[List[WorkerBase]] = None

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend,
                   executor_in=engine.executor_in,
                   executor_out=engine.executor_out)

    def thread_target(self, rank: int):
        # Is there a better way to avoid loading the model multiple times?
        # Load to cpu first?
        worker_kwargs = dict(engine_config=self.engine_config,
                             attn_backend=self.attn_backend,
                             envs={'CUDA_VISIBLE_DEVICES': str(rank)})
        worker_kwargs.update(module=self.workflow.Worker)
        worker = create_worker(**worker_kwargs)
        worker.init_device()
        worker.load_model()

        if self.engine_config.scheduler_config.scheduling == "double_buffer":
            execute_loop = double_buffer_execute_loop
        else:
            execute_loop = simple_execute_loop

        execute_loop(worker, self.executor_in, self.executor_out,
                     self.output_to_cpu)

    def ensure_start_execute_loop(self):
        if self.workers is None:
            self.workers = []
            for rank in range(
                    self.engine_config.parallel_config.data_parallel_size):
                worker = Thread(target=self.thread_target,
                                args=(rank, ),
                                daemon=True)
                worker.start()
                self.workers.append(worker)
            atexit.register(self.shutdown_execute_loop)

    def shutdown_execute_loop(self):
        if self.workers is not None:
            for worker in self.workers:
                self.executor_in.put(None)
            for worker in self.workers:
                worker.join()
            self.workers = None
            atexit.unregister(self.shutdown_execute_loop)
