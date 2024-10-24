import atexit
import queue
from queue import Queue
from threading import Thread
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.wde.core.config import EngineConfig
from vllm.wde.core.layers.attention import AttentionBackend
from vllm.wde.core.llm_engine import LLMEngine
from vllm.wde.core.schema.execute_io import ExecuteInput, ExecuteOutput
from vllm.wde.core.worker import WorkerBase, create_worker
from vllm.wde.core.workflow import Workflow

logger = init_logger(__name__)


class GPUExecutor:
    support_scheduling = ["sync_scheduling"]

    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: AttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self.output_to_cpu = False
        self._init_executor()

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend)

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """

        worker_kwargs = dict(
            engine_config=self.engine_config,
            attn_backend=self.attn_backend,
        )
        worker_kwargs.update(module=self.workflow.Worker)

        self.worker = create_worker(**worker_kwargs)
        self.worker.init_device()
        self.worker.load_model()

    def execute_model(self,
                      executor_input: ExecuteInput) -> Optional[ExecuteOutput]:
        executor_input.model_input.to(self.worker.device)
        output = self.worker(executor_input)
        if self.output_to_cpu:
            output.to("cpu")
        return output

    def shutdown_execute_loop(self):
        pass


class GPUAsyncExecutor(GPUExecutor):
    support_scheduling = ["async_scheduling"]

    def __init__(self, engine_config: EngineConfig, workflow: Workflow,
                 attn_backend: AttentionBackend, executor_in: Queue,
                 executor_out: Queue) -> None:
        super().__init__(engine_config, workflow, attn_backend)
        self.executor_in = executor_in
        self.executor_out = executor_out

        self.executor_thread: Optional[Thread] = None

        if self.engine_config.scheduler_config.scheduling == "double_buffer":
            self.execute_loop = double_buffer_execute_loop
        else:
            self.execute_loop = simple_execute_loop

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend,
                   executor_in=engine.executor_in,
                   executor_out=engine.executor_out)

    def ensure_start_execute_loop(self):
        if self.executor_thread is None or not self.executor_thread.is_alive():
            self.executor_thread = Thread(target=self.execute_loop,
                                          args=(self.worker, self.executor_in,
                                                self.executor_out,
                                                self.output_to_cpu),
                                          daemon=True)
            self.executor_thread.start()
            atexit.register(self.shutdown_execute_loop)

    def shutdown_execute_loop(self):
        if self.executor_thread.is_alive():
            self.executor_in.put(None)
            self.executor_thread.join()
            atexit.unregister(self.shutdown_execute_loop)


def simple_execute_loop(worker: WorkerBase,
                        executor_in: Queue,
                        executor_out: Queue,
                        output_to_cpu: bool = False):

    def execute_model(executor_input: ExecuteInput) -> Optional[ExecuteOutput]:
        executor_input.model_input.to(worker.device)
        output = worker(executor_input)
        if output_to_cpu:
            output.to("cpu")
        return output

    while True:
        o = executor_in.get()
        if o is None:
            break

        scheduler_output, executor_input = o
        executor_output = execute_model(executor_input)
        if output_to_cpu:
            executor_output.to("cpu")
        executor_out.put((scheduler_output, executor_output))


def double_buffer_execute_loop(worker: WorkerBase,
                               executor_in: Queue,
                               executor_out: Queue,
                               output_to_cpu: bool = False):
    from dataclasses import dataclass

    from vllm.wde.core.schema.engine_io import SchedulerOutput

    @dataclass
    class Task:
        scheduler_output: SchedulerOutput
        executor_input: ExecuteInput
        executor_output: Optional[ExecuteOutput]

        @classmethod
        def get(cls, block):
            o = executor_in.get(block)
            if o is None:
                return None

            scheduler_output, executor_input = o

            task = cls(scheduler_output=scheduler_output,
                       executor_input=executor_input,
                       executor_output=None)
            return task

    current_task: Optional[Task] = None
    next_task: Optional[Task] = None
    compute_stream = torch.cuda.Stream()
    io_stream = torch.cuda.Stream()

    go_on = True
    while go_on:
        if current_task is None:
            current_task = Task.get(block=True)
            if current_task is None:
                break

            with torch.cuda.stream(compute_stream):
                current_task.executor_input.model_input.to(worker.device,
                                                           non_blocking=True)
                current_task.executor_output = worker(
                    current_task.executor_input)
                end_compute = torch.cuda.Event()
        else:
            with torch.cuda.stream(compute_stream):
                end_compute = torch.cuda.Event()

        try:
            next_task = Task.get(block=False)
            if next_task is None:
                go_on = False
            else:
                with torch.cuda.stream(io_stream):
                    next_task.executor_input.model_input.to(worker.device,
                                                            non_blocking=True)

                compute_stream.wait_stream(io_stream)

                with torch.cuda.stream(compute_stream):
                    next_task.executor_output = worker(
                        next_task.executor_input)
        except queue.Empty:
            pass

        end_compute.wait()
        if output_to_cpu:
            with torch.cuda.stream(io_stream):
                current_task.executor_output.to("cpu", non_blocking=True)
                io_stream.synchronize()
        executor_out.put(
            (current_task.scheduler_output, current_task.executor_output))

        current_task = next_task
        next_task = None
