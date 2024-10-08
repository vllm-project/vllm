import importlib
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Type

import torch

from vllm.attention.prefill_only.abstract import PrefillOnlyAttentionBackend
from vllm.config import (DeviceConfig, EngineConfig, LoadConfig, ModelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.prefill_only.execute_io import (
    ExecuteInput, ExecuteOutput, PrefillOnlyExecuteInput)
from vllm.model_executor.utils import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import (enable_trace_function_call_for_thread,
                        update_environment_variables)

from .prefill_only_model_runner import ModelRunner

logger = init_logger(__name__)


class WorkerBase(ABC):

    @abstractmethod
    def __call__(self, execute_input: ExecuteInput) -> ExecuteOutput:
        raise NotImplementedError


class WorkerWrapperBase:
    """
    The whole point of this class is to lazily initialize the worker.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.

    If worker_class_fn is specified, it will be executed to get the worker
    class.
    Otherwise, the worker class will be obtained by dynamically importing it
    using worker_module_name and worker_class_name.
    """

    def __init__(
        self,
        worker_module_name: str,
        worker_class_name: str,
        trust_remote_code: bool = False,
        worker_class_fn: Optional[Callable[[],
                                           Type[WorkerBase]]] = None) -> None:
        self.worker_module_name = worker_module_name
        self.worker_class_name = worker_class_name
        self.worker_class_fn = worker_class_fn
        self.worker: Optional[WorkerBase] = None
        if trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

    @staticmethod
    def update_environment_variables(envs: Dict[str, str]) -> None:
        key = 'CUDA_VISIBLE_DEVICES'
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, *args, **kwargs):
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        enable_trace_function_call_for_thread()

        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        if self.worker_class_fn:
            worker_class = self.worker_class_fn()
        else:
            mod = importlib.import_module(self.worker_module_name)
            worker_class = getattr(mod, self.worker_class_name)

        self.worker = worker_class(*args, **kwargs)
        assert self.worker is not None

    def execute_method(self, method, *args, **kwargs):
        try:
            target = self if self.worker is None else self.worker
            executor = getattr(target, method)
            return executor(*args, **kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (f"Error executing method {method}. "
                   "This might cause deadlock in distributed execution.")
            logger.exception(msg)
            raise e


def create_worker(module, envs=None, **kwargs):
    module_name, class_name = module.split(":")
    wrapper = WorkerWrapperBase(
        worker_module_name=module_name,
        worker_class_name=class_name,
    )
    if envs:
        wrapper.update_environment_variables(envs)

    wrapper.init_worker(**kwargs)
    return wrapper.worker


class Worker(WorkerBase):

    def __init__(
        self,
        engine_config: EngineConfig,
        attn_backend: PrefillOnlyAttentionBackend,
    ) -> None:
        self.model_config: ModelConfig = engine_config.model_config
        self.scheduler_config: SchedulerConfig = (
            engine_config.scheduler_config)
        self.device_config: DeviceConfig = engine_config.device_config
        self.load_config: LoadConfig = engine_config.load_config
        self.device = self.device_config.device
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner = ModelRunner(self.model_config,
                                        self.scheduler_config,
                                        self.device_config, self.load_config,
                                        attn_backend)

    def init_device(self) -> None:
        from vllm.model_executor.prefill_only.utils import (
            fix_distributed_environment)

        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")

        fix_distributed_environment()

        # Set random seed.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode
    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode
    def __call__(self,
                 execute_input: PrefillOnlyExecuteInput) -> ExecuteOutput:
        output = self.model_runner.execute_model(execute_input.model_input)
        return output


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = current_platform.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
