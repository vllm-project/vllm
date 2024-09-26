import importlib
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Type

from vllm.logger import init_logger
from vllm.utils import (enable_trace_function_call_for_thread,
                        update_environment_variables)
from vllm.wde.core.schema.execute_io import ExecuteInput

logger = init_logger(__name__)


class WorkerBase(ABC):

    @abstractmethod
    def __call__(self, execute_input: ExecuteInput):
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