import dataclasses
import importlib
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch

from vllm.distributed import broadcast_tensor_dict, get_pp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SamplerOutput)
from vllm.utils import (enable_trace_function_call_for_thread, is_hip,
                        update_environment_variables)
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

logger = init_logger(__name__)


class WorkerBase(ABC):
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    @abstractmethod
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    @abstractmethod
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the GPU KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_gpu_blocks, num_cpu_blocks], where num_gpu_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        while True:
            output = self.execute_model(execute_model_req=None)
            if output is None:
                return None

    @abstractmethod
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        raise NotImplementedError

    @abstractmethod
    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    @abstractmethod
    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_loras(self) -> Set[int]:
        raise NotImplementedError


class LoraNotSupportedWorkerBase(WorkerBase):
    """Partial implementation of WorkerBase that raises exceptions when LoRA
    methods are invoked.
    """

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def remove_lora(self, lora_id: int) -> bool:
        raise ValueError(f"{type(self)} does not support LoRA")

    def pin_lora(self, lora_id: int) -> bool:
        return ValueError(
            f"{type(self)} does not support LoRA")  # type: ignore

    def list_loras(self) -> Set[int]:
        raise ValueError(f"{type(self)} does not support LoRA")


@dataclasses.dataclass(frozen=True)
class WorkerInput:
    """Local inputs to each worker. May contain device-specific data. These
    fields should be broadcastable to other workers.
    """

    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None
    virtual_engine: int = 0

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["WorkerInput"],
        tensor_dict: Dict[str, Any],
    ) -> "WorkerInput":
        """
        Pop fields from the given tensor_dict and populate a new instance of
        WorkerInput.
        """
        return cls(
            num_seq_groups=tensor_dict.pop("num_seq_groups"),
            blocks_to_swap_in=tensor_dict.pop("blocks_to_swap_in"),
            blocks_to_swap_out=tensor_dict.pop("blocks_to_swap_out"),
            blocks_to_copy=tensor_dict.pop("blocks_to_copy"),
            virtual_engine=tensor_dict["virtual_engine"],
        )

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Extract broadcastable fields.
        """
        tensor_dict = {
            "num_seq_groups": self.num_seq_groups,
            "blocks_to_swap_in": self.blocks_to_swap_in,
            "blocks_to_swap_out": self.blocks_to_swap_out,
            "blocks_to_copy": self.blocks_to_copy,
            "virtual_engine": self.virtual_engine,
        }

        return tensor_dict


class LocalOrDistributedWorkerBase(WorkerBase):
    """
    Partial implementation of WorkerBase that has a default `execute_model`
    definition to perform metadata transfer between workers when in distributed
    mode. Subclasses of this interface should use model runners that inherit
    from ModelRunnerBase, and should only need to implement worker-local logic.
    If custom control plane logic is needed to transfer metadata, or if the
    model runner cannot inherit from ModelRunnerBase, use WorkerBase instead.
    """
    is_driver_worker: bool
    model_runner: ModelRunnerBase

    @property
    @abstractmethod
    def do_metadata_broadcast(self) -> bool:
        """
        Used by the default `execute_model` to check whether broadcast is
        needed to transfer request inputs from the driver worker to other
        workers in the TP group. If WorkerBase subclass only supports
        single-worker execution, then this method should return False.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        """
        Gets the list of kv caches to pass to the worker's model runner. Each
        element in the list is a kv cache corresponding to a particular virtual
        engine (PP stream). Used by the default `execute_model`. If the worker's
        model runner does not follow the ModelRunnerBase interface, then inherit
        from WorkerBase instead.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        """
        Prepare the inputs to WorkerBase.execute_worker from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        """
        raise NotImplementedError

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        if self.is_driver_worker:
            if execute_model_req is None:
                if self.do_metadata_broadcast:
                    # This signals that there's no more requests to process for
                    # now. All workers are running infinite loop with
                    # broadcast_tensor_dict, and it stops the loop when the
                    # driver broadcasts an empty input. Send an empty input to
                    # notify all other workers to stop their execution loop.
                    broadcast_tensor_dict({}, src=0)
                return None

            worker_input: WorkerInput = self.prepare_worker_input(
                execute_model_req=execute_model_req)
            model_input: ModelRunnerInputBase = (
                self.model_runner.prepare_model_input(
                    execute_model_req.seq_group_metadata_list,
                    execute_model_req.virtual_engine,
                    execute_model_req.finished_requests_ids))
            num_steps = execute_model_req.num_steps

            if self.do_metadata_broadcast:
                broadcast_data = worker_input.as_broadcastable_tensor_dict()
                broadcast_data.update(
                    model_input.as_broadcastable_tensor_dict())
                broadcast_data["num_steps"] = num_steps
                broadcast_tensor_dict(broadcast_data, src=0)
        else:
            assert self.do_metadata_broadcast
            broadcast_data = broadcast_tensor_dict(src=0)
            if not broadcast_data:
                return None

            num_steps = broadcast_data.pop("num_steps")
            worker_input = WorkerInput.from_broadcasted_tensor_dict(
                broadcast_data)
            model_input = (
                self.model_runner.
                make_model_input_from_broadcasted_tensor_dict(broadcast_data))

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict())

        output = self.model_runner.execute_model(
            model_input, self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None, intermediate_tensors,
            num_steps)

        if not get_pp_group().is_last_rank:
            get_pp_group().send_tensor_dict(output.tensors)
            return [None]

        # Worker only supports single-step execution. Wrap the output in a
        # list to conform to interface.
        return output


class WorkerWrapperBase:
    """
    The whole point of this class is to lazily initialize the worker.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(self,
                 worker_module_name: str,
                 worker_class_name: str,
                 trust_remote_code: bool = False) -> None:
        self.worker_module_name = worker_module_name
        self.worker_class_name = worker_class_name
        self.worker = None
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
            if is_hip():
                hip_env_var = "HIP_VISIBLE_DEVICES"
                if hip_env_var in os.environ:
                    logger.warning(
                        "Ignoring pre-set environment variable `%s=%s` as "
                        "%s has also been set, which takes precedence.",
                        hip_env_var, os.environ[hip_env_var], key)
                os.environ.pop(hip_env_var, None)
        update_environment_variables(envs)

    def init_worker(self, *args, **kwargs):
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        enable_trace_function_call_for_thread()

        # see https://github.com/NVIDIA/nccl/issues/1234
        os.environ['NCCL_CUMEM_ENABLE'] = '0'

        mod = importlib.import_module(self.worker_module_name)
        worker_class = getattr(mod, self.worker_class_name)
        self.worker = worker_class(*args, **kwargs)

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
