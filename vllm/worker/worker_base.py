import dataclasses
import importlib
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch

from vllm.config import ObservabilityConfig, VllmConfig
from vllm.distributed import broadcast_tensor_dict, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.utils import (enable_trace_function_call_for_thread,
                        update_environment_variables)
from vllm.worker.model_runner_base import (BroadcastableModelInput,
                                           ModelRunnerBase,
                                           ModelRunnerInputBase)

logger = init_logger(__name__)


class WorkerBase(ABC):
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

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

    @current_platform.inference_mode()
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
    num_steps: int = 1

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
            num_steps=tensor_dict.pop("num_steps"),
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
            "num_steps": self.num_steps,
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
    observability_config: Optional[ObservabilityConfig] = None

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

    def _get_worker_input_from_broadcast(
        self
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """ Get the worker input from the broadcasted tensor dict. """
        assert self.do_metadata_broadcast
        assert not self.is_driver_worker
        broadcast_data = broadcast_tensor_dict(src=0)
        if not broadcast_data:
            return None

        worker_input = WorkerInput.from_broadcasted_tensor_dict(broadcast_data)
        model_input = (
            self.model_runner.make_model_input_from_broadcasted_tensor_dict(
                broadcast_data))

        kwargs = extract_previous_hidden_states(broadcast_data)

        return model_input, worker_input, kwargs

    def _get_driver_input_and_broadcast(
        self, execute_model_req: ExecuteModelRequest
    ) -> Tuple[BroadcastableModelInput, WorkerInput, Dict[str, torch.Tensor]]:
        """ Get the driver input and broadcast it to other workers.  """
        assert self.is_driver_worker

        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.virtual_engine,
                execute_model_req.finished_requests_ids))

        kwargs = extract_previous_hidden_states(execute_model_req)

        if self.do_metadata_broadcast:
            broadcast_data = worker_input.as_broadcastable_tensor_dict()
            broadcast_data.update(model_input.as_broadcastable_tensor_dict())
            broadcast_data.update(kwargs)
            broadcast_tensor_dict(broadcast_data, src=0)

        if execute_model_req.async_callback:
            model_input = dataclasses.replace(  # type: ignore
                model_input,
                async_callback=execute_model_req.async_callback)

        return model_input, worker_input, kwargs

    def prepare_input(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> Optional[Tuple[BroadcastableModelInput, WorkerInput, Dict[
            str, torch.Tensor]]]:
        """
        Prepare the inputs to ModelRunner and workers.
        """
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
            return self._get_driver_input_and_broadcast(execute_model_req)
        else:
            return self._get_worker_input_from_broadcast()

    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        inputs = self.prepare_input(execute_model_req)
        if inputs is None:
            return None

        model_input, worker_input, kwargs = inputs
        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                orig_model_execute_time = intermediate_tensors.tensors.get(
                    "model_execute_time", torch.tensor(0)).item()

        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
            **kwargs,
        )

        model_execute_time = time.perf_counter() - start_time
        if not get_pp_group().is_last_rank:
            # output is IntermediateTensors
            if (self.observability_config is not None
                    and self.observability_config.collect_model_execute_time):
                output.tensors["model_execute_time"] = torch.tensor(
                    model_execute_time + orig_model_execute_time)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return [None]
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None
    ) -> Optional[List[SamplerOutput]]:
        """
        Execute model in Single Program Multiple Data (SPMD) fashion.
        All workers take the same request, prepare the input and
        execute the model.
        """
        assert execute_model_req is not None, (
            "_execute_model_spmd() requires each worker to take in an "
            "ExecuteModelRequest")
        worker_input: WorkerInput = self.prepare_worker_input(
            execute_model_req=execute_model_req)
        model_input: ModelRunnerInputBase = (
            self.model_runner.prepare_model_input(
                execute_model_req.seq_group_metadata_list))

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if worker_input.num_seq_groups == 0:
            return []

        kwargs = extract_previous_hidden_states(execute_model_req)

        return self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache[worker_input.virtual_engine]
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            **kwargs,
        )


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

        from vllm.plugins import load_general_plugins
        load_general_plugins()

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


def extract_previous_hidden_states(
        data: Union[ExecuteModelRequest, Dict[str, torch.Tensor]]) -> \
            Dict[str, torch.Tensor]:
    """If data contains previous_hidden_states, extract it. This returns a dict
    which can be used directly as additional kwargs in any following 
    execute_model calls. This is used in draft models like EAGLE."""
    output = {}

    # When called from non-driver worker, data is dict but when called from
    # driver worker, data is ExecuteModelRequest.
    if isinstance(data, dict):
        if "previous_hidden_states" in data:
            output["previous_hidden_states"] = data["previous_hidden_states"]
    elif data.previous_hidden_states is not None:
        output["previous_hidden_states"] = data.previous_hidden_states\
            .hidden_states

    return output
