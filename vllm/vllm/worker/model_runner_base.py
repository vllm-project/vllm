# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Type,
                    TypeVar)

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.model_executor import SamplingMetadata

logger = init_logger(__name__)

T = TypeVar('T', bound="BroadcastableModelInput")


def _add_attn_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Any],
        attn_metadata: Optional["AttentionMetadata"]) -> None:
    """
    Helper method to update tensor_dict with broadcastable
    AttentionMetadata fields.
    """
    if attn_metadata is not None:
        tensor_dict.update(attn_metadata.asdict_zerocopy())


def _init_attn_metadata_from_tensor_dict(
    attn_backend: "AttentionBackend",
    tensor_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper method to initialize AttentionMetadata based on an
    AttentionBackend and broadcastable AttentionMetadata fields.
    """
    # Extract the fields used to create AttentionMetadata.
    valid_attn_kwargs = {}
    for field in dataclasses.fields(attn_backend.get_metadata_cls()):
        if field.name in tensor_dict:
            if field.name == "input_positions":
                valid_attn_kwargs[field.name] = tensor_dict[field.name]
            else:
                valid_attn_kwargs[field.name] = tensor_dict.pop(field.name)

    attn_metadata = attn_backend.make_metadata(**valid_attn_kwargs)
    tensor_dict["attn_metadata"] = attn_metadata
    return tensor_dict


def _init_sampling_metadata_from_tensor_dict(  # type: ignore
        tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper method to initialize SamplingMetadata based on broadcastable
    SamplingMetadata fields.
    """
    from vllm.model_executor import SamplingMetadata

    selected_token_indices = tensor_dict.pop("selected_token_indices", None)
    # An empty SamplingMetadata to signal that the worker should skip
    # sampling.
    if selected_token_indices is not None:
        tensor_dict["sampling_metadata"] = SamplingMetadata(
            seq_groups=None,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=None,
            num_prompts=0,
        )
    return tensor_dict


def _add_sampling_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Any],
        sampling_metadata: Optional["SamplingMetadata"]) -> None:
    """
    Helper method to update tensor_dict with broadcastable
    SamplingMetadata fields.
    """
    if sampling_metadata is not None:
        tensor_dict["selected_token_indices"] = (
            sampling_metadata.selected_token_indices)


def _init_frozen_model_input_from_tensor_dict(
        frozen_model_input_cls: Type["ModelRunnerInputBase"],
        tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper method to initialize a frozen ModelInput based on broadcastable
    """
    valid_tensor_kwargs = {}
    for field in dataclasses.fields(frozen_model_input_cls):
        val = tensor_dict.pop(field.name, None)
        if val is not None:
            valid_tensor_kwargs[field.name] = val

    frozen_model_input = frozen_model_input_cls(**valid_tensor_kwargs)
    tensor_dict["frozen_model_input"] = frozen_model_input
    return tensor_dict


class BroadcastableModelInput(ABC):

    @abstractmethod
    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_broadcasted_tensor_dict(
        cls: Type[T],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> T:
        """
        Pop fields from the given tensor_dict and populate a new instance of
        BroadcastableModelInput.
        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class ModelRunnerInputBase(BroadcastableModelInput):
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelRunnerInputBase objects.

    Model runners that support multi-GPU execution should define a
    ModelRunnerInputBase subclass, add their required fields, and specify how to
    serialize/deserialize a ModelInput for broadcast between workers.
    """
    pass


class ModelRunnerInputBuilderBase(ABC, Generic[T]):
    """A builder to create ModelRunnerInputBase objects.
  """

    @abstractmethod
    def prepare(self,
                finished_requests_ids: Optional[List[str]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_seq_group(self, seq_group_metadata):
        """TBA"""
        raise NotImplementedError

    @abstractmethod
    def build(self, *args, **kwargs) -> T:
        """Build metadata with on-device tensors."""
        raise NotImplementedError


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding
    ModelRunnerInputBase subclass.
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

    # Map of request_id -> generator used for seeded random sampling
    generators: Dict[str, torch.Generator] = {}

    @abstractmethod
    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> T:
        """
        Make an instance of a ModelRunnerInputBase from the broadcasted tensor
        dict.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
    ) -> T:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[List[SamplerOutput]]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError

    def get_generators(self, finished_request_ids: Optional[List[str]] = None):
        """
        Return dict of per-request generators used for random sampling.
        """

        # Clean up generators from completed requests
        if finished_request_ids:
            for request_id in finished_request_ids:
                self.generators.pop(request_id, None)

        return self.generators


class ModelRunnerWrapperBase:
    """
    The whole point of this class is to lazily initialize the model_runner.
    """

    def __init__(
        self,
        model_runner: ModelRunnerBase,
    ) -> None:
        self.model_runner: ModelRunnerBase = model_runner

    def __getattr__(self, attr):
        return getattr(self.model_runner, attr)


class InputProcessingError(Exception):
    """This exception is raised when an error occurs preparing the inputs for
    a single sequence group.
    This allows the engine to gracefully handle errors with a single sequence
    group without having to fail the entire batch.
    """

    def __init__(self, request_id, message):
        """request_id is the id of the offending sequence group"""
        self.request_id = request_id
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return "Failed to prepare inputs for sequence group with request id: " \
                f"{self.request_id}, Error: {self.message}"
