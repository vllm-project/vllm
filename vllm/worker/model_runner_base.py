import dataclasses
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple,
                    Type, TypeVar, Union)

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.model_executor import SamplingMetadata

T = TypeVar('T', bound="ModelInput")


@dataclasses.dataclass(frozen=True)
class ModelInput:
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelInput objects.

    Model runners should define a ModelInput subclass and add their required
    fields. For distributed executors, any fields that should be sent during a
    broadcast op should also be added to the broadcastable_fields. During
    execution, these fields will be extracted from the source copy and
    broadcasted to all workers using broadcast_tensor_dict.

    Some fields may have values that cannot be broadcasted with this method
    because they require some special serialization/deserialization, e.g., a
    Python class like SamplingMetadata. For these fields, override
    as_broadcastable_tensor_dict to return the custom serialized values and
    override _get_init_kwargs to perform the custom deserialization (
    ModelInputForGPU for an example).
    """

    @property
    def broadcastable_fields(self) -> Tuple[str, ...]:
        """
        Return fields to broadcast to all workers from driver. The value of
        each field must be broadcastable using broadcast_tensor_dict (i.e.
        either a tensor, or a Python primitive like int). During the broadcast,
        the listed fields will be extracted from the source copy and then
        passed to `new()` to create a copy on the destination(s).
        """
        raise NotImplementedError()

    @classmethod
    def _get_init_kwargs(cls: Type[T], **kwargs) -> Dict[str, Any]:
        """
        Helper method to extract all dataclass fields from the given kwargs.
        Override for fields that require some custom deserialization.
        """
        init_kwargs = {}
        for field in dataclasses.fields(cls):
            val = kwargs.get(field.name, None)
            if val is not None:
                init_kwargs[field.name] = val
        return init_kwargs

    @classmethod
    def new(cls: Type[T], **kwargs) -> T:
        """
        Create a new instance of this class. Populate the new instance with
        the given kwargs.
        """
        kwargs = cls._get_init_kwargs(**kwargs)
        return cls(**kwargs)

    def replace(self: T, **kwargs) -> T:
        """
        Replace current fields with fields in kwargs.
        """
        valid_kwargs = self.__class__._get_init_kwargs(**kwargs)
        return dataclasses.replace(self, **valid_kwargs)

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        tensor_dict: Dict[str, Union[int, torch.Tensor]] = {}
        for field in self.broadcastable_fields:
            val = getattr(self, field, None)
            if val is not None:
                tensor_dict[field] = val

        return tensor_dict

    @staticmethod
    def _add_attn_metadata_broadcastable_dict(
            tensor_dict: Dict[str, Union[int, torch.Tensor]],
            attn_metadata: Optional["AttentionMetadata"]) -> None:
        """
        Helper method to update tensor_dict with broadcastable
        AttentionMetadata fields.
        """
        if attn_metadata is not None:
            tensor_dict.update(attn_metadata.asdict_zerocopy())

    @staticmethod
    def _init_attn_metadata_from_kwargs(
            attn_backend: Optional["AttentionBackend"] = None,
            attn_metadata: Optional["AttentionMetadata"] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Helper method to initialize AttentionMetadata based on an
        AttentionBackend and broadcastable AttentionMetadata fields.
        """
        if attn_metadata is None and attn_backend is not None:
            # Extract the fields used to create AttentionMetadata.
            valid_attn_kwargs = {}
            for field in dataclasses.fields(attn_backend.get_metadata_cls()):
                val = kwargs.pop(field.name, None)
                if val is not None:
                    valid_attn_kwargs[field.name] = val

            attn_metadata = attn_backend.make_metadata(**valid_attn_kwargs)
        if attn_metadata is not None:
            kwargs["attn_metadata"] = attn_metadata
        return kwargs

    @staticmethod
    def _init_sampling_metadata_from_kwargs(  # type: ignore
            selected_token_indices: Optional[torch.Tensor] = None,
            sampling_metadata: Optional["SamplingMetadata"] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Helper method to initialize SamplingMetadata based on broadcastable
        SamplingMetadata fields.
        """
        if sampling_metadata is None and selected_token_indices is not None:
            from vllm.model_executor import SamplingMetadata

            # An empty SamplingMetadata to signal that the worker should skip
            # sampling.
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )
        if sampling_metadata is not None:
            kwargs["sampling_metadata"] = sampling_metadata
        return kwargs

    @staticmethod
    def _add_sampling_metadata_broadcastable_dict(
            tensor_dict: Dict[str, Union[int, torch.Tensor]],
            sampling_metadata: Optional["SamplingMetadata"]) -> None:
        """
        Helper method to update tensor_dict with broadcastable
        SamplingMetadata fields.
        """
        if sampling_metadata is not None:
            tensor_dict["selected_token_indices"] = (
                sampling_metadata.selected_token_indices)


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding ModelInput
    subclass.
    """

    @abstractmethod
    def make_model_input(self,
                         make_attn_metadata: bool = False,
                         **model_input_fields) -> T:
        """
        Make an instance of a ModelInput from the given fields. If
        make_attn_metadata=True, then AttentionMetadata will be created from
        fields extracted from model_input_fields.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> T:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError
