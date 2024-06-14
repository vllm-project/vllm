"""Worker-local model inputs. These define the inputs to different model
runners."""
import dataclasses
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type,
                    TypeVar, Union)

import torch

from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.lora.layers import LoRAMapping
    from vllm.model_executor import SamplingMetadata
    from vllm.model_executor.pooling_metadata import PoolingMetadata


def _init_attn_metadata_from_kwargs(
        attn_backend: Optional["AttentionBackend"] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
        **kwargs) -> Dict[str, Any]:
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


def _add_attn_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Union[int, torch.Tensor]],
        attn_metadata: Optional["AttentionMetadata"]) -> None:
    if attn_metadata is not None:
        tensor_dict.update(attn_metadata.asdict_zerocopy())


def _init_sampling_metadata_from_kwargs(  # type: ignore
        selected_token_indices: Optional[torch.Tensor] = None,
        sampling_metadata: Optional["SamplingMetadata"] = None,
        **kwargs) -> Dict[str, Any]:
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


def _add_sampling_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Union[int, torch.Tensor]],
        sampling_metadata: Optional["SamplingMetadata"]) -> None:
    if sampling_metadata is not None:
        tensor_dict["selected_token_indices"] = (
            sampling_metadata.selected_token_indices)


T = TypeVar('T', bound="ModelInput")


@dataclasses.dataclass(frozen=True)
class ModelInput:
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelInput objects.

    Model runners should inherit from this class and add their required fields.
    For distributed executors, any fields that should be sent during a
    broadcast op should also be added to the broadcastable_fields. During
    execution, these fields will be extracted from the source copy and
    broadcasted to all workers using broadcast_tensor_dict.

    Some fields may have values that cannot be broadcasted with this method
    because they require some special serialization/deserialization, e.g., a
    Python class like SamplingMetadata. For these fields, override
    as_broadcastable_tensor_dict to return the custom serialized values and
    override _get_init_kwargs to perform the custom deserialization (
    GPUModelInput for an example).
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
    def _get_init_kwargs(cls, **kwargs) -> Dict[str, Any]:
        """
        Helper method to extract all dataclass fields from the given kwargs.
        Override for fields that require some custom deserialization.
        """
        return kwargs

    @classmethod
    def new(cls: Type[T], clone: Optional["ModelInput"] = None, **kwargs) -> T:
        """
        Create a new instance of this class. Copy fields from `clone` if
        provided. Populate the new instance with the given kwargs.
        """
        clone_kwargs = {}
        if clone is not None:
            for field in dataclasses.fields(clone):
                val = getattr(clone, field.name)
                if val is not None:
                    clone_kwargs[field.name] = val
            clone_kwargs = cls._get_init_kwargs(**clone_kwargs)

        kwargs = cls._get_init_kwargs(**kwargs)
        return cls(**clone_kwargs, **kwargs)

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


@dataclasses.dataclass(frozen=True)
class CPUModelInput(ModelInput):
    """
    Used by the CPUModelRunner.
    """
    num_seq_groups: Optional[int] = None
    blocks_to_copy: Optional[torch.Tensor] = None

    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    multi_modal_kwargs: Optional[Dict[str, torch.Tensor]] = None

    attn_metadata: Optional["AttentionMetadata"] = None
    sampling_metadata: Optional["SamplingMetadata"] = None

    @property
    def broadcastable_fields(self) -> Tuple[str, ...]:
        return (
            "num_seq_groups",
            "blocks_to_copy",
            "input_tokens",
            "input_positions",
            "multi_modal_kwargs",
        )

    @classmethod
    def _get_init_kwargs(  # type: ignore
            cls, **kwargs) -> Dict[str, Any]:
        kwargs = _init_attn_metadata_from_kwargs(**kwargs)
        kwargs = _init_sampling_metadata_from_kwargs(**kwargs)
        return super()._get_init_kwargs(**kwargs)

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = super().as_broadcastable_tensor_dict()
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict


@dataclasses.dataclass(frozen=True)
class GPUModelInput(ModelInput):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    num_seq_groups: Optional[int] = None
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None

    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None
    multi_modal_kwargs: Optional[Dict[str, torch.Tensor]] = None

    attn_metadata: Optional["AttentionMetadata"] = None

    @property
    def broadcastable_fields(self) -> Tuple[str, ...]:
        return (
            "num_seq_groups",
            "blocks_to_swap_in",
            "blocks_to_swap_out",
            "blocks_to_copy",
            "input_tokens",
            "input_positions",
            "lora_requests",
            "lora_mapping",
            "multi_modal_kwargs",
        )

    @classmethod
    def _get_init_kwargs(  # type: ignore
            cls, **kwargs) -> Dict[str, Any]:
        kwargs = _init_attn_metadata_from_kwargs(**kwargs)
        return super()._get_init_kwargs(**kwargs)

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = super().as_broadcastable_tensor_dict()
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict


@dataclasses.dataclass(frozen=True)
class GPUModelInputWithPoolingMetadata(GPUModelInput):
    """
    Used by the EmbeddingModelRunner.
    """
    pooling_metadata: Optional["PoolingMetadata"] = None


@dataclasses.dataclass(frozen=True)
class GPUModelInputWithSamplingMetadata(GPUModelInput):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None

    @classmethod
    def _get_init_kwargs(  # type: ignore
            cls, **kwargs) -> Dict[str, Any]:
        kwargs = _init_sampling_metadata_from_kwargs(**kwargs)
        return super()._get_init_kwargs(**kwargs)

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = super().as_broadcastable_tensor_dict()
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict


@dataclasses.dataclass(frozen=True)
class ModelInputForNeuron(ModelInput):
    """
    Used by the NeuronModelRunner.
    """
    num_seq_groups: Optional[int] = None

    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_block_ids: Optional[torch.Tensor] = None
    sampling_metadata: Optional["SamplingMetadata"] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        raise NotImplementedError("ModelInputForNeuron cannot be broadcast.")
