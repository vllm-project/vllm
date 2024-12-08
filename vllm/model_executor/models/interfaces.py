from typing import (TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional,
                    Protocol, Type, Union, overload, runtime_checkable)

import torch
from typing_extensions import TypeIs, TypeVar

from vllm.logger import init_logger
from vllm.utils import supports_kw

from .interfaces_base import is_pooling_model

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.multimodal.inputs import NestedTensors  # noqa: F401
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

T = TypeVar("T", default="NestedTensors")


@runtime_checkable
class SupportsMultiModal(Protocol):
    """The interface required for all multi-modal models."""

    supports_multimodal: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports multi-modal inputs.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def get_multimodal_embeddings(self, **kwargs) -> Optional[T]:
        """
        Returns multimodal embeddings generated from multimodal kwargs 
        to be merged with text embeddings.

        The output embeddings must be one of the following formats:
        - A list or tuple of 2D tensors, where each tensor corresponds to 
          each input image.
        - A single 3D tensor, with the batch dimension grouping the 2D tensors.
        """
        ...

    # Only for models that support v0 chunked prefill
    # TODO(ywang96): Remove this overload once v0 is deprecated
    @overload
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[T] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        ...

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[T] = None,
    ) -> torch.Tensor:
        """
        Returns the input embeddings merged from the text embeddings from 
        input_ids and the multimodal embeddings generated from multimodal 
        kwargs.
        """
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsMultiModalType(Protocol):
    supports_multimodal: Literal[True]


@overload
def supports_multimodal(
        model: Type[object]) -> TypeIs[Type[SupportsMultiModal]]:
    ...


@overload
def supports_multimodal(model: object) -> TypeIs[SupportsMultiModal]:
    ...


def supports_multimodal(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsMultiModal]], TypeIs[SupportsMultiModal]]:
    if isinstance(model, type):
        return isinstance(model, _SupportsMultiModalType)

    return isinstance(model, SupportsMultiModal)


@runtime_checkable
class SupportsLoRA(Protocol):
    """The interface required for all models that support LoRA."""

    supports_lora: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports LoRA.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    packed_modules_mapping: ClassVar[Dict[str, List[str]]]
    supported_lora_modules: ClassVar[List[str]]
    embedding_modules: ClassVar[Dict[str, str]]
    embedding_padding_modules: ClassVar[List[str]]


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: Dict[str, List[str]]
    supported_lora_modules: List[str]
    embedding_modules: Dict[str, str]
    embedding_padding_modules: List[str]


@overload
def supports_lora(model: Type[object]) -> TypeIs[Type[SupportsLoRA]]:
    ...


@overload
def supports_lora(model: object) -> TypeIs[SupportsLoRA]:
    ...


def supports_lora(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsLoRA]], TypeIs[SupportsLoRA]]:
    result = _supports_lora(model)

    if not result:
        lora_attrs = (
            "packed_modules_mapping",
            "supported_lora_modules",
            "embedding_modules",
            "embedding_padding_modules",
        )
        missing_attrs = tuple(attr for attr in lora_attrs
                              if not hasattr(model, attr))

        if getattr(model, "supports_lora", False):
            if missing_attrs:
                logger.warning(
                    "The model (%s) sets `supports_lora=True`, "
                    "but is missing LoRA-specific attributes: %s",
                    model,
                    missing_attrs,
                )
        else:
            if not missing_attrs:
                logger.warning(
                    "The model (%s) contains all LoRA-specific attributes, "
                    "but does not set `supports_lora=True`.", model)

    return result


def _supports_lora(model: Union[Type[object], object]) -> bool:
    if isinstance(model, type):
        return isinstance(model, _SupportsLoRAType)

    return isinstance(model, SupportsLoRA)


@runtime_checkable
class SupportsPP(Protocol):
    """The interface required for all models that support pipeline parallel."""

    supports_pp: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pipeline parallel.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "IntermediateTensors":
        """Called when PP rank > 0 for profiling purposes."""
        ...

    def forward(
        self,
        *,
        intermediate_tensors: Optional["IntermediateTensors"],
    ) -> Union[torch.Tensor, "IntermediateTensors"]:
        """
        Accept :class:`IntermediateTensors` when PP rank > 0.

        Return :class:`IntermediateTensors` only for the last PP rank.
        """
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsPPType(Protocol):
    supports_pp: Literal[True]

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "IntermediateTensors":
        ...

    def forward(
        self,
        *,
        intermediate_tensors: Optional["IntermediateTensors"],
    ) -> Union[torch.Tensor, "IntermediateTensors"]:
        ...


@overload
def supports_pp(model: Type[object]) -> TypeIs[Type[SupportsPP]]:
    ...


@overload
def supports_pp(model: object) -> TypeIs[SupportsPP]:
    ...


def supports_pp(
    model: Union[Type[object], object],
) -> Union[bool, TypeIs[Type[SupportsPP]], TypeIs[SupportsPP]]:
    supports_attributes = _supports_pp_attributes(model)
    supports_inspect = _supports_pp_inspect(model)

    if supports_attributes and not supports_inspect:
        logger.warning(
            "The model (%s) sets `supports_pp=True`, but does not accept "
            "`intermediate_tensors` in its `forward` method", model)

    if not supports_attributes:
        pp_attrs = ("make_empty_intermediate_tensors", )
        missing_attrs = tuple(attr for attr in pp_attrs
                              if not hasattr(model, attr))

        if getattr(model, "supports_pp", False):
            if missing_attrs:
                logger.warning(
                    "The model (%s) sets `supports_pp=True`, "
                    "but is missing PP-specific attributes: %s",
                    model,
                    missing_attrs,
                )
        else:
            if not missing_attrs:
                logger.warning(
                    "The model (%s) contains all PP-specific attributes, "
                    "but does not set `supports_pp=True`.", model)

    return supports_attributes and supports_inspect


def _supports_pp_attributes(model: Union[Type[object], object]) -> bool:
    if isinstance(model, type):
        return isinstance(model, _SupportsPPType)

    return isinstance(model, SupportsPP)


def _supports_pp_inspect(model: Union[Type[object], object]) -> bool:
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

    return supports_kw(model_forward, "intermediate_tensors")


@runtime_checkable
class HasInnerState(Protocol):
    """The interface required for all models that has inner state."""

    has_inner_state: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has inner state.
        Models that has inner state usually need access to the scheduler_config
        for max_num_seqs, etc. True for e.g. both Mamba and Jamba.
    """


@runtime_checkable
class _HasInnerStateType(Protocol):
    has_inner_state: ClassVar[Literal[True]]


@overload
def has_inner_state(model: object) -> TypeIs[HasInnerState]:
    ...


@overload
def has_inner_state(model: Type[object]) -> TypeIs[Type[HasInnerState]]:
    ...


def has_inner_state(
    model: Union[Type[object], object]
) -> Union[TypeIs[Type[HasInnerState]], TypeIs[HasInnerState]]:
    if isinstance(model, type):
        return isinstance(model, _HasInnerStateType)

    return isinstance(model, HasInnerState)


@runtime_checkable
class IsAttentionFree(Protocol):
    """The interface required for all models like Mamba that lack attention,
    but do have state whose size is constant wrt the number of tokens."""

    is_attention_free: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has no attention.
        Used for block manager and attention backend selection.
        True for Mamba but not Jamba.
    """


@runtime_checkable
class _IsAttentionFreeType(Protocol):
    is_attention_free: ClassVar[Literal[True]]


@overload
def is_attention_free(model: object) -> TypeIs[IsAttentionFree]:
    ...


@overload
def is_attention_free(model: Type[object]) -> TypeIs[Type[IsAttentionFree]]:
    ...


def is_attention_free(
    model: Union[Type[object], object]
) -> Union[TypeIs[Type[IsAttentionFree]], TypeIs[IsAttentionFree]]:
    if isinstance(model, type):
        return isinstance(model, _IsAttentionFreeType)

    return isinstance(model, IsAttentionFree)


@runtime_checkable
class SupportsCrossEncoding(Protocol):
    """The interface required for all models that support cross encoding."""

    supports_cross_encoding: ClassVar[Literal[True]] = True


@overload
def supports_cross_encoding(
        model: Type[object]) -> TypeIs[Type[SupportsCrossEncoding]]:
    ...


@overload
def supports_cross_encoding(model: object) -> TypeIs[SupportsCrossEncoding]:
    ...


def _supports_cross_encoding(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsCrossEncoding]], TypeIs[SupportsCrossEncoding]]:

    if isinstance(model, type):
        return isinstance(model, SupportsCrossEncoding)

    return isinstance(model, SupportsCrossEncoding)


def supports_cross_encoding(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsCrossEncoding]], TypeIs[SupportsCrossEncoding]]:
    return is_pooling_model(model) and _supports_cross_encoding(model)
