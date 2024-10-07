from typing import (TYPE_CHECKING, ClassVar, Dict, List, Literal, Optional,
                    Protocol, Type, Union, overload, runtime_checkable)

import torch
from typing_extensions import TypeIs

from vllm.logger import init_logger
from vllm.utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import LoRAConfig, MultiModalConfig, SchedulerConfig
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


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

    def __init__(self, *, multimodal_config: "MultiModalConfig") -> None:
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsMultiModalType(Protocol):
    supports_multimodal: Literal[True]

    def __call__(self, *, multimodal_config: "MultiModalConfig") -> None:
        ...


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

    # lora_config is None when LoRA is not enabled
    def __init__(self, *, lora_config: Optional["LoRAConfig"] = None) -> None:
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: Dict[str, List[str]]
    supported_lora_modules: List[str]
    embedding_modules: Dict[str, str]
    embedding_padding_modules: List[str]

    def __call__(self, *, lora_config: Optional["LoRAConfig"] = None) -> None:
        ...


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
        for max_num_seqs ,etc... (Currently only used by Jamba)
    """

    def __init__(self,
                 *,
                 scheduler_config: Optional["SchedulerConfig"] = None) -> None:
        ...


@runtime_checkable
class _HasInnerStateType(Protocol):
    has_inner_state: ClassVar[Literal[True]]

    def __init__(self,
                 *,
                 scheduler_config: Optional["SchedulerConfig"] = None) -> None:
        ...


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
