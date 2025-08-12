# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, MutableSequence
from typing import (TYPE_CHECKING, ClassVar, Literal, Optional, Protocol,
                    Union, overload, runtime_checkable)

import numpy as np
import torch
from torch import Tensor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from typing_extensions import Self, TypeIs

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.inputs import TokensPrompt
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.utils import supports_kw

from .interfaces_base import is_pooling_model

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

MultiModalEmbeddings = Union[list[Tensor], Tensor, tuple[Tensor, ...]]
"""
The output embeddings must be one of the following formats:

- A list or tuple of 2D tensors, where each tensor corresponds to
    each input multimodal data item (e.g, image).
- A single 3D tensor, with the batch dimension grouping the 2D tensors.
"""


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

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """
        Get the placeholder text for the `i`th `modality` item in the prompt.
        """
        ...

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        """
        Returns multimodal embeddings generated from multimodal kwargs 
        to be merged with text embeddings.

        Note:
            The returned multimodal embeddings must be in the same order as
            the appearances of their corresponding multimodal data item in the
            input prompt.
        """
        ...

    def get_language_model(self) -> torch.nn.Module:
        """
        Returns the underlying language model used for text generation.

        This is typically the `torch.nn.Module` instance responsible for 
        processing the merged multimodal embeddings and producing hidden states

        Returns:
            torch.nn.Module: The core language model component.
        """
        ...

    # Only for models that support v0 chunked prefill
    # TODO(ywang96): Remove this overload once v0 is deprecated
    @overload
    def get_input_embeddings(
        self,
        input_ids: Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        ...

    # TODO: Remove this overload once v0 is deprecated
    @overload
    def get_input_embeddings(
        self,
        input_ids: Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> Tensor:
        ...

    def get_input_embeddings(
        self,
        input_ids: Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        # Only necessary so that the v0 overload is valid
        # TODO: Remove attn_metadata once v0 is deprecated
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        """
        Returns the input embeddings merged from the text embeddings from 
        input_ids and the multimodal embeddings generated from multimodal 
        kwargs.
        """
        ...


@overload
def supports_multimodal(
        model: type[object]) -> TypeIs[type[SupportsMultiModal]]:
    ...


@overload
def supports_multimodal(model: object) -> TypeIs[SupportsMultiModal]:
    ...


def supports_multimodal(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsMultiModal]], TypeIs[SupportsMultiModal]]:
    return getattr(model, "supports_multimodal", False)


@runtime_checkable
class SupportsMultiModalWithRawInput(SupportsMultiModal, Protocol):
    """The interface required for all multi-modal models."""

    supports_multimodal_raw_input: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports multi-modal inputs and processes
    them in their raw form and not embeddings.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """


@overload
def supports_multimodal_raw_input(
        model: object) -> TypeIs[SupportsMultiModalWithRawInput]:
    ...


@overload
def supports_multimodal_raw_input(
        model: type[object]) -> TypeIs[type[SupportsMultiModalWithRawInput]]:
    ...


def supports_multimodal_raw_input(
    model: Union[type[object], object]
) -> Union[TypeIs[type[SupportsMultiModalWithRawInput]],
           TypeIs[SupportsMultiModalWithRawInput]]:
    return getattr(model, "supports_multimodal_raw_input", False)


@runtime_checkable
class SupportsScoreTemplate(Protocol):
    """The interface required for all models that support score template."""

    supports_score_template: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports score template.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    @classmethod
    def get_score_template(cls, query: str, document: str) -> Optional[str]:
        """
        Generate a full prompt by populating the score template with query and document content.
        """ # noqa: E501
        ...

    @classmethod
    def post_process_tokens(cls, prompt: TokensPrompt) -> None:
        """
        Perform architecture-specific manipulations on the input tokens.
        """
        ...


@overload
def supports_score_template(
        model: type[object]) -> TypeIs[type[SupportsScoreTemplate]]:
    ...


@overload
def supports_score_template(model: object) -> TypeIs[SupportsScoreTemplate]:
    ...


def supports_score_template(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsScoreTemplate]], TypeIs[SupportsScoreTemplate]]:
    return getattr(model, "supports_score_template", False)


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
    # The `embedding_module` and `embedding_padding_modules`
    # are empty by default.
    embedding_modules: ClassVar[dict[str, str]] = {}
    embedding_padding_modules: ClassVar[list[str]] = []
    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {}


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: dict[str, list[str]]
    embedding_modules: dict[str, str]
    embedding_padding_modules: list[str]


@overload
def supports_lora(model: type[object]) -> TypeIs[type[SupportsLoRA]]:
    ...


@overload
def supports_lora(model: object) -> TypeIs[SupportsLoRA]:
    ...


def supports_lora(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsLoRA]], TypeIs[SupportsLoRA]]:
    result = _supports_lora(model)

    if not result:
        lora_attrs = (
            "packed_modules_mapping",
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


def _supports_lora(model: Union[type[object], object]) -> bool:
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
    ) -> Union[Tensor, "IntermediateTensors"]:
        """
        Accept [`IntermediateTensors`][vllm.sequence.IntermediateTensors] when
        PP rank > 0.

        Return [`IntermediateTensors`][vllm.sequence.IntermediateTensors] only
        for the last PP rank.
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
    ) -> Union[Tensor, "IntermediateTensors"]:
        ...


@overload
def supports_pp(model: type[object]) -> TypeIs[type[SupportsPP]]:
    ...


@overload
def supports_pp(model: object) -> TypeIs[SupportsPP]:
    ...


def supports_pp(
    model: Union[type[object], object],
) -> Union[bool, TypeIs[type[SupportsPP]], TypeIs[SupportsPP]]:
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


def _supports_pp_attributes(model: Union[type[object], object]) -> bool:
    if isinstance(model, type):
        return isinstance(model, _SupportsPPType)

    return isinstance(model, SupportsPP)


def _supports_pp_inspect(model: Union[type[object], object]) -> bool:
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


@overload
def has_inner_state(model: object) -> TypeIs[HasInnerState]:
    ...


@overload
def has_inner_state(model: type[object]) -> TypeIs[type[HasInnerState]]:
    ...


def has_inner_state(
    model: Union[type[object], object]
) -> Union[TypeIs[type[HasInnerState]], TypeIs[HasInnerState]]:
    return getattr(model, "has_inner_state", False)


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


@overload
def is_attention_free(model: object) -> TypeIs[IsAttentionFree]:
    ...


@overload
def is_attention_free(model: type[object]) -> TypeIs[type[IsAttentionFree]]:
    ...


def is_attention_free(
    model: Union[type[object], object]
) -> Union[TypeIs[type[IsAttentionFree]], TypeIs[IsAttentionFree]]:
    return getattr(model, "is_attention_free", False)


@runtime_checkable
class IsHybrid(Protocol):
    """The interface required for all models like Jamba that have both
    attention and mamba blocks, indicates that 
    hf_config has 'layers_block_type'"""

    is_hybrid: ClassVar[Literal[True]] = True
    """
        A flag that indicates this model has both mamba and attention blocks
        , also indicates that the model's hf_config has 
        'layers_block_type' """

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config
            use_v1: Get shapes for V1 (or V0)

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        ...


@overload
def is_hybrid(model: object) -> TypeIs[IsHybrid]:
    ...


@overload
def is_hybrid(model: type[object]) -> TypeIs[type[IsHybrid]]:
    ...


def is_hybrid(
    model: Union[type[object], object]
) -> Union[TypeIs[type[IsHybrid]], TypeIs[IsHybrid]]:
    return getattr(model, "is_hybrid", False)


@runtime_checkable
class MixtureOfExperts(Protocol):
    """
    Check if the model is a mixture of experts (MoE) model.
    """

    expert_weights: MutableSequence[Iterable[Tensor]]
    """
    Expert weights saved in this rank.

    The first dimension is the layer, and the second dimension is different
    parameters in the layer, e.g. up/down projection weights.
    """

    num_moe_layers: int
    """Number of MoE layers in this model."""

    num_expert_groups: int
    """Number of expert groups in this model."""

    num_logical_experts: int
    """Number of logical experts in this model."""

    num_physical_experts: int
    """Number of physical experts in this model."""

    num_local_physical_experts: int
    """Number of local physical experts in this model."""

    num_routed_experts: int
    """Number of routed experts in this model."""

    num_shared_experts: int
    """Number of shared experts in this model."""

    num_redundant_experts: int
    """Number of redundant experts in this model."""

    def set_eplb_state(
        self,
        expert_load_view: Tensor,
        logical_to_physical_map: Tensor,
        logical_replica_count: Tensor,
    ) -> None:
        """
        Register the EPLB state in the MoE model.
        
        Since these are views of the actual EPLB state, any changes made by
        the EPLB algorithm are automatically reflected in the model's behavior
        without requiring additional method calls to set new states.

        You should also collect model's `expert_weights` here instead of in
        the weight loader, since after initial weight loading, further
        processing like quantization may be applied to the weights.

        Args:
            expert_load_view: A view of the expert load metrics tensor.
            logical_to_physical_map: Mapping from logical to physical experts.
            logical_replica_count: Count of replicas for each logical expert.
        """
        ...

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        ...


def is_mixture_of_experts(model: object) -> TypeIs[MixtureOfExperts]:
    return isinstance(model, MixtureOfExperts)


@runtime_checkable
class HasNoOps(Protocol):
    has_noops: ClassVar[Literal[True]] = True


@overload
def has_noops(model: object) -> TypeIs[HasNoOps]:
    ...


@overload
def has_noops(model: type[object]) -> TypeIs[type[HasNoOps]]:
    ...


def has_noops(
    model: Union[type[object], object]
) -> Union[TypeIs[type[HasNoOps]], TypeIs[HasNoOps]]:
    return getattr(model, "has_noops", False)


@runtime_checkable
class SupportsCrossEncoding(Protocol):
    """The interface required for all models that support cross encoding."""

    supports_cross_encoding: ClassVar[Literal[True]] = True


@overload
def supports_cross_encoding(
        model: type[object]) -> TypeIs[type[SupportsCrossEncoding]]:
    ...


@overload
def supports_cross_encoding(model: object) -> TypeIs[SupportsCrossEncoding]:
    ...


def _supports_cross_encoding(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsCrossEncoding]], TypeIs[SupportsCrossEncoding]]:
    return getattr(model, "supports_cross_encoding", False)


def supports_cross_encoding(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsCrossEncoding]], TypeIs[SupportsCrossEncoding]]:
    return is_pooling_model(model) and _supports_cross_encoding(model)


def default_pooling_type(pooling_type: str) -> object:
    """Set default_pooling_type decorator. """

    def func(model: object):
        model.default_pooling_type = pooling_type
        return model

    return func


def get_default_pooling_type(model: Union[type[object], object]) -> str:
    return getattr(model, "default_pooling_type", "LAST")


class SupportsQuant:
    """The interface required for all models that support quantization."""

    hf_to_vllm_mapper: ClassVar[Optional["WeightsMapper"]] = None
    packed_modules_mapping: ClassVar[Optional[dict[str, list[str]]]] = None
    quant_config: Optional[QuantizationConfig] = None

    def __new__(cls, *args, **kwargs) -> Self:
        instance = super().__new__(cls)

        # find config passed in arguments
        quant_config = cls._find_quant_config(*args, **kwargs)
        if quant_config is not None:

            # attach config to model for general use
            instance.quant_config = quant_config

            # apply model mappings to config for proper config-model matching
            if (hf_to_vllm_mapper := instance.hf_to_vllm_mapper) is not None:
                instance.quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
            if instance.packed_modules_mapping is not None:
                instance.quant_config.packed_modules_mapping.update(
                    instance.packed_modules_mapping)

        return instance

    @staticmethod
    def _find_quant_config(*args, **kwargs) -> Optional[QuantizationConfig]:
        """Find quant config passed through model constructor args"""
        from vllm.config import VllmConfig  # avoid circular import

        args_values = list(args) + list(kwargs.values())
        for arg in args_values:
            if isinstance(arg, VllmConfig):
                return arg.quant_config

            if isinstance(arg, QuantizationConfig):
                return arg

        return None


@runtime_checkable
class SupportsTranscription(Protocol):
    """The interface required for all models that support transcription."""
    # Mapping from ISO639_1 language codes: language names
    supported_languages: ClassVar[Mapping[str, str]]

    supports_transcription: ClassVar[Literal[True]] = True

    supports_transcription_only: ClassVar[bool] = False
    """
    Transcription models can opt out of text generation by setting this to
    `True`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # language codes in supported_languages
        # that don't exist in the full language map
        invalid = set(cls.supported_languages) - set(LANGUAGES.keys())
        if invalid:
            raise ValueError(
                f"{cls.__name__}.supported_languages contains invalid "
                f"language codes: {sorted(invalid)}\n. "
                f"Valid choices are: {sorted(LANGUAGES.keys())}")

    @classmethod
    def get_generation_prompt(cls, audio: np.ndarray,
                              stt_config: SpeechToTextConfig,
                              model_config: ModelConfig,
                              language: Optional[str], task_type: str,
                              request_prompt: str) -> PromptType:
        """Get the prompt for the ASR model.
        The model has control over the construction, as long as it
        returns a valid PromptType."""
        ...

    @classmethod
    def get_other_languages(cls) -> Mapping[str, str]:
        # other possible language codes from the whisper map
        return {
            k: v
            for k, v in LANGUAGES.items() if k not in cls.supported_languages
        }

    @classmethod
    def validate_language(cls, language: Optional[str]) -> Optional[str]:
        """
        Ensure the language specified in the transcription request 
        is a valid ISO 639-1 language code. If the request language is 
        valid, but not natively supported by the model, trigger a 
        warning (but not an exception).
        """
        if language is None or language in cls.supported_languages:
            return language
        elif language in cls.get_other_languages():
            logger.warning(
                "Language %r is not natively supported by %s; "
                "results may be less accurate. Supported languages: %r",
                language,
                cls.__name__,
                list(cls.supported_languages.keys()),
            )
            return language
        else:
            raise ValueError(
                f"Unsupported language: {language!r}.  Must be one of "
                f"{list(cls.supported_languages.keys())}.")

    @classmethod
    def get_speech_to_text_config(
            cls, model_config: ModelConfig,
            task_type: Literal["transcribe",
                               "translate"]) -> SpeechToTextConfig:
        """Get the speech to text config for the ASR model."""
        ...

    @classmethod
    def get_num_audio_tokens(cls, audio_duration_s: float,
                             stt_config: SpeechToTextConfig,
                             model_config: ModelConfig) -> Optional[int]:
        """
        Map from audio duration to number of audio tokens produced by the ASR 
        model, without running a forward pass.
        This is used for estimating the amount of processing for this audio.
        """
        return None


@overload
def supports_transcription(
        model: type[object]) -> TypeIs[type[SupportsTranscription]]:
    ...


@overload
def supports_transcription(model: object) -> TypeIs[SupportsTranscription]:
    ...


def supports_transcription(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsTranscription]], TypeIs[SupportsTranscription]]:
    return getattr(model, "supports_transcription", False)


@runtime_checkable
class SupportsV0Only(Protocol):
    """Models with this interface are not compatible with V1 vLLM."""

    supports_v0_only: ClassVar[Literal[True]] = True


@overload
def supports_v0_only(model: type[object]) -> TypeIs[type[SupportsV0Only]]:
    ...


@overload
def supports_v0_only(model: object) -> TypeIs[SupportsV0Only]:
    ...


def supports_v0_only(
    model: Union[type[object], object],
) -> Union[TypeIs[type[SupportsV0Only]], TypeIs[SupportsV0Only]]:
    return getattr(model, "supports_v0_only", False)
