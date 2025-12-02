# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable, Mapping, MutableSequence, Set
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    overload,
    runtime_checkable,
)

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from typing_extensions import Self, TypeIs

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.inputs import TokensPrompt
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.func_utils import supports_kw

from .interfaces_base import VllmModel, is_pooling_model

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.registry import _ProcessorFactories
    from vllm.sequence import IntermediateTensors
else:
    VllmConfig = object
    WeightsMapper = object
    MultiModalFeatureSpec = object
    _ProcessorFactories = object
    IntermediateTensors = object

logger = init_logger(__name__)

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]
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

    supports_multimodal_raw_input_only: ClassVar[bool] = False
    """
    A flag that indicates this model supports multi-modal inputs and processes
    them in their raw form and not embeddings.
    """

    supports_encoder_tp_data: ClassVar[bool] = False
    """
    A flag that indicates whether this model supports
    `multimodal_config.mm_encoder_tp_mode="data"`.
    """

    merge_by_field_config: ClassVar[bool] = False
    """
    A flag that indicates which implementation of
    `vllm.multimodal.utils.group_mm_kwargs_by_modality` to use.
    """

    multimodal_cpu_fields: ClassVar[Set[str]] = frozenset()
    """
    A set indicating CPU-only multimodal fields.
    """

    _processor_factory: ClassVar[_ProcessorFactories]
    """
    Set internally by `MultiModalRegistry.register_processor`.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """
        Get the placeholder text for the `i`th `modality` item in the prompt.
        """
        ...

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """
        Returns multimodal embeddings generated from multimodal kwargs
        to be merged with text embeddings.

        Note:
            The returned multimodal embeddings must be in the same order as
            the appearances of their corresponding multimodal data item in the
            input prompt.
        """
        if hasattr(self, "get_multimodal_embeddings"):
            logger.warning_once(
                "`get_multimodal_embeddings` for vLLM models is deprecated and will be "
                "removed in v0.13.0 or v1.0.0, whichever is earlier. Please rename "
                "this method to `embed_multimodal`."
            )
            return self.get_multimodal_embeddings(**kwargs)

    def get_language_model(self) -> VllmModel:
        """
        Returns the underlying language model used for text generation.

        This is typically the `torch.nn.Module` instance responsible for
        processing the merged multimodal embeddings and producing hidden states

        Returns:
            torch.nn.Module: The core language model component.
        """
        ...

    @overload
    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...

    @overload
    def embed_input_ids(
        self,
        input_ids: Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        *,
        is_multimodal: torch.Tensor,
        handle_oov_mm_token: bool = False,
    ) -> Tensor: ...

    def _embed_text_input_ids(
        self,
        input_ids: Tensor,
        embed_input_ids: Callable[[Tensor], Tensor],
        *,
        is_multimodal: Tensor | None,
        handle_oov_mm_token: bool,
    ) -> Tensor:
        if handle_oov_mm_token and is_multimodal is not None:
            is_text = ~is_multimodal
            text_embeds = embed_input_ids(input_ids[is_text])

            return torch.empty(
                (input_ids.shape[0], text_embeds.shape[1]),
                dtype=text_embeds.dtype,
                device=text_embeds.device,
            ).masked_scatter_(is_text.unsqueeze_(-1), text_embeds)

        return embed_input_ids(input_ids)

    def embed_input_ids(
        self,
        input_ids: Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> Tensor:
        """
        Apply token embeddings to `input_ids`.

        If `multimodal_embeddings` is passed, scatter them into
        `input_ids` according to the mask `is_multimodal`.

        In case the multi-modal token IDs exceed the vocabulary size of
        the language model, you can set `handle_oov_mm_token=False`
        to avoid calling the language model's `embed_input_ids` method
        on those tokens. Note however that doing so increases memory usage
        as an additional buffer is needed to hold the input embeddings.
        """
        from .utils import _merge_multimodal_embeddings

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if is_multimodal is None:
            raise ValueError(
                "`embed_input_ids` now requires `is_multimodal` arg, "
                "please update your model runner according to "
                "https://github.com/vllm-project/vllm/pull/16229."
            )

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )


@runtime_checkable
class SupportsMultiModalPruning(Protocol):
    """The interface required for models that support returning both input
    embeddings and positions. Model may require custom positions for dynamic
    pruning of multimodal embeddings.
    """

    supports_multimodal_pruning: ClassVar[Literal[True]] = True

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[MultiModalEmbeddings, Tensor, int]:
        """
        Update part of input mrope positions (starting with
        num_computed_tokens index). Original mrope_positions are computed
        for unpruned sequence and becomes incorrect once pruning occurs,
        so once we prune media tokens we should reflect this in the
        mrope_positions before we feed it to LLM.

        Args:
            input_ids: (N,) All input tokens of the prompt containing
                entire sequence.
            multimodal_embeddings: Tuple of multimodal embeddings that
                fits into the prefill chunk that is being processed.
            mrope_positions: Existing mrope positions (3, N) for entire
                sequence
            num_computed_tokens: A number of computed tokens so far.

        Returns:
            Tuple of (multimodal_embeddings, mrope_positions,
                mrope_position_delta).
        """
        ...


@overload
def supports_multimodal(model: type[object]) -> TypeIs[type[SupportsMultiModal]]: ...


@overload
def supports_multimodal(model: object) -> TypeIs[SupportsMultiModal]: ...


def supports_multimodal(
    model: type[object] | object,
) -> TypeIs[type[SupportsMultiModal]] | TypeIs[SupportsMultiModal]:
    return getattr(model, "supports_multimodal", False)


def supports_multimodal_raw_input_only(model: type[object] | object) -> bool:
    return getattr(model, "supports_multimodal_raw_input_only", False)


def supports_multimodal_encoder_tp_data(model: type[object] | object) -> bool:
    return getattr(model, "supports_encoder_tp_data", False)


@overload
def supports_multimodal_pruning(
    model: type[object],
) -> TypeIs[type[SupportsMultiModalPruning]]: ...


@overload
def supports_multimodal_pruning(model: object) -> TypeIs[SupportsMultiModalPruning]: ...


def supports_multimodal_pruning(
    model: type[object] | object,
) -> TypeIs[type[SupportsMultiModalPruning]] | TypeIs[SupportsMultiModalPruning]:
    return getattr(model, "supports_multimodal_pruning", False)


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
    def get_score_template(cls, query: str, document: str) -> str | None:
        """
        Generate a full prompt by populating the score template with query and document content.
        """  # noqa: E501
        ...

    @classmethod
    def post_process_tokens(cls, prompt: TokensPrompt) -> None:
        """
        Perform architecture-specific manipulations on the input tokens.
        """
        ...


@overload
def supports_score_template(
    model: type[object],
) -> TypeIs[type[SupportsScoreTemplate]]: ...


@overload
def supports_score_template(model: object) -> TypeIs[SupportsScoreTemplate]: ...


def supports_score_template(
    model: type[object] | object,
) -> TypeIs[type[SupportsScoreTemplate]] | TypeIs[SupportsScoreTemplate]:
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
    is_3d_moe_weight: ClassVar[bool] = False
    # The `embedding_module` and `embedding_padding_modules`
    # are empty by default.
    embedding_modules: ClassVar[dict[str, str]] = {}
    packed_modules_mapping: dict[str, list[str]] = {}


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: dict[str, list[str]]
    embedding_modules: dict[str, str]


@overload
def supports_lora(model: type[object]) -> TypeIs[type[SupportsLoRA]]: ...


@overload
def supports_lora(model: object) -> TypeIs[SupportsLoRA]: ...


def supports_lora(
    model: type[object] | object,
) -> TypeIs[type[SupportsLoRA]] | TypeIs[SupportsLoRA]:
    result = _supports_lora(model)

    if not result:
        lora_attrs = (
            "packed_modules_mapping",
            "embedding_modules",
        )
        missing_attrs = tuple(attr for attr in lora_attrs if not hasattr(model, attr))

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
                    "but does not set `supports_lora=True`.",
                    model,
                )

    return result


def _supports_lora(model: type[object] | object) -> bool:
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
    ) -> IntermediateTensors:
        """Called when PP rank > 0 for profiling purposes."""
        ...

    def forward(
        self,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
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
    ) -> IntermediateTensors: ...

    def forward(
        self,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> Tensor | IntermediateTensors: ...


@overload
def supports_pp(model: type[object]) -> TypeIs[type[SupportsPP]]: ...


@overload
def supports_pp(model: object) -> TypeIs[SupportsPP]: ...


def supports_pp(
    model: type[object] | object,
) -> bool | TypeIs[type[SupportsPP]] | TypeIs[SupportsPP]:
    supports_attributes = _supports_pp_attributes(model)
    supports_inspect = _supports_pp_inspect(model)

    if supports_attributes and not supports_inspect:
        logger.warning(
            "The model (%s) sets `supports_pp=True`, but does not accept "
            "`intermediate_tensors` in its `forward` method",
            model,
        )

    if not supports_attributes:
        pp_attrs = ("make_empty_intermediate_tensors",)
        missing_attrs = tuple(attr for attr in pp_attrs if not hasattr(model, attr))

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
                    "but does not set `supports_pp=True`.",
                    model,
                )

    return supports_attributes and supports_inspect


def _supports_pp_attributes(model: type[object] | object) -> bool:
    if isinstance(model, type):
        return isinstance(model, _SupportsPPType)

    return isinstance(model, SupportsPP)


def _supports_pp_inspect(model: type[object] | object) -> bool:
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
def has_inner_state(model: object) -> TypeIs[HasInnerState]: ...


@overload
def has_inner_state(model: type[object]) -> TypeIs[type[HasInnerState]]: ...


def has_inner_state(
    model: type[object] | object,
) -> TypeIs[type[HasInnerState]] | TypeIs[HasInnerState]:
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
def is_attention_free(model: object) -> TypeIs[IsAttentionFree]: ...


@overload
def is_attention_free(model: type[object]) -> TypeIs[type[IsAttentionFree]]: ...


def is_attention_free(
    model: type[object] | object,
) -> TypeIs[type[IsAttentionFree]] | TypeIs[IsAttentionFree]:
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
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        ...


@overload
def is_hybrid(model: object) -> TypeIs[IsHybrid]: ...


@overload
def is_hybrid(model: type[object]) -> TypeIs[type[IsHybrid]]: ...


def is_hybrid(
    model: type[object] | object,
) -> TypeIs[type[IsHybrid]] | TypeIs[IsHybrid]:
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

    moe_layers: Iterable[nn.Module]
    """List of MoE layers in this model."""

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
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None: ...


def is_mixture_of_experts(model: object) -> TypeIs[MixtureOfExperts]:
    return (
        isinstance(model, MixtureOfExperts) and getattr(model, "num_moe_layers", 0) > 0
    )


@runtime_checkable
class HasNoOps(Protocol):
    has_noops: ClassVar[Literal[True]] = True


@overload
def has_noops(model: object) -> TypeIs[HasNoOps]: ...


@overload
def has_noops(model: type[object]) -> TypeIs[type[HasNoOps]]: ...


def has_noops(
    model: type[object] | object,
) -> TypeIs[type[HasNoOps]] | TypeIs[HasNoOps]:
    return getattr(model, "has_noops", False)


@runtime_checkable
class SupportsMambaPrefixCaching(Protocol):
    """The interface for models whose mamba layers support prefix caching.

    This is currently experimental.
    """

    supports_mamba_prefix_caching: ClassVar[Literal[True]] = True


@overload
def supports_mamba_prefix_caching(
    model: object,
) -> TypeIs[SupportsMambaPrefixCaching]: ...


@overload
def supports_mamba_prefix_caching(
    model: type[object],
) -> TypeIs[type[SupportsMambaPrefixCaching]]: ...


def supports_mamba_prefix_caching(
    model: type[object] | object,
) -> TypeIs[type[SupportsMambaPrefixCaching]] | TypeIs[SupportsMambaPrefixCaching]:
    return getattr(model, "supports_mamba_prefix_caching", False)


@runtime_checkable
class SupportsCrossEncoding(Protocol):
    """The interface required for all models that support cross encoding."""

    supports_cross_encoding: ClassVar[Literal[True]] = True


@overload
def supports_cross_encoding(
    model: type[object],
) -> TypeIs[type[SupportsCrossEncoding]]: ...


@overload
def supports_cross_encoding(model: object) -> TypeIs[SupportsCrossEncoding]: ...


def _supports_cross_encoding(
    model: type[object] | object,
) -> TypeIs[type[SupportsCrossEncoding]] | TypeIs[SupportsCrossEncoding]:
    return getattr(model, "supports_cross_encoding", False)


def supports_cross_encoding(
    model: type[object] | object,
) -> TypeIs[type[SupportsCrossEncoding]] | TypeIs[SupportsCrossEncoding]:
    return is_pooling_model(model) and _supports_cross_encoding(model)


class SupportsQuant:
    """The interface required for all models that support quantization."""

    hf_to_vllm_mapper: ClassVar[WeightsMapper | None] = None
    packed_modules_mapping: ClassVar[dict[str, list[str]] | None] = None
    quant_config: QuantizationConfig | None = None

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
                    instance.packed_modules_mapping
                )

        return instance

    @staticmethod
    def _find_quant_config(*args, **kwargs) -> QuantizationConfig | None:
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
    supports_segment_timestamp: ClassVar[bool] = False
    """
    Enables the segment timestamp option for supported models by setting this to `True`.
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
                f"Valid choices are: {sorted(LANGUAGES.keys())}"
            )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Get the prompt for the ASR model.
        The model has control over the construction, as long as it
        returns a valid PromptType."""
        ...

    @classmethod
    def get_other_languages(cls) -> Mapping[str, str]:
        # other possible language codes from the whisper map
        return {k: v for k, v in LANGUAGES.items() if k not in cls.supported_languages}

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
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
                f"{list(cls.supported_languages.keys())}."
            )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
        """Get the speech to text config for the ASR model."""
        ...

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        """
        Map from audio duration to number of audio tokens produced by the ASR
        model, without running a forward pass.
        This is used for estimating the amount of processing for this audio.
        """
        return None


@overload
def supports_transcription(
    model: type[object],
) -> TypeIs[type[SupportsTranscription]]: ...


@overload
def supports_transcription(model: object) -> TypeIs[SupportsTranscription]: ...


def supports_transcription(
    model: type[object] | object,
) -> TypeIs[type[SupportsTranscription]] | TypeIs[SupportsTranscription]:
    return getattr(model, "supports_transcription", False)


@runtime_checkable
class SupportsEagleBase(Protocol):
    """Base interface for models that support EAGLE-based speculative decoding."""

    has_own_lm_head: bool = False
    """
    A flag that indicates this model has trained its own lm_head.
    """

    has_own_embed_tokens: bool = False
    """
    A flag that indicates this model has trained its own input embeddings.
    """


@overload
def supports_any_eagle(model: type[object]) -> TypeIs[type[SupportsEagleBase]]: ...


@overload
def supports_any_eagle(model: object) -> TypeIs[SupportsEagleBase]: ...


def supports_any_eagle(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagleBase]] | TypeIs[SupportsEagleBase]:
    """Check if model supports any EAGLE variant (1, 2, or 3)."""
    return supports_eagle(model) or supports_eagle3(model)


@runtime_checkable
class SupportsEagle(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-1 and EAGLE-2 speculative decoding."""

    supports_eagle: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-1 and EAGLE-2 
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """


@overload
def supports_eagle(model: type[object]) -> TypeIs[type[SupportsEagle]]: ...


@overload
def supports_eagle(model: object) -> TypeIs[SupportsEagle]: ...


def supports_eagle(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagle]] | TypeIs[SupportsEagle]:
    return isinstance(model, SupportsEagle)


@runtime_checkable
class SupportsEagle3(SupportsEagleBase, Protocol):
    """The interface required for models that support
    EAGLE-3 speculative decoding."""

    supports_eagle3: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports EAGLE-3 
    speculative decoding.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """
        Set which layers should output auxiliary
        hidden states for EAGLE-3.

        Args:
            layers: Tuple of layer indices that should output auxiliary
                hidden states.
        """
        ...

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """
        Get the layer indices that should output auxiliary hidden states
        for EAGLE-3.

        Returns:
            Tuple of layer indices for auxiliary hidden state outputs.
        """
        ...


@overload
def supports_eagle3(model: type[object]) -> TypeIs[type[SupportsEagle3]]: ...


@overload
def supports_eagle3(model: object) -> TypeIs[SupportsEagle3]: ...


def supports_eagle3(
    model: type[object] | object,
) -> TypeIs[type[SupportsEagle3]] | TypeIs[SupportsEagle3]:
    return isinstance(model, SupportsEagle3)


@runtime_checkable
class SupportsMRoPE(Protocol):
    """The interface required for all models that support M-RoPE."""

    supports_mrope: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports M-RoPE.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> tuple[torch.Tensor, int]:
        """
        Get M-RoPE input positions and delta value for this specific model.

        This method should be implemented by each model that supports M-RoPE
        to provide model-specific logic for computing input positions.

        Args:
            input_tokens: List of input token IDs
            mm_features: Information about each multi-modal data item

        Returns:
            Tuple of `(llm_positions, mrope_position_delta)`
            - llm_positions: Tensor of shape `[3, num_tokens]` with T/H/W positions
            - mrope_position_delta: Delta for position calculations
        """
        ...


@overload
def supports_mrope(model: type[object]) -> TypeIs[type[SupportsMRoPE]]: ...


@overload
def supports_mrope(model: object) -> TypeIs[SupportsMRoPE]: ...


def supports_mrope(
    model: type[object] | object,
) -> TypeIs[type[SupportsMRoPE]] | TypeIs[SupportsMRoPE]:
    return isinstance(model, SupportsMRoPE)


@runtime_checkable
class SupportsXDRoPE(Protocol):
    """The interface required for all models that support XD-RoPE."""

    supports_xdrope: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports XD-RoPE.

    Note:
        There is no need to redefine this flag if this class is in the
        XDRope of your model class.
    """

    def get_xdrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> torch.Tensor:
        """
        Get XD-RoPE input positions and delta value for this specific model.

        This method should be implemented by each model that supports XD-RoPE
        to provide model-specific logic for computing input positions.

        Args:
            input_tokens: List of input token IDs
            mm_features: Information about each multi-modal data item

        Returns:
            llm_positions: Tensor of shape `[xdrope_dim, num_tokens]` with
            4D(P/W/H/T) or 3D(W/H/T) positions.
        """
        ...


@overload
def supports_xdrope(model: type[object]) -> TypeIs[type[SupportsXDRoPE]]: ...


@overload
def supports_xdrope(model: object) -> TypeIs[SupportsXDRoPE]: ...


def supports_xdrope(
    model: type[object] | object,
) -> TypeIs[type[SupportsXDRoPE]] | TypeIs[SupportsXDRoPE]:
    return isinstance(model, SupportsXDRoPE)
