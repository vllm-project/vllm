# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, get_type_hints, overload

import torch
from typing_extensions import TypeVar

from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    EmbeddingItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.tokenizers import TokenizerLike
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.func_utils import get_allowed_kwarg_only_overrides
from vllm.utils.jsontree import JSONTree, json_map_leaves
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    from vllm.config import ModelConfig
    from vllm.renderers import TokenizeParams
else:
    PreTrainedConfig = object
    BatchFeature = object
    ProcessorMixin = object

    ModelConfig = object
    TokenizeParams = object

logger = init_logger(__name__)

# HuggingFace processors accept a shared flat namespace in addition to the
# nested processor kwarg scopes ``text_kwargs`` / ``images_kwargs`` /
# ``videos_kwargs`` / ``audio_kwargs``. vLLM recognizes these scopes when
# resolving ``mm_processor_kwargs``.
_HF_PROCESSOR_KWARG_SCOPES = (
    "text_kwargs",
    "images_kwargs",
    "videos_kwargs",
    "audio_kwargs",
)


def _merge_scoped_mm_processor_value(
    flat_value: object,
    scoped_value: object,
) -> object:
    """Merge a flat value with an existing, more specific scoped value.

    Mappings are merged recursively; otherwise the scoped value replaces the
    flat value.
    """
    if isinstance(flat_value, Mapping) and isinstance(scoped_value, Mapping):
        # Merge recursively so scoped leaves override matching flat leaves
        # without dropping unrelated flat entries.
        merged = dict(flat_value)
        for key, value in scoped_value.items():
            if key in merged:
                merged[key] = _merge_scoped_mm_processor_value(merged[key], value)
            else:
                merged[key] = value
        return merged
    return scoped_value


def _resolve_mm_processor_kwargs(
    kwargs: Mapping[str, object],
    supported_mm_processor_kwargs: Mapping[str, set[str]] | None = None,
) -> dict[str, object]:
    """Resolve flat processor kwargs into HuggingFace processor kwarg scopes.

    With ``supported_mm_processor_kwargs``, each scope lists the flat keys it
    supports. Matching flat keys are routed to every scope that supports them,
    existing scoped values take precedence, and keys not supported by any scope
    remain flat.

    Without ``supported_mm_processor_kwargs``, flat keys already represented in
    at least one existing mapping-valued scope are removed from the shared flat
    namespace; all other flat keys are left unchanged because there is no
    information to determine which scopes should receive them.
    """
    resolved = dict(kwargs)

    # Without ``supported_mm_processor_kwargs``, remove flat entries already
    # represented in at least one existing mapping-valued scope. There is no
    # information to route the remaining flat entries.
    if supported_mm_processor_kwargs is None:
        # Collect all keys represented in existing mapping-valued scopes.
        # Non-mapping scope values do not contain scoped kwargs and are left
        # untouched.
        represented_scoped_keys: set[str] = set()
        for scoped_key in _HF_PROCESSOR_KWARG_SCOPES:
            scoped_kwargs = resolved.get(scoped_key)
            if isinstance(scoped_kwargs, Mapping):
                represented_scoped_keys.update(scoped_kwargs)

        # Remove flat entries for keys already represented in a scope.
        for key in represented_scoped_keys:
            if key not in _HF_PROCESSOR_KWARG_SCOPES:
                resolved.pop(key, None)
        return resolved

    # With ``supported_mm_processor_kwargs``, collect the flat kwargs before
    # routing each one to the processor kwarg scopes that support it.
    flat_kwargs = {
        key: value
        for key, value in resolved.items()
        if key not in _HF_PROCESSOR_KWARG_SCOPES
    }

    for key, flat_value in flat_kwargs.items():
        # Find every processor kwarg scope that supports this flat key. If none
        # supports it, keep the key in the shared flat namespace.
        supported_scopes = [
            scoped_key
            for scoped_key in _HF_PROCESSOR_KWARG_SCOPES
            if key in supported_mm_processor_kwargs.get(scoped_key, set())
        ]
        if not supported_scopes:
            continue

        # Add the flat value to each supported destination scope.
        for scoped_key in supported_scopes:
            # Start from a copy of the existing scoped kwargs, or an empty scope
            # if absent.
            if scoped_key in resolved:
                scoped_kwargs = resolved[scoped_key]
                # A scope must be a mapping before keys can be added to it. Do
                # not replace an explicit non-mapping value.
                if not isinstance(scoped_kwargs, Mapping):
                    raise TypeError(f"`{scoped_key}` must be a mapping")
                scoped_kwargs = dict(scoped_kwargs)
            else:
                scoped_kwargs = {}

            # Add the flat value to this scope. Existing scoped values take
            # precedence.
            if key in scoped_kwargs:
                scoped_kwargs[key] = _merge_scoped_mm_processor_value(
                    flat_value, scoped_kwargs[key]
                )
            else:
                # Assign the flat value to this scope. For mappings, assign a
                # copy so the same mapping object is not shared between the flat
                # value and multiple scopes.
                scoped_kwargs[key] = (
                    dict(flat_value) if isinstance(flat_value, Mapping) else flat_value
                )
            resolved[scoped_key] = scoped_kwargs

        # After routing to every destination, remove the flat copy.
        resolved.pop(key)

    return resolved


@dataclass
class TimingContext:
    """Helper class to record execution times during multi-modal processing."""

    enabled: bool = True
    """If disabled, `TimingContext.record` becomes a no-op."""

    stage_secs: dict[str, float] = field(default_factory=dict)
    """The execution time (in seconds) for each processing stage."""

    @property
    def total_secs(self) -> float:
        return sum(self.stage_secs.values())

    @contextmanager
    def record(self, stage: str):
        """Record the execution time for a processing stage."""
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.stage_secs.setdefault(stage, 0.0)
            self.stage_secs[stage] += elapsed

    def get_stats_dict(self):
        stats_dict = {
            f"{stage}_secs": time_s for stage, time_s in self.stage_secs.items()
        }
        stats_dict["preprocessor_total_secs"] = self.total_secs

        return stats_dict


_T = TypeVar("_T")
_C = TypeVar("_C", bound=PreTrainedConfig, default=PreTrainedConfig)
_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)


@dataclass(frozen=True)
class InputProcessingContext:
    """Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: ModelConfig
    """The configuration of the model."""

    tokenizer: TokenizerLike | None
    """The tokenizer used to tokenize the inputs."""

    def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "You cannot pass text prompts when `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    @overload
    def get_hf_config(self, /) -> PreTrainedConfig: ...

    @overload
    def get_hf_config(
        self,
        typ: type[_C] | tuple[type[_C], ...],
        /,
    ) -> _C: ...

    def get_hf_config(
        self,
        typ: type[Any] | tuple[type[Any], ...] | None = None,
        /,
    ) -> Any:
        """Get the HuggingFace configuration
        (`transformers.PreTrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the configuration is not of the specified type.

        """
        if typ is None:
            from transformers.configuration_utils import PreTrainedConfig

            typ = PreTrainedConfig

        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, typ):
            raise TypeError(
                "Invalid type of HuggingFace config. "
                f"Expected type: {typ}, but "
                f"found type: {type(hf_config)}"
            )

        return hf_config

    def get_hf_image_processor_config(self) -> dict[str, Any]:
        """Get the HuggingFace image processor configuration of the model."""
        return self.model_config.hf_image_processor_config

    def get_mm_config(self):
        """Get the multimodal config of the model.

        Raises:
            RuntimeError: If the model is not a multimodal model.

        """
        mm_config = self.model_config.multimodal_config
        if mm_config is None:
            raise RuntimeError("Not a multimodal model")

        return mm_config

    @overload
    def get_hf_processor(self, /, **kwargs: object) -> ProcessorMixin: ...

    @overload
    def get_hf_processor(
        self,
        typ: type[_P] | tuple[type[_P], ...],
        /,
        **kwargs: object,
    ) -> _P: ...

    def get_hf_processor(
        self,
        typ: type[Any] | tuple[type[Any], ...] | None = None,
        /,
        **kwargs: object,
    ) -> Any:
        """Get the HuggingFace processor
        (`transformers.ProcessorMixin`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the processor is not of the specified type.

        """
        if typ is None:
            from transformers.processing_utils import ProcessorMixin

            typ = ProcessorMixin

        tokenizer = self.tokenizer
        if is_mistral_tokenizer(tokenizer):
            tokenizer = tokenizer.transformers_tokenizer  # type: ignore[union-attr]

        merged_kwargs = self.get_merged_mm_kwargs(kwargs)
        merged_kwargs.pop("tokenizer", None)

        return cached_processor_from_config(
            self.model_config,
            processor_cls=typ,
            tokenizer=tokenizer,
            **merged_kwargs,
        )

    def init_processor(
        self,
        typ: type[_T],
        /,
        **kwargs: object,
    ) -> _T:
        """Initialize a HuggingFace-like processor class, merging the
        keyword arguments with those in the model's configuration.
        """
        merged_kwargs = self.get_merged_mm_kwargs(kwargs)

        return typ(**merged_kwargs)

    def _postprocess_output(
        self,
        output: JSONTree,
    ) -> JSONTree:
        # "torch_shm" puts tensors on a torch.multiprocessing queue, which
        # shares device tensors by CUDA IPC handle, so a device-side processor
        # can hand `pixel_values` straight to the worker. Every other transport
        # serializes host bytes, so the result has to be copied back first.
        keep_on_device = (
            self.model_config.get_multimodal_config().mm_tensor_ipc == "torch_shm"
        )

        def _postprocess_one(x: object):
            if not isinstance(x, torch.Tensor):
                return x

            # Bind to a Tensor-typed local: reassigning the `object`-typed
            # parameter would discard the isinstance narrowing.
            tensor = x

            # This mimics the behavior of transformers.BatchFeature
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=self.model_config.dtype)

            if not tensor.is_cpu and not keep_on_device:
                tensor = tensor.cpu()

            return tensor

        return json_map_leaves(_postprocess_one, output)

    def get_merged_mm_kwargs(
        self,
        kwargs: Mapping[str, object],
        *,
        supported_mm_processor_kwargs: Mapping[str, set[str]] | None = None,
    ) -> dict[str, Any]:
        """Merge configured and request ``mm_processor_kwargs``.

        When ``supported_mm_processor_kwargs`` is provided, flat keys are matched
        against the keys supported by each HuggingFace processor kwarg scope.
        Matching flat keys are routed to every scope that supports them and
        removed from the shared flat namespace; keys not supported by any scope
        remain flat.

        Without ``supported_mm_processor_kwargs``, flat keys already represented
        in at least one existing mapping-valued scope are removed from the shared
        flat namespace; all other flat keys are left unchanged because there is
        no information to determine which scopes should receive them.
        """
        mm_config = self.model_config.get_multimodal_config()
        merged = mm_config.merge_mm_processor_kwargs(kwargs)
        return _resolve_mm_processor_kwargs(
            merged,
            supported_mm_processor_kwargs=supported_mm_processor_kwargs,
        )

    def call_hf_processor(
        self,
        hf_processor: Callable[..., BatchFeature] | ProcessorMixin,
        data: Mapping[str, object],
        kwargs: Mapping[str, object] = {},
    ) -> BatchFeature:
        """Call `hf_processor` on the prompt `data`
        (text, image, audio...) with configurable options `kwargs`.
        """
        assert callable(hf_processor)

        merged_kwargs = self.get_merged_mm_kwargs(kwargs)

        allowed_kwargs = get_allowed_kwarg_only_overrides(
            hf_processor,
            merged_kwargs,
            requires_kw_only=False,
            allow_var_kwargs=True,
        )
        allowed_kwargs.setdefault("return_tensors", "pt")

        try:
            output = hf_processor(**data, **allowed_kwargs)
        except Exception as exc:
            msg = (
                f"Failed to apply {type(hf_processor).__name__} "
                f"on data={data} with kwargs={allowed_kwargs}"
            )

            raise ValueError(msg) from exc

        # this emulates output.to(dtype=self.model_config.dtype)
        from transformers.feature_extraction_utils import BatchFeature

        if isinstance(output, BatchFeature):
            output_ = self._postprocess_output(output.data)
            return BatchFeature(output_)  # type: ignore

        logger.warning_once(
            "%s did not return `BatchFeature`. "
            "Make sure to match the behaviour of `ProcessorMixin` when "
            "implementing custom processors.",
            type(hf_processor).__name__,
        )

        return self._postprocess_output(output)  # type: ignore


class BaseProcessingInfo:
    """Base class to provide the information necessary for data processing."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__()

        self.ctx = ctx

    @property
    def model_id(self) -> str:
        return self.ctx.model_config.model

    def get_tokenizer(self) -> TokenizerLike:
        return self.ctx.get_tokenizer()

    def get_hf_config(self) -> PreTrainedConfig:
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        """Subclasses can override this method to handle
        specific kwargs from model config or user inputs.
        """
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        """Return supported kwarg names for each HF processor kwargs scope."""
        processor = self.get_hf_processor()
        processor_kwargs = get_type_hints(processor.valid_processor_kwargs)

        supported = {
            "text_kwargs": set(processor_kwargs["text_kwargs"].__annotations__),
        }
        if "image" in self.supported_mm_limits:
            supported["images_kwargs"] = set(
                processor.image_processor.valid_kwargs.__annotations__
            )
        if "video" in self.supported_mm_limits:
            supported["videos_kwargs"] = set(
                processor.video_processor.valid_kwargs.__annotations__
            )
        if "audio" in self.supported_mm_limits:
            supported["audio_kwargs"] = set(
                processor_kwargs["audio_kwargs"].__annotations__
            )

        return supported

    @cached_property
    def supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        """Supported kwarg names for each HF processor kwargs scope."""
        return self.get_supported_mm_processor_kwargs()

    def _merge_and_resolve_mm_processor_kwargs(
        self,
        mm_kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        """Merge configured and request ``mm_processor_kwargs``.

        Flat kwargs are routed into the HuggingFace processor kwarg scopes that
        support them after the configured/request merge. When a routed flat value
        conflicts with an existing scoped value, the scoped value takes precedence.
        """
        return self.ctx.get_merged_mm_kwargs(
            mm_kwargs,
            supported_mm_processor_kwargs=self.supported_mm_processor_kwargs,
        )

    def get_default_tok_params(self) -> TokenizeParams:
        """Construct the default parameters for tokenization."""
        from vllm.renderers import TokenizeParams

        model_config = self.ctx.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_tok_params(self) -> TokenizeParams:
        return self.get_default_tok_params()

    def _get_expected_hidden_size(self) -> int | None:
        """Get expected hidden size for embedding validation if `mm_embeds` are enabled.

        This validates hidden dimensions to prevent a vulnerability where embeddings
        with correct `ndim` but wrong `shape` could cause crashes at inference time.
        """
        model_config = self.ctx.model_config
        mm_config = model_config.get_multimodal_config()

        if mm_config.enable_mm_embeds:
            return model_config.get_inputs_embeds_size()

        return None

    @property
    def allow_missing_mm_embeddings(self) -> bool:
        """Whether pre-computed embedding tensors may be omitted."""
        mm_config = self.ctx.model_config.multimodal_config
        return mm_config is not None and mm_config.allow_missing_mm_embeddings

    def get_data_parser(self) -> MultiModalDataParser:
        """Constructs a parser to preprocess multi-modal data items
        before passing them to
        [`_get_hf_mm_inputs`][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_inputs].

        You can support additional modalities by creating a subclass
        of [`MultiModalDataParser`][vllm.multimodal.parse.MultiModalDataParser]
        that has additional subparsers.
        """
        return MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    @cached_property
    def data_parser(self) -> MultiModalDataParser:
        return self.get_data_parser()

    @property
    def skip_prompt_length_check(self) -> bool:
        return False

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """Return the maximum supported number of items for each modality.

        A value of `None` means unlimited number of items.

        Omitting a modality from the returned dictionary means that
        it is not supported at all.
        """
        raise NotImplementedError

    @cached_property
    def supported_mm_limits(self) -> Mapping[str, int | None]:
        """The maximum supported number of items for each modality."""
        return self.get_supported_mm_limits()

    @cached_property
    def allowed_mm_limits(self) -> Mapping[str, int]:
        """The maximum allowed number of items for each modality."""
        mm_config = self.ctx.get_mm_config()

        allowed_limits = dict[str, int]()
        for modality, supported_limit in self.supported_mm_limits.items():
            user_limit = mm_config.get_limit_per_prompt(modality)

            allowed_limits[modality] = (
                user_limit
                if supported_limit is None
                else min(user_limit, supported_limit)
            )

        return allowed_limits

    def validate_num_items(self, modality: str, num_items: int) -> None:
        """Raise `ValueError` if the number of input items for the given modality
        is invalid.
        """
        supported_limit = self.supported_mm_limits.get(modality, 0)
        allowed_limit = self.allowed_mm_limits.get(modality, 0)

        if supported_limit is None:
            supported_limit = allowed_limit

        limit = min(supported_limit, allowed_limit)

        if num_items > limit:
            msg = f"At most {limit} {modality}(s) may be provided in one prompt."

            if num_items <= supported_limit:
                msg += " Set `--limit-mm-per-prompt` to increase this limit."

            raise VLLMValidationError(msg, parameter=modality)

    def parse_mm_data(
        self,
        mm_data: MultiModalDataDict,
        *,
        validate: bool = True,
    ) -> MultiModalDataItems:
        """Normalize [`MultiModalDataDict`][vllm.inputs.MultiModalDataDict]
        to [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems]
        before passing them to
        [`_get_hf_mm_inputs`][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_inputs].
        """
        mm_items = self.data_parser.parse_mm_data(mm_data)

        if validate:
            mm_config = self.ctx.get_mm_config()

            for modality, items in mm_items.items():
                if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
                    if not mm_config.enable_mm_embeds:
                        raise ValueError(
                            f"You must set `--enable-mm-embeds` to input "
                            f"`{modality}_embeds`"
                        )
                    if mm_config.get_limit_per_prompt(modality) == 0:
                        logger.debug(
                            "Skipping count validation for modality "
                            "'%s' (embeddings with limit=0)",
                            modality,
                        )
                        continue
                self.validate_num_items(modality, len(items))

        return mm_items

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Return the maximum number of tokens per item of for each modality.

        When `None` (the default) is returned, vLLM will generate dummy inputs
        (images/videos) at maximum possible sizes and process them to determine
        the maximum token count per modality.

        This approach works but can be very slow for certain models (e.g.,
        Qwen2.5-VL), leading to very long startup time. For better performance,
        each model can override this method to return pre-computed maximum token
        counts, avoiding the need for dummy input generation and processing.

        Note:
            The maximum number of tokens per item of each modality returned
            from this function should respect the model's maximum sequence
            length and the maximum number of items of each modality allowed,
            and agree with dummy inputs (images/videos) at maximum possible
            sizes.

        """
        return None
