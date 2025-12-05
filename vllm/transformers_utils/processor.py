# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import inspect
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast, get_args, get_type_hints

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoVideoProcessor,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.video_processing_utils import BaseVideoProcessor
from typing_extensions import TypeVar

from vllm.transformers_utils.gguf_utils import is_gguf
from vllm.transformers_utils.utils import convert_model_repo_to_path
from vllm.utils.func_utils import get_allowed_kwarg_only_overrides

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)
_V = TypeVar("_V", bound=BaseVideoProcessor, default=BaseVideoProcessor)


class HashableDict(dict):
    """
    A dictionary that can be hashed by lru_cache.
    """

    # NOTE: pythonic dict is not hashable,
    # we override on it directly for simplicity
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))


class HashableList(list):
    """
    A list that can be hashed by lru_cache.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))


def _get_processor_factory_fn(processor_cls: type | tuple[type, ...]):
    if isinstance(processor_cls, tuple) or processor_cls == ProcessorMixin:
        return AutoProcessor.from_pretrained
    if hasattr(processor_cls, "from_pretrained"):
        return processor_cls.from_pretrained

    return processor_cls


@lru_cache
def _collect_dynamic_keys_from_processing_kwargs(kwargs_cls: type) -> set[str]:
    dynamic_kwargs: set[str] = set()
    if kwargs_cls is None:
        return dynamic_kwargs
    # get kwargs annotations in processor
    # merge text_kwargs / images_kwargs / videos_kwargs / audio_kwargs
    kwargs_type_annotations = get_type_hints(kwargs_cls)
    for kw_type in ("text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"):
        if kw_type in kwargs_type_annotations:
            kw_annotations = get_type_hints(kwargs_type_annotations[kw_type])
            for kw_name in kw_annotations:
                dynamic_kwargs.add(kw_name)
    dynamic_kwargs |= {"text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"}
    return dynamic_kwargs


def _merge_mm_kwargs(
    model_config: "ModelConfig",
    processor_cls: type | tuple[type, ...],
    /,
    **kwargs,
):
    mm_config = model_config.get_multimodal_config()
    merged_kwargs = mm_config.merge_mm_processor_kwargs(kwargs)

    factory = _get_processor_factory_fn(processor_cls)
    allowed_kwargs = get_allowed_kwarg_only_overrides(
        factory,
        merged_kwargs,
        requires_kw_only=False,
        allow_var_kwargs=True,
    )
    # NOTE: Pythonic dict is not hashable and will raise unhashable type
    # error when calling `cached_get_processor`, therefore we need to
    # wrap it to a hashable dict.
    for key, value in allowed_kwargs.items():
        if isinstance(value, dict):
            allowed_kwargs[key] = HashableDict(value)
        if isinstance(value, list):
            allowed_kwargs[key] = HashableList(value)

    return allowed_kwargs


def get_processor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    processor_cls: type[_P] | tuple[type[_P], ...] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    """Load a processor for the given model name via HuggingFace."""
    if revision is None:
        revision = "main"
    try:
        processor_name = convert_model_repo_to_path(processor_name)
        if isinstance(processor_cls, tuple) or processor_cls == ProcessorMixin:
            processor = AutoProcessor.from_pretrained(
                processor_name,
                *args,
                revision=revision,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        elif issubclass(processor_cls, ProcessorMixin):
            processor = processor_cls.from_pretrained(
                processor_name,
                *args,
                revision=revision,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            # Processors that are standalone classes unrelated to HF
            processor = processor_cls(*args, **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(processor, processor_cls):
        raise TypeError(
            "Invalid type of HuggingFace processor. "
            f"Expected type: {processor_cls}, but "
            f"found type: {type(processor)}"
        )

    return processor


cached_get_processor = lru_cache(get_processor)


@lru_cache
def get_processor_kwargs_from_processor(processor: _P) -> set[str]:
    try:
        # get kwargs annotations in processor
        call_kwargs = inspect.signature(type(processor).__call__).parameters.get(
            "kwargs"
        )
        call_kwargs_annotations = call_kwargs.annotation if call_kwargs else None
        # if the processor has explicit kwargs annotation, use it
        if call_kwargs_annotations not in (None, inspect._empty):
            # get_type_hints will parse all type annotations at runtime,
            # and if an annotation refers to a type or
            # name that hasn’t been imported or defined, it will raise an error.
            # So we use __annotations__ to get the raw annotations directly.
            return _collect_dynamic_keys_from_processing_kwargs(
                get_args(call_kwargs_annotations)[0]
            )
        # otherwise, try to get from ProcessingKwargs
        else:
            module_name = type(processor).__module__
            mod = importlib.import_module(module_name)
            # find *ProcessingKwargs in the module
            processor_kwargs: set[str] = set()
            for name, obj in vars(mod).items():
                if name.endswith("ProcessingKwargs"):
                    processor_kwargs = (
                        processor_kwargs
                        | _collect_dynamic_keys_from_processing_kwargs(obj)
                    )
            return processor_kwargs
    except Exception:
        return set()


def cached_get_processor_without_dynamic_kwargs(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    processor_cls: type[_P] | tuple[type[_P], ...] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    # Step 1: use default kwargs to get a temporary processor instance
    processor = cached_get_processor(
        processor_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        processor_cls=processor_cls,  # type: ignore[arg-type]
    )

    # Step 2: use temporary processor collect dynamic keys
    dynamic_keys = get_processor_kwargs_from_processor(processor)

    # Step 3: use dynamic_keys filter kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in dynamic_keys}

    # Step 4: use filtered kwargs to get final processor instance
    final_processor = cached_get_processor(
        processor_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        processor_cls=processor_cls,  # type: ignore[arg-type]
        **filtered_kwargs,
    )

    return final_processor


def cached_processor_from_config(
    model_config: "ModelConfig",
    processor_cls: type[_P] | tuple[type[_P], ...] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    if is_gguf(model_config.model):
        assert not is_gguf(model_config.tokenizer), (
            "For multimodal GGUF models, the original tokenizer "
            "should be used to correctly load processor."
        )
        model = model_config.tokenizer
        revision = model_config.tokenizer_revision
    else:
        model = model_config.model
        revision = model_config.revision

    return cached_get_processor_without_dynamic_kwargs(
        model,
        revision=revision,
        trust_remote_code=model_config.trust_remote_code,
        processor_cls=processor_cls,  # type: ignore[arg-type]
        **_merge_mm_kwargs(model_config, processor_cls, **kwargs),
    )


def get_feature_extractor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an audio feature extractor for the given model name
    via HuggingFace."""
    try:
        processor_name = convert_model_repo_to_path(processor_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            processor_name,
            *args,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the feature extractor. If the feature "
                "extractor is a custom extractor not yet available in the "
                "HuggingFace transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e
    return cast(FeatureExtractionMixin, feature_extractor)


cached_get_feature_extractor = lru_cache(get_feature_extractor)


def cached_feature_extractor_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    return cached_get_feature_extractor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        **_merge_mm_kwargs(model_config, AutoFeatureExtractor, **kwargs),
    )


def get_image_processor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an image processor for the given model name via HuggingFace."""
    try:
        processor_name = convert_model_repo_to_path(processor_name)
        processor = AutoImageProcessor.from_pretrained(
            processor_name,
            *args,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the image processor. If the image processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(BaseImageProcessor, processor)


cached_get_image_processor = lru_cache(get_image_processor)


def cached_image_processor_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    if is_gguf(model_config.model):
        assert not is_gguf(model_config.tokenizer), (
            "For multimodal GGUF models, the original tokenizer "
            "should be used to correctly load image processor."
        )
        model = model_config.tokenizer
        revision = model_config.tokenizer_revision
    else:
        model = model_config.model
        revision = model_config.revision
    return cached_get_image_processor(
        model,
        revision=revision,
        trust_remote_code=model_config.trust_remote_code,
        **_merge_mm_kwargs(model_config, AutoImageProcessor, **kwargs),
    )


def get_video_processor(
    processor_name: str,
    *args: Any,
    revision: str | None = None,
    trust_remote_code: bool = False,
    processor_cls_overrides: type[_V] | None = None,
    **kwargs: Any,
):
    """Load a video processor for the given model name via HuggingFace."""
    try:
        processor_name = convert_model_repo_to_path(processor_name)
        processor_cls = processor_cls_overrides or AutoVideoProcessor
        processor = processor_cls.from_pretrained(
            processor_name,
            *args,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoVideoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the video processor. If the video processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(BaseVideoProcessor, processor)


cached_get_video_processor = lru_cache(get_video_processor)


def cached_video_processor_from_config(
    model_config: "ModelConfig",
    processor_cls: type[_V] | None = None,
    **kwargs: Any,
):
    return cached_get_video_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        processor_cls_overrides=processor_cls,  # type: ignore[arg-type]
        **_merge_mm_kwargs(model_config, AutoVideoProcessor, **kwargs),
    )
