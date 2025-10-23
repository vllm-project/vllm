# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import lru_cache
import importlib
import inspect
from typing import TYPE_CHECKING, Any, cast, get_args

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

from vllm.utils.functools import get_allowed_kwarg_only_overrides

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
def _collect_dynamic_keys_from_processing_kwargs(kwargs_cls) -> set[str]:
    dynamic_kwargs: set[str] = set()
    if kwargs_cls is None:
        return dynamic_kwargs
    # get kwargs annotations in processor
    # merge text_kwargs / images_kwargs / videos_kwargs / audio_kwargs
    kwargs_type_annotations = getattr(kwargs_cls, "__annotations__", {})
    for kw_type in ("text_kwargs", "images_kwargs", "videos_kwargs", "audio_kwargs"):
        if kw_type in kwargs_type_annotations:
            kw_annotations = kwargs_type_annotations[kw_type].__annotations__
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
    # try to get specific processor dynamic kwargs cached in _PROC_CLASS_CACHE
    cached_dynamic_kwargs = _get_cached_dynamic_keys(model_config, processor_cls)
    # Get basic processor dynamic kwargs if not cached
    if cached_dynamic_kwargs is None:
        try:
            from transformers.processing_utils import ProcessingKwargs

            kwargs_cls = ProcessingKwargs
        except Exception:
            kwargs_cls = None
        mm_config.mm_processor_dynamic_kwargs = (
            _collect_dynamic_keys_from_processing_kwargs(kwargs_cls)
        )
    # Dynamic-wins: filter out dynamic (call-time) keys from constructor kwargs
    if mm_config.mm_processor_dynamic_kwargs:
        allowed_kwargs = {
            k: v
            for k, v in allowed_kwargs.items()
            if k not in mm_config.mm_processor_dynamic_kwargs
        }
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


def cached_processor_from_config(
    model_config: "ModelConfig",
    processor_cls: type[_P] | tuple[type[_P], ...] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    processor = cached_get_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        processor_cls=processor_cls,  # type: ignore[arg-type]
        **_merge_mm_kwargs(model_config, processor_cls, **kwargs),
    )
    # After processor instantiation, refine cached dynamic keys
    try:
        info = _update_cache_from_instance(model_config, processor_cls, processor)
        mm_config = model_config.get_multimodal_config()

        mm_config.mm_processor_dynamic_kwargs = (
            mm_config.mm_processor_dynamic_kwargs | info.dynamic_keys
            if mm_config.mm_processor_dynamic_kwargs else info.dynamic_keys
        )

    except Exception:
        # Best-effort; should never break processor loading.
        pass
    return processor


@dataclass(frozen=True)
class _ResolvedProcessorInfo:
    processor_type: type
    dynamic_keys: set[str]


# Cache key is (model_id, revision, trust_remote_code, processor_cls_identity)
_PROC_CLASS_CACHE: dict[tuple, _ResolvedProcessorInfo] = {}


def _proc_resolution_key(model_config: "ModelConfig", processor_cls: type[_P] | tuple[type[_P], ...] = ProcessorMixin) -> tuple:
    return (
        model_config.model,
        model_config.revision,
        bool(model_config.trust_remote_code),
        getattr(processor_cls, "__name__", str(processor_cls)),
    )


@lru_cache
def get_processorkwargs_from_processor(processor: _P) -> set[str]:
    try:
        call_kwargs = type(processor).__call__.__annotations__.get("kwargs", None)
        # if the processor has explicit kwargs annotation, use it
        if call_kwargs is not None:
            processor_kwargs = get_args(call_kwargs)[0].__annotations__.keys()
            return set(processor_kwargs)
        # otherwise, try to get from ProcessingKwargs
        else:
            module_name = type(processor).__module__
            mod = importlib.import_module(module_name)
            # find *ProcessingKwargs in the module
            processor_kwargs = set()
            for name, obj in vars(mod).items():
                if name.endswith("ProcessingKwargs"):
                    processor_kwargs = processor_kwargs | _collect_dynamic_keys_from_processing_kwargs(obj)
            return processor_kwargs
    except Exception as e:
        return set()


def _update_cache_from_instance(
        model_config: "ModelConfig",
        processor_cls: type[_P] | tuple[type[_P], ...],
        processor: ProcessorMixin
) -> _ResolvedProcessorInfo:
    """
    After instantiation (e.g., via AutoProcessor), record the concrete class
    and its dynamic keys to the cache keyed by (config, processor_cls identity).
    """
    key = _proc_resolution_key(model_config, processor_cls)
    # Extract kwargs from processorkwargs, including those model specific ones
    dynamic_kwargs = get_processorkwargs_from_processor(processor)
    info = _ResolvedProcessorInfo(processor_type=type(processor), dynamic_keys=dynamic_kwargs)
    _PROC_CLASS_CACHE[key] = info
    return info


def _get_cached_dynamic_keys(
        model_config: "ModelConfig",
        processor_cls: type[_P] | tuple[type[_P], ...] = ProcessorMixin
) -> set[str] | None:
    key = _proc_resolution_key(model_config, processor_cls)
    info = _PROC_CLASS_CACHE.get(key)
    return set(info.dynamic_keys) if info is not None else None


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
    return cached_get_image_processor(
        model_config.model,
        revision=model_config.revision,
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
