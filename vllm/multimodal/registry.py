# SPDX-License-Identifier: Apache-2.0

import functools
import json
from collections import UserDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Optional, Protocol, TypeVar

import torch.nn as nn

from vllm.envs import VLLM_MM_INPUT_CACHE_GIB
from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               cached_tokenizer_from_config)
from vllm.utils import ClassRegistry

from .audio import AudioPlugin
from .base import MultiModalInputMapper, MultiModalPlugin, MultiModalTokensCalc
from .image import ImagePlugin
from .inputs import MultiModalDataDict, MultiModalKwargs, NestedTensors
from .processing import (BaseMultiModalProcessor, BaseProcessingInfo,
                         ProcessingCache)
from .profiling import (BaseDummyInputsBuilder, DummyDecoderData,
                        DummyEncoderData, MultiModalProfiler)
from .video import VideoPlugin

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

N = TypeVar("N", bound=type[nn.Module])
_I = TypeVar("_I", bound=BaseProcessingInfo)
_I_co = TypeVar("_I_co", bound=BaseProcessingInfo, covariant=True)


class ProcessingInfoFactory(Protocol[_I_co]):
    """Constructs a :class:`MultiModalProcessor` instance from the context."""

    def __call__(
        self,
        ctx: InputProcessingContext,
    ) -> _I_co:
        ...


class DummyInputsBuilderFactory(Protocol[_I]):
    """
    Constructs a :class:`BaseDummyInputsBuilder` instance from the context.
    """

    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]:
        ...


class MultiModalProcessorFactory(Protocol[_I]):
    """Constructs a :class:`MultiModalProcessor` instance from the context."""

    def __call__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: Optional[ProcessingCache] = None,
    ) -> BaseMultiModalProcessor[_I]:
        ...


@dataclass(frozen=True)
class _ProcessorFactories(Generic[_I]):
    info: ProcessingInfoFactory[_I]
    processor: MultiModalProcessorFactory[_I]
    dummy_inputs: DummyInputsBuilderFactory[_I]

    def build_processor(
        self,
        ctx: InputProcessingContext,
        *,
        cache: Optional[ProcessingCache] = None,
    ):
        info = self.info(ctx)
        dummy_inputs_builder = self.dummy_inputs(info)
        return self.processor(info, dummy_inputs_builder, cache=cache)


class _MultiModalLimits(UserDict["ModelConfig", dict[str, int]]):
    """
    Wraps `_limits_by_model` for a more informative error message
    when attempting to access a model that does not exist.
    """

    def __getitem__(self, key: "ModelConfig") -> dict[str, int]:
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            msg = (f"Cannot find `mm_limits` for model={key.model}. Did you "
                   "forget to call `init_mm_limits_per_prompt`?")
            raise KeyError(msg) from exc


class MultiModalRegistry:
    """
    A registry that dispatches data processing according to the model.
    """

    DEFAULT_PLUGINS = (ImagePlugin(), AudioPlugin(), VideoPlugin())

    def __init__(
            self,
            *,
            plugins: Sequence[MultiModalPlugin] = DEFAULT_PLUGINS) -> None:
        self._plugins = {p.get_data_key(): p for p in plugins}

        self._processor_factories = ClassRegistry[nn.Module,
                                                  _ProcessorFactories]()

        # This is used for non-multimodal models
        self._disabled_limits_per_plugin = {k: 0 for k in self._plugins}

        self._limits_by_model = _MultiModalLimits()

        self._processing_cache = ProcessingCache(VLLM_MM_INPUT_CACHE_GIB)

    def register_plugin(self, plugin: MultiModalPlugin) -> None:
        """
        Register a multi-modal plugin so it can be recognized by vLLM.
        """
        data_type_key = plugin.get_data_key()

        if data_type_key in self._plugins:
            logger.warning(
                "A plugin is already registered for data type %s, "
                "and will be overwritten by the new plugin %s.", data_type_key,
                plugin)

        self._plugins[data_type_key] = plugin

    def _get_plugin(self, data_type_key: str):
        plugin = self._plugins.get(data_type_key)
        if plugin is not None:
            return plugin

        msg = f"Unknown multi-modal data type: {data_type_key}"
        raise NotImplementedError(msg)

    def register_input_mapper(
        self,
        data_type_key: str,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper for a specific modality to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self._get_plugin(data_type_key).register_input_mapper(mapper)

    def register_image_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper for image data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper("image", mapper)

    def map_input(
        self,
        model_config: "ModelConfig",
        data: MultiModalDataDict,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> MultiModalKwargs:
        """
        Apply an input mapper to the data passed to the model.

        The data belonging to each modality is passed to the corresponding
        plugin which in turn converts the data into into keyword arguments
        via the input mapper registered for that model.

        See :meth:`MultiModalPlugin.map_input` for more details.

        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        merged_dict = dict[str, NestedTensors]()

        for data_key, data_value in data.items():
            plugin = self._get_plugin(data_key)

            num_items = len(data_value) if isinstance(data_value, list) else 1
            max_items = self._limits_by_model[model_config][data_key]
            if num_items > max_items:
                raise ValueError(
                    f"You set '{json.dumps({data_key: max_items})}' (or "
                    "defaulted to 1) in `--limit-mm-per-prompt`, but found "
                    f"{num_items} items in the same prompt.")

            input_dict = plugin.map_input(model_config, data_value,
                                          mm_processor_kwargs)
            for input_key, input_tensor in input_dict.items():
                if input_key in merged_dict:
                    raise ValueError(f"The input mappers (keys={set(data)}) "
                                     f"resulted in a conflicting keyword "
                                     f"argument to `forward()`: {input_key}")

                merged_dict[input_key] = input_tensor

        return MultiModalKwargs(merged_dict)

    def create_input_mapper(self, model_config: "ModelConfig"):
        """
        Create an input mapper (see :meth:`map_input`) for a specific model.
        """
        # NOTE - we currently make the assumption that if a model has multiple
        # supported modalities, they take the same kwargs. For the default,
        # this could be an issue in the future if it falls back to two HF
        # resources and we can't inspect the signature easily since it's
        # getting initialized through the autoclass.
        #
        # If this is a problem in the future, we should revisit it, but since
        # it potentially introduces a lot of complexity for a currently
        # uncommon case, we do not for simplicity of both use & implementation
        return functools.partial(self.map_input, model_config)

    def register_max_multimodal_tokens(
        self,
        data_type_key: str,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of tokens, corresponding to a single
        instance of multimodal data belonging to a specific modality, that are
        passed to the language model for a model class.
        """
        return self._get_plugin(data_type_key) \
            .register_max_multimodal_tokens(max_mm_tokens)

    def register_max_image_tokens(
        self,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of image tokens, corresponding to a single
        image, that are passed to the language model for a model class.
        """
        return self.register_max_multimodal_tokens("image", max_mm_tokens)

    def get_max_tokens_per_item_by_modality(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based 
        on underlying model configuration.
        """
        if self.has_processor(model_config):
            processor = self.create_processor(model_config, disable_cache=True)
            profiler = MultiModalProfiler(processor)

            seq_len = model_config.max_model_len
            mm_limits = self.get_mm_limits_per_prompt(model_config)

            return profiler.get_mm_max_tokens(
                seq_len,
                {
                    modality: 1
                    for modality, limit in mm_limits.items() if limit > 0
                },
            )

        return {
            key: plugin.get_max_multimodal_tokens(model_config)
            for key, plugin in self._plugins.items()
        }

    def get_max_tokens_per_item_by_nonzero_modality(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens per data item from each modality based
        on underlying model configuration, excluding modalities that user 
        explicitly disabled via `limit_mm_per_prompt`.

        Note:
            This is currently directly used only in V1 for profiling the memory 
            usage of a model.
        """
        mm_limits = self.get_mm_limits_per_prompt(model_config)

        return {
            key: max_tokens_per_mm_item
            for key, max_tokens_per_mm_item in
            self.get_max_tokens_per_item_by_modality(model_config).items()
            if mm_limits[key] > 0
        }

    def get_max_tokens_by_modality(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, int]:
        """
        Get the maximum number of tokens from each modality
        for profiling the memory usage of a model.

        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.

        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        mm_limits = self.get_mm_limits_per_prompt(model_config)

        return {
            key: mm_limits[key] * max_tokens_per_mm_item
            for key, max_tokens_per_mm_item in
            self.get_max_tokens_per_item_by_modality(model_config).items()
        }

    def get_max_multimodal_tokens(self, model_config: "ModelConfig") -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.

        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.

        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        return sum(self.get_max_tokens_by_modality(model_config).values())

    def init_mm_limits_per_prompt(
        self,
        model_config: "ModelConfig",
    ) -> None:
        """
        Initialize the maximum number of multi-modal input instances for each
        modality that are allowed per prompt for a model class.
        """
        if model_config in self._limits_by_model:
            logger.warning(
                "`mm_limits` has already been set for model=%s, and will "
                "be overwritten by the new values.", model_config.model)

        multimodal_config = model_config.multimodal_config
        if multimodal_config is None:
            limits_per_plugin = self._disabled_limits_per_plugin
        else:
            config_limits_per_plugin = multimodal_config.limit_per_prompt

            extra_keys = config_limits_per_plugin.keys() - self._plugins.keys()
            if extra_keys:
                logger.warning(
                    "Detected extra keys in `--limit-mm-per-prompt` which "
                    "are not registered as multi-modal plugins: %s. "
                    "They will be ignored.", extra_keys)

            # NOTE: Currently the default is set to 1 for each plugin
            # TODO: Automatically determine the limits based on budget
            # once more models support multi-image inputs
            limits_per_plugin = {
                key: multimodal_config.get_limit_per_prompt(key)
                for key in self._plugins
            }

        self._limits_by_model[model_config] = limits_per_plugin

    def get_mm_limits_per_prompt(
        self,
        model_config: "ModelConfig",
    ) -> Mapping[str, int]:
        """
        Get the maximum number of multi-modal input instances for each modality
        that are allowed per prompt for a model class.

        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        if self.has_processor(model_config):
            processor = self.create_processor(model_config, disable_cache=True)
            profiler = MultiModalProfiler(processor)
            return profiler.get_mm_limits()

        return self._limits_by_model[model_config]

    def register_processor(
        self,
        processor: MultiModalProcessorFactory[_I],
        *,
        info: ProcessingInfoFactory[_I],
        dummy_inputs: DummyInputsBuilderFactory[_I],
    ):
        """
        Register a multi-modal processor to a model class. The processor
        is constructed lazily, hence a factory method should be passed.

        When the model receives multi-modal data, the provided function is
        invoked to transform the data into a dictionary of model inputs.

        See also:
            :ref:`mm-processing`
        """

        def wrapper(model_cls: N) -> N:
            if self._processor_factories.contains(model_cls, strict=True):
                logger.warning(
                    "Model class %s already has a multi-modal processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._processor_factories[model_cls] = _ProcessorFactories(
                info=info,
                dummy_inputs=dummy_inputs,
                processor=processor,
            )

            return model_cls

        return wrapper

    def _get_model_cls(self, model_config: "ModelConfig"):
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        return model_cls

    def has_processor(self, model_config: "ModelConfig") -> bool:
        """
        Test whether a multi-modal processor is defined for a specific model.

        See also:
            :ref:`mm-processing`
        """
        return self._get_model_cls(model_config) in self._processor_factories

    def create_processor(
        self,
        model_config: "ModelConfig",
        *,
        tokenizer: Optional[AnyTokenizer] = None,
        disable_cache: Optional[bool] = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        Create a multi-modal processor for a specific model and tokenizer.

        See also:
            :ref:`mm-processing`
        """
        if tokenizer is None:
            tokenizer = cached_tokenizer_from_config(model_config)
        if disable_cache is None:
            disable_cache = model_config.disable_mm_preprocessor_cache

        model_cls = self._get_model_cls(model_config)
        factories = self._processor_factories[model_cls]

        ctx = InputProcessingContext(model_config, tokenizer)
        cache = None if disable_cache else self._processing_cache

        return factories.build_processor(ctx, cache=cache)

    def get_decoder_dummy_data(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Optional[Mapping[str, int]] = None,
    ) -> DummyDecoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.
        """
        processor = self.create_processor(model_config, disable_cache=True)
        profiler = MultiModalProfiler(processor)
        dummy_data = profiler.get_decoder_dummy_data(seq_len, mm_counts)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            raise AssertionError(
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but found {len(token_ids)} tokens instead.")

        return dummy_data

    def get_encoder_dummy_data(
        self,
        model_config: "ModelConfig",
        seq_len: int,
        mm_counts: Optional[Mapping[str, int]] = None,
    ) -> DummyEncoderData:
        """
        Create dummy data for profiling the memory usage of a model.

        The model is identified by ``model_config``.
        """
        processor = self.create_processor(model_config, disable_cache=True)
        profiler = MultiModalProfiler(processor)
        dummy_data = profiler.get_encoder_dummy_data(seq_len, mm_counts)

        # Having more tokens is over-conservative but otherwise fine
        token_ids = dummy_data.prompt_token_ids
        if len(token_ids) < seq_len:
            logger.warning_once(
                f"Expected at least {seq_len} dummy encoder tokens for "
                f"profiling, but found {len(token_ids)} tokens instead.")

        return dummy_data
