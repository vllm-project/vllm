import functools
from collections import UserDict
from typing import (TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional,
                    Sequence, Type, TypeVar)

import torch.nn as nn
from typing_extensions import TypeAlias

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import ClassRegistry

from .audio import AudioPlugin
from .base import MultiModalInputMapper, MultiModalPlugin, MultiModalTokensCalc
from .image import ImagePlugin
from .inputs import MultiModalDataDict, MultiModalKwargs, NestedTensors
from .processing import MultiModalProcessor
from .video import VideoPlugin

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

N = TypeVar("N", bound=Type[nn.Module])

MultiModalProcessorFactory: TypeAlias = Callable[[InputProcessingContext],
                                                 MultiModalProcessor]
"""
Constructs a :class:`MultiModalProcessor` instance from the context.

The processing metadata should be derived from the context.
"""


class _MultiModalLimits(UserDict["ModelConfig", Dict[str, int]]):
    """
    Wraps `_limits_by_model` for a more informative error message
    when attempting to access a model that does not exist.
    """

    def __getitem__(self, key: "ModelConfig") -> Dict[str, int]:
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            msg = (f"Cannot find `mm_limits` for model={key.model}. Did you "
                   "forget to call `init_mm_limits_per_prompt`?")
            raise KeyError(msg) from exc


class MultiModalRegistry:
    """
    A registry that dispatches data processing to the
    :class:`~vllm.multimodal.MultiModalPlugin` for each modality.
    """

    DEFAULT_PLUGINS = (ImagePlugin(), AudioPlugin(), VideoPlugin())

    def __init__(
            self,
            *,
            plugins: Sequence[MultiModalPlugin] = DEFAULT_PLUGINS) -> None:
        self._plugins = {p.get_data_key(): p for p in plugins}

        self._processor_factories = ClassRegistry[nn.Module,
                                                  MultiModalProcessorFactory]()

        # This is used for non-multimodal models
        self._disabled_limits_per_plugin = {k: 0 for k in self._plugins}

        self._limits_by_model = _MultiModalLimits()

    def register_plugin(self, plugin: MultiModalPlugin) -> None:
        """
        Register a multi-modal plugin so it can be recognized by vLLM.

        See also:
            :ref:`adding_multimodal_plugin`
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
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
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
        merged_dict: Dict[str, NestedTensors] = {}

        for data_key, data_value in data.items():
            plugin = self._get_plugin(data_key)

            num_items = len(data_value) if isinstance(data_value, list) else 1
            max_items = self._limits_by_model[model_config][data_key]
            if num_items > max_items:
                raise ValueError(
                    f"You set {data_key}={max_items} (or defaulted to 1) in "
                    f"`--limit-mm-per-prompt`, but found {num_items} items "
                    "in the same prompt.")

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

    def get_max_multimodal_tokens(self, model_config: "ModelConfig") -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.

        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.

        Note:
            This should be called after :meth:`init_mm_limits_per_prompt`.
        """
        limits_per_plugin = self._limits_by_model[model_config]

        return sum((limits_per_plugin[key] *
                    plugin.get_max_multimodal_tokens(model_config))
                   for key, plugin in self._plugins.items())

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
                key: config_limits_per_plugin.get(key, 1)
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
        return self._limits_by_model[model_config]

    def register_processor(
        self,
        factory: MultiModalProcessorFactory,
    ):
        """
        Register a multi-modal processor to a model class.

        When the model receives multi-modal data, the provided function is
        invoked to transform the data into a dictionary of model inputs.

        See also:
            - :ref:`input_processing_pipeline`
            - :ref:`enabling_multimodal_inputs`
        """

        def wrapper(model_cls: N) -> N:
            if model_cls in self._processor_factories:
                logger.warning(
                    "Model class %s already has an input mapper "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._processor_factories[model_cls] = factory

            return model_cls

        return wrapper

    def has_processor(self, model_config: "ModelConfig") -> bool:
        """
        Test whether a multi-modal processor is defined for a specific model.
        """
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        return model_cls in self._processor_factories

    def create_processor(
        self,
        model_config: "ModelConfig",
        tokenizer: AnyTokenizer,
    ) -> MultiModalProcessor:
        """
        Create a multi-modal processor for a specific model and tokenizer.
        """

        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        processor_factory = self._processor_factories[model_cls]

        ctx = InputProcessingContext(model_config, tokenizer)
        return processor_factory(ctx)
