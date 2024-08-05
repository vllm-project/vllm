import functools
from typing import Dict, Optional, Sequence

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger

from .base import (MultiModalDataDict, MultiModalInputMapper, MultiModalInputs,
                   MultiModalPlugin, MultiModalTokensCalc)
from .image import ImagePlugin

logger = init_logger(__name__)


class MultiModalRegistry:
    """
    A registry that dispatches data processing to the
    :class:`~vllm.multimodal.MultiModalPlugin` for each modality.
    """

    DEFAULT_PLUGINS = (ImagePlugin(), )

    def __init__(
            self,
            *,
            plugins: Sequence[MultiModalPlugin] = DEFAULT_PLUGINS) -> None:
        self._plugins = {p.get_data_key(): p for p in plugins}

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

    def map_input(self, model_config: ModelConfig,
                  data: MultiModalDataDict) -> MultiModalInputs:
        """
        Apply an input mapper to the data passed to the model.

        The data belonging to each modality is passed to the corresponding
        plugin which in turn converts the data into into keyword arguments
        via the input mapper registered for that model.

        See :meth:`MultiModalPlugin.map_input` for more details.
        """
        merged_dict: Dict[str, torch.Tensor] = {}

        for data_key, data_value in data.items():
            input_dict = self._get_plugin(data_key) \
                .map_input(model_config, data_value)

            for input_key, input_tensor in input_dict.items():
                if input_key in merged_dict:
                    raise ValueError(f"The input mappers (keys={set(data)}) "
                                     f"resulted in a conflicting keyword "
                                     f"argument to `forward()`: {input_key}")

                merged_dict[input_key] = input_tensor

        return MultiModalInputs(merged_dict)

    def create_input_mapper(self, model_config: ModelConfig):
        """
        Create an input mapper (see :meth:`map_input`) for a specific model.
        """
        return functools.partial(self.map_input, model_config)

    def register_max_multimodal_tokens(
        self,
        data_type_key: str,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of tokens, belonging to a
        specific modality, input to the language model for a model class.
        """
        return self._get_plugin(data_type_key) \
            .register_max_multimodal_tokens(max_mm_tokens)

    def register_max_image_tokens(
        self,
        max_mm_tokens: Optional[MultiModalTokensCalc] = None,
    ):
        """
        Register the maximum number of image tokens
        input to the language model for a model class.
        """
        return self.register_max_multimodal_tokens("image", max_mm_tokens)

    def get_max_multimodal_tokens(self, model_config: ModelConfig) -> int:
        """
        Get the maximum number of multi-modal tokens
        for profiling the memory usage of a model.
        
        See :meth:`MultiModalPlugin.get_max_multimodal_tokens` for more details.
        """
        return sum(
            plugin.get_max_multimodal_tokens(model_config)
            for plugin in self._plugins.values())
