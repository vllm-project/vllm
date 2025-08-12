import functools
from typing import Dict, Optional, Sequence

import torch

from vllm.config import ModelConfig
from vllm.logger import init_logger

from .base import (MultiModalDataDict, MultiModalInputMapper, MultiModalInputs,
                   MultiModalPlugin)
from .image import ImagePlugin

logger = init_logger(__name__)


class MultiModalRegistry:
    """
    A registry to dispatch data processing
    according to its modality and the target model.

    The registry handles both external and internal data input.
    """

    DEFAULT_PLUGINS = (ImagePlugin(), )

    def __init__(
            self,
            *,
            plugins: Sequence[MultiModalPlugin] = DEFAULT_PLUGINS) -> None:
        self._plugins = {p.get_data_key(): p for p in plugins}

    def register_plugin(self, plugin: MultiModalPlugin) -> None:
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

    def register_image_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper for image data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper("image", mapper)

    def _process_input(self, key: str, value: object,
                       model_config: ModelConfig) -> MultiModalInputs:
        plugin = self._plugins.get(key)
        if plugin:
            return plugin.map_input(model_config, value)
        msg = f"Unknown multi-modal data type: {key}"
        raise NotImplementedError(msg)

    def register_input_mapper(
        self,
        data_type: str,
        mapper: Optional[MultiModalInputMapper] = None,
    ):
        """
        Register an input mapper for a specific modality to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        plugin = self._plugins.get(data_type)
        if not plugin:
            msg = f"Unknown multi-modal data type: {data_type}"
            raise NotImplementedError(msg)
        return plugin.register_input_mapper(mapper)

    def register_image_input(self,
                             mapper: Optional[MultiModalInputMapper] = None):
        """
        Register an input mapper for image pixel data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper("image", mapper)

    def map_input(self, model_config: ModelConfig,
                  data: MultiModalDataDict) -> MultiModalInputs:
        """
        Apply an input mapper to the data passed to the model.
        
        See :meth:`MultiModalPlugin.map_input` for more details.
        """
        merged_dict: Dict[str, torch.Tensor] = {}

        for data_key, data_value in data.items():
            input_dict = self._process_input(data_key, data_value,
                                             model_config)

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

    def get_num_input_tokens(self):
        """
        Get the number of input tokens for profiling purposes.
        """
        # TODO: Provide this number on a per model basis.
        return 3000
