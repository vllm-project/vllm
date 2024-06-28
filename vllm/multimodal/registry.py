import functools
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, Union

from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger

from .base import (EXTERNAL_MM_DATA_TYPE, MultiModalData,
                   MultiModalInputMapper, MultiModalPlugin)
from .image import ImageData, ImagePlugin

logger = init_logger(__name__)

D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type[nn.Module])


class MultiModalRegistry:
    """
    A registry to dispatch data processing
    according to its modality and the target model.

    The registry handles both external and internal data input.
    """

    DEFAULT_PLUGINS = (ImagePlugin(), )

    def __init__(self,
                 *,
                 plugins: Sequence[MultiModalPlugin[Any]] = DEFAULT_PLUGINS
                 ) -> None:
        self._plugins_by_internal_data_type = {
            p.get_internal_data_type(): p
            for p in plugins
        }
        self._plugins_by_external_data_type = {
            p.get_external_data_type(): p
            for p in plugins
        }

    def register_plugin(self, plugin: MultiModalPlugin[Any]) -> None:
        data_type = plugin.get_internal_data_type()

        if data_type in self._plugins_by_internal_data_type:
            logger.warning(
                "A plugin is already registered for data type %s, "
                "and will be overwritten by the new plugin %s.", data_type,
                plugin)

        self._plugins_by_internal_data_type[data_type] = plugin

    def register_image_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper[ImageData]] = None,
    ):
        """
        Register an input mapper for image pixel data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper(ImageData, mapper)

    def _process_external_input(self, key, value, model_config: ModelConfig):
        plugin = self._get_plugin_for_external_data_type(key, type(value))
        if plugin:
            return plugin.map_input(model_config,
                                    plugin.get_internal_data_type()(value))
        msg = f"Unknown multi-modal data type: {type(value)}"
        raise NotImplementedError(msg)

    def _get_plugin_for_external_data_type(self, key: str,
                                           data_type: Type[Any]):
        for typ in data_type.mro():
            plugin = self._plugins_by_external_data_type.get((key, typ))
            if plugin is not None:
                return plugin

        msg = f"No plugin found for key {key} and type {data_type}"
        raise NotImplementedError(msg)

    def _get_plugin_for_internal_data_type(self,
                                           data_type: Type[MultiModalData]):
        for typ in data_type.mro():
            plugin = self._plugins_by_internal_data_type.get(typ)
            if plugin is not None:
                return plugin

        msg = f"Unknown multi-modal data type: {data_type}"
        raise NotImplementedError(msg)

    def register_input_mapper(
        self,
        data_type: Type[D],
        mapper: Optional[MultiModalInputMapper[D]] = None,
    ):
        """
        Register an input mapper for a specific modality to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self._get_plugin_for_internal_data_type(data_type) \
            .register_input_mapper(mapper)

    def register_image_input(self,
                             mapper: Optional[
                                 MultiModalInputMapper[ImageData]] = None):
        """
        Register an input mapper for image pixel data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper(ImageData, mapper)

    def map_input(self, model_config: ModelConfig,
                  data: Union[MultiModalData, Dict[str,
                                                   EXTERNAL_MM_DATA_TYPE]]):
        """
        Apply an input mapper to a :class:`~MultiModalData` instance passed
        to the model.
        
        See :meth:`MultiModalPlugin.map_input` for more details.
        """
        if isinstance(data, MultiModalData):
            return self._get_plugin_for_internal_data_type(type(data)) \
                .map_input(model_config, data)
        else:
            result_list = [
                self._process_external_input(k, v, model_config)
                for k, v in data.items()
            ]
            return {k: v for d in result_list for k, v in d.items()}

    def create_input_mapper(self, model_config: ModelConfig):
        """
        Create an input mapper (see :meth:`map_input`) for a specific model.
        """
        return functools.partial(self.map_input, model_config)
