import functools
from typing import Any, Optional, Sequence, Type, TypeVar

from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger

from .base import MultiModalData, MultiModalInputMapper, MultiModalPlugin
from .image import (ImageFeatureData, ImageFeaturePlugin, ImagePixelData,
                    ImagePixelPlugin)

logger = init_logger(__name__)

D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type[nn.Module])


class MultiModalRegistry:
    """
    A registry to dispatch data processing
    according to its modality and the target model.
    """

    DEFAULT_PLUGINS = (ImageFeaturePlugin(), ImagePixelPlugin())

    def __init__(
        self,
        *,
        plugins: Sequence[MultiModalPlugin[Any]] = DEFAULT_PLUGINS,
    ) -> None:
        self._plugins_by_data_type = {p.get_data_type(): p for p in plugins}

    def register_plugin(self, plugin: MultiModalPlugin[Any]) -> None:
        data_type = plugin.get_data_type()

        if data_type in self._plugins_by_data_type:
            logger.warning(
                "A plugin is already registered for data type %s, "
                "and will be overwritten by the new plugin %s.", data_type,
                plugin)

        self._plugins_by_data_type[data_type] = plugin

    def _get_plugin_for_data_type(self, data_type: Type[MultiModalData]):
        for typ in data_type.mro():
            plugin = self._plugins_by_data_type.get(typ)
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
        return self._get_plugin_for_data_type(data_type) \
            .register_input_mapper(mapper)

    def register_image_pixel_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper[ImagePixelData]] = None,
    ):
        """
        Register an input mapper for image pixel data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper(ImagePixelData, mapper)

    def register_image_feature_input_mapper(
        self,
        mapper: Optional[MultiModalInputMapper[ImageFeatureData]] = None,
    ):
        """
        Register an input mapper for image feature data to a model class.

        See :meth:`MultiModalPlugin.register_input_mapper` for more details.
        """
        return self.register_input_mapper(ImageFeatureData, mapper)

    def map_input(self, model_config: ModelConfig, data: MultiModalData):
        """
        Apply an input mapper to a :class:`~MultiModalData` instance passed
        to the model.
        
        See :meth:`MultiModalPlugin.map_input` for more details.
        """
        return self._get_plugin_for_data_type(type(data)) \
            .map_input(model_config, data)

    def create_input_mapper(self, model_config: ModelConfig):
        """
        Create an input mapper (see :meth:`map_input`) for a specific model.
        """
        return functools.partial(self.map_input, model_config)
