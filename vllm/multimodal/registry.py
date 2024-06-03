import functools
from typing import Any, Optional, Sequence, Type, TypeVar

from torch import nn

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger

from .base import MultiModalData, MultiModalInputProcessor, MultiModalPlugin
from .image import (ImageFeatureData, ImageFeaturePlugin, ImagePixelData,
                    ImagePixelPlugin)

logger = init_logger(__name__)

D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type[nn.Module])


class MultiModalRegistry:
    """
    This registry is used by model runners to dispatch data processing
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

    def register_input(
            self,
            data_type: Type[D],
            processor: Optional[MultiModalInputProcessor[D]] = None):
        """
        Register an input processor for a specific modality to a model class.

        See :meth:`MultiModalPlugin.register_input_processor` for more details.
        """
        return self._get_plugin_for_data_type(data_type) \
            .register_input_processor(processor)

    def register_image_pixel_input(
            self,
            processor: Optional[
                MultiModalInputProcessor[ImagePixelData]] = None):
        """
        Register an input processor for image pixel data to a model class.

        See :meth:`MultiModalPlugin.register_input_processor` for more details.
        """
        return self.register_input(ImagePixelData, processor)

    def register_image_feature_input(
        self,
        processor: Optional[
            MultiModalInputProcessor[ImageFeatureData]] = None):
        """
        Register an input processor for image feature data to a model class.

        See :meth:`MultiModalPlugin.register_input_processor` for more details.
        """
        return self.register_input(ImageFeatureData, processor)

    def process_input(self, data: MultiModalData, model_config: ModelConfig,
                      vlm_config: VisionLanguageConfig):
        """
        Apply an input processor to a :class:`~MultiModalData` instance passed
        to the model.
        
        See :meth:`MultiModalPlugin.process_input` for more details.
        """
        return self._get_plugin_for_data_type(type(data)) \
            .process_input(data, model_config, vlm_config)

    def create_input_processor(self, model_config: ModelConfig,
                               vlm_config: VisionLanguageConfig):
        """
        Create an input processor (see :meth:`process_input`) for a
        specific model.
        """
        return functools.partial(self.process_input,
                                 model_config=model_config,
                                 vlm_config=vlm_config)
