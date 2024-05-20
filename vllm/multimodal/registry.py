import functools
from typing import (TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence,
                    Tuple, Type, TypeVar)

from vllm.config import ModelConfig, VisionLanguageConfig
from vllm.logger import init_logger

from .base import MultiModalData, MultiModalPlugin
from .image import (ImageFeatureData, ImageFeaturePlugin, ImagePixelData,
                    ImagePixelPlugin)

if TYPE_CHECKING:
    import torch
    from torch import nn

    from vllm.sequence import SequenceData

logger = init_logger(__name__)

D = TypeVar("D", bound=MultiModalData)
N = TypeVar("N", bound=Type["nn.Module"])

MultiModalInputProcessor = Callable[[D, ModelConfig, VisionLanguageConfig],
                                    Dict[str, "torch.Tensor"]]
MultiModalDummyFactory = Callable[[int, ModelConfig, VisionLanguageConfig],
                                  Tuple["SequenceData", MultiModalData]]


class MultiModalRegistry:

    DEFAULT_PLUGINS = (ImageFeaturePlugin(), ImagePixelPlugin())

    def __init__(self,
                 *,
                 plugins: Sequence[MultiModalPlugin[Any]] = DEFAULT_PLUGINS
                 ) -> None:
        self._plugins_by_data_type = {p.get_data_type(): p for p in plugins}
        self._dummy_factories_by_model_type: Dict[Type["nn.Module"],
                                                  MultiModalDummyFactory] = {}

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

    def register_dummy_data(self, factory: MultiModalDummyFactory):

        def wrapper(model_cls: N) -> N:
            if model_cls in self._dummy_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._dummy_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def dummy_data_for_profiling(self, seq_len: int, model_config: ModelConfig,
                                 vlm_config: VisionLanguageConfig):
        """Creates dummy data for :meth:`vllm.worker.model_runner.profile_run`.
        """
        model_cls = MultiModalPlugin.get_model_cls(model_config)
        dummy_factory = self._dummy_factories_by_model_type.get(model_cls)
        if dummy_factory is None:
            msg = f"No dummy data defined for model class: {model_cls}"
            raise NotImplementedError(msg)

        return dummy_factory(seq_len, model_config, vlm_config)

    def register_image_pixel_input(
            self,
            processor: Optional[
                MultiModalInputProcessor[ImagePixelData]] = None):
        return self._get_plugin_for_data_type(ImagePixelData) \
            .register_input_processor(processor)

    def register_image_feature_input(
        self,
        processor: Optional[
            MultiModalInputProcessor[ImageFeatureData]] = None):
        return self._get_plugin_for_data_type(ImageFeatureData) \
            .register_input_processor(processor)

    def process_input(self, data: MultiModalData, model_config: ModelConfig,
                      vlm_config: VisionLanguageConfig):
        return self._get_plugin_for_data_type(type(data)) \
            .process_input(data, model_config, vlm_config)

    def create_input_processor(self, model_config: ModelConfig,
                               vlm_config: VisionLanguageConfig):
        return functools.partial(self.process_input,
                                 model_config=model_config,
                                 vlm_config=vlm_config)


MM_REGISTRY = MultiModalRegistry()
"""The global registry for multi-modal data."""
