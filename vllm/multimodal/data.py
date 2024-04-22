"""The types of multi-modal data supported by vLLM."""
import torch
from PIL import Image


class MultiModalData:
    """To add a new data type, create a subclass of
    :class:`vllm.multimodal.processor.MultiModalData`
    and :class:`vllm.multimodal.processor.MultiModalProcessorBaseRegistry`,
    then update `vllm.multimodal.registry.MultiModalRegistry`
    to handle the newly defined registry."""
    pass


class ImagePixelData(MultiModalData):

    def __init__(self, image: Image.Image) -> None:
        # So that this class can be created inside the Image context manager
        image.load()

        self.image = image


class ImageFeatureData(MultiModalData):

    def __init__(self, image_features: torch.Tensor) -> None:
        self.image_features = image_features
