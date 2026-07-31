# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer
from transformers.image_utils import SizeDict
from transformers.models.siglip2.image_processing_siglip2 import (
    Siglip2ImageProcessor,
    get_image_size_for_max_num_patches,
)
from transformers.processing_utils import ProcessorMixin, Unpack


class Siglip2ImageProcessorNoUpscaleKwargs(  # type: ignore[call-arg]
    Siglip2ImageProcessor.valid_kwargs,
    total=False,
):
    min_num_patches: int


class Siglip2ImageProcessorNoUpscale(Siglip2ImageProcessor):
    """SigLIP2 NaFlex processor that does not upscale already-large images.

    Vendored from the `microsoft/Phi-4-reasoning-vision-15B` remote code, whose
    module-level imports break on Transformers v5.4+.
    """

    valid_kwargs = Siglip2ImageProcessorNoUpscaleKwargs
    min_num_patches = 1

    def __init__(self, **kwargs: Unpack[Siglip2ImageProcessorNoUpscaleKwargs]):
        super().__init__(**kwargs)

    def resize(self, image, size, resample=None, **kwargs):
        # NOTE: `size` is ignored on purpose. The caller
        # (`Siglip2ImageProcessor._preprocess`) always sizes for
        # `max_num_patches`, but this checkpoint keeps an image at its native
        # patch grid whenever that grid already lies within
        # [min_num_patches, max_num_patches], and only resizes to the nearer
        # bound otherwise. Everything else (rescale, normalize, patchify,
        # pad to `max_num_patches`, spatial shapes) is inherited unchanged.
        patch_size = self.patch_size
        height, width = image.shape[-2], image.shape[-1]
        num_patches = max((width // patch_size) * (height // patch_size), 1)
        num_patches = min(max(num_patches, self.min_num_patches), self.max_num_patches)

        height, width = get_image_size_for_max_num_patches(
            height, width, patch_size, max_num_patches=num_patches
        )
        size = SizeDict(height=height, width=width)

        return super().resize(image, size, resample=resample, **kwargs)


class Phi4VisionRProcessor(ProcessorMixin):
    """Processor for `microsoft/Phi-4-reasoning-vision-15B`.

    Text and image handling are inherited from :class:`ProcessorMixin`; only
    component construction needs overriding.
    """

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    # The checkpoint names `Siglip2ImageProcessorNoUpscale` in
    # `preprocessor_config.json` but has no `AutoImageProcessor` entry in its
    # `auto_map`, so the default auto resolution cannot find it.
    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        component_kwargs = dict(kwargs)
        component_kwargs.pop("trust_remote_code", None)
        tokenizer = component_kwargs.pop("tokenizer", None)

        image_processor = Siglip2ImageProcessorNoUpscale.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=False,
                **component_kwargs,
            )

        return [image_processor, tokenizer]


__all__ = [
    "Phi4VisionRProcessor",
]
