# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from abc import ABC
from collections.abc import Iterable

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, PretrainedConfig
from transformers.image_processing_utils_fast import BaseImageProcessorFast

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.models.internvl import (
    BaseInternVLDummyInputsBuilder,
    BaseInternVLMultiModalProcessor,
    BaseInternVLProcessingInfo,
    InternVLImageEmbeddingInputs,
    InternVLImageInputs,
    InternVLImagePixelInputs,
    InternVLProcessor,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_image_processor_from_config
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

IMG_START = "<img>"
IMG_END = "</img>"
IMG_CONTEXT = "<image>"


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: convert_image_mode(img, "RGB")),
            T.Resize(
                (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
        ]
    )


# adapted from https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height

    for rw, rh in target_ratios:
        target_aspect_ratio = rw / rh
        size_factor = min((rw * rh * image_size * image_size) / area, 0.6)
        ratio_closeness = min(
            target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
        )
        factor = size_factor * ratio_closeness

        if factor > best_factor:
            best_factor = factor
            best_ratio = (rw, rh)

    return best_ratio


def calculate_nemotron_vl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]:
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks, target_width, target_height


def dynamic_preprocess_nemotron_vl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_nemotron_vl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def get_nemotron_vl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def image_to_pixel_values_nemotron_vl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
    target_ratios = get_nemotron_vl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)

    images = dynamic_preprocess_nemotron_vl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
    return pixel_values


class NemotronVLProcessor(InternVLProcessor):
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        image_processor: BaseImageProcessorFast,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        ABC.__init__(self)
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        image_size: int = config.force_image_size
        patch_size: int = config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = 1
        assert isinstance(min_dynamic_patch, int)

        if max_dynamic_patch is None:
            max_dynamic_patch = self.image_processor.max_num_tiles
        assert isinstance(max_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = True
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = self.image_processor.use_thumbnail

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_nemotron_vl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_nemotron_vl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            )
            for image in images
        ]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
            )
            image_inputs = {
                "pixel_values_flat": torch.cat(pixel_values_lst),
                "image_num_patches": torch.tensor(
                    [len(item) for item in pixel_values_lst]
                ),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token
                image_repl = self.get_image_repl(feature_size, num_patches)
                NVL_IMAGE_CONTEXT = image_repl.full.replace(
                    "<image>", "<NVL_IMG_CONTEXT>"
                )
                text = [t.replace("<image>", NVL_IMAGE_CONTEXT, 1) for t in text]
            text = [t.replace("<NVL_IMG_CONTEXT>", IMG_CONTEXT) for t in text]
        return text, image_inputs

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)


class NemotronVLProcessingInfo(BaseInternVLProcessingInfo):
    """Processing info for Nemotron VL models."""

    def get_hf_processor(self, **kwargs: object) -> NemotronVLProcessor:
        return self.ctx.init_processor(
            NemotronVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            image_processor=self.get_image_processor(),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object):
        return cached_image_processor_from_config(
            self.ctx.model_config,
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    BaseInternVLMultiModalProcessor[NemotronVLProcessingInfo],
    info=NemotronVLProcessingInfo,
    dummy_inputs=BaseInternVLDummyInputsBuilder[NemotronVLProcessingInfo],
)
class LlamaNemotronVLChatModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    merge_by_field_config = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.llm_arch_name = config.text_config.architectures[0]
        self.vision_model = self._init_vision_model(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.mlp1 = self._init_mlp1(config)

        self.img_context_token_id = None

        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _patch_quant_config(
        self, config: PretrainedConfig, quant_config: QuantizationConfig
    ):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config", None)
            if (not quant_config.modules_to_not_convert) and (
                llm_quant_config is not None
            ):
                quant_config.modules_to_not_convert.append("vision_model")

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ):
        return AutoModel.from_config(config.vision_config, trust_remote_code=True)

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Module:
        vit_hidden_size = config.vit_hidden_size
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, bias=True
            ),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                vision_projection_hidden_size,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1/blob/main/modeling.py#L177
        vit_embeds = self.vision_model(x=pixel_values).features
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> InternVLImageInputs | None:
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_token_id = kwargs["image_token_id"]
        if isinstance(image_token_id, torch.Tensor):
            image_token_id = image_token_id.flatten().unique().item()

        assert isinstance(image_token_id, int)
        self.img_context_token_id = image_token_id

        if pixel_values_flat is not None:
            return InternVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=pixel_values_flat,
                num_patches=image_num_patches,
                resolve_bindings={
                    "h": self.config.force_image_size,
                    "w": self.config.force_image_size,
                },
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values_flat", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)

        return modalities

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        self.visual_token_mask = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(image_embeddings)

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            self._set_visual_token_mask(input_ids)

        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> IntermediateTensors:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update({"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ## Ignore registered_buffers
        ## see https://huggingface.co/nvidia/C-RADIOv2-H/blob/main/input_conditioner.py#L28 # noqa: E501
        skip_substrs = ["norm_mean", "norm_std"]
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mlp1",
            tower_model="vision_model",
        )
