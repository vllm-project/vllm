# adapted from https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/modeling_h2ovl_chat.py
# https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/image_process.py
# --------------------------------------------------------
# H2OVL-Mississippi
# Copyright (c) 2024 H2O.AI
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
from functools import partial
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import PretrainedConfig

from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, InputContext,
                         token_inputs)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.utils import is_list_of

from .intern_vit import InternVisionModel
from .internvl import (IMG_CONTEXT, IMG_END, IMG_START, InternVLChatModel,
                       InternVLInputPipeline, build_transform,
                       find_closest_aspect_ratio, get_internvl_num_patches)


# modified to include blocks generated in second pass
def calculate_num_blocks(
    orig_width: int,
    orig_height: int,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
    prior_aspect_ratio=None,
) -> Tuple[int, int, int, Tuple[int, int]]:
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # if prior_aspect_ratio is provided, filter the target ratios
    if prior_aspect_ratio is not None:
        target_ratios = [
            ratio for ratio in target_ratios if prior_aspect_ratio[0] %
            ratio[0] != 0 and prior_aspect_ratio[1] % ratio[1] != 0
        ]

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # add thumbnail image if num_blocks > 1
    if use_thumbnail and blocks > 1:
        blocks += 1
    return blocks, target_width, target_height, target_aspect_ratio


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def dynamic_preprocess(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height, target_aspect_ratio = (
        calculate_num_blocks(
            orig_width,
            orig_height,
            min_num,
            max_num,
            image_size,
            use_thumbnail=False,
        ))
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
    return processed_images, target_aspect_ratio


# new dynamic_preprocess2 with prior_aspect_ratio
def dynamic_preprocess2(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
    prior_aspect_ratio: Tuple[int, int],
) -> List[Image.Image]:
    orig_width, orig_height = image.size

    # calculate the number of blocks based on prior aspect ratio
    blocks, target_width, target_height, _ = calculate_num_blocks(
        orig_width,
        orig_height,
        min_num,
        max_num,
        image_size,
        use_thumbnail=False,
        prior_aspect_ratio=prior_aspect_ratio,
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


def load_image1(image: Image.Image, input_size=448, min_num=1, max_num=6):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        min_num=min_num,
        max_num=max_num,
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio


def load_image2(
    image: Image.Image,
    input_size=448,
    min_num=1,
    max_num=6,
    target_aspect_ratio=None,
):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(
        image,
        image_size=input_size,
        use_thumbnail=True,
        min_num=min_num,
        max_num=max_num,
        prior_aspect_ratio=target_aspect_ratio,
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def image_to_pixel_values(
    image: Image.Image,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
    use_MSAC: bool,
) -> torch.Tensor:
    # when MSAC is turned on, we need to process the image twice
    if use_MSAC:
        pixel_values, target_aspect_ratio = load_image1(image,
                                                        input_size=input_size,
                                                        min_num=min_num,
                                                        max_num=max_num)
        pixel_values2 = load_image2(
            image,
            input_size=input_size,
            min_num=min_num,
            max_num=max_num,
            target_aspect_ratio=target_aspect_ratio,
        )
        pixel_values = torch.cat(
            [pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

    else:
        transform = build_transform(input_size=input_size)
        images, _ = dynamic_preprocess(
            image,
            min_num=min_num,
            max_num=max_num,
            image_size=input_size,
            use_thumbnail=use_thumbnail,
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

    return pixel_values


def image_to_pixel_values_wrapper(hf_config: PretrainedConfig,
                                  max_dynamic_patch: Optional[int] = None):
    image_size = hf_config.vision_config.image_size
    min_num = hf_config.min_dynamic_patch
    if max_dynamic_patch is None:
        max_dynamic_patch = hf_config.max_dynamic_patch
    use_thumbnail = hf_config.use_thumbnail
    use_MSAC = hf_config.use_msac
    return partial(
        image_to_pixel_values,
        input_size=image_size,
        min_num=min_num,
        max_num=max_dynamic_patch,
        use_thumbnail=use_thumbnail,
        use_MSAC=use_MSAC,
    )


def get_max_internvl_image_tokens(ctx: InputContext,
                                  *,
                                  max_dynamic_patch: Optional[int] = None):
    """
    Calculate the maximum number of tokens with/without MSAC and thumbnail
    """
    hf_config = ctx.get_hf_config()
    use_thumbnail = hf_config.use_thumbnail
    use_MSAC = hf_config.use_msac

    if max_dynamic_patch is None:
        max_dynamic_patch = hf_config.max_dynamic_patch

    num_patches = get_internvl_num_patches(hf_config)

    coefficient = 2 if use_MSAC else 1
    num_blocks = coefficient * max_dynamic_patch + (1 if use_thumbnail else 0)

    return num_blocks * num_patches


class H2OVLInputPipeline(InternVLInputPipeline):

    def __init__(self):
        super().__init__(IMG_START, IMG_END, IMG_CONTEXT)

    def input_processor(
        self,
        ctx: InputContext,
        inputs: DecoderOnlyInputs,
        *,
        max_dynamic_patch: Optional[int] = None,
    ) -> DecoderOnlyInputs:
        multi_modal_data = inputs.get("multi_modal_data")
        if multi_modal_data is None or "image" not in multi_modal_data:
            return inputs

        model_config = ctx.model_config
        hf_config = ctx.get_hf_config()

        image_data = multi_modal_data["image"]
        num_patches = get_internvl_num_patches(hf_config)

        # can only get the total blocks num after the image fully processed
        num_blocks_calculator = image_to_pixel_values_wrapper(
            hf_config, max_dynamic_patch=max_dynamic_patch)

        if isinstance(image_data, Image.Image):
            num_blocks = num_blocks_calculator(image_data).shape[0]
            image_feature_sizes = [num_blocks * num_patches]

        elif is_list_of(image_data, Image.Image):
            # Do not use MSAC for multi images
            hf_config.use_msac = False
            image_feature_sizes = []
            for image in image_data:
                num_blocks = num_blocks_calculator(image).shape[0]
                image_feature_sizes.append(num_blocks * num_patches)

        elif isinstance(image_data, torch.Tensor):
            num_images, image_feature_size, hidden_size = image_data.shape
            image_feature_sizes = [image_feature_size]
        else:
            raise TypeError(f"Invalid image type: {type(image_data)}")

        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )

        prompt = inputs.get("prompt")
        prompt_token_ids = inputs["prompt_token_ids"]
        if prompt is None:
            prompt = tokenizer.decode(prompt_token_ids)

        new_prompt = self._expand_image_prompt(prompt, image_feature_sizes,
                                               num_patches)
        new_prompt_token_ids = tokenizer.encode(new_prompt)

        return token_inputs(
            prompt=prompt,
            prompt_token_ids=new_prompt_token_ids,
            multi_modal_data=multi_modal_data,
        )

    def input_mapper(
        self,
        ctx: InputContext,
        data: object,
        *,
        max_dynamic_patch: Optional[int] = None,
    ):
        hf_config = ctx.get_hf_config()

        image_pixel_values_mapper = image_to_pixel_values_wrapper(
            hf_config, max_dynamic_patch)

        if isinstance(data, Image.Image):
            data = image_pixel_values_mapper(data)
            data = data.unsqueeze(0)
        elif is_list_of(data, Image.Image):
            hf_config.use_msac = False
            data = [image_pixel_values_mapper(img) for img in data]

        else:
            return MultiModalInputs({"image_embeds": data})
        model_config = ctx.model_config
        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        image_token_id = tokenizer.encode(
            self.img_context_token,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]

        return MultiModalInputs({
            "pixel_values": data,
            "image_token_id": image_token_id
        })


input_pipeline = H2OVLInputPipeline()


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_pipeline.input_mapper)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_internvl_image_tokens)
@INPUT_REGISTRY.register_dummy_data(input_pipeline.dummy_data)
@INPUT_REGISTRY.register_input_processor(input_pipeline.input_processor)
class H2OVLChatModel(InternVLChatModel):

    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = (config.vision_config.num_hidden_layers +
                                     vision_feature_layer + 1)
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            msg = "Monolith mode is not applicable to H2OVL"
            raise NotImplementedError(msg)
