# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/image_process.py
import math

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from transformers import AutoProcessor, BatchFeature, LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

# TODO(Isotr0py): change modes for variants
# see: https://github.com/deepseek-ai/DeepSeek-OCR/blob/8cf003d38821fa1b19c73da3bd1b0dc262ea8136/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py#L1-L6
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# TODO(Isotr0py): Expose as mm_kwargs
MIN_CROPS = 2
MAX_CROPS = 6  # max:9; If your GPU memory is small, it is recommended to set it to 6.


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def calculate_aspect_ratios(
    min_num: int = MIN_CROPS, max_num: int = MAX_CROPS
) -> list[tuple[int, int]]:
    target_ratios: set[tuple[int, int]] = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    sorted_target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    return sorted_target_ratios


def count_tiles(
    orig_width,
    orig_height,
    min_num=MIN_CROPS,
    max_num=MAX_CROPS,
    image_size=640,
    use_thumbnail=False,
):
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = calculate_aspect_ratios(min_num, max_num)

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    return target_aspect_ratio


def dynamic_preprocess(
    image, min_num=MIN_CROPS, max_num=MAX_CROPS, image_size=640, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = calculate_aspect_ratios(min_num, max_num)

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

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


class ImageTransform:
    def __init__(
        self,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines = [T.ToTensor()]

        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))

        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        x = self.transform(pil_img)
        return x


class DeepseekOCRProcessor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        self.image_size = IMAGE_SIZE
        self.base_size = BASE_SIZE
        self.patch_size = 16
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = 4

        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # must set this，padding side with make a difference in batch inference # noqa: E501

        # add the pad_token as special token to use 'tokenizer.pad_token'
        # and 'tokenizer.pad_token_id'
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})

        # add image token
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(
            tokenizer,
            **kwargs,
        )

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
        self,
        prompt: str,
        images: list[Image.Image],
        crop_mode: bool = CROP_MODE,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            images (List[ImageType]): the list of images;
            crop_mode (bool): if True, then crop the image;

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - pixel_values (torch.FloatTensor): [n_patches, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert prompt is not None and images is not None, (
            "prompt and images must be used at the same time."
        )

        sft_format = prompt

        (
            input_ids,
            pixel_values,
            images_crop,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
            _,
        ) = self.tokenize_with_images(
            conversation=sft_format,
            images=images,
            bos=True,
            eos=True,
            cropping=crop_mode,
        )

        prepare = BatchFeature(
            data=dict(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_crop=images_crop,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
                num_image_tokens=num_image_tokens,
            ),
            tensor_type="pt",
        )
        return prepare

    def __call__(
        self,
        *,
        prompt: str,
        images: list[Image.Image],
        crop_mode: bool = CROP_MODE,
        **kwargs,
    ):
        prepare = self.process_one(
            prompt=prompt,
            images=images,
            crop_mode=crop_mode,
        )

        return prepare

    def tokenize_with_images(
        self,
        conversation: str,
        images: list[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        """Tokenize text with <image> tags."""

        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        images_list, images_crop_list, images_seq_mask, images_spatial_crop = (
            [],
            [],
            [],
            [],
        )
        image_shapes = []
        num_image_tokens = []
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            image_shapes.append(image.size)

            images_crop_raw = []
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
            elif cropping:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, image_size=IMAGE_SIZE
                )
            else:
                crop_ratio = [1, 1]

            if self.image_size <= 640 and not cropping:
                image = image.resize((self.image_size, self.image_size))

            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            num_width_tiles, num_height_tiles = crop_ratio
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            if num_width_tiles > 1 or num_height_tiles > 1:
                for cropped_image in images_crop_raw:
                    images_crop_list.append(self.image_transform(cropped_image))

            num_queries = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            num_queries_base = math.ceil(
                (self.base_size // self.patch_size) / self.downsample_ratio
            )

            tokenized_image = (
                [self.image_token_id] * num_queries_base + [self.image_token_id]
            ) * num_queries_base
            tokenized_image += [self.image_token_id]
            if num_width_tiles > 1 or num_height_tiles > 1:
                local_row = [self.image_token_id] * (num_queries * num_width_tiles + 1)
                tokenized_image += local_row * (num_queries * num_height_tiles)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))

        """process the last text split"""
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos and eos tokens"""
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(images_seq_mask), (
            f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} "
            f"is not equal to images_seq_mask's length {len(images_seq_mask)}."
        )

        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)

        assert (
            len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)
        ), (
            f"tokenized_str's length {len(tokenized_str)}, "
            f"input_ids' length {len(masked_tokenized_str)}, "
            f"images_seq_mask's length {len(images_seq_mask)}, are not equal."
        )

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = (
            self.ignore_id
        )
        input_ids[input_ids < 0] = self.pad_id

        # Remove the ending eos token
        assert input_ids[-1] == self.eos_id
        input_ids = input_ids[:-1]
        target_ids = target_ids[:-1]
        images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            pixel_values = torch.zeros((0, 3, self.base_size, self.base_size))
            images_spatial_crop = torch.zeros((0, 2), dtype=torch.long)
            images_crop = torch.zeros((0, 3, self.image_size, self.image_size))
        else:
            pixel_values = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((0, 3, self.image_size, self.image_size))

        input_ids = input_ids.unsqueeze(0)

        return (
            input_ids,
            pixel_values,
            images_crop,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
            image_shapes,
        )


AutoProcessor.register("DeepseekOCRProcessor", DeepseekOCRProcessor)
