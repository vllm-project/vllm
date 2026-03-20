# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product
from math import ceil

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BatchFeature, ProcessorMixin, TensorType

from vllm.tokenizers import TokenizerLike

MAX_IMAGE_SIZE: int = 3024

ImageWithPatches = tuple[Image.Image, list[Image.Image], list[bool]]


class Step3VisionProcessor:
    def __init__(self, size, interpolation_mode="bicubic", patch_size=None):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = patch_size if patch_size is not None else size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize(
                    (size, size),
                    interpolation=InterpolationMode.BICUBIC
                    if interpolation_mode == "bicubic"
                    else InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ]
        )

        self.patch_transform = (
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.Resize(
                        (patch_size, patch_size),
                        interpolation=InterpolationMode.BICUBIC
                        if interpolation_mode == "bicubic"
                        else InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ]
            )
            if patch_size is not None
            else None
        )

    def __call__(self, image, is_patch=False):
        if is_patch:
            assert self.patch_transform is not None
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}

        return {"pixel_values": self.transform(image).unsqueeze(0)}


class ImagePatcher:
    def __init__(self, enable_patch: bool = True) -> None:
        self.enable_patch = enable_patch

    def determine_window_size(self, long: int, short: int) -> int:
        if long < 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    def slide_window(
        self,
        width: int,
        height: int,
        sizes: list[tuple[int, int]],
        steps: list[tuple[int, int]],
        img_rate_thr: float = 0.6,
    ) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
        assert 1 >= img_rate_thr >= 0, "The `in_rate_thr` should lie in 0~1"
        windows = []
        # Sliding windows.
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step

            x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w

            y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h

            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)

        return [
            (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
            for box in windows
        ], (x_num, y_num)

    def square_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    def get_image_size_for_padding(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:
        ratio = img_width / img_height
        if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(img_height, img_width)
            return new_size, new_size
        return img_width, img_height

    def get_image_size_for_preprocess(
        self, img_width: int, img_height: int
    ) -> tuple[int, int]:
        if max(img_height, img_width) > MAX_IMAGE_SIZE:
            scale_factor = MAX_IMAGE_SIZE / max(img_height, img_width)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
        return img_width, img_height

    def get_image_size_for_crop(
        self, img_width: int, img_height: int, window_size: int
    ):
        w_ratio = img_width / window_size
        h_ratio = img_height / window_size

        if w_ratio < 1:
            width_new = img_width
        else:
            decimal_w = w_ratio - img_width // window_size
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window_size * w_ratio
        if h_ratio < 1:
            height_new = img_height
        else:
            decimal_h = h_ratio - img_height // window_size
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window_size * h_ratio
        return int(width_new), int(height_new)

    def patch_crop(self, img: Image.Image, i: int, j: int, th: int, tw: int):
        target = img.crop((j, i, j + tw, i + th))
        return target

    def get_num_patches(self, img_width: int, img_height: int) -> tuple[int, int]:
        img_width, img_height = self.get_image_size_for_padding(img_width, img_height)
        img_width, img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        window_size = self.determine_window_size(
            max(img_height, img_width), min(img_height, img_width)
        )
        if window_size == 0 or not self.enable_patch:
            return 0, 0
        else:
            img_width, img_height = self.get_image_size_for_crop(
                img_width, img_height, window_size
            )
            center_list, (x_num, y_num) = self.slide_window(
                img_width,
                img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            full_rows = (len(center_list) - 1) // x_num + 1
            if len(center_list) > 0 and len(center_list) % x_num == 0:
                full_rows -= 1
            return len(center_list), full_rows

    def __call__(
        self, img: Image.Image
    ) -> tuple[Image.Image, list[Image.Image], list[bool]]:
        img_width, img_height = img.size
        new_img_width, new_img_height = self.get_image_size_for_padding(
            img_width, img_height
        )
        if new_img_width != img_width or new_img_height != img_height:
            img = self.square_pad(img)
            img_width, img_height = img.size

        new_img_width, new_img_height = self.get_image_size_for_preprocess(
            img_width, img_height
        )
        img = img.resize((new_img_width, new_img_height), Image.Resampling.BILINEAR)
        window_size = self.determine_window_size(
            max(new_img_height, new_img_width), min(new_img_height, new_img_width)
        )

        if window_size == 0 or not self.enable_patch:
            return img, [], []
        else:
            new_img_width, new_img_height = self.get_image_size_for_crop(
                new_img_width, new_img_height, window_size
            )
            if (new_img_width, new_img_height) != (img_width, img_height):
                img_for_crop = img.resize(
                    (new_img_width, new_img_height), Image.Resampling.BILINEAR
                )
            else:
                img_for_crop = img

            patches = []
            newlines = []
            center_list, (x_num, y_num) = self.slide_window(
                new_img_width,
                new_img_height,
                [(window_size, window_size)],
                [(window_size, window_size)],
            )
            for patch_id, center_lf_point in enumerate(center_list):
                x, y, patch_w, patch_h = center_lf_point
                big_patch = self.patch_crop(img_for_crop, y, x, patch_h, patch_w)
                patches.append(big_patch)
                if (patch_id + 1) % x_num == 0:
                    newlines.append(patch_id)

            if newlines and newlines[-1] == len(patches) - 1:
                newlines.pop()

            return (
                img,
                patches,
                [i in newlines for i in range(len(patches))],
            )


class Step3VLImageProcessor:
    def __init__(
        self,
        image_size: int = 728,
        patch_size: int = 504,
        num_image_feature_size: int = 169,
        num_patch_feature_size: int = 81,
        enable_patch: bool = True,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_image_feature_size = num_image_feature_size
        self.num_patch_feature_size = num_patch_feature_size
        self.image_preprocessor = Step3VisionProcessor(
            image_size, "bilinear", patch_size
        )
        self.patcher = ImagePatcher(enable_patch=enable_patch)

    def get_num_image_tokens(self, img_width: int, img_height: int) -> int:
        num_patches, num_newlines = self.patcher.get_num_patches(img_width, img_height)

        return (
            num_patches * (self.num_patch_feature_size + 2)
            + self.num_image_feature_size
            + 2
            + num_newlines
        )

    def _split_images(self, images: list[Image.Image]) -> list[ImageWithPatches]:
        result = []
        for img in images:
            result.append(self.patcher(img))
        return result

    def _convert_images_to_pixel_values(
        self,
        images: list[Image.Image],
        is_patch: bool = False,
    ) -> list[torch.Tensor]:
        return [
            self.image_preprocessor(img, is_patch=is_patch)["pixel_values"]
            for img in images
        ]

    def __call__(
        self,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        split_images_data = self._split_images(images)
        pixel_values_lst = []
        patch_pixel_values_lst = []
        patch_newline_mask_lst = []
        num_patches = []
        for raw_img, img_patches, patch_newline_mask in split_images_data:
            pixel_values_lst.extend(self._convert_images_to_pixel_values([raw_img]))
            num_patches.append(len(img_patches))
            patch_pixel_values_lst.extend(
                self._convert_images_to_pixel_values(img_patches, is_patch=True)
            )
            patch_newline_mask_lst.extend(patch_newline_mask)

        pixel_values = torch.cat(pixel_values_lst)
        patch_size = self.patch_size
        image_inputs = {
            "pixel_values": pixel_values,
            "num_patches": num_patches,
            "patch_pixel_values": (
                torch.cat(patch_pixel_values_lst)
                if patch_pixel_values_lst
                else pixel_values.new_empty((0, 3, patch_size, patch_size))
            ),
            "patch_newline_mask": torch.tensor(
                patch_newline_mask_lst, dtype=torch.bool
            ),
        }
        return BatchFeature(image_inputs, tensor_type=return_tensors)


class Step3VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: Step3VLImageProcessor,
        tokenizer: TokenizerLike,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.image_start_token = image_start_token = "<im_start>"
        self.image_end_token = image_end_token = "<im_end>"
        self.patch_start_token = patch_start_token = "<patch_start>"
        self.patch_end_token = patch_end_token = "<patch_end>"
        self.patch_newline_token = patch_newline_token = "<patch_newline>"
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(image_end_token)
        self.patch_start_token_id = tokenizer.convert_tokens_to_ids(patch_start_token)
        self.patch_end_token_id = tokenizer.convert_tokens_to_ids(patch_end_token)
        self.patch_newline_token_id = tokenizer.convert_tokens_to_ids(
            patch_newline_token
        )

        self.image_token = image_token = "<im_patch>"
        self.image_feature_tokens = image_token * image_processor.num_image_feature_size
        self.patch_feature_tokens = image_token * image_processor.num_patch_feature_size

        self.image_token_id = image_token_id = tokenizer.convert_tokens_to_ids(
            image_token
        )
        self.image_feature_token_ids = [
            image_token_id
        ] * image_processor.num_image_feature_size
        self.patch_feature_token_ids = [
            image_token_id
        ] * image_processor.num_patch_feature_size

    def _get_patch_repl_text(
        self,
        num_patches: int,
        patch_newline_mask: list[bool],
    ) -> str:
        assert len(patch_newline_mask) == num_patches

        parts = []
        for i in range(num_patches):
            parts.extend(
                [
                    self.patch_start_token,
                    self.patch_feature_tokens,
                    self.patch_end_token,
                ]
            )
            if patch_newline_mask[i]:
                parts.append(self.patch_newline_token)

        return "".join(parts)

    def _get_patch_repl_ids(
        self,
        num_patches: int,
        patch_newline_mask: list[bool],
    ) -> list[int]:
        assert len(patch_newline_mask) == num_patches

        parts = []
        for i in range(num_patches):
            parts.extend(
                [
                    self.patch_start_token_id,
                    *self.patch_feature_token_ids,
                    self.patch_end_token_id,
                ]
            )
            if patch_newline_mask[i]:
                parts.append(self.patch_newline_token_id)

        return parts

    def _get_image_repl_text(
        self,
        num_images: int,
    ) -> str:
        parts = [
            self.image_start_token,
            self.image_feature_tokens,
            self.image_end_token,
        ] * num_images

        return "".join(parts)

    def _get_image_repl_ids(
        self,
        num_images: int,
    ) -> list[int]:
        part = [
            self.image_start_token_id,
            *self.image_feature_token_ids,
            self.image_end_token_id,
        ]
        return part * num_images

    def get_image_repl_feature_text(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: list[bool],
    ) -> str:
        patch_repl = self._get_patch_repl_text(num_patches, patch_new_line_idx)
        image_repl = self._get_image_repl_text(num_images)
        return patch_repl + image_repl

    def get_image_repl_feature_ids(
        self,
        num_images: int,
        num_patches: int,
        patch_new_line_idx: list[bool],
    ) -> list[int]:
        patch_repl = self._get_patch_repl_ids(num_patches, patch_new_line_idx)
        image_repl = self._get_image_repl_ids(num_images)
        return patch_repl + image_repl

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)

        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                return_tensors=return_tensors,
            )
            num_patches = image_inputs["num_patches"]
            patch_newline_mask = image_inputs["patch_newline_mask"]
        else:
            image_inputs = {}
            num_patches = []
            patch_newline_mask = []

        if text is not None:
            if not isinstance(text, list):
                text = [text]

            if image_inputs:
                image_repl_str_lst = []
                start = 0
                for n_patches in num_patches:
                    image_repl_str = self.get_image_repl_feature_text(
                        1, n_patches, patch_newline_mask[start : start + n_patches]
                    )
                    image_repl_str_lst.append(image_repl_str)

                    start += n_patches

                text = [
                    self.replace_placeholder(t, self.image_token, image_repl_str_lst)
                    for t in text
                ]

            text_inputs = self.tokenizer(text)
        else:
            text_inputs = {}

        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )
