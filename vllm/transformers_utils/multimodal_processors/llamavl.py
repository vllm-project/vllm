from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor

import torch
from typing import List, Tuple
from PIL import Image
from functools import partial

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import math
from functools import reduce
from typing import Any, Tuple

import numpy as np
import torch
import torchvision.transforms as tv
from PIL import Image
from torchvision.transforms import functional as F

IMAGE_RES = 224

class TorchBF16Context:

    def __enter__(self):
        self.prev_dtype = torch.get_default_dtype()
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev_dtype == torch.float32:
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            raise ValueError("Unsupported dtype")

class VariableSizeImageTransform(object):
    """
    The variable size image transform will resize the image dynamically
    based on the image aspect ratio and the number of image chunks we allow.
    The algorithm will not upsample low-res images to fit a certain aspect
    ratio, because that leads to a significant degradation in image quality.
    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image chunks, it will find the closest aspect ratio that
    is allowed within 16 image chunks, i.e., 2:5 = 2 horizontal patches and
    5 vertical patches, giving a total of 10 chunks.
    The image will then be resized to products of the base size (default is
    224px because MetaCLIP takes that), so in this case it will  be resized to
    2*224:5*224 = 448:1120, where we maintain the original aspect ratio and
    pad with the mean value for the rest. This approach minimizes the amount
    of padding required for any arbitrary resolution.
    The final output will therefore be of shape (11, 3, 224, 224), where 10
    patches are coming from the resizing and chunking, and the first patch
    is a downsampled version of the image that preserves aspect ratios.
    """

    def __init__(self, size: int = IMAGE_RES) -> None:
        self.size = size
        self.to_tensor = tv.ToTensor()
        self._mean = (0.48145466, 0.4578275, 0.40821073)
        self._std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = tv.Normalize(
            mean=self._mean,
            std=self._std,
            inplace=True,
        )

    @staticmethod
    def _factors(n: int):
        """Return all factors of a number."""
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )

    def _find_supported_aspect_ratios(self, num_chunks: int):
        """
        This function computes all the allowed aspect ratios for a fixed
        number of input chunks.
        For example, with `num_chunks=5`, it will return:
        {
            0.2: [(1, 5)],
            5.0: [(5, 1)],
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.3333333333333333: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
        }
        """
        asp_dict = {}
        for chunk_size in range(num_chunks, 0, -1):
            _factors = sorted(VariableSizeImageTransform._factors(chunk_size))
            _asp_ratios = [(x, chunk_size // x) for x in _factors]
            for ratio in _asp_ratios:
                k = ratio[0] / ratio[1]
                if k not in asp_dict:
                    asp_dict[k] = [ratio]
                else:
                    asp_dict[k].append(ratio)
        return asp_dict

    def _find_closest_aspect_ratio(
        self, num_chunks: int, img_width: int, img_height: int
    ) -> Tuple:
        """
        Given an image width, height and target number of chunks
        this function will find the closest supported aspect ratio.
        """
        tgt_ar = img_width / img_height
        asp_dict = self._find_supported_aspect_ratios(num_chunks)
        cl_d, cl_p = 1e23, None
        if tgt_ar >= 1:
            cl_p = min(
                [k for k in asp_dict.keys() if k <= tgt_ar],
                key=lambda x: abs(x - tgt_ar),
            )
            v = asp_dict[cl_p]
            # select width
            widths = [(idx, self.size * vv[0]) for idx, vv in enumerate(v)]
            tgt_idx = max(widths, key=lambda x: x[1])[0]
        else:
            cl_p = min(
                [k for k in asp_dict.keys() if k > tgt_ar],
                key=lambda x: abs(1 / x - 1 / tgt_ar),
            )
            v = asp_dict[cl_p]
            # select height
            heights = [(idx, self.size * vv[1]) for idx, vv in enumerate(v)]
            tgt_idx = max(heights, key=lambda x: x[1])[0]
        out = v[tgt_idx]
        return out

    def _resize(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        # Resize longer edge to given size.
        w, h = image.size
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = target_width
            new_h = math.floor(new_w / scale)
        else:
            # height >= width
            new_h = target_height
            new_w = math.floor(new_h * scale)

        image = F.resize(image, (new_h, new_w))
        return image

    def _resize_max_side_to_size(
        self,
        image: Image.Image,
    ) -> Image.Image:
        # Resize longer edge to given size.
        w, h = image.size
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = max(self.size, w)
            new_h = math.floor(new_w / scale)
        else:
            # height >= width
            new_h = max(self.size, h)
            new_w = math.floor(new_h * scale)

        image = F.resize(image, (new_h, new_w))
        return image

    def _pad(self, image: Image.Image, new_width: int, new_height: int) -> Image.Image:
        mean_per_channel = tuple(
            np.clip(np.array(image).mean(axis=(0, 1)), 0, 255).astype(np.uint8)
        )
        new_im = Image.new(mode="RGB", size=(new_height, new_width), color=(0, 0, 0))  # type: ignore
        new_im.paste(image)
        return new_im

    def _split(self, image: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        # Split image into number of required tiles (width x height)
        num_channels, height, width = image.size()
        image = image.view(num_channels, nch, height // nch, ncw, width // ncw)
        # Permute dimensions to reorder the axes
        image = image.permute(1, 3, 0, 2, 4).contiguous()
        # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
        image = image.view(ncw * nch, num_channels, height // nch, width // ncw)
        return image

    def _fit_image_to_canvas(
        self, num_chunks: int, img_width: int, img_height: int
    ) -> Any:
        """
        Given an image width, height and target number of chunks this function will see if the image
        can be fit into any of the canvases that can be build from arranging the tiles in a grid.
        If the image can be fit onto several canvases, it will return the canvas where the shorter edge
        of the image will be largest.
        """
        # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
        optimal_canvas = None
        optimal_image_width_height = None

        scale = img_width / img_height

        # Gather all potential supported image resolutions and iterate through them to find best match
        potential_arrangements = [
            item
            for sublist in self._find_supported_aspect_ratios(num_chunks).values()
            for item in sublist
        ]
        current_gap = 1e23
        for n_w, n_h in potential_arrangements:
            # Compute the canvas size
            canvas_width, canvas_height = n_w * self.size, n_h * self.size

            # Check if image can fit into the canvas without downsampling
            if canvas_width >= img_width and canvas_height >= img_height:
                # If we did not find a good canvas yet, we will use the current one
                if optimal_canvas is None:
                    # Set optimal canvas and determine the actual image height and width in the canvas with aspect ratio preserving resampling
                    optimal_canvas = (n_w, n_h)
                    optimal_image_width_height = (n_w * self.size, n_h * self.size)
                else:
                    # Find closest fit based on gap
                    image_width_height = (n_w * self.size, n_h * self.size)
                    gap = abs(img_width - image_width_height[0]) + abs(
                        img_height - image_width_height[1]
                    )
                    if gap < current_gap:
                        # If the gap is smaller than the previous one, we will update our optimal canvas and image width height
                        optimal_canvas = (n_w, n_h)
                        optimal_image_width_height = image_width_height
                        current_gap = gap
        return optimal_canvas

    def __call__(self, image: Image.Image, max_num_chunks: int) -> Tuple[Any, Any]:
        assert max_num_chunks > 0
        assert isinstance(image, Image.Image), type(image)

        import numpy as np
        w, h = image.size
        # Check if the image can be fit to the canvas without downsampling
        ar = self._fit_image_to_canvas(
            num_chunks=max_num_chunks, img_width=w, img_height=h
        )
        if ar is None:
            # If we did not find a canvas, we have to find the closest aspect ratio and downsample the image
            ar = self._find_closest_aspect_ratio(
                num_chunks=max_num_chunks, img_width=w, img_height=h
            )
            image = self._resize(image, ar[0] * self.size, ar[1] * self.size)
        else:
            image = self._resize_max_side_to_size(image)

            arr = np.array(image)
        
        image = self._pad(image, ar[1] * self.size, ar[0] * self.size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self._split(image, ar[0], ar[1])  # type: ignore
        return image, ar


def _stack_images(
    images: List[List[Image.Image]],
    max_num_chunks: int,
    image_res: int,
    max_num_images: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes a list of list of images and stacks them into a tensor.
    This function is needed since images can be of completely
    different resolutions and aspect ratios.
    """
    out_images, out_num_chunks = [], []
    for imgs_sample in images:
        out_images_i = torch.zeros(
            max_num_images,
            max_num_chunks,
            3,
            image_res,
            image_res,
        )
        _num_chunks = []
        for j, chunks_image in enumerate(imgs_sample):
            out_images_i[j, : chunks_image.shape[0]] = chunks_image
            _num_chunks.append(chunks_image.shape[0])
        out_images.append(out_images_i)
        out_num_chunks.append(_num_chunks)
    return torch.stack(out_images), out_num_chunks

class LlamaVLImageProcessor(BaseImageProcessor):
    def __init__(self, name, *args, **kwargs):
        if "11B" in name:
            self.vision_chunk_size = 448
        elif "90B" in name:
            self.vision_chunk_size = 560
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.vision_max_num_chunks = 4
        self.max_num_chunks = self.vision_max_num_chunks
        self.image_transform = partial(
            VariableSizeImageTransform(size=self.vision_chunk_size),
            max_num_chunks=self.vision_max_num_chunks,
        )
    def preprocess(self, images, **kwargs) -> BatchFeature:
        with TorchBF16Context():
            print("[warning] mask unsupported due to lack of example, replace with official release in the future")
            # assert len(images) == len(
            #     batch_masks
            # ), "Images and masks must have the same length"

            # preprocess is called for each batch now, so add batch dimension here.
            images = [images]

            max_num_images = max(len(x) for x in images)
            bsz = len(images)

            if max_num_images == 0:
                data = {'pixel_values': None}
            else:
                images_and_aspect_ratios = [
                    [self.image_transform(im) for im in row] for row in images
                ]
                transformed_images = [
                    [x[0] for x in row] for row in images_and_aspect_ratios
                ]

                aspect_ratios = torch.ones(bsz, max_num_images, 2, dtype=torch.int64)
                for i, row in enumerate(images_and_aspect_ratios):
                    if len(row) > 0:
                        aspect_ratios[i, : len(row)] = torch.stack(
                            [torch.tensor(x[1]) for x in row]
                        )
                assert bsz == 1, "the below code is not for batched images"
                data = {
                    'pixel_values': transformed_images[0],
                    'aspect_ratios': aspect_ratios[0],
                }
                # print("transformed_images", transformed_images)
                # for i, row in enumerate(transformed_images):
                #     for j, x in enumerate(row):
                #         print(i, j, x.shape)
                # print("aspect_ratios", aspect_ratios)
                # stacked_images, num_chunks = _stack_images(
                #     transformed_images,
                #     self.vision_max_num_chunks,
                #     self.vision_chunk_size,
                #     max_num_images,
                # )
                # print("stacked_images", stacked_images.shape)
                # print("num_chunks", num_chunks)
        return BatchFeature(data, tensor_type=None)