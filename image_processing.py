# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
import math

import torch
from einops import rearrange
from torchvision import transforms as T
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode


IMAGENET_PIXEL_MEAN = [0.485, 0.456, 0.406]
IMAGENET_PIXEL_STD = [0.229, 0.224, 0.225]
SIGLIP_PIXEL_MEAN = [0.5, 0.5, 0.5]
SIGLIP_PIXEL_STD = [0.5, 0.5, 0.5]
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
RADIO_G_PIXEL_MEAN = [0.4850, 0.4560, 0.4060]
RADIO_G_PIXEL_STD = [0.2230, 0.2240, 0.2250]


pixel_statistics = {
    "clip": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "siglip": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
    "internvit": (IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD),
    "radio": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "radio-g": (RADIO_G_PIXEL_MEAN, RADIO_G_PIXEL_STD),
    "cradio-g": (CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    "internvit300M": (IMAGENET_PIXEL_MEAN, IMAGENET_PIXEL_STD),
    "huggingface": (SIGLIP_PIXEL_MEAN, SIGLIP_PIXEL_STD),
}


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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


def find_closest_area_weighted_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the best number of tiles based on the aspect ratio and the area covered by the tiles.
    """
    best_factor = float('-inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        factor_based_on_area_n_ratio = (
            min((ratio[0]*ratio[1]*image_size*image_size)/ area, 0.6) *
            min(target_aspect_ratio/aspect_ratio, aspect_ratio/target_aspect_ratio))
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio
    return best_ratio


def process_images(sample_imgs, patch_dim, dynamic_resolution, batch_mode=False):
    """Process a batch of images for multimodal training or evaluation.
    
    This function handles image preprocessing with support for both static and dynamic 
    resolution processing. For dynamic resolution, it rearranges images into patches 
    and computes cumulative sequence lengths for efficient batching.
    
    Args:
        sample_imgs (List[torch.Tensor]): List of image tensors with shape (C, H, W).
        patch_dim (int): Dimension of each patch (e.g., 14 for 14x14 patches).
        dynamic_resolution (bool): Whether to use dynamic resolution processing.
            If True, images are rearranged into patches with variable sequence lengths.
            If False, images are simply stacked into a batch tensor.
        batch_mode (bool, optional): Whether this is being called from training batch processing.
            If True, wraps tensors in additional list dimension for consistency with batch format.
            If False, returns tensors directly as used in evaluation. Defaults to False.
    
    Returns:
        tuple: A 4-tuple containing:
            - images (torch.Tensor): Processed image tensor.
                For dynamic resolution: shape (1, total_patches, patch_features) if batch_mode=False,
                                       or shape (1, total_patches, patch_features) if batch_mode=True
                For static resolution: shape (batch_size, C, H, W)
            - imgs_sizes (torch.Tensor): Image sizes tensor with shape (N, 2)
                containing [width, height] for each image, or [[0,0]] if no images.
            - vision_cu_lengths (torch.Tensor or None): Cumulative sequence lengths
                for dynamic resolution. Shape (batch_size + 1,) for evaluation mode,
                or shape (1, batch_size + 1) for batch mode. None for static resolution.
            - vision_max_lengths (torch.Tensor or None): Maximum sequence length
                among all images for dynamic resolution. Scalar tensor for evaluation mode,
                or shape (1,) for batch mode. None for static resolution.
    
    Note:
        This function is designed for processing one microbatch at a time for dynamic resolution.
    """
    vision_cu_lengths = None
    vision_max_lengths = None
    
    if len(sample_imgs) > 0:
        imgs_sizes = torch.tensor([[img.shape[1], img.shape[2]] for img in sample_imgs], dtype=torch.int32)
        if dynamic_resolution:
            def rearrange_img(x):
                py = x.shape[-2] // patch_dim
                px = x.shape[-1] // patch_dim
                x = rearrange(x, 'c (py yy) (px xx) -> (py px) (c yy xx)',
                                    py=py, yy=patch_dim,
                                    px=px, xx=patch_dim,
                )
                return x
            imgs = [rearrange_img(img) for img in sample_imgs]

            current_length = 0
            max_length = 0
            vision_cu_lengths = [0]
            for img in imgs:
                if max_length < img.shape[0]:
                    max_length = img.shape[0] 
                current_length += img.shape[0]
                vision_cu_lengths.append(current_length)
            
            vision_cu_lengths = torch.tensor(vision_cu_lengths, dtype=torch.int32)
            vision_max_lengths = torch.tensor(max_length, dtype=torch.int32)
            
            # For batch mode, wrap in additional dimension for consistency
            if batch_mode:
                vision_cu_lengths = vision_cu_lengths.unsqueeze(0)  # Shape: (1, batch_size + 1)
                vision_max_lengths = vision_max_lengths.unsqueeze(0)  # Shape: (1,)
            
            imgs = torch.cat(imgs, dim=0)
            images = imgs.unsqueeze(0)
        else:
            images = torch.stack(sample_imgs)
    else:
        imgs_sizes = torch.tensor([[0,0]], dtype=torch.int32)
        if len(sample_imgs) == 0 and batch_mode:
            # For batch mode when no images, use appropriate dummy tensor
            images = torch.tensor([[0]], dtype=torch.float32)
        else:
            images = torch.stack(sample_imgs)

    return images, imgs_sizes, vision_cu_lengths, vision_max_lengths


class ImageTransform:
    """Image transformation."""

    def __init__(self, input_size, vision_model_type, *, dynamic_resolution=False, res_step=16, min_num_patches=1, max_num_patches=128,  pixel_shuffle=False, min_side=None, conv_merging=False, match_tiling_dynamic_resolution=False, masked_tiling_dynamic_resolution=False, thumbnail_area_threshold=0.8):
        self._transform = _build_transform(input_size, vision_model_type)
        self._vision_model_type = vision_model_type
        self._dynamic_resolution = dynamic_resolution
        self._res_step = res_step
        self._min_num_patches = min_num_patches
        self._max_num_patches = max_num_patches
        self._pixel_shuffle = pixel_shuffle
        self._min_side = min_side
        self._conv_merging = conv_merging
        self._match_tiling_dynamic_resolution = match_tiling_dynamic_resolution
        self._masked_tiling_dynamic_resolution = masked_tiling_dynamic_resolution
        self._thumbnail_area_threshold = thumbnail_area_threshold

    def __call__(self, img, img_h, img_w, use_tiling=False, max_num_tiles=1, use_thumbnail=False, augment=False, find_closest_aspect_ratio_fn=find_closest_aspect_ratio, is_video=False):
        assert not augment, "Image augmentation not implemented."
        if use_tiling:
            assert img_h == img_w, "dynamic tiling expects equal tile height and width"
            imgs = dynamic_preprocess(
                img, min_num=1, max_num=max_num_tiles, image_size=img_h, use_thumbnail=use_thumbnail,
                find_closest_aspect_ratio_fn=find_closest_aspect_ratio_fn)
            imgs = [self._transform(img) for img in imgs]
        elif self._masked_tiling_dynamic_resolution:
            assert img_h == img_w, "masked tiling dynamic resolution expects equal tile height and width"
            assert "radio" in self._vision_model_type, "Masked tiling dynamic resolution is only supported for radio models"

            # Use tiling logic to determine tile grid (nx, ny)
            orig_width, orig_height = img.size
            aspect_ratio = orig_width / orig_height

            target_ratios = set(
                (i, j) for n in range(1, max_num_tiles + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                if i * j <= max_num_tiles and i * j >= 1
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            tiling = find_closest_aspect_ratio_fn(
                aspect_ratio, target_ratios, orig_width, orig_height, img_h
            )

            # Resize and split into tiles of size (img_h x img_h)
            target_width = img_h * tiling[0]
            target_height = img_w * tiling[1]
            blocks = tiling[0] * tiling[1]

            resized_img = img.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // img_h)) * img_h,
                    (i // (target_width // img_h)) * img_h,
                    ((i % (target_width // img_h)) + 1) * img_h,
                    ((i // (target_width // img_h)) + 1) * img_h,
                )
                tile_img = resized_img.crop(box)
                processed_images.append(tile_img)
            assert len(processed_images) == blocks

            # Optional thumbnail
            if use_thumbnail and blocks != 1:
                thumbnail_img = img.resize((img_h, img_h))
                processed_images.append(thumbnail_img)

            pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ])
            imgs = [transform(im) for im in processed_images]
        elif self._match_tiling_dynamic_resolution:
            assert img_h == img_w, "match tiling dynamic resolution expects equal tile height and width"
            assert "radio" in self._vision_model_type, "Match tiling dynamic resolution is only supported for radio models"
            
            # Use tiling logic to determine optimal dimensions
            orig_width, orig_height = img.size
            aspect_ratio = orig_width / orig_height
            
            # Calculate target ratios (same logic as tiling)
            target_ratios = set(
                (i, j) for n in range(1, max_num_tiles + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num_tiles and i * j >= 1)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            
            # Find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio_fn(
                aspect_ratio, target_ratios, orig_width, orig_height, img_h)
            
            # Calculate the target width and height using tiling logic
            target_width = img_h * target_aspect_ratio[0]
            target_height = img_w * target_aspect_ratio[1]
            
            # Resize image to target dimensions (same as tiling, but don't split)
            resized_img = img.resize((target_width, target_height))
            
            # Process as single dynamic resolution image
            pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ])
            processed_images = [resized_img]
            
            # Add thumbnail if use_thumbnail=True and there's more than 1 tile
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
            if use_thumbnail and blocks != 1:
                thumbnail_img = img.resize((img_h, img_h))
                processed_images.append(thumbnail_img)
            
            imgs = [transform(img) for img in processed_images]
        elif self._dynamic_resolution:
            pixel_mean, pixel_std = pixel_statistics[self._vision_model_type]
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.ToTensor(),
                T.Normalize(mean=pixel_mean, std=pixel_std),
            ])
            processed_img = dynamic_res_preprocess(img, min_patches=self._min_num_patches, max_patches=self._max_num_patches, res_step=self._res_step, pixel_shuffle=self._pixel_shuffle, min_side=self._min_side, conv_merging=self._conv_merging, is_video=is_video)
            processed_images = [processed_img]
            
            # Add thumbnail if enabled and image area is below threshold
            if use_thumbnail:
                # Calculate areas
                processed_width, processed_height = processed_img.size
                resized_area = processed_width * processed_height
                thumbnail_area = img_h * img_h  # img_h should be square thumbnail size
                area_ratio = resized_area / thumbnail_area
                
                # Only add thumbnail if resized image area is less than threshold % of thumbnail area
                if area_ratio < self._thumbnail_area_threshold:
                    thumbnail_img = img.resize((img_h, img_h))  # Use square thumbnail with img_h size
                    processed_images.append(thumbnail_img)
            
            imgs = [transform(img) for img in processed_images]
        else:
            imgs = [self._transform(img)]

        return imgs


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L702
# Copyright (c) 2023 OpenGVLab.
def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False,
    find_closest_aspect_ratio_fn=find_closest_aspect_ratio):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_fn(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_res_preprocess(image, min_patches=1, max_patches=128, res_step=16, factor_max=1., pixel_shuffle=False, min_side=None, conv_merging=False, is_video=False):
    """Preprocess an image with dynamic resolution for vision transformers.
    
    This function resizes an image to optimize the number of patches while respecting
    constraints on minimum/maximum patches, minimum side length, and compatibility
    with pixel shuffle or convolution merging operations.
    
    The algorithm works by:
    1. Computing the initial patch grid size based on the image dimensions and res_step
    2. Scaling the patch grid to fit within the max_patches constraint
    3. Ensuring the result has at least min_patches
    4. Optionally enforcing a minimum side length constraint
    5. Rounding patch dimensions to even numbers for pixel_shuffle/conv_merging compatibility
    6. Resizing the image to the computed target dimensions
    
    Args:
        image (PIL.Image): Input image to preprocess.
        min_patches (int, optional): Minimum number of patches required. Defaults to 1.
        max_patches (int, optional): Maximum number of patches allowed. Defaults to 128.
        res_step (int, optional): Resolution step size (patch dimension). Defaults to 16.
        factor_max (float, optional): Maximum scaling factor to apply. Defaults to 1.0.
        pixel_shuffle (bool, optional): Whether to ensure compatibility with pixel shuffle
            operations by rounding to even patch dimensions. Defaults to False.
        min_side (int, optional): Minimum side length in pixels. If specified, ensures
            at least one side meets this constraint. Defaults to None.
        conv_merging (bool, optional): Whether to ensure compatibility with convolution
            merging by rounding to even patch dimensions. Defaults to False.
    
    Returns:
        PIL.Image: Resized image with dimensions optimized for patch-based processing.
            The output dimensions will be (target_patch_width * res_step, target_patch_height * res_step).
    
    Note:
        The function preserves aspect ratio as much as possible while satisfying all constraints.
        When constraints conflict (e.g., min_side vs max_patches), the function prioritizes
        staying within max_patches while maximizing the image size.
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("example.jpg")  # 800x600 image
        >>> resized_img = dynamic_res_preprocess(img, min_patches=4, max_patches=64, res_step=14)
        >>> # Returns image resized to maintain aspect ratio with 4-64 patches of size 14x14
    """
    orig_width, orig_height = image.size

    closest_patch_height = round(orig_height / res_step + 0.5)
    closest_patch_width = round(orig_width / res_step + 0.5)
    patches = closest_patch_height * closest_patch_width

    factor = min(math.sqrt(max_patches / patches), factor_max)
    target_patch_height = math.floor(factor * closest_patch_height)
    target_patch_width = math.floor(factor * closest_patch_width)

    if target_patch_height * target_patch_width < min_patches:
        up_factor = math.sqrt(min_patches / (target_patch_height * target_patch_width))
        target_patch_height = math.ceil(up_factor * target_patch_height)
        target_patch_width = math.ceil(up_factor * target_patch_width)

    if min_side is not None and min(target_patch_width, target_patch_height) * res_step < min_side:
        if target_patch_width <= target_patch_height:
            up_factor = min_side / (target_patch_width * res_step)
            new_patch_height = math.ceil(up_factor * target_patch_height)
            new_patch_width = math.ceil(up_factor * target_patch_width)

            if new_patch_height * new_patch_width > max_patches:
                # If only one side can be min_side, make as big as possible at native aspect ratio while staying below max_patches
                if max(max_patches // new_patch_width, 1) * res_step < min_side:
                    up_factor = math.sqrt(max_patches / (target_patch_height * target_patch_width))
                    target_patch_height = math.floor(up_factor * target_patch_height)
                    target_patch_width = math.floor(up_factor * target_patch_width)
                target_patch_width = new_patch_width
                target_patch_height = max(max_patches // new_patch_width, 1)
            else:
                target_patch_height = new_patch_height
                target_patch_width = new_patch_width
        else:
            up_factor = min_side / (target_patch_height * res_step)
            new_patch_height = math.ceil(up_factor * target_patch_height)
            new_patch_width = math.ceil(up_factor * target_patch_width)

            if new_patch_height * new_patch_width > max_patches:
                # If only one side can be min_side, make as big as possible at native aspect ratio while staying below max_patches
                if max(max_patches // new_patch_height, 1) * res_step < min_side:
                    up_factor = math.sqrt(max_patches / (target_patch_height * target_patch_width))
                    target_patch_height = math.floor(up_factor * target_patch_height)
                    target_patch_width = math.floor(up_factor * target_patch_width)
                else:
                    target_patch_height = new_patch_height
                    target_patch_width = max(max_patches // new_patch_height, 1)
            else:
                target_patch_height = new_patch_height
                target_patch_width = new_patch_width

    # Round patch grid to be divisible by 2 (pixel-shuffle OR conv-merging)
    # or by 4 when BOTH are enabled (two successive 2x reductions)
    if pixel_shuffle or conv_merging:
        required_divisor = 4 if (pixel_shuffle and conv_merging) else 2

        rem_h = target_patch_height % required_divisor
        if rem_h != 0:
            inc_h = required_divisor - rem_h
            if (target_patch_height + inc_h) * target_patch_width <= max_patches:
                target_patch_height += inc_h
            else:
                target_patch_height = max(1, target_patch_height - rem_h)

        rem_w = target_patch_width % required_divisor
        if rem_w != 0:
            inc_w = required_divisor - rem_w
            if target_patch_height * (target_patch_width + inc_w) <= max_patches:
                target_patch_width += inc_w
            else:
                target_patch_width = max(1, target_patch_width - rem_w)
    assert target_patch_height * target_patch_width <= max_patches

    #TEMP: hacky way to process video same as in training
    if is_video:
        # max_patches = 1024
        # min_patches = 512
        target_patch_width = 32
        target_patch_height = 32
        

    # resize the image
    resized_img = image.resize((target_patch_width * res_step, target_patch_height * res_step))

    return resized_img



# Based on https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/clip.py#L79
# and https://github.com/OpenGVLab/InternVL/blob/aa521e6eb1df4cf153aa4118fcf13e673c055d46/internvl_chat/internvl/train/dataset.py#L276
def _build_transform(input_size, vision_model_type):
    if vision_model_type in ("siglip", "internvit", "internvit300M", "radio", "radio-g", "cradio-g"):
        pixel_mean, pixel_std = pixel_statistics[vision_model_type]

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std)
        ])
    elif vision_model_type == "clip":
        pixel_mean, pixel_std = pixel_statistics[vision_model_type]

        transform = Compose([
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std),
        ])
    elif vision_model_type.startswith("hf://"):
        from megatron.core.models.huggingface.module import get_hf_model_type

        model_type = get_hf_model_type(vision_model_type)
        if "siglip" in model_type:
            from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor

            processor = SiglipImageProcessor(size={"height": input_size, "width": input_size})

            def transform(x):
                x = x.convert("RGB") if x.mode != "RGB" else x
                x = processor(x, return_tensors="pt")
                return x["pixel_values"][0]
        else:
            raise NotImplementedError(f"image processing not defined for huggingface model {vision_model_type}")
    else:
        raise NotImplementedError(f"image processing not defined for vision model {vision_model_type}")

    return transform
