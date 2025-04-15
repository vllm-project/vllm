from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from typing import Optional, Union, Tuple, Dict, List, Iterable
from transformers.image_transforms import to_channel_dimension_format, PaddingMode
from transformers.image_utils import ChannelDimension, to_numpy_array, make_list_of_images, get_image_size, infer_channel_dimension_format
from transformers.utils import TensorType
from PIL import Image
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torch
from transformers.utils import (
    TensorType,
    is_torch_device,
    is_torch_dtype,
    requires_backends,
)

from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, RandomResizedCrop, Resize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import Image
import torch
import numpy as np
import os
processor_for_vllm = int(os.getenv("PROCESSOR_FOR_VLLM", 0))

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit 

def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches

def image_size_to_num_patches(image_size, grid_pinpoints, patch_size):
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    width, height = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches

def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    width, height = select_best_resolution(image_size, grid_pinpoints)
    return width // patch_size, height // patch_size


# custom transform
class KeeyRatioResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return keepratio_resize(image, self.size)

def keepratio_resize(image, size, return_scale=False):
    # Resize the image to keep the ratio
    w, h = image.size
    resized_w, resized_h = size
    if w / h > resized_w / resized_h:
        # resize and pad to the right and left
        new_h = int(resized_w*h/w)
        resized_image = image.resize((resized_w, new_h), Image.BICUBIC)

        image = Image.new('RGB', (resized_w, resized_h), (0, 0, 0))
        pad_h = (resized_h - new_h) // 2
        image.paste(resized_image, (0, pad_h))
        scale = resized_w / w
        #image.paste(resized_image, (0, 0))
    else:
        # resize and pad to the top and bottom
        new_w = int(resized_h*w/h)
        resized_image = image.resize((new_w, resized_h), Image.BICUBIC)
        image = Image.new('RGB', (resized_w, resized_h), (0, 0, 0))
        #image.paste(resized_image, (0, 0))
        pad_w = (resized_w - new_w) // 2
        image.paste(resized_image, (pad_w, 0))
        scale = resized_h / h
    if return_scale:
        return image, scale
    return image

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(img_h, img_w, image_mean=(0.48145466, 0.4578275, 0.40821073), image_std=(0.26862954, 0.26130258, 0.27577711)):
    return Compose([
        # ToPILImage(),
        #RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0), interpolation=BICUBIC),
        #Resize((img_h, img_w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(image_mean, image_std),
    ])


def get_hw_multiple_of(image_size, multiple, max_size=None):
    w, h = image_size
    new_w = w if w % multiple == 0 else w + (multiple - w % multiple)
    new_h = h if h % multiple == 0 else h + (multiple - h % multiple)
    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0
        if new_w > max_w or new_h > max_h:
            # ratio = min(max_w / new_w, max_h / new_h)
            # new_w = int(new_w * ratio)
            # new_h = int(new_h * ratio)
            new_w = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)

            new_w = new_w if new_w % multiple == 0 else new_w + (multiple - new_w % multiple)
            new_h = new_h if new_h % multiple == 0 else new_h + (multiple - new_h % multiple)
        assert new_w % multiple == 0 and new_h % multiple == 0
        assert new_w <= max_w and new_h <= max_h
    return new_w, new_h

def resize_multiple_of(image, multiple, max_size=None):
    """
    Resize the image to the multiple of a number.

    Args:
        image (PIL.Image.Image): The input image.
        multiple (int): The number to which the image should be resized.

    Returns:
        PIL.Image.Image: The resized image.
    """
    width, height = image.size
    new_width, new_height = get_hw_multiple_of((width, height), multiple, max_size)
    return image.resize((new_width, new_height), Image.BICUBIC)



class CustomBatchFeature(BatchFeature):
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        # Do the tensor conversion in batch
        for key, value in self.items():
            if key == "pixel_values":
                for i, image in enumerate(value):
                    if not is_tensor(image):
                        tensor = as_tensor(image)
                        self[key][i] = tensor
                continue
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        return self

    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch  # noqa

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            if k == "pixel_values":
                new_data[k] = [v[i].to(*args, **kwargs) for i in range(len(v))]
                continue
            # check if v is a floating point
            if torch.is_floating_point(v):
                # cast and send to device
                new_data[k] = v.to(*args, **kwargs)
            elif device is not None:
                new_data[k] = v.to(device=device)
            else:
                new_data[k] = v
        self.data = new_data
        return self


def as_tensor(value):
    if isinstance(value, (list, tuple)) and len(value) > 0:
        if isinstance(value[0], np.ndarray):
            value = np.array(value)
        elif (
            isinstance(value[0], (list, tuple))
            and len(value[0]) > 0
            and isinstance(value[0][0], np.ndarray)
        ):
            value = np.array(value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    else:
        return torch.tensor(value)

class ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int], Dict[str, int]]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        process_image_mode: Optional[str] = 'resize',
        patch_size: Optional[int] = 14,
        image_grid_pinpoints: List = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.size = size # (width, height)
        self.image_mean = image_mean
        self.image_std = image_std
        self.process_image_mode = process_image_mode
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.image_grid_pinpoints = image_grid_pinpoints
        self.patch_size = patch_size

    def preprocess(self,
                    images,
                    return_tensors: Optional[Union[str, TensorType]] = None,
                    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
                    input_data_format: Optional[Union[str, ChannelDimension]] = None,
                    **kwargs,
                    ):
        if self.process_image_mode == 'resize':
            return self.resize_preprocess(images, return_tensors, data_format, input_data_format, **kwargs)
        elif self.process_image_mode == 'anyres':
            if processor_for_vllm == 1:
                return self.anyres_for_vllm_preprocess(images, return_tensors, data_format, input_data_format, **kwargs)
            return self.anyres_preprocess(images, return_tensors, data_format, input_data_format, **kwargs)
        elif self.process_image_mode == 'keepratio_resize':
            return self.keepratio_resize_preprocess(images, return_tensors, data_format, input_data_format, **kwargs)
        elif self.process_image_mode == 'dynamic_res':
            return self.dynamic_res_preprocess(images, return_tensors, data_format, input_data_format, **kwargs)
        else:
            raise ValueError(f"Invalid process_image_mode: {self.process_image_mode}")
    
    def resize_preprocess(self, images, return_tensors: Optional[Union[str, TensorType]] = None, data_format: Optional[ChannelDimension] = ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs):
        images = make_list_of_images(images)
        all_images = []
        for image in images:
            resized_image = image.resize(self.size, Image.BICUBIC)
            transform_img = _transform(self.size[1], self.size[0], self.image_mean, self.image_std)(resized_image)
            all_images.append(to_numpy_array(transform_img))

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in all_images
        ]

        data = {"pixel_values": images}
        return CustomBatchFeature(data=data, tensor_type=return_tensors)

    def keepratio_resize_preprocess(self, images, return_tensors: Optional[Union[str, TensorType]] = None, data_format: Optional[ChannelDimension] = ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs):
        images = make_list_of_images(images)
        all_images = []
        for image in images:
            resized_image = keepratio_resize(image, self.size)
            transform_img = _transform(self.size[1], self.size[0], self.image_mean, self.image_std)(resized_image)
            all_images.append(to_numpy_array(transform_img))

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            for image in all_images
        ]

        data = {"pixel_values": images}
        return CustomBatchFeature(data=data, tensor_type=return_tensors)

    def dynamic_res_preprocess(self, images, return_tensors: Optional[Union[str, TensorType]] = None, data_format: Optional[ChannelDimension] = ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs):
        images = make_list_of_images(images)
        all_images = []
        image_sizes = []
        for image in images:
            ori_w, ori_h = image.size
            image_sizes.append([ori_h, ori_w])
            resized_image = resize_multiple_of(image, self.patch_size, max_size=self.size)
            resized_w, resized_h = resized_image.size
            transform_img = _transform(resized_h, resized_w, self.image_mean, self.image_std)(resized_image)
            all_images.append(to_numpy_array(transform_img))

        images = [
            as_tensor(to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format))
            for image in all_images
        ]

        # data = {"pixel_values": images, "image_sizes": as_tensor(image_sizes)}
        # return data
        data = {"pixel_values": images, "image_sizes": image_sizes}
        #return BatchFeature(data=data, data_format=data_format, tensor_type=return_tensors)
        
        return CustomBatchFeature(data=data, tensor_type=return_tensors)

    def get_image_patches(
        self,
        data: Image,
        image_grid_pinpoints,
    ):
        if not isinstance(image_grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")


        best_resolution = select_best_resolution(data.size, image_grid_pinpoints)

        resized_data, scale = keepratio_resize(data, best_resolution, return_scale=True)
        resized_data = divide_to_patches(resized_data, self.size[0])
        ori_data = data.resize(self.size, Image.BICUBIC)
        data = [ori_data] + resized_data
        return data
    
    def pad(
        self,
        image: np.ndarray,
        padding: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
        mode: PaddingMode = PaddingMode.CONSTANT,
        constant_values: Union[float, Iterable[float]] = 0.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pads the `image` with the specified `padding` and `mode`. Padding can be in the (`height`, `width`)
        dimension of in the (`num_patches`) dimension. In the second case an iterable if tuples is expected
        as input.

        Args:
            image (`np.ndarray`):
                The image to pad.
            padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
                Padding to apply to the edges of the height, width axes. Can be one of three formats:
                - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
                - `((before, after),)` yields same before and after pad for height and width.
                - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            `np.ndarray`: The padded image.

        """

        # call the general `pad` if padding on `height/width`, otherwise it's the `num_patched` dim
        if isinstance(padding, int) or len(padding) != 4:
            return pad(image, padding, mode, constant_values, data_format, input_data_format)

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        if mode == PaddingMode.CONSTANT:
            image = np.pad(image, padding, mode="constant", constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )
        return image

    def _pad_for_batching(
        self,
        pixel_values: List[np.ndarray],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[np.ndarray]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            List[`np.ndarray`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            self.pad(
                image,
                padding=((0, max_patch - image.shape[0]), (0, 0), (0, 0), (0, 0)),
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in pixel_values
        ]

        return pixel_values

    def anyres_for_vllm_preprocess(self, images, return_tensors: Optional[Union[str, TensorType]] = None, data_format: Optional[ChannelDimension] = ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]] = None, do_pad: Optional[bool] = None, **kwargs):
        
        images = make_list_of_images(images)
        new_images = []
        image_sizes = []

        for image in images:
            ori_w, ori_h = image.size
            image_sizes.append([ori_h, ori_w])
            image_patches = self.get_image_patches(
                image,
                self.image_grid_pinpoints
            )
            all_images = []
            for image in image_patches:
                transform_img = _transform(self.size[0], self.size[1], self.image_mean, self.image_std)(image)
                img_array = to_numpy_array(transform_img)
                img_array = to_channel_dimension_format(img_array, data_format, input_channel_dim=input_data_format)
                all_images.append(img_array)
                #new_images.append(img_array)
            pixel_values = np.array(all_images)
            new_images.append(pixel_values)
        

        new_images = self._pad_for_batching(new_images)

        data = {"pixel_values": new_images, "image_sizes": image_sizes}
        return BatchFeature(data=data, tensor_type=return_tensors)

    
    def anyres_preprocess(self, images, return_tensors: Optional[Union[str, TensorType]] = None, data_format: Optional[ChannelDimension] = ChannelDimension.FIRST, input_data_format: Optional[Union[str, ChannelDimension]] = None, do_pad: Optional[bool] = None, **kwargs):
        
        images = make_list_of_images(images)
        new_images = []
        image_sizes = []

        for image in images:
            ori_w, ori_h = image.size
            image_sizes.append([ori_h, ori_w])
            image_patches = self.get_image_patches(
                image,
                self.image_grid_pinpoints
            )
            #all_images = []
            for image in image_patches:
                transform_img = _transform(self.size[0], self.size[1], self.image_mean, self.image_std)(image)
                img_array = to_numpy_array(transform_img)
                img_array = to_channel_dimension_format(img_array, data_format, input_channel_dim=input_data_format)
                #all_images.append(img_array)
                new_images.append(img_array)
            #pixel_values = np.array(all_images)
            #new_images.append(pixel_values)
        
       #  if do_pad:
        new_images = self._pad_for_batching(new_images)

        data = {"pixel_values": new_images, "image_sizes": image_sizes}
        return CustomBatchFeature(data=data, tensor_type=return_tensors)