# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import datetime
import os
import tempfile
import urllib.request
from collections.abc import Sequence
from typing import Any

import albumentations
import numpy as np
import rasterio
import regex as re
import torch
from einops import rearrange
from terratorch.datamodules import Sen1Floods11NonGeoDataModule

from vllm.config import VllmConfig
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor

from .types import DataModuleConfig, ImagePrompt, ImageRequestOutput

logger = init_logger(__name__)

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99

DEFAULT_INPUT_INDICES = [0, 1, 2, 3, 4, 5]

datamodule_config: DataModuleConfig = {
    "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    "batch_size": 16,
    "constant_scale": 0.0001,
    "data_root": "/dccstor/geofm-finetuning/datasets/sen1floods11",
    "drop_last": True,
    "no_data_replace": 0.0,
    "no_label_replace": -1,
    "num_workers": 8,
    "test_transform": [
        albumentations.Resize(
            always_apply=False, height=448, interpolation=1, p=1, width=448
        ),
        albumentations.pytorch.ToTensorV2(
            transpose_mask=False, always_apply=True, p=1.0
        ),
    ],
}


def save_geotiff(image: torch.Tensor, meta: dict, out_format: str) -> str | bytes:
    """Save multi-band image in Geotiff file.

    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """
    if out_format == "path":
        # create temp file
        file_path = os.path.join(os.getcwd(), "prediction.tiff")
        with rasterio.open(file_path, "w", **meta) as dest:
            for i in range(image.shape[0]):
                dest.write(image[i, :, :], i + 1)

        return file_path
    elif out_format == "b64_json":
        with tempfile.NamedTemporaryFile() as tmpfile:
            with rasterio.open(tmpfile.name, "w", **meta) as dest:
                for i in range(image.shape[0]):
                    dest.write(image[i, :, :], i + 1)

            file_data = tmpfile.read()
            return base64.b64encode(file_data)

    else:
        raise ValueError("Unknown output format")


def _convert_np_uint8(float_image: torch.Tensor):
    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)

    return image


def read_geotiff(
    file_path: str | None = None,
    path_type: str | None = None,
    file_data: bytes | None = None,
) -> tuple[torch.Tensor, dict, tuple[float, float] | None]:
    """Read all bands from *file_path* and return image + meta info.

    Args:
        file_path: path to image file.

    Returns:
        np.ndarray with shape (bands, height, width)
        meta info dict
    """

    if all([x is None for x in [file_path, path_type, file_data]]):
        raise Exception("All input fields to read_geotiff are None")
    write_to_file: bytes | None = None
    path: str | None = None
    if file_data is not None:
        # with tempfile.NamedTemporaryFile() as tmpfile:
        #     tmpfile.write(file_data)
        #     path = tmpfile.name

        write_to_file = file_data
    elif file_path is not None and path_type == "url":
        resp = urllib.request.urlopen(file_path)
        # with tempfile.NamedTemporaryFile() as tmpfile:
        #     tmpfile.write(resp.read())
        #     path = tmpfile.name
        write_to_file = resp.read()
    elif file_path is not None and path_type == "path":
        path = file_path
    elif file_path is not None and path_type == "b64_json":
        image_data = base64.b64decode(file_path)
        # with tempfile.NamedTemporaryFile() as tmpfile:
        #     tmpfile.write(image_data)
        #     path = tmpfile.name
        write_to_file = image_data
    else:
        raise Exception("Wrong combination of parameters to read_geotiff")

    with tempfile.NamedTemporaryFile() as tmpfile:
        path_to_use = None
        if write_to_file:
            tmpfile.write(write_to_file)
            path_to_use = tmpfile.name
        elif path:
            path_to_use = path

        with rasterio.open(path_to_use) as src:
            img = src.read()
            meta = src.meta
            try:
                coords = src.lnglat()
            except Exception:
                # Cannot read coords
                coords = None

    return img, meta, coords


def load_image(
    data: list[str],
    path_type: str,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    indices: list[int] | None | None = None,
):
    """Build an input example by loading images in *file_paths*.

    Args:
        file_paths: list of file paths .
        mean: list containing mean values for each band in the
              images in *file_paths*.
        std: list containing std values for each band in the
             images in *file_paths*.

    Returns:
        np.array containing created example
        list of meta info for each image in *file_paths*
    """

    imgs = []
    metas = []
    temporal_coords = []
    location_coords = []

    for file in data:
        # if isinstance(file, bytes):
        #     img, meta, coords = read_geotiff(file_data=file)
        # else:
        img, meta, coords = read_geotiff(file_path=file, path_type=path_type)
        # Rescaling (don't normalize on nodata)
        img = np.moveaxis(img, 0, -1)  # channels last for rescaling
        if indices is not None:
            img = img[..., indices]
        if mean is not None and std is not None:
            img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

        imgs.append(img)
        metas.append(meta)
        if coords is not None:
            location_coords.append(coords)

        try:
            match = re.search(r"(\d{7,8}T\d{6})", file)
            if match:
                year = int(match.group(1)[:4])
                julian_day = match.group(1).split("T")[0][4:]
                if len(julian_day) == 3:
                    julian_day = int(julian_day)
                else:
                    julian_day = (
                        datetime.datetime.strptime(julian_day, "%m%d")
                        .timetuple()
                        .tm_yday
                    )
                temporal_coords.append([year, julian_day])
        except Exception:
            logger.exception("Could not extract timestamp for %s", file)

    imgs = np.stack(imgs, axis=0)  # num_frames, H, W, C
    imgs = np.moveaxis(imgs, -1, 0).astype("float32")  # C, num_frames, H, W
    imgs = np.expand_dims(imgs, axis=0)  # add batch di

    return imgs, temporal_coords, location_coords, metas


class PrithviMultimodalDataProcessor(IOProcessor[ImagePrompt, ImageRequestOutput]):
    indices = [0, 1, 2, 3, 4, 5]

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        self.datamodule = Sen1Floods11NonGeoDataModule(
            data_root=datamodule_config["data_root"],
            batch_size=datamodule_config["batch_size"],
            num_workers=datamodule_config["num_workers"],
            bands=datamodule_config["bands"],
            drop_last=datamodule_config["drop_last"],
            test_transform=datamodule_config["test_transform"],
        )
        self.img_size = 512
        self.h1 = 1
        self.w1 = 1
        self.original_h = 512
        self.original_w = 512
        self.batch_size = 1
        self.meta_data = None
        self.requests_cache: dict[str, dict[str, Any]] = {}
        self.indices = DEFAULT_INPUT_INDICES

    def parse_data(self, data: object) -> ImagePrompt:
        if isinstance(data, dict):
            return ImagePrompt(**data)

        raise ValueError("Prompt data should be an `ImagePrompt`")

    def pre_process(
        self,
        prompt: ImagePrompt,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        image_data = dict(prompt)

        if request_id:
            self.requests_cache[request_id] = {
                "out_format": image_data["out_data_format"],
            }

        input_data, temporal_coords, location_coords, meta_data = load_image(
            data=[image_data["data"]],
            indices=self.indices,
            path_type=image_data["data_format"],
        )

        self.meta_data = meta_data[0]

        if input_data.mean() > 1:
            input_data = input_data / 10000  # Convert to range 0-1

        self.original_h, self.original_w = input_data.shape[-2:]
        pad_h = (self.img_size - (self.original_h % self.img_size)) % self.img_size
        pad_w = (self.img_size - (self.original_w % self.img_size)) % self.img_size
        input_data = np.pad(
            input_data,
            ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )

        batch = torch.tensor(input_data)
        windows = batch.unfold(3, self.img_size, self.img_size).unfold(
            4, self.img_size, self.img_size
        )
        self.h1, self.w1 = windows.shape[3:5]
        windows = rearrange(
            windows,
            "b c t h1 w1 h w -> (b h1 w1) c t h w",
            h=self.img_size,
            w=self.img_size,
        )

        # Split into batches if number of windows > batch_size
        num_batches = (
            windows.shape[0] // self.batch_size
            if windows.shape[0] > self.batch_size
            else 1
        )
        windows = torch.tensor_split(windows, num_batches, dim=0)

        if temporal_coords:
            temporal_coords = torch.tensor(temporal_coords).unsqueeze(0)
        else:
            temporal_coords = None
        if location_coords:
            location_coords = torch.tensor(location_coords[0]).unsqueeze(0)
        else:
            location_coords = None

        prompts = []
        for window in windows:
            # Apply standardization
            window = self.datamodule.test_transform(
                image=window.squeeze().numpy().transpose(1, 2, 0)
            )
            window = self.datamodule.aug(window)["image"]
            prompts.append(
                {
                    "prompt_token_ids": [1],
                    "multi_modal_data": {
                        "image": {
                            "pixel_values": window.to(torch.float16)[0],
                            "location_coords": location_coords.to(torch.float16),
                        }
                    },
                }
            )

        return prompts

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> ImageRequestOutput:
        pred_imgs_list = []

        if request_id and (request_id in self.requests_cache):
            out_format = self.requests_cache[request_id]["out_format"]
        else:
            out_format = "b64_json"

        for output in model_output:
            y_hat = output.outputs.data.argmax(dim=0)
            pred = torch.nn.functional.interpolate(
                y_hat[None, None, ...].float(),
                size=self.img_size,
                mode="nearest",
            )
            pred_imgs_list.append(pred)

        pred_imgs: torch.Tensor = torch.concat(pred_imgs_list, dim=0)

        # Build images from patches
        pred_imgs = rearrange(
            pred_imgs,
            "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
            h=self.img_size,
            w=self.img_size,
            b=1,
            c=1,
            h1=self.h1,
            w1=self.w1,
        )

        # Cut padded area back to original size
        pred_imgs = pred_imgs[..., : self.original_h, : self.original_w]

        # Squeeze (batch size 1)
        pred_imgs = pred_imgs[0]

        if not self.meta_data:
            raise ValueError("No metadata available for the current task")
        self.meta_data.update(count=1, dtype="uint8", compress="lzw", nodata=0)
        out_data = save_geotiff(
            _convert_np_uint8(pred_imgs), self.meta_data, out_format
        )

        return ImageRequestOutput(
            type=out_format,
            format="tiff",
            data=out_data,
        )
