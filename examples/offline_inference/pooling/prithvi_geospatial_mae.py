# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import datetime
import os

import albumentations
import numpy as np
import rasterio
import regex as re
import torch
from einops import rearrange
from terratorch.datamodules import Sen1Floods11NonGeoDataModule

from vllm import LLM

torch.set_default_dtype(torch.float16)

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
OFFSET = 0
PERCENTILE = 99

datamodule_config = {
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


class PrithviMAE:
    def __init__(self, model):
        self.model = LLM(
            model=model,
            skip_tokenizer_init=True,
            dtype="float16",
            enforce_eager=True,
            model_impl="terratorch",
            enable_mm_embeds=True,
        )

    def run(self, input_data, location_coords):
        # merge the inputs into one data structure
        if input_data is not None and input_data.dtype == torch.float32:
            input_data = input_data.to(torch.float16)
            input_data = input_data[0]

        mm_data = {
            "pixel_values": input_data,
            "location_coords": location_coords,
        }

        prompt = {"prompt_token_ids": [1], "multi_modal_data": mm_data}
        outputs = self.model.encode(prompt, pooling_task="plugin", use_tqdm=False)

        return outputs[0].outputs.data


def generate_datamodule():
    datamodule = Sen1Floods11NonGeoDataModule(
        data_root=datamodule_config["data_root"],
        batch_size=datamodule_config["batch_size"],
        num_workers=datamodule_config["num_workers"],
        bands=datamodule_config["bands"],
        drop_last=datamodule_config["drop_last"],
        test_transform=datamodule_config["test_transform"],
    )

    return datamodule


def process_channel_group(orig_img, channels):
    """
    Args:
        orig_img: torch.Tensor representing original image (reference)
        with shape = (bands, H, W).
        channels: list of indices representing RGB channels.

    Returns:
        torch.Tensor with shape (num_channels, height, width)
        for original image
    """

    orig_img = orig_img[channels, ...]
    valid_mask = torch.ones_like(orig_img, dtype=torch.bool)
    valid_mask[orig_img == NO_DATA_FLOAT] = False

    # Rescale (enhancing contrast)
    max_value = max(3000, np.percentile(orig_img[valid_mask], PERCENTILE))
    min_value = OFFSET

    orig_img = torch.clamp((orig_img - min_value) / (max_value - min_value), 0, 1)

    # No data as zeros
    orig_img[~valid_mask] = 0

    return orig_img


def read_geotiff(file_path: str):
    """Read all bands from *file_path* and return image + meta info.

    Args:
        file_path: path to image file.

    Returns:
        np.ndarray with shape (bands, height, width)
        meta info dict
    """

    with rasterio.open(file_path) as src:
        img = src.read()
        meta = src.meta
        try:
            coords = src.lnglat()
        except Exception:
            # Cannot read coords
            coords = None

    return img, meta, coords


def save_geotiff(image, output_path: str, meta: dict):
    """Save multi-band image in Geotiff file.

    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """

    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)

    return


def _convert_np_uint8(float_image: torch.Tensor):
    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)

    return image


def load_example(
    file_paths: list[str],
    mean: list[float] = None,
    std: list[float] = None,
    indices: list[int] | None = None,
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

    for file in file_paths:
        img, meta, coords = read_geotiff(file)

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
        except Exception as e:
            print(f"Could not extract timestamp for {file} ({e})")

    imgs = np.stack(imgs, axis=0)  # num_frames, H, W, C
    imgs = np.moveaxis(imgs, -1, 0).astype("float32")  # C, num_frames, H, W
    imgs = np.expand_dims(imgs, axis=0)  # add batch di

    return imgs, temporal_coords, location_coords, metas


def run_model(
    input_data,
    temporal_coords,
    location_coords,
    model,
    datamodule,
    img_size,
    lightning_model=None,
):
    # Reflect pad if not divisible by img_size
    original_h, original_w = input_data.shape[-2:]
    pad_h = (img_size - (original_h % img_size)) % img_size
    pad_w = (img_size - (original_w % img_size)) % img_size
    input_data = np.pad(
        input_data, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
    )

    # Build sliding window

    batch_size = 1
    # batch = torch.tensor(input_data, device="cpu")
    batch = torch.tensor(input_data)
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    h1, w1 = windows.shape[3:5]
    windows = rearrange(
        windows, "b c t h1 w1 h w -> (b h1 w1) c t h w", h=img_size, w=img_size
    )

    # Split into batches if number of windows > batch_size
    num_batches = windows.shape[0] // batch_size if windows.shape[0] > batch_size else 1
    windows = torch.tensor_split(windows, num_batches, dim=0)

    if temporal_coords:
        temporal_coords = torch.tensor(temporal_coords).unsqueeze(0)
    else:
        temporal_coords = None
    if location_coords:
        location_coords = torch.tensor(location_coords[0]).unsqueeze(0)
    else:
        location_coords = None

    # Run Prithvi-EO-V2-300M-TL-Sen1Floods11
    pred_imgs = []
    for x in windows:
        # Apply standardization
        x = datamodule.test_transform(image=x.squeeze().numpy().transpose(1, 2, 0))
        x = datamodule.aug(x)["image"]

        with torch.no_grad():
            pred = model.run(x, location_coords=location_coords)
        y_hat = pred.argmax(dim=1)

        y_hat = torch.nn.functional.interpolate(
            y_hat.unsqueeze(1).float(), size=img_size, mode="nearest"
        )

        pred_imgs.append(y_hat)

    pred_imgs = torch.concat(pred_imgs, dim=0)

    # Build images from patches
    pred_imgs = rearrange(
        pred_imgs,
        "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
        h=img_size,
        w=img_size,
        b=1,
        c=1,
        h1=h1,
        w1=w1,
    )

    # Cut padded area back to original size
    pred_imgs = pred_imgs[..., :original_h, :original_w]

    # Squeeze (batch size 1)
    pred_imgs = pred_imgs[0]

    return pred_imgs


def main(
    data_file: str,
    model: str,
    output_dir: str,
    rgb_outputs: bool,
    input_indices: list[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    model_obj = PrithviMAE(model=model)
    datamodule = generate_datamodule()
    img_size = 512  # Size of Sen1Floods11

    input_data, temporal_coords, location_coords, meta_data = load_example(
        file_paths=[data_file],
        indices=input_indices,
    )

    meta_data = meta_data[0]  # only one image

    if input_data.mean() > 1:
        input_data = input_data / 10000  # Convert to range 0-1

    channels = [
        datamodule_config["bands"].index(b) for b in ["RED", "GREEN", "BLUE"]
    ]  # BGR -> RGB

    pred = run_model(
        input_data, temporal_coords, location_coords, model_obj, datamodule, img_size
    )
    # Save pred
    meta_data.update(count=1, dtype="uint8", compress="lzw", nodata=0)
    pred_file = os.path.join(
        output_dir, f"pred_{os.path.splitext(os.path.basename(data_file))[0]}.tiff"
    )
    save_geotiff(_convert_np_uint8(pred), pred_file, meta_data)

    # Save image + pred
    meta_data.update(count=3, dtype="uint8", compress="lzw", nodata=0)

    if input_data.mean() < 1:
        input_data = input_data * 10000  # Scale to 0-10000

    rgb_orig = process_channel_group(
        orig_img=torch.Tensor(input_data[0, :, 0, ...]),
        channels=channels,
    )
    rgb_orig = rgb_orig.to(torch.float32)

    pred[pred == 0.0] = np.nan
    img_pred = rgb_orig * 0.7 + pred * 0.3
    img_pred[img_pred.isnan()] = rgb_orig[img_pred.isnan()]

    img_pred_file = os.path.join(
        output_dir, f"rgb_pred_{os.path.splitext(os.path.basename(data_file))[0]}.tiff"
    )
    save_geotiff(
        image=_convert_np_uint8(img_pred),
        output_path=img_pred_file,
        meta=meta_data,
    )

    # Save image rgb
    if rgb_outputs:
        name_suffix = os.path.splitext(os.path.basename(data_file))[0]
        rgb_file = os.path.join(
            output_dir,
            f"original_rgb_{name_suffix}.tiff",
        )
        save_geotiff(
            image=_convert_np_uint8(rgb_orig),
            output_path=rgb_file,
            meta=meta_data,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MAE run inference", add_help=False)

    parser.add_argument(
        "--data_file",
        type=str,
        default="./India_900498_S2Hand.tif",
        help="Path to the file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM",
        help="Path to a checkpoint file to load from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the directory where to save outputs.",
    )
    parser.add_argument(
        "--input_indices",
        default=[1, 2, 3, 8, 11, 12],
        type=int,
        nargs="+",
        help="""
        0-based indices of the six Prithvi channels to be selected from the input.
        By default selects [1,2,3,8,11,12] for S2L1C data.
        """,
    )
    parser.add_argument(
        "--rgb_outputs",
        action="store_true",
        help="If present, output files will only contain RGB channels. "
        "Otherwise, all bands will be saved.",
    )
    args = parser.parse_args()

    main(**vars(args))
