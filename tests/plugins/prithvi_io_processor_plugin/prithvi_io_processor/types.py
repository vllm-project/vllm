# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, TypedDict

import albumentations
from pydantic import BaseModel


class DataModuleConfig(TypedDict):
    bands: list[str]
    batch_size: int
    constant_scale: float
    data_root: str
    drop_last: bool
    no_data_replace: float
    no_label_replace: int
    num_workers: int
    test_transform: list[albumentations.core.transforms_interface.BasicTransform]


class ImagePrompt(BaseModel):
    data_format: Literal["b64_json", "bytes", "url", "path"]
    """
    This is the data type for the input image
    """

    image_format: str
    """
    This is the image format (e.g., jpeg, png, etc.)
    """

    out_data_format: Literal["b64_json", "url"]

    data: Any
    """
    Input image data
    """


class ImageRequestOutput(BaseModel):
    """
    The output data of an image request to vLLM.

    Args:
        type (str): The data content type [path, object]
        format (str): The image format (e.g., jpeg, png, etc.)
        data (Any): The resulting data.
    """

    type: Literal["path", "b64_json"]
    format: str
    data: str
