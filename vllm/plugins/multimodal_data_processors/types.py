# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, Optional, Union

from typing_extensions import TypedDict


class ImagePrompt(TypedDict):

    data_format: Literal["b64_json", "bytes", "url"]
    """
    This is the data type for the input image
    """

    image_format: str
    """
    This is the image format (e.g., jpeg, png, etc.)
    """

    out_format: Literal["b64_json", "url"]

    data: Any
    """
    Input image data
    """


MultiModalPromptType = Union[ImagePrompt]


class MultiModalRequestOutput:
    """
    The output data of a multimodal request to vLLM. 
    A request generating anything but text

    Args:
        data (Any): The resulting data
    """

    def __init__(
        self,
        data: Any,
        request_id: Optional[str] = None,
    ):
        self.data = data
        self.request_id = request_id


class ImageRequestOutput(MultiModalRequestOutput):
    """
    The output data of an image request to vLLM. 

    Args:
        type (str): The data content type [path, object]
        format (str): The image format (e.g., jpeg, png, etc.)
        data (Any): The resulting data.
    """

    def __init__(
        self,
        type: Literal["path", "object"],
        format: str,
        data: Any,
        request_id: Optional[str] = None,
    ):
        super().__init__(data, request_id)
        self.type: Literal["path", "object"] = type
        self.format = format
        self.request_id = request_id

    def __repr__(self) -> str:
        return (f"ImageRequestOutput,"
                f"type={self.type},"
                f"format={self.format}")
