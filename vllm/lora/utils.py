# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import huggingface_hub
from huggingface_hub.utils import HfHubHTTPError, HFValidationError

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.

    If the lora_path is identified as a Hugging Face model identifier,
    it will download the model and return the local snapshot path.
    Otherwise, it treats the lora_path as a local file path and
    converts it to an absolute path.

    Parameters:
    lora_path (str): The path to the lora model, which can be an absolute path,
                     a relative path, or a Hugging Face model identifier.

    Returns:
    str: The resolved absolute local path to the lora model.
    """

    # Check if the path is an absolute path. Return it no matter exists or not.
    if os.path.isabs(lora_path):
        return lora_path

    # If the path starts with ~, expand the user home directory.
    if lora_path.startswith("~"):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # If the path does not exist locally.
    if envs.VLLM_USE_MODELSCOPE:
        # If using ModelScope, we assume the path is a ModelScope repo.
        from modelscope.hub.snapshot_download import InvalidParameter, snapshot_download
        from requests import HTTPError

        download_fn = lambda: snapshot_download(model_id=lora_path)
        download_exceptions = (HTTPError, InvalidParameter)
        error_log = "Error downloading the ModelScope model"
    else:
        # Otherwise, we assume the path is a Hugging Face Hub repo.
        download_fn = lambda: huggingface_hub.snapshot_download(repo_id=lora_path)
        download_exceptions = (HfHubHTTPError, HFValidationError)
        error_log = "Error downloading the HuggingFace model"

    try:
        local_snapshot_path = download_fn()
    except download_exceptions:
        # Handle errors that may occur during the download.
        # Return original path instead of throwing error here.
        logger.exception(error_log)
        return lora_path

    return local_snapshot_path
