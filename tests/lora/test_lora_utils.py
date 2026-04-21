# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

from huggingface_hub.utils import HfHubHTTPError

from vllm.lora.utils import (
    get_adapter_absolute_path,
)


# Unit tests for get_adapter_absolute_path
@patch("os.path.isabs")
def test_get_adapter_absolute_path_absolute(mock_isabs):
    path = "/absolute/path/to/lora"
    mock_isabs.return_value = True
    assert get_adapter_absolute_path(path) == path


@patch("os.path.expanduser")
def test_get_adapter_absolute_path_expanduser(mock_expanduser):
    # Path with ~ that needs to be expanded
    path = "~/relative/path/to/lora"
    absolute_path = "/home/user/relative/path/to/lora"
    mock_expanduser.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("os.path.exists")
@patch("os.path.abspath")
def test_get_adapter_absolute_path_local_existing(mock_abspath, mock_exist):
    # Relative path that exists locally
    path = "relative/path/to/lora"
    absolute_path = "/absolute/path/to/lora"
    mock_exist.return_value = True
    mock_abspath.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("huggingface_hub.snapshot_download")
@patch("os.path.exists")
def test_get_adapter_absolute_path_huggingface(mock_exist, mock_snapshot_download):
    # Hugging Face model identifier
    path = "org/repo"
    absolute_path = "/mock/snapshot/path"
    mock_exist.return_value = False
    mock_snapshot_download.return_value = absolute_path
    assert get_adapter_absolute_path(path) == absolute_path


@patch("huggingface_hub.snapshot_download")
@patch("os.path.exists")
def test_get_adapter_absolute_path_huggingface_error(
    mock_exist, mock_snapshot_download
):
    # Hugging Face model identifier with download error
    path = "org/repo"
    mock_exist.return_value = False
    mock_snapshot_download.side_effect = HfHubHTTPError(
        "failed to query model info",
        response=MagicMock(),
    )
    assert get_adapter_absolute_path(path) == path
