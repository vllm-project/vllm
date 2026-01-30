# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for model repo interaction."""

import fnmatch
import json
import os
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TypeVar

import huggingface_hub
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub import list_repo_files as hf_list_repo_files
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


_R = TypeVar("_R")


def with_retry(
    func: Callable[[], _R],
    log_msg: str,
    max_retries: int = 2,
    retry_delay: int = 2,
) -> _R:
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("%s: %s", log_msg, e)
                raise
            logger.warning(
                "%s: %s, retrying %d of %d", log_msg, e, attempt + 1, max_retries
            )
            time.sleep(retry_delay)
            retry_delay *= 2

    raise AssertionError("Should not be reached")


# @cache doesn't cache exceptions
@cache
def list_repo_files(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    def lookup_files() -> list[str]:
        # directly list files if model is local
        if (local_path := Path(repo_id)).exists():
            return [
                str(file.relative_to(local_path))
                for file in local_path.rglob("*")
                if file.is_file()
            ]
        # if model is remote, use hf_hub api to list files
        try:
            if envs.VLLM_USE_MODELSCOPE:
                from vllm.transformers_utils.utils import modelscope_list_repo_files

                return modelscope_list_repo_files(
                    repo_id,
                    revision=revision,
                    token=os.getenv("MODELSCOPE_API_TOKEN", None),
                )
            return hf_list_repo_files(
                repo_id, revision=revision, repo_type=repo_type, token=token
            )
        except huggingface_hub.errors.OfflineModeIsEnabled:
            # Don't raise in offline mode,
            # all we know is that we don't have this
            # file cached.
            return []

    return with_retry(lookup_files, "Error retrieving file list")


def list_filtered_repo_files(
    model_name_or_path: str,
    allow_patterns: list[str],
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    try:
        all_files = list_repo_files(
            repo_id=model_name_or_path,
            revision=revision,
            token=token,
            repo_type=repo_type,
        )
    except Exception:
        logger.error(
            "Error retrieving file list. Please ensure your `model_name_or_path`"
            "`repo_type`, `token` and `revision` arguments are correctly set. "
            "Returning an empty list."
        )
        return []

    file_list = []
    # Filter patterns on filenames
    for pattern in allow_patterns:
        file_list.extend(
            [
                file
                for file in all_files
                if fnmatch.fnmatch(os.path.basename(file), pattern)
            ]
        )
    return file_list


def file_exists(
    repo_id: str,
    file_name: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> bool:
    # `list_repo_files` is cached and retried on error, so this is more efficient than
    # huggingface_hub.file_exists default implementation when looking for multiple files
    file_list = list_repo_files(
        repo_id, repo_type=repo_type, revision=revision, token=token
    )
    return file_name in file_list


# In offline mode the result can be a false negative
def file_or_path_exists(
    model: str | Path, config_name: str, revision: str | None
) -> bool:
    if (local_path := Path(model)).exists():
        return (local_path / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(
        repo_id=model, filename=config_name, revision=revision
    )
    if isinstance(cached_filepath, str):
        # The config file exists in cache - we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.

    # Call HF to check if the file exists
    return file_exists(str(model), config_name, revision=revision)


def get_model_path(model: str | Path, revision: str | None = None):
    if os.path.exists(model):
        return model
    assert huggingface_hub.constants.HF_HUB_OFFLINE
    common_kwargs = {
        "local_files_only": huggingface_hub.constants.HF_HUB_OFFLINE,
        "revision": revision,
    }

    if envs.VLLM_USE_MODELSCOPE:
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(model_id=model, **common_kwargs)

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model, **common_kwargs)


def get_hf_file_bytes(
    file_name: str, model: str | Path, revision: str | None = "main"
) -> bytes | None:
    """Get file contents from HuggingFace repository as bytes."""
    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)

    if file_path is None:
        hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path, "rb") as file:
            return file.read()

    return None


def try_get_local_file(
    model: str | Path, file_name: str, revision: str | None = "main"
) -> Path | None:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(
                repo_id=model, filename=file_name, revision=revision
            )
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except ValueError:
            ...
    return None


def get_hf_file_to_dict(
    file_name: str, model: str | Path, revision: str | None = "main"
):
    """
    Downloads a file from the Hugging Face Hub and returns
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model.

    Returns:
    - config_dict (dict): A dictionary containing
    the contents of the downloaded file.
    """

    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)

    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            return None
        except (
            RepositoryNotFoundError,
            RevisionNotFoundError,
            EntryNotFoundError,
            LocalEntryNotFoundError,
        ) as e:
            logger.debug("File or repository not found in hf_hub_download", e)
            return None
        except HfHubHTTPError as e:
            logger.warning(
                "Cannot connect to Hugging Face Hub. Skipping file download for '%s':",
                file_name,
                exc_info=e,
            )
            return None
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)

    return None
