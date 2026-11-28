# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_SAFETENSORS_HEADER_BYTES = 64 << 20
"""Upper bound on a safetensors JSON header. Real headers are a few KB even
for large shards; a larger declared length means a corrupt or hostile file."""


def read_safetensors_header(path: str | PathLike) -> bytes:
    """Read a safetensors length prefix and JSON header with size checks."""
    with open(path, "rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) != 8:
            raise ValueError(
                f"Invalid safetensors header in {path}: expected an 8-byte "
                f"length prefix, got {len(size_bytes)} bytes"
            )
        (header_len,) = struct.unpack("<Q", size_bytes)
        file_size = os.fstat(f.fileno()).st_size
        if header_len > file_size - 8:
            raise ValueError(
                f"Invalid safetensors header in {path}: declared length "
                f"{header_len} exceeds the {file_size}-byte file"
            )
        if header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"Invalid safetensors header in {path}: declared length "
                f"{header_len} exceeds the maximum of "
                f"{MAX_SAFETENSORS_HEADER_BYTES} bytes"
            )
        header = f.read(header_len)
        if len(header) != header_len:
            raise ValueError(
                f"Invalid safetensors header in {path}: expected {header_len} "
                f"bytes, got {len(header)}"
            )
        return size_bytes + header


def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("s3://")


def is_gcs(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("gs://")


def is_azure(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("az://")


def is_cloud_storage(model_or_path: str) -> bool:
    return is_s3(model_or_path) or is_gcs(model_or_path) or is_azure(model_or_path)


def without_trust_remote_code(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs without trust_remote_code without modifying original dict."""
    if "trust_remote_code" not in kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k != "trust_remote_code"}


def modelscope_list_repo_files(
    repo_id: str,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    """List files in a modelscope repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)

    # same as huggingface_hub.list_repo_files
    return [
        file["Path"]
        for file in api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True
        )
    ]


def _maybe_json_dict(path: str | PathLike) -> dict[str, str]:
    with open(path) as f:
        try:
            return json.loads(f.read())
        except Exception:
            return dict[str, str]()


def _maybe_space_split_dict(path: str | PathLike) -> dict[str, str]:
    parsed_dict = dict[str, str]()
    with open(path) as f:
        for line in f.readlines():
            try:
                model_name, redirect_name = line.strip().split()
                parsed_dict[model_name] = redirect_name
            except Exception:
                pass
    return parsed_dict


@cache
def maybe_model_redirect(model: str) -> str:
    """Use model_redirect to redirect the model name to a local folder.

    Args:
        model: hf model name

    Returns:
        maybe redirect to a local folder

    """
    model_redirect_path = envs.VLLM_MODEL_REDIRECT_PATH

    if not model_redirect_path:
        return model

    if not Path(model_redirect_path).exists():
        return model

    redirect_dict = _maybe_json_dict(model_redirect_path) or _maybe_space_split_dict(
        model_redirect_path
    )
    if redirect_model := redirect_dict.get(model):
        logger.info("model redirect: [ %s ] -> [ %s ]", model, redirect_model)
        return redirect_model

    return model


def parse_safetensors_file_metadata(path: str | PathLike) -> dict[str, Any]:
    return json.loads(read_safetensors_header(path)[8:].decode("utf-8"))


def convert_model_repo_to_path(model_repo: str) -> str:
    """When VLLM_USE_MODELSCOPE is True convert a model
    repository string to a Path str."""
    if not envs.VLLM_USE_MODELSCOPE or Path(model_repo).exists():
        return model_repo
    from modelscope.utils.file_utils import get_model_cache_root

    return os.path.join(get_model_cache_root(), model_repo)
