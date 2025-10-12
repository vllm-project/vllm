# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

from vllm.envs import VLLM_MODEL_REDIRECT_PATH
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("s3://")


def check_gguf_file(model: Union[str, PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    try:
        with model.open("rb") as f:
            header = f.read(4)

        return header == b"GGUF"
    except Exception as e:
        logger.debug("Error reading file %s: %s", model, e)
        return False


def modelscope_list_repo_files(
    repo_id: str,
    revision: Optional[str] = None,
    token: Union[str, bool, None] = None,
) -> list[str]:
    """List files in a modelscope repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    # same as huggingface_hub.list_repo_files
    files = [
        file["Path"]
        for file in api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True
        )
        if file["Type"] == "blob"
    ]
    return files


def _maybe_json_dict(path: Union[str, PathLike]) -> dict[str, str]:
    with open(path) as f:
        try:
            return json.loads(f.read())
        except Exception:
            return dict[str, str]()


def _maybe_space_split_dict(path: Union[str, PathLike]) -> dict[str, str]:
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
    """
    Use model_redirect to redirect the model name to a local folder.

    :param model: hf model name
    :return: maybe redirect to a local folder
    """

    model_redirect_path = VLLM_MODEL_REDIRECT_PATH

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


def parse_safetensors_file_metadata(path: Union[str, PathLike]) -> dict[str, Any]:
    with open(path, "rb") as f:
        length_of_metadata = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(length_of_metadata).decode("utf-8"))
        return metadata


def is_oci_model_with_tag(model: str) -> bool:
    """
    Detect if model name is an OCI reference with explicit tag or digest.

    Returns True for OCI references with explicit tag/digest:
    - username/model:tag
    - username/model:v1.0
    - registry.io/username/model:tag
    - registry.io/username/model@sha256:digest

    Returns False for:
    - username/model (ambiguous - could be HuggingFace or OCI with implicit tag)
    - local/path/to/model (local filesystem paths)
    - model (single name without repository)

    This allows automatic detection of OCI format when the reference is
    unambiguous (has explicit tag/digest), while requiring explicit
    load_format="oci" for ambiguous cases.

    Args:
        model: Model name or path to check

    Returns:
        True if the model name matches OCI reference pattern with tag/digest
    """
    import regex as re

    # Return False for local paths that exist on filesystem
    if os.path.exists(model):
        return False

    # Pattern explanation:
    # ^                                    - Start of string
    # (?:(?:[^/]+\.[^/]+|[^/]+:[0-9]+)/)? - Optional registry:
    #                                        - either with domain (contains dot)
    #                                        - or with port (hostname:port)
    # [^/]+/                               - Repository owner/namespace (required slash)
    # [^/:@]+                              - Repository name (no slashes, colons, or @)
    # [:@]                                 - Tag separator (: or @)
    # .+                                   - Tag or digest content
    # $                                    - End of string
    #
    # This matches:
    # - username/repo:tag
    # - username/repo@sha256:abc
    # - registry.io/username/repo:tag
    # - registry.io:5000/username/repo:tag (registry with port)
    # - localhost:8080/username/repo:tag (hostname with port)
    #
    # Does NOT match:
    # - username/repo (no tag)
    # - /path/to/model:tag (starts with /)
    # - ./relative/path (starts with .)
    # - model (no slash)
    pattern = r"^(?:(?:[^/]+\.[^/]+|[^/]+:[0-9]+)/)?[^/]+/[^/:@]+[:@].+$"

    return bool(re.match(pattern, model))
