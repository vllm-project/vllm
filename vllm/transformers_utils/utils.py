# SPDX-License-Identifier: Apache-2.0
import re
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union


def is_remote_url(url: Union[str, Path]) -> bool:
    """
    Check if the URL is a remote URL of the format:
    <connector_type>://<host>:<port>/<model_name>
    """
    if isinstance(url, Path):
        return False

    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    return m is not None


def parse_connector_type(url: str) -> str:
    """
    Parse the connector type from the URL of the format:
    <connector_type>://<path>
    """
    pattern = r"(.+)://(.*)"
    m = re.match(pattern, url)
    if m is None:
        return ""

    return m.group(1)


def check_gguf_file(model: Union[str, PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"


def modelscope_list_repo_files(
    repo_id: str,
    revision: Optional[str] = None,
    token: Union[str, bool, None] = None,
) -> List[str]:
    """List files in a modelscope repo."""
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)
    # same as huggingface_hub.list_repo_files
    files = [
        file['Path'] for file in api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True)
        if file['Type'] == 'blob'
    ]
    return files
