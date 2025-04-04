# SPDX-License-Identifier: Apache-2.0

from functools import cache
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

from vllm.envs import VLLM_MODEL_REDIRECT_PATH
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith('s3://')


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

    with open(model_redirect_path) as f:
        for line in f.readlines():
            try:
                model_name, redirect_name = line.split("\t")
                if model == model_name:
                    redirect_name = redirect_name.strip()
                    logger.info("model redirect: [ %s ] -> [ %s ]", model,
                                redirect_name)
                    return redirect_name
            except Exception:
                pass

    return model
