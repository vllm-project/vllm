# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
from typing import Optional, Union

from transformers.dynamic_module_utils import get_cached_module_file

import vllm.envs as envs

logger = logging.getLogger(__name__)


def get_dynamic_module_file(
    class_reference: str,
    pretrained_model_name_or_path: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> str:
    """
    As [transformers.dynamic_module_utils.get_class_from_dynamic_module][],
    but only makes sure that the module has been downloaded.
    """
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path

    module_file, class_name = class_reference.split(".")

    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision

    return get_cached_module_file(
        repo_id,
        module_file + ".py",
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=code_revision,
        local_files_only=local_files_only,
        repo_type=repo_type,
    )


def try_get_dynamic_module_file(
    class_reference: str,
    pretrained_model_name_or_path: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    repo_type: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """
    As [transformers.dynamic_module_utils.get_class_from_dynamic_module][],
    ignoring any errors.
    """
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path

    module_file, class_name = class_reference.split(".")

    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision

    try:
        return get_cached_module_file(
            repo_id,
            module_file + ".py",
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=code_revision,
            local_files_only=local_files_only,
            repo_type=repo_type,
        )
    except Exception:
        location = "ModelScope" if envs.VLLM_USE_MODELSCOPE else "HF Hub"

        logger.exception(
            "Unable to load %s from %s on %s. This means that Transformers "
            "backend will not work for this model.",
            class_reference,
            pretrained_model_name_or_path,
            location,
        )

        return None
