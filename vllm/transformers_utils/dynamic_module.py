# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Optional, Union

from transformers.dynamic_module_utils import get_class_from_dynamic_module

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def try_get_class_from_dynamic_module(
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
    warn_on_fail: bool = True,
    **kwargs,
) -> Optional[type]:
    """
    As [transformers.dynamic_module_utils.get_class_from_dynamic_module][],
    but ignoring any errors.
    """
    try:
        return get_class_from_dynamic_module(
            class_reference,
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            repo_type=repo_type,
            code_revision=code_revision,
            **kwargs,
        )
    except Exception:
        location = "ModelScope" if envs.VLLM_USE_MODELSCOPE else "HF Hub"

        if warn_on_fail:
            logger.warning(
                "Unable to load %s from %s on %s.",
                class_reference,
                pretrained_model_name_or_path,
                location,
                exc_info=True,
            )

        return None
