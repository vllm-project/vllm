# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from transformers.dynamic_module_utils import (
    get_class_from_dynamic_module,
    resolve_trust_remote_code,
)

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def try_get_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: str,
    trust_remote_code: bool,
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    repo_type: str | None = None,
    code_revision: str | None = None,
    warn_on_fail: bool = True,
    **kwargs,
) -> type | None:
    """
    As `transformers.dynamic_module_utils.get_class_from_dynamic_module`,
    but ignoring any errors.
    """
    try:
        resolve_trust_remote_code(
            trust_remote_code,
            pretrained_model_name_or_path,
            has_local_code=False,
            has_remote_code=True,
        )

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
