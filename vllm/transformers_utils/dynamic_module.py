# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Optional, Union
from unittest.mock import patch

from transformers.dynamic_module_utils import get_cached_module_file


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
    but only makes sure that the module has been downloaded without checking
    imports within the module.
    """
    if "--" in class_reference:
        repo_id, class_reference = class_reference.split("--")
    else:
        repo_id = pretrained_model_name_or_path

    module_file, class_name = class_reference.split(".")

    if code_revision is None and pretrained_model_name_or_path == repo_id:
        code_revision = revision

    with patch("transformers.dynamic_module_utils.check_imports",
               lambda _: []):
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
