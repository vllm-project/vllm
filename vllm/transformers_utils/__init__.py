# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs

# Bind Hugging Face's snapshot downloader before ModelScope patches Hub functions.
from vllm.model_hub import hf_snapshot_download as _hf_snapshot_download  # noqa: F401

if envs.VLLM_USE_MODELSCOPE:
    try:
        # Patch here, before each import happens
        import modelscope
        from packaging import version

        # patch_hub begins from modelscope>=1.18.1
        if version.parse(modelscope.__version__) <= version.parse("1.18.0"):
            raise ImportError(
                "Using vLLM with ModelScope needs modelscope>=1.18.1, please "
                "install by `pip install modelscope -U`"
            )
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()
    except ImportError as err:
        raise ImportError(
            "Please install modelscope>=1.18.1 via "
            "`pip install modelscope>=1.18.1` to use ModelScope."
        ) from err
