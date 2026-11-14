# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import envs


def is_modelscope_hub_enabled() -> bool:
    """Use global Hub routing when target and draft do not use different sources.

    With different sources, ModelConfig resolves ModelScope models to local
    paths, leaving remote repo IDs for the original Hugging Face loaders.
    """
    return envs.VLLM_USE_MODELSCOPE and envs.VLLM_DRAFT_USE_MODELSCOPE is not False


if is_modelscope_hub_enabled():
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
