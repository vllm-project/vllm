# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings

from vllm import envs
from vllm.transformers_utils.modelscope_utils import (
    modelscope_is_available,
    warn_modelscope_fallback,
)

if envs.VLLM_USE_MODELSCOPE:
    if not modelscope_is_available():
        warn_modelscope_fallback("vllm.transformers_utils")
    else:
        try:
            # Patch here, before each import happens.
            import modelscope
            from packaging import version

            # patch_hub begins from modelscope>=1.18.1.
            if version.parse(modelscope.__version__) <= version.parse("1.18.0"):
                warnings.warn(
                    "ModelScope < 1.18.1 is installed; falling back to "
                    "Hugging Face Hub."
                )
            else:
                from modelscope.utils.hf_util import patch_hub

                # Patch hub to download models from modelscope to speed up.
                patch_hub()
        except ImportError:
            warnings.warn(
                "ModelScope import failed; falling back to Hugging Face Hub."
            )
