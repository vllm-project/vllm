# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 platform entry point.

The implementation lives under ``nvidia/`` and ``amd/``; this module picks the
right one for the current platform and re-exports the model class used by the
model registry.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

__all__ = ["MistralLarge4ForCausalLM"]

if TYPE_CHECKING:
    from .nvidia.dspark import MistralLarge4DSparkForCausalLM
    from .nvidia.model import MistralLarge4ForCausalLM
elif current_platform.is_cuda():
    from .nvidia.dspark import MistralLarge4DSparkForCausalLM  # noqa: E402
    from .nvidia.model import MistralLarge4ForCausalLM  # noqa: E402

    __all__.append("MistralLarge4DSparkForCausalLM")
elif current_platform.is_rocm():
    from .amd.dspark import (
        MistralLarge4DSparkForCausalLM,  # type: ignore[assignment]  # noqa: F401
    )
    from .amd.model import MistralLarge4ForCausalLM  # type: ignore[assignment]

    __all__.append("MistralLarge4DSparkForCausalLM")
else:
    raise NotImplementedError(
        "Mistral Large 4 is only supported on NVIDIA and AMD GPUs."
    )
