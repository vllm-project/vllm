# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Specialized model implementations.

Each entry maps a vLLM architecture name to a (module_path, class_name)
tuple, exactly like the main model registry.  When
``VLLM_USE_SPECIALIZED_MODELS=1`` the main registry merges these entries
so they take priority over the generic implementations.

To add a new specialized model:
  1. Create a sub-package under this directory.
  2. Add the architecture -> (module, class) mapping to ``_MODELS`` below.
"""

_MODELS: dict[str, tuple[str, str]] = {
    "KimiK25ForConditionalGeneration": (
        "vllm.model_executor.specialized_models.kimi_k2_5_nvfp4",
        "KimiK25ForConditionalGeneration",
    ),
}


def get_specialized_models() -> dict[str, tuple[str, str]]:
    """Return the specialized model registry."""
    return _MODELS
