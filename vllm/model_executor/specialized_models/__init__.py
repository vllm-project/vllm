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

from __future__ import annotations

# ── Model list ───────────────────────────────────────────────────────
# Maps architecture name -> (fully-qualified module, class name).
# When the flag is enabled, these override the corresponding entries
# in the main registry.
_MODELS: dict[str, tuple[str, str]] = {
    "DeepseekV32ForCausalLM": (
        "vllm.model_executor.specialized_models.deepseek_v3_2_nvfp4",
        "DeepseekV32ForCausalLM",
    ),
    "DeepSeekMTPModel": (
        "vllm.model_executor.specialized_models.deepseek_v3_2_nvfp4",
        "DeepSeekMTP",
    ),
}


def get_specialized_models() -> dict[str, tuple[str, str]]:
    """Return the specialized model registry."""
    return _MODELS
