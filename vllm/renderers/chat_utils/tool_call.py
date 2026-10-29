# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import ModelConfig
from vllm.utils import random_uuid

_KIMI_MODEL_TYPES = ("kimi_k2", "kimi_k25", "kimi_k3")


def get_tool_call_id_type(model_config: ModelConfig) -> str:
    """Return the tool-call ID type for a given model configuration."""
    hf_overrides = getattr(model_config, "hf_overrides", None)
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_types = (
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
    )
    if any(model_type in _KIMI_MODEL_TYPES for model_type in model_types) or (
        isinstance(hf_overrides, dict)
        and hf_overrides.get("model_type") in _KIMI_MODEL_TYPES
    ):
        return "kimi_k2"
    return "random"


def make_tool_call_id(id_type: str = "random", func_name=None, idx=None):
    if id_type == "kimi_k2":
        return f"functions.{func_name}:{idx}"
    else:
        # by default return random
        return f"chatcmpl-tool-{random_uuid()}"
