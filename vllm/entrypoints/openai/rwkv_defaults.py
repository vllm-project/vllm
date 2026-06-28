# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth

RWKV_TOOL_CALL_PARSER = "rwkv"
RWKV_DEFAULT_STOPS = ("\nUser:", "\n### User")
RWKV_DEFAULT_STOP_TOKEN_IDS = (0,)


def _is_rwkv_tokenizer_mode(tokenizer_mode: Any) -> bool:
    return isinstance(tokenizer_mode, str) and tokenizer_mode.lower() == "rwkv"


def is_rwkv_model_config(model_config: Any) -> bool:
    if _is_rwkv_tokenizer_mode(getattr(model_config, "tokenizer_mode", None)):
        return True
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "rwkv7"


def is_rwkv_model_arg(model: str | Path | None) -> bool:
    if model is None:
        return False
    try:
        return build_rwkv7_config_from_pth(model) is not None
    except ValueError:
        return False


def resolve_rwkv_tool_parser(
    *,
    tool_parser: str | None,
    enable_auto_tools: bool,
    model_config: Any | None = None,
    tokenizer_mode: str | None = None,
    model: str | Path | None = None,
) -> str | None:
    if tool_parser is not None or not enable_auto_tools:
        return tool_parser
    if model_config is not None and is_rwkv_model_config(model_config):
        return RWKV_TOOL_CALL_PARSER
    if _is_rwkv_tokenizer_mode(tokenizer_mode) or is_rwkv_model_arg(model):
        return RWKV_TOOL_CALL_PARSER
    return tool_parser


def apply_rwkv_default_sampling_params(
    default_sampling_params: dict[str, Any],
    model_config: Any,
) -> None:
    if not is_rwkv_model_config(model_config):
        return
    if "stop" not in default_sampling_params:
        default_sampling_params["stop"] = list(RWKV_DEFAULT_STOPS)
    if "stop_token_ids" not in default_sampling_params:
        default_sampling_params["stop_token_ids"] = list(RWKV_DEFAULT_STOP_TOKEN_IDS)
