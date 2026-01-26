# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from vllm.logger import init_logger

logger = init_logger(__file__)

CHAT_TEMPLATES_DIR = Path(__file__).parent

ChatTemplatePath: TypeAlias = Path | Callable[[str], Path | None]


def _get_qwen_chat_template_fallback(tokenizer_name_or_path: str) -> Path | None:
    if tokenizer_name_or_path.endswith("-Chat"):
        return CHAT_TEMPLATES_DIR / "template_chatml.jinja"

    return CHAT_TEMPLATES_DIR / "template_basic.jinja"


def _get_minicpmv_chat_template_fallback(tokenizer_name_or_path: str) -> Path | None:
    # MiniCPM-V-4.5 version uses a dedicated template
    if "4.5" in tokenizer_name_or_path or "4_5" in tokenizer_name_or_path:
        return CHAT_TEMPLATES_DIR / "template_minicpmv45.jinja"

    # Other versions use chatml template
    return CHAT_TEMPLATES_DIR / "template_chatml.jinja"


_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK: dict[str, ChatTemplatePath] = {
    "blip-2": CHAT_TEMPLATES_DIR / "template_blip2.jinja",
    "chameleon": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "clip": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "deepseek_ocr": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
    "deepseek_vl_v2": CHAT_TEMPLATES_DIR / "template_deepseek_vl2.jinja",
    "fuyu": CHAT_TEMPLATES_DIR / "template_fuyu.jinja",
    "minicpmv": _get_minicpmv_chat_template_fallback,
    "paligemma": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "qwen": _get_qwen_chat_template_fallback,
    "siglip": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "siglip2": CHAT_TEMPLATES_DIR / "template_basic.jinja",
}

# Model types that require a custom tool chat template to properly handle
# special characters (like parentheses) in tool parameter descriptions.
# These templates are used instead of the HuggingFace tokenizer's template
# when tools are provided in the request.
_MODEL_TYPE_TO_TOOL_CHAT_TEMPLATE: dict[str, ChatTemplatePath] = {
    "minimax_m2": CHAT_TEMPLATES_DIR / "template_minimax_m2.jinja",
}


def register_chat_template_fallback_path(
    model_type: str,
    chat_template: ChatTemplatePath,
) -> None:
    if model_type in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Model type %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.",
            model_type,
            chat_template,
        )

    _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK[model_type] = chat_template


def get_chat_template_fallback_path(
    model_type: str,
    tokenizer_name_or_path: str,
) -> Path | None:
    chat_template = _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK.get(model_type)
    if callable(chat_template):
        chat_template = chat_template(tokenizer_name_or_path)

    if chat_template is None:
        return None

    return chat_template


def get_tool_chat_template_path(
    model_type: str,
    tokenizer_name_or_path: str,
) -> Path | None:
    """
    Get a custom tool chat template for models that have issues with the
    default HuggingFace tokenizer template when handling tools.

    This is used for models like MiniMax-M2 where the default template
    doesn't properly handle special characters (e.g., parentheses) in
    tool parameter descriptions.

    Args:
        model_type: The model type (e.g., "minimax_m2")
        tokenizer_name_or_path: The tokenizer name or path

    Returns:
        Path to the tool chat template, or None if no custom template exists
    """
    chat_template = _MODEL_TYPE_TO_TOOL_CHAT_TEMPLATE.get(model_type)
    if callable(chat_template):
        chat_template = chat_template(tokenizer_name_or_path)

    if chat_template is None:
        return None

    return chat_template
