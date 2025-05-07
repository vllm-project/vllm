# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Callable, Optional, Union

from vllm.logger import init_logger

logger = init_logger(__file__)

CHAT_TEMPLATES_DIR = Path(__file__).parent

ChatTemplatePath = Union[Path, Callable[[str], Optional[Path]]]


def _get_qwen_chat_template_fallback(
        tokenizer_name_or_path: str) -> Optional[Path]:
    if tokenizer_name_or_path.endswith("-Chat"):
        return CHAT_TEMPLATES_DIR / "template_chatml.jinja"

    return CHAT_TEMPLATES_DIR / "template_basic.jinja"


# yapf: disable
_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK: dict[str, ChatTemplatePath] = {
    "blip-2": CHAT_TEMPLATES_DIR / "template_blip2.jinja",
    "chameleon": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "deepseek_vl_v2": CHAT_TEMPLATES_DIR / "template_deepseek_vl2.jinja",
    "florence2": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "fuyu": CHAT_TEMPLATES_DIR / "template_fuyu.jinja",
    "paligemma": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "qwen": _get_qwen_chat_template_fallback,
}
# yapf: enable


def register_chat_template_fallback_path(
    model_type: str,
    chat_template: ChatTemplatePath,
) -> None:
    if model_type in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Model type %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.", model_type,
            chat_template)

    _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK[model_type] = chat_template


def get_chat_template_fallback_path(
    model_type: str,
    tokenizer_name_or_path: str,
) -> Optional[Path]:
    chat_template = _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK.get(model_type)
    if callable(chat_template):
        chat_template = chat_template(tokenizer_name_or_path)

    if chat_template is None:
        return None

    return chat_template
