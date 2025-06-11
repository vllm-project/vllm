# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TemplateInfo:
    path: ChatTemplatePath
    override_exists: bool = False

    def get_path(self, tokenizer_name_or_path: str) -> Optional[Path]:
        if callable(self.path):
            return self.path(tokenizer_name_or_path)

        return self.path


# yapf: disable
_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK: dict[str, TemplateInfo] = {
    "blip-2": TemplateInfo(CHAT_TEMPLATES_DIR / "template_blip2.jinja"),
    "chameleon": TemplateInfo(CHAT_TEMPLATES_DIR / "template_basic.jinja"),
    "deepseek_vl_v2": TemplateInfo(CHAT_TEMPLATES_DIR / "template_deepseek_vl2.jinja"), # noqa: E501
    "florence2": TemplateInfo(CHAT_TEMPLATES_DIR / "template_basic.jinja"),
    "fuyu": TemplateInfo(CHAT_TEMPLATES_DIR / "template_fuyu.jinja"),
    "paligemma": TemplateInfo(CHAT_TEMPLATES_DIR / "template_basic.jinja"),
    "qwen": TemplateInfo(_get_qwen_chat_template_fallback),
    "qwen2_audio": TemplateInfo(CHAT_TEMPLATES_DIR / "template_qwen2_audio.jinja",  # noqa: E501
                                override_exists=True),
}
# yapf: enable


def register_chat_template_fallback_path(
    model_type: str,
    chat_template: ChatTemplatePath,
    override_exists: bool = False,
) -> None:
    if model_type in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Model type %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.", model_type,
            chat_template)

    _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK[model_type] = TemplateInfo(
        chat_template, override_exists=override_exists)


def get_chat_template_fallback(model_type: str) -> Optional[TemplateInfo]:
    chat_template_info = _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK.get(model_type)

    if chat_template_info is None:
        return None

    return chat_template_info


def get_chat_template_fallback_path(
    model_type: str,
    tokenizer_name_or_path: str,
) -> Optional[Path]:
    chat_template_info = get_chat_template_fallback(model_type)

    if chat_template_info is None:
        return None

    return chat_template_info.get_path(tokenizer_name_or_path)
