# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from typing_extensions import override

from .hf import HfRenderer

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatTemplateContentFormat

    from .params import ChatParams


class HYV4Renderer(HfRenderer):
    """HF rendering with string content for the default HYV4 VL template."""

    @override
    def _resolve_content_format(
        self, params: "ChatParams"
    ) -> "ChatTemplateContentFormat":
        if params.chat_template is None:
            given_format = params.chat_template_content_format
            return "string" if given_format == "auto" else given_format
        return super()._resolve_content_format(params)
