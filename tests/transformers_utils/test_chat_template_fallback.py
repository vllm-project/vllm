# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.transformers_utils.chat_templates.registry import (
    get_chat_template_fallback_path,
)


@pytest.mark.parametrize(
    ("name_or_path", "expected"),
    [
        ("openbmb/MiniCPM-V-4_5", "template_minicpmv45.jinja"),
        ("openbmb/MiniCPM-V-4.5", "template_minicpmv45.jinja"),
        ("openbmb/MiniCPM-V-4.5-int4", "template_minicpmv45.jinja"),
        ("openbmb/MiniCPM-V-4.0", "template_chatml.jinja"),
        ("openbmb/MiniCPM-V-2_6", "template_chatml.jinja"),
        # "4.5" as a substring of a longer number must not select the 4.5
        # template.
        ("openbmb/MiniCPM-V-14.5", "template_chatml.jinja"),
        ("/runs/2.4.5/MiniCPM-V-4.0", "template_chatml.jinja"),
    ],
)
def test_minicpmv_chat_template_fallback_version_match(name_or_path, expected):
    path = get_chat_template_fallback_path("minicpmv", name_or_path)
    assert path is not None
    assert path.name == expected
