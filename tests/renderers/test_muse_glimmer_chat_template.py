# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the MuseGlimmer chat template's reasoning strength line."""

from pathlib import Path

import jinja2.sandbox
import pytest

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "tool_chat_template_muse_glimmer.jinja"
)


@pytest.fixture(scope="module")
def glimmer_template():
    template_str = TEMPLATE_PATH.read_text()
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    return env.from_string(template_str)


def _render(template, **kwargs):
    kwargs.setdefault("bos_token", "")
    kwargs.setdefault("messages", [{"role": "user", "content": "Hello"}])
    kwargs.setdefault("add_generation_prompt", True)
    return template.render(**kwargs)


class TestMuseGlimmerReasoningStrength:
    def test_defaults_to_high(self, glimmer_template):
        result = _render(glimmer_template)
        assert "Reasoning strength: high." in result

    def test_reasoning_strength_kwarg(self, glimmer_template):
        result = _render(glimmer_template, reasoning_strength="low")
        assert "Reasoning strength: low." in result

    @pytest.mark.parametrize(
        ("effort", "strength"),
        [
            ("none", "low"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "xhigh"),
        ],
    )
    def test_reasoning_effort_fallback(self, glimmer_template, effort, strength):
        """reasoning_effort is always forwarded to template kwargs by
        build_chat_params(); the template maps it to the strength line so
        the standard OpenAI parameter works without chat_template_kwargs.
        Efforts outside MuseGlimmer's supported low/medium/high/xhigh
        levels clamp to the nearest supported one."""
        result = _render(glimmer_template, reasoning_effort=effort)
        assert f"Reasoning strength: {strength}." in result

    def test_reasoning_strength_wins_over_effort(self, glimmer_template):
        result = _render(
            glimmer_template,
            reasoning_strength="low",
            reasoning_effort="high",
        )
        assert "Reasoning strength: low." in result
