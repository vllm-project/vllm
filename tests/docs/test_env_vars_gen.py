# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the env-vars docs generator.

These tests run the generator's pure-Python rendering helpers (no mkdocs)
and assert structural properties of the output.
"""

from __future__ import annotations

import pytest
import regex as re

import vllm.envs as envs


@pytest.fixture
def rendered():
    """Run the generator and return the produced markdown string."""
    from docs.mkdocs.plugins.gen_env_vars import render_env_vars_page

    return render_env_vars_page()


def test_warning_admonition_is_first(rendered):
    # The k8s warning must appear before any sub-model heading.
    warn_idx = rendered.find("!!! warning")
    first_h2 = rendered.find("\n## ")
    assert warn_idx >= 0, "Expected '!!! warning' admonition in output"
    assert first_h2 > warn_idx, "Warning must precede first sub-model heading"


def test_h1_present(rendered):
    assert rendered.startswith("# Environment Variables\n"), (
        "Page must start with '# Environment Variables'"
    )


def test_every_known_env_var_appears_exactly_once(rendered):
    # Each env var must appear as an H3 heading exactly once.
    for name in envs._VAR_TO_PATH:
        heading = f"### `{name}`"
        count = rendered.count(heading)
        assert count == 1, f"Expected exactly one '### `{name}`' heading, found {count}"


def test_no_extra_h3_headings(rendered):
    # All H3 headings should be env var names from the registry.
    h3_pattern = re.compile(r"^### `([^`]+)`", re.MULTILINE)
    headings = set(h3_pattern.findall(rendered))
    registry = set(envs._VAR_TO_PATH)
    extra = headings - registry
    assert not extra, f"Unexpected H3 headings (not in registry): {sorted(extra)}"


def test_each_submodel_appears_as_h2(rendered):
    # Every sub-model in Settings.model_fields should have an H2 entry.
    from pydantic_settings import BaseSettings

    h2_pattern = re.compile(r"^## (\w+)", re.MULTILINE)
    h2_names = set(h2_pattern.findall(rendered))
    for sub_attr, sub_field in envs.Settings.model_fields.items():
        sub_cls = sub_field.annotation
        if not (isinstance(sub_cls, type) and issubclass(sub_cls, BaseSettings)):
            continue
        assert sub_cls.__name__ in h2_names, (
            f"Sub-model {sub_cls.__name__} not rendered as H2"
        )


def test_vllm_port_renders(rendered):
    section = _extract_section(rendered, "VLLM_PORT")
    assert "**Type:**" in section
    assert "**Default:**" in section
    # Default for VLLM_PORT is None
    assert "Default:** `None`" in section


def test_vllm_logging_level_renders(rendered):
    section = _extract_section(rendered, "VLLM_LOGGING_LEVEL")
    assert "Default:** `'INFO'`" in section


def test_vllm_do_not_track_mentions_fallback(rendered):
    # AliasChoices fields should expose the fallback alias somewhere
    # in their section (either via Field.description or generator-injected note).
    section = _extract_section(rendered, "VLLM_DO_NOT_TRACK")
    assert "DO_NOT_TRACK" in section


def test_path_defaults_use_stable_display_strings(rendered):
    # VLLM_CACHE_ROOT etc. must NOT bake the build machine's home directory
    # into the rendered docs.
    section = _extract_section(rendered, "VLLM_CACHE_ROOT")
    assert "Default:**" in section
    # Stable form
    assert "~/.cache/vllm" in section or "$XDG_CACHE_HOME" in section
    # No bake-in
    assert "/Users/" not in section
    assert "/home/" not in section


def test_rpc_base_path_uses_stable_display_string(rendered):
    section = _extract_section(rendered, "VLLM_RPC_BASE_PATH")
    assert "<system tempdir>" in section


def _extract_section(rendered: str, env_name: str) -> str:
    """Return the markdown subsection for env_name (heading + content
    until the next H2 or H3)."""
    heading = f"### `{env_name}`"
    start = rendered.find(heading)
    assert start >= 0, f"{heading} not found"
    # Find the next H2 or H3
    after = rendered[start + len(heading) :]
    next_h3 = after.find("\n### ")
    next_h2 = after.find("\n## ")
    candidates = [x for x in (next_h3, next_h2) if x >= 0]
    end = min(candidates) if candidates else len(after)
    return rendered[start : start + len(heading) + end]
