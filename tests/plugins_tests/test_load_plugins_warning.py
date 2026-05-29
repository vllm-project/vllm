# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the warning surfaced when a name in ``VLLM_PLUGINS`` does
not match any entry-point discovered for the requested group.

The motivating scenario is a developer doing ``pip install -e .`` whose
package ships an entry-point under ``vllm.general_plugins`` but who
also has a stale ``.egg-info`` from a previous ``python setup.py
develop`` shadowing the new ``.dist-info``. Discovery returns the
stale dist (without the entry-point), the plugin is silently skipped,
and the user sees the downstream "Model architectures are not
supported" error with no breadcrumb.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import vllm.plugins as plugins_mod


class _FakeEntryPoint:
    """Minimal duck-typed stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(self, name: str):
        self.name = name
        self.value = f"fake_module:{name}"

    def load(self):
        return lambda: None


def _patched_entry_points(names):
    """Return a callable mimicking ``importlib.metadata.entry_points``."""

    def _ep(group: str):
        del group  # the fake doesn't care
        return [_FakeEntryPoint(n) for n in names]

    return _ep


class _RecordingHandler(logging.Handler):
    """Handler that captures every record seen, independent of pytest's
    `caplog` (vLLM disables logger propagation, which bypasses caplog)."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _capture_records():
    """Attach a recording handler to the vllm.plugins logger."""
    handler = _RecordingHandler()
    plugins_mod.logger.addHandler(handler)
    return handler


def _detach(handler):
    plugins_mod.logger.removeHandler(handler)


def test_warning_emitted_for_missing_plugin():
    handler = _capture_records()
    try:
        with (
            patch(
                "vllm.plugins.envs.VLLM_PLUGINS",
                ["does_not_exist", "lora_filesystem_resolver"],
            ),
            patch(
                "importlib.metadata.entry_points",
                _patched_entry_points(["lora_filesystem_resolver"]),
            ),
        ):
            plugins_mod.load_plugins_by_group(plugins_mod.DEFAULT_PLUGINS_GROUP)
    finally:
        _detach(handler)

    matches = [
        r
        for r in handler.records
        if "does_not_exist" in r.getMessage() and r.levelno >= logging.WARNING
    ]
    assert matches, (
        "expected a WARNING mentioning the missing plugin name, "
        f"got records: {[r.getMessage() for r in handler.records]}"
    )


def test_no_warning_when_all_allowed_plugins_resolve():
    handler = _capture_records()
    try:
        with (
            patch("vllm.plugins.envs.VLLM_PLUGINS", ["lora_filesystem_resolver"]),
            patch(
                "importlib.metadata.entry_points",
                _patched_entry_points(["lora_filesystem_resolver"]),
            ),
        ):
            plugins_mod.load_plugins_by_group(plugins_mod.DEFAULT_PLUGINS_GROUP)
    finally:
        _detach(handler)

    bad = [r for r in handler.records if "VLLM_PLUGINS requested" in r.getMessage()]
    assert not bad, (
        "no warning expected when every allowed plugin resolves, "
        f"got: {[r.getMessage() for r in bad]}"
    )


def test_no_warning_when_vllm_plugins_unset():
    handler = _capture_records()
    try:
        with (
            patch("vllm.plugins.envs.VLLM_PLUGINS", None),
            patch(
                "importlib.metadata.entry_points",
                _patched_entry_points(["lora_filesystem_resolver"]),
            ),
        ):
            plugins_mod.load_plugins_by_group(plugins_mod.DEFAULT_PLUGINS_GROUP)
    finally:
        _detach(handler)

    bad = [r for r in handler.records if "VLLM_PLUGINS requested" in r.getMessage()]
    assert not bad, (
        "no warning expected when VLLM_PLUGINS is unset, "
        f"got: {[r.getMessage() for r in bad]}"
    )
