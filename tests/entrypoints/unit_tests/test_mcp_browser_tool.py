# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HarmonyBrowserTool backend selection."""

import sys
from types import ModuleType

import pytest

import vllm.entrypoints.mcp.tool as tool_module
from vllm.entrypoints.mcp.tool import HarmonyBrowserTool

pytestmark = pytest.mark.skip_global_cleanup


class FakeExaBackend:
    def __init__(self, source, api_key):
        self.source = source
        self.api_key = api_key


class FakeYouComBackend:
    def __init__(self, source):
        self.source = source


class FakeSimpleBrowserTool:
    def __init__(self, backend):
        self.backend = backend


@pytest.fixture
def gpt_oss_stub(monkeypatch):
    """Stub the lazily imported gpt_oss modules and the install check."""
    for var in ("BROWSER_BACKEND", "EXA_API_KEY", "YDC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    backend_mod = ModuleType("gpt_oss.tools.simple_browser.backend")
    backend_mod.__dict__["ExaBackend"] = FakeExaBackend
    backend_mod.__dict__["YouComBackend"] = FakeYouComBackend
    browser_mod = ModuleType("gpt_oss.tools.simple_browser")
    browser_mod.__dict__["SimpleBrowserTool"] = FakeSimpleBrowserTool

    for name, mod in (
        ("gpt_oss", ModuleType("gpt_oss")),
        ("gpt_oss.tools", ModuleType("gpt_oss.tools")),
        ("gpt_oss.tools.simple_browser", browser_mod),
        ("gpt_oss.tools.simple_browser.backend", backend_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(tool_module, "validate_gpt_oss_install", lambda: None)


def test_default_backend_is_exa(gpt_oss_stub, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    tool = HarmonyBrowserTool()
    assert tool.enabled
    assert isinstance(tool.browser_tool.backend, FakeExaBackend)
    assert tool.browser_tool.backend.api_key == "exa-key"


def test_youcom_backend_opt_in(gpt_oss_stub, monkeypatch):
    monkeypatch.setenv("BROWSER_BACKEND", "youcom")
    monkeypatch.setenv("YDC_API_KEY", "ydc-key")
    tool = HarmonyBrowserTool()
    assert tool.enabled
    assert isinstance(tool.browser_tool.backend, FakeYouComBackend)


@pytest.mark.parametrize(
    "backend,key_var", [("exa", "EXA_API_KEY"), ("youcom", "YDC_API_KEY")]
)
def test_missing_api_key_disables_browsing(gpt_oss_stub, monkeypatch, backend, key_var):
    monkeypatch.setenv("BROWSER_BACKEND", backend)
    tool = HarmonyBrowserTool()
    assert not tool.enabled
    assert not hasattr(tool, "browser_tool")


def test_unknown_backend_disables_browsing(gpt_oss_stub, monkeypatch):
    monkeypatch.setenv("BROWSER_BACKEND", "not-a-backend")
    tool = HarmonyBrowserTool()
    assert not tool.enabled
    assert not hasattr(tool, "browser_tool")


def test_missing_gpt_oss_disables_browsing(gpt_oss_stub, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "exa-key")

    def raise_import_error():
        raise ImportError("no gpt_oss")

    monkeypatch.setattr(tool_module, "validate_gpt_oss_install", raise_import_error)
    tool = HarmonyBrowserTool()
    assert not tool.enabled
    assert not hasattr(tool, "browser_tool")
