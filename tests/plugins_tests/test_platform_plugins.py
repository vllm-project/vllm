# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import pytest
import torch

from vllm.plugins import load_general_plugins


def test_platform_plugins():
    # simulate workload by running an example
    import runpy

    current_file = __file__
    import os

    example_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
        "examples",
        "basic/offline_inference/basic.py",
    )
    runpy.run_path(example_file)

    # check if the plugin is loaded correctly
    from vllm.platforms import _init_trace, current_platform

    assert current_platform.device_name == "DummyDevice", (
        f"Expected DummyDevice, got {current_platform.device_name}, "
        "possibly because current_platform is imported before the plugin"
        f" is loaded. The first import:\n{_init_trace}"
    )


def test_oot_custom_op(default_vllm_config, monkeypatch: pytest.MonkeyPatch):
    # simulate workload by running an example
    load_general_plugins()
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    layer = RotaryEmbedding(16, 16, 16, 16, True, torch.float16)
    assert layer.__class__.__name__ == "DummyRotaryEmbedding", (
        f"Expected DummyRotaryEmbedding, got {layer.__class__.__name__}, "
        "possibly because the custom op is not registered correctly."
    )
    assert hasattr(layer, "addition_config"), (
        "Expected DummyRotaryEmbedding to have an 'addition_config' attribute, "
        "which is set by the custom op."
    )


def test_broken_platform_plugin_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    from vllm.platforms import resolve_current_platform_cls_qualname

    def broken_plugin() -> str | None:
        raise ModuleNotFoundError("no module named 'totally_broken_plugin'")

    monkeypatch.setattr(
        "vllm.platforms.load_plugins_by_group",
        lambda group: {"broken": broken_plugin},
    )

    with caplog.at_level(logging.WARNING):
        resolve_current_platform_cls_qualname()

    assert any(
        "broken" in record.getMessage() and "failed to load" in record.getMessage()
        for record in caplog.records
    ), "Expected a warning naming the broken plugin, but none was logged"
