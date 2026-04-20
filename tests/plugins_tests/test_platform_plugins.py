# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


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
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    # Define and register an OOT RotaryEmbedding inline so the test is
    # self-contained (works with or without the plugin pip-installed).
    # monkeypatch.setitem ensures op_registry_oot is restored after the test.
    class DummyRotaryEmbedding(RotaryEmbedding):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.addition_config = True

        def forward_oot(self, *args, **kwargs):
            return super().forward_oot(*args, **kwargs)

    monkeypatch.setitem(op_registry_oot, "RotaryEmbedding", DummyRotaryEmbedding)

    layer = RotaryEmbedding(16, 16, 16, 16, True, torch.float16)
    assert layer.__class__.__name__ == "DummyRotaryEmbedding", (
        f"Expected DummyRotaryEmbedding, got {layer.__class__.__name__}, "
        "possibly because the custom op is not registered correctly."
    )
    assert hasattr(layer, "addition_config"), (
        "Expected DummyRotaryEmbedding to have an 'addition_config' attribute, "
        "which is set by the custom op."
    )
