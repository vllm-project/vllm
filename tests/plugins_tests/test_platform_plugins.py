# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.attention.selector import get_attn_backend
from vllm.plugins import load_general_plugins
from vllm.utils import STR_BACKEND_ENV_VAR, STR_INVALID_VAL


def test_platform_plugins():
    # simulate workload by running an example
    import runpy
    current_file = __file__
    import os
    example_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
        "examples", "offline_inference/basic/basic.py")
    runpy.run_path(example_file)

    # check if the plugin is loaded correctly
    from vllm.platforms import _init_trace, current_platform
    assert current_platform.device_name == "DummyDevice", (
        f"Expected DummyDevice, got {current_platform.device_name}, "
        "possibly because current_platform is imported before the plugin"
        f" is loaded. The first import:\n{_init_trace}")


def test_oot_attention_backend(monkeypatch: pytest.MonkeyPatch):
    # ignore the backend env variable if it is set
    with monkeypatch.context() as m:
        m.setenv(STR_BACKEND_ENV_VAR, STR_INVALID_VAL)
        backend = get_attn_backend(16, torch.float16, "auto", 16, False)
        assert backend.get_name() == "Dummy_Backend"


def test_oot_custom_op(monkeypatch: pytest.MonkeyPatch):
    # simulate workload by running an example
    load_general_plugins()
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    layer = RotaryEmbedding(16, 16, 16, 16, True, torch.float16)
    assert layer.__class__.__name__ == "DummyRotaryEmbedding", (
        f"Expected DummyRotaryEmbedding, got {layer.__class__.__name__}, "
        "possibly because the custom op is not registered correctly.")
    assert hasattr(layer, "addition_config"), (
        "Expected DummyRotaryEmbedding to have an 'addition_config' attribute, "
        "which is set by the custom op.")
