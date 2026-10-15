# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

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


def test_import_vllm_does_not_resolve_platform():
    # Platform plugins must be loaded after `import vllm` completes; resolving
    # current_platform during it runs plugins against a half-imported vLLM.
    # Needs a fresh interpreter, since vllm is already imported here.
    code = (
        "import sys, vllm\n"
        "p = sys.modules.get('vllm.platforms')\n"
        "assert p is None or p._current_platform is None, p._init_trace\n"
        "from vllm.platforms import current_platform\n"
        "assert current_platform.device_name == 'DummyDevice'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
