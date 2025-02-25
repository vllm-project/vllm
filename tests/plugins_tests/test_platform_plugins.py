# SPDX-License-Identifier: Apache-2.0

import torch

from tests.kernels.utils import override_backend_env_variable
from vllm.attention.selector import get_attn_backend
from vllm.utils import STR_INVALID_VAL


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


def test_oot_attention_backend(monkeypatch):
    # ignore the backend env variable if it is set
    override_backend_env_variable(monkeypatch, STR_INVALID_VAL)
    backend = get_attn_backend(16, torch.float16, torch.float16, 16, False)
    assert backend.get_name() == "Dummy_Backend"
