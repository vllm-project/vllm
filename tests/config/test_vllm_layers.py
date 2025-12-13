# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import (
    VllmConfig,
    get_layers_from_vllm_config,
    resolve_layers_from_vllm_config,
)


class _DummyLayer:
    pass


def test_get_layers_from_vllm_config_skips_missing_entries():
    cfg = VllmConfig()
    cfg.compilation_config.static_forward_context = {
        "layer_present": _DummyLayer(),
    }

    layers = get_layers_from_vllm_config(
        cfg, _DummyLayer, ["layer_present", "layer_missing"]
    )
    assert list(layers.keys()) == ["layer_present"]

    layers_with_missing, missing = resolve_layers_from_vllm_config(
        cfg, _DummyLayer, ["layer_present", "layer_missing"]
    )
    assert list(layers_with_missing.keys()) == ["layer_present"]
    assert missing == ["layer_missing"]
