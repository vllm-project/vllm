# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.gemma4_mm import _get_tower_quant_config


class _DummyQuantConfig:
    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name


@pytest.mark.parametrize(
    "hidden_size, intermediate_size, expected",
    [
        (768, 4304, False),
        (1024, 4096, True),
        (1024, None, True),
    ],
)
def test_tower_quantization_uses_tower_dimensions(
    hidden_size: int, intermediate_size: int | None, expected: bool
):
    dimensions = {"hidden_size": hidden_size}
    if intermediate_size is not None:
        dimensions["intermediate_size"] = intermediate_size
    tower_config = SimpleNamespace(**dimensions)
    quant_config = _DummyQuantConfig("auto_gptq")

    result = _get_tower_quant_config(tower_config, quant_config)

    assert (result is quant_config) is expected


def test_audio_tower_quantization_uses_hidden_size_without_intermediate_size():
    tower_config = SimpleNamespace(hidden_size=1024, output_proj_dims=1536)
    quant_config = _DummyQuantConfig("auto_gptq")

    assert _get_tower_quant_config(tower_config, quant_config) is quant_config


@pytest.mark.parametrize("name", ["bitsandbytes", "torchao", "compressed-tensors"])
def test_tower_quantization_keeps_arbitrary_dimension_methods(name: str):
    tower_config = SimpleNamespace(hidden_size=768, intermediate_size=4304)
    quant_config = _DummyQuantConfig(name)

    assert _get_tower_quant_config(tower_config, quant_config) is quant_config


def test_tower_quantization_handles_missing_config():
    tower_config = SimpleNamespace(hidden_size=1024, intermediate_size=4096)

    assert _get_tower_quant_config(tower_config, None) is None
