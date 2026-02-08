# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.mistral import (
    _remap_mistral_quantization_args,
)


@pytest.mark.parametrize("quant_method", ["compressed-tensors", "compressed_tensors"])
def test_remap_mistral_quantization_args_compressed_tensors(
    quant_method: str,
) -> None:
    config = {
        "quantization": {
            "quant_method": quant_method,
            "format": "float-quantized",
            "config_groups": {},
        }
    }

    remapped = _remap_mistral_quantization_args(config)

    assert "quantization" not in remapped
    assert remapped["quantization_config"]["quant_method"] == "compressed-tensors"
    assert remapped["quantization_config"]["format"] == "float-quantized"


def test_remap_mistral_quantization_args_unknown_raises() -> None:
    config = {"quantization": {"quant_method": "unknown"}}

    with pytest.raises(ValueError, match="Found unknown quantization"):
        _remap_mistral_quantization_args(config)
