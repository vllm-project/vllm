import pytest

from vllm.model_executor.model_loader.weight_utils import (
    resolve_hf_overrides_for_quant,
)


def test_dict_passthrough():
    assert resolve_hf_overrides_for_quant({"a": 1}) == {"a": 1}


def test_none_becomes_empty():
    assert resolve_hf_overrides_for_quant(None) == {}


def test_callable_with_hf_config():
    def compose(cfg):
        return {"quantization_config_dict_json": cfg}

    assert resolve_hf_overrides_for_quant(compose, hf_config="draft") == {
        "quantization_config_dict_json": "draft"
    }


def test_callable_zero_arg():
    assert resolve_hf_overrides_for_quant(lambda: {"k": 1}) == {"k": 1}


def test_invalid_type_still_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        resolve_hf_overrides_for_quant(["not-a-dict"])
