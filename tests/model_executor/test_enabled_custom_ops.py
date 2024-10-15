import os
from typing import List

import pytest

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import (GeluAndMul,
                                                   ReLUSquaredActivation,
                                                   SiluAndMul)
from vllm.model_executor.layers.layernorm import RMSNorm


@pytest.mark.parametrize(
    "env, torch_level, ops_enabled, default_on",
    [
        # Default values based on compile level
        ("", 0, [True] * 4, True),
        ("", 1, [True] * 4, True),
        ("", 2, [True] * 4, True),  # All by default
        ("", 3, [False] * 4, False),
        ("", 4, [False] * 4, False),  # None by default
        # Explicitly enabling/disabling
        # Default: all
        ("+rms_norm,-silu_and_mul", 0, [True, False, True, True], True),  # All but SiluAndMul
        ("none,-rms_norm,+relu2", 0, [False, False, False, True], False),  # Only ReLUSquaredActivation
        ("all,-silu_and_mul", 1, [True, False, True, True], True),  # All but SiluAndMul
        ("-relu2", 1, [True, True, True, False], True),  # All but ReLUSquaredActivation
        ("none,-relu2,+gelu_and_mul,+silu_and_mul", 2,[False, True, True, False], False),  # GeluAndMul and SiluAndMul
        ("-rms_norm", 2, [False, True, True, True], True),  # All but RMSNorm
        # Default: none
        ("-silu_and_mul,+relu2", 3, [False, False, False, True], False),  # Only ReLUSquaredActivation
        ("all,-rms_norm", 4, [False, True, True, True], True),  # All but RMSNorm
    ])
def test_enabled_ops(env: str, torch_level: int, ops_enabled: List[bool],
                     default_on: bool):
    os.environ["VLLM_CUSTOM_OPS"] = env
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(torch_level)

    # Reset default_on (computed once):
    CustomOp.default_on.cache_clear()

    assert CustomOp.default_on() == default_on

    assert RMSNorm(1024)._enabled() == ops_enabled[0]
    assert SiluAndMul()._enabled() == ops_enabled[1]
    assert GeluAndMul()._enabled() == ops_enabled[2]
    assert ReLUSquaredActivation()._enabled() == ops_enabled[3]


@pytest.mark.parametrize(
    "env", ["all,none", "all,+rms_norm,all", "+rms_norm,-rms_norm"])
def test_enabled_ops_invalid(env: str):
    os.environ["VLLM_CUSTOM_OPS"] = env
    CustomOp.default_on.cache_clear()

    with pytest.raises(AssertionError):
        RMSNorm(1024)._enabled()
