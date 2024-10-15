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
        #
        # Default: all
        #
        # All but SiluAndMul
        ("+rms_norm,-silu_and_mul", 0, [1, 0, 1, 1], True),
        # Only ReLUSquaredActivation
        ("none,-rms_norm,+relu2", 0, [0, 0, 0, 1], False),
        # All but SiluAndMul
        ("all,-silu_and_mul", 1, [1, 0, 1, 1], True),
        # All but ReLUSquaredActivation
        ("-relu2", 1, [1, 1, 1, 0], True),
        # GeluAndMul and SiluAndMul
        ("none,-relu2,+gelu_and_mul,+silu_and_mul", 2, [0, 1, 1, 0], False),
        # All but RMSNorm
        ("-rms_norm", 2, [0, 1, 1, 1], True),
        #
        # Default: none
        #
        # Only ReLUSquaredActivation
        ("-silu_and_mul,+relu2", 3, [0, 0, 0, 1], False),
        # All but RMSNorm
        ("all,-rms_norm", 4, [0, 1, 1, 1], True),
    ])
def test_enabled_ops(env: str, torch_level: int, ops_enabled: List[int],
                     default_on: bool):
    os.environ["VLLM_CUSTOM_OPS"] = env
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(torch_level)

    # Reset default_on (computed once):
    CustomOp.default_on.cache_clear()

    assert CustomOp.default_on() == default_on

    ops_enabled = [bool(x) for x in ops_enabled]

    assert RMSNorm(1024).enabled() == ops_enabled[0]
    assert CustomOp.op_registry["rms_norm"].enabled() == ops_enabled[0]

    assert SiluAndMul().enabled() == ops_enabled[1]
    assert CustomOp.op_registry["silu_and_mul"].enabled() == ops_enabled[1]

    assert GeluAndMul().enabled() == ops_enabled[2]
    assert CustomOp.op_registry["gelu_and_mul"].enabled() == ops_enabled[2]

    assert ReLUSquaredActivation().enabled() == ops_enabled[3]
    assert CustomOp.op_registry["relu2"].enabled() == ops_enabled[3]


@pytest.mark.parametrize(
    "env", ["all,none", "all,+rms_norm,all", "+rms_norm,-rms_norm"])
def test_enabled_ops_invalid(env: str):
    os.environ["VLLM_CUSTOM_OPS"] = env
    CustomOp.default_on.cache_clear()

    with pytest.raises(AssertionError):
        RMSNorm(1024).enabled()
