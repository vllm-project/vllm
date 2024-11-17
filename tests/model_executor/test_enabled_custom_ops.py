import os
from typing import List

import pytest

from vllm.config import CompilationConfig, VllmConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import (GeluAndMul,
                                                   ReLUSquaredActivation,
                                                   SiluAndMul)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.plugins import set_current_vllm_config


# Registered subclass for test
@CustomOp.register("relu3")
class Relu3(ReLUSquaredActivation):
    pass


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
        # Only ReLU3
        ("none,-rms_norm,+relu3", 0, [0, 0, 0, 1], False),
        # All but SiluAndMul
        ("all,-silu_and_mul", 1, [1, 0, 1, 1], True),
        # All but ReLU3 (even if ReLU2 is on)
        ("-relu3,relu2", 1, [1, 1, 1, 0], True),
        # GeluAndMul and SiluAndMul
        ("none,-relu3,+gelu_and_mul,+silu_and_mul", 2, [0, 1, 1, 0], False),
        # All but RMSNorm
        ("-rms_norm", 2, [0, 1, 1, 1], True),
        #
        # Default: none
        #
        # Only ReLU3
        ("-silu_and_mul,+relu3", 3, [0, 0, 0, 1], False),
        # All but RMSNorm
        ("all,-rms_norm", 4, [0, 1, 1, 1], True),
    ])
def test_enabled_ops(env: str, torch_level: int, ops_enabled: List[int],
                     default_on: bool):
    os.environ["VLLM_TORCH_COMPILE_LEVEL"] = str(torch_level)
    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        custom_ops=env.split(",")))
    with set_current_vllm_config(vllm_config):
        assert CustomOp.default_on() == default_on

        ops_enabled = [bool(x) for x in ops_enabled]

        assert RMSNorm(1024).enabled() == ops_enabled[0]
        assert CustomOp.op_registry["rms_norm"].enabled() == ops_enabled[0]

        assert SiluAndMul().enabled() == ops_enabled[1]
        assert CustomOp.op_registry["silu_and_mul"].enabled() == ops_enabled[1]

        assert GeluAndMul().enabled() == ops_enabled[2]
        assert CustomOp.op_registry["gelu_and_mul"].enabled() == ops_enabled[2]

        # If registered, subclasses should follow their own name
        assert Relu3().enabled() == ops_enabled[3]
        assert CustomOp.op_registry["relu3"].enabled() == ops_enabled[3]

        # Unregistered subclass
        class SiluAndMul2(SiluAndMul):
            pass

        # Subclasses should not require registration
        assert SiluAndMul2().enabled() == SiluAndMul().enabled()


@pytest.mark.parametrize(
    "env", ["all,none", "all,+rms_norm,all", "+rms_norm,-rms_norm"])
def test_enabled_ops_invalid(env: str):
    with pytest.raises(Exception):  # noqa
        vllm_config = VllmConfig(compilation_config=CompilationConfig(
            custom_ops=env.split(",")))
        with set_current_vllm_config(vllm_config):
            RMSNorm(1024).enabled()
