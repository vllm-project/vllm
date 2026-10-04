# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest
import torch

import vllm.config.compilation as compilation
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.model_executor import custom_op as classic_custom_op
from vllm.model_executor.custom_op import CustomOp, op_registry
from vllm.model_executor.hw_agnostic import custom_op as hw_agnostic_custom_op
from vllm.model_executor.layers.activation import (
    GeluAndMul,
    ReLUSquaredActivation,
    SiluAndMul,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    dispatch_topk_sigmoid_func,
    dispatch_topk_softmax_func,
    vllm_topk_sigmoid,
    vllm_topk_softmax,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform

RMS_NORM_SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]


# Registered subclass for test
@CustomOp.register("relu3")
class Relu3(ReLUSquaredActivation):
    pass


# Dummy class used as a registry value. Only the keys matter here.
class DummyOp:
    pass


# Names held by only one registry. The real hw-agnostic layers use rms_norm and
# silu_and_mul, which the classic stack also has, so those cannot show which
# registry a name came from. The oot registries stay empty without a plugin, so
# the tests fill them in.
HW_ONLY_IN_TREE = "hw_only_op"
HW_ONLY_OOT = "HwOnlyOotOp"
CLASSIC_OOT_ONLY = "ClassicOotOnlyOp"


@pytest.mark.parametrize(
    "env, compilation_mode, backend, ops_enabled, default_on",
    [
        # Default values based on compile level
        # - All by default (no Inductor compilation)
        (None, 0, "eager", [True] * 4, True),
        (None, 1, "eager", [True] * 4, True),
        (None, 2, "eager", [True] * 4, True),
        (None, 3, "eager", [True] * 4, True),
        # - None by default (with Inductor)
        (None, 0, "inductor", [True] * 4, True),
        # - None by default (with Inductor)
        (None, 1, "inductor", [False] * 4, False),
        (None, 2, "inductor", [False] * 4, False),
        (None, 3, "inductor", [False] * 4, False),
        # Explicitly enabling/disabling
        #
        # Default: all
        #
        # All but SiluAndMul
        ("+rms_norm,-silu_and_mul", 0, "inductor", [1, 0, 1, 1], True),
        # Only ReLU3
        ("none,-rms_norm,+relu3", 1, "eager", [0, 0, 0, 1], False),
        # All but SiluAndMul
        ("all,-silu_and_mul", 2, "inductor", [1, 0, 1, 1], True),
        # All but ReLU3 (even if ReLU2 is on)
        ("-relu3,+relu2", 3, "eager", [1, 1, 1, 0], True),
        # RMSNorm and SiluAndMul
        ("none,-relu3,+rms_norm,+silu_and_mul", 3, "eager", [1, 1, 0, 0], False),
        # All but RMSNorm
        ("-rms_norm", 3, "eager", [0, 1, 1, 1], True),
        #
        # Default: none
        #
        # Only ReLU3
        ("none,+relu3", 3, "inductor", [0, 0, 0, 1], False),
        # All but RMSNorm
        ("all,-rms_norm", 3, "inductor", [0, 1, 1, 1], True),
    ],
)
def test_enabled_ops(
    env: str | None,
    compilation_mode: int,
    backend: str,
    ops_enabled: list[int],
    default_on: bool,
):
    custom_ops = env.split(",") if env else []
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            backend=backend, mode=compilation_mode, custom_ops=custom_ops
        )
    )
    get_cached_compilation_config.cache_clear()
    with set_current_vllm_config(vllm_config):
        assert CustomOp.default_on() == default_on

        ops_enabled = [bool(x) for x in ops_enabled]

        assert RMSNorm(1024).enabled() == ops_enabled[0]
        assert op_registry["rms_norm"].enabled() == ops_enabled[0]

        assert SiluAndMul().enabled() == ops_enabled[1]
        assert op_registry["silu_and_mul"].enabled() == ops_enabled[1]

        assert GeluAndMul().enabled() == ops_enabled[2]
        assert op_registry["gelu_and_mul"].enabled() == ops_enabled[2]

        # If registered, subclasses should follow their own name
        assert Relu3().enabled() == ops_enabled[3]
        assert op_registry["relu3"].enabled() == ops_enabled[3]

        # Unregistered subclass
        class SiluAndMul2(SiluAndMul):
            pass

        # Subclasses should not require registration
        assert SiluAndMul2().enabled() == SiluAndMul().enabled()


@pytest.mark.parametrize(
    "env", ["all,none", "all,+rms_norm,all", "+rms_norm,-rms_norm"]
)
def test_enabled_ops_invalid(env: str):
    with pytest.raises(Exception):  # noqa
        vllm_config = VllmConfig(
            compilation_config=CompilationConfig(custom_ops=env.split(","))
        )
        with set_current_vllm_config(vllm_config):
            RMSNorm(1024).enabled()


def _add_hw_only_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(hw_agnostic_custom_op.op_registry, HW_ONLY_IN_TREE, DummyOp)
    monkeypatch.setitem(hw_agnostic_custom_op.op_registry_oot, HW_ONLY_OOT, DummyOp)
    monkeypatch.setitem(classic_custom_op.op_registry_oot, CLASSIC_OOT_ONLY, DummyOp)
    # Read the flag live, even if the envs cache is enabled.
    envs.disable_envs_cache()


@pytest.mark.parametrize("hw_agnostic", [False, True])
def test_get_known_op_names_reads_registry_keys(
    monkeypatch: pytest.MonkeyPatch, hw_agnostic: bool
):
    _add_hw_only_ops(monkeypatch)
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1" if hw_agnostic else "0")

    names = classic_custom_op.get_known_op_names()

    # Classic keys always count, from both of its registries.
    assert "rms_norm" in names
    assert CLASSIC_OOT_ONLY in names

    # hw-agnostic keys count only when the flag is on, from both registries.
    hw_names = {HW_ONLY_IN_TREE, HW_ONLY_OOT}
    if hw_agnostic:
        assert hw_names <= names
    else:
        assert hw_names.isdisjoint(names)


@pytest.mark.parametrize(
    "hw_agnostic,expected",
    [
        (False, "doesn't exist (or wasn't imported/registered)"),
        (True, "not present in model"),
    ],
)
def test_custom_op_log_check_uses_hw_agnostic_names(
    monkeypatch: pytest.MonkeyPatch, hw_agnostic: bool, expected: str
):
    _add_hw_only_ops(monkeypatch)
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1" if hw_agnostic else "0")

    compilation_config = CompilationConfig(custom_ops=["all", f"+{HW_ONLY_IN_TREE}"])
    # The op is registered but unused by this model, so the warning wording
    # shows whether the hw-agnostic keys were visible.
    compilation_config.enabled_custom_ops = Counter({"rms_norm": 1})

    warnings: list[tuple] = []
    monkeypatch.setattr(
        compilation.logger, "warning_once", lambda *args: warnings.append(args)
    )

    compilation_config.custom_op_log_check()

    assert len(warnings) == 1
    # args are (message, op_name, missing_str, enable_str, op)
    assert warnings[0][2] == expected


@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_topk_softmax_dispatch(use_rocm_aiter: bool):
    topk_func = dispatch_topk_softmax_func(use_rocm_aiter)

    if current_platform.is_rocm() and use_rocm_aiter:
        assert topk_func == rocm_aiter_ops.topk_softmax
    else:
        assert topk_func == vllm_topk_softmax


@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_topk_sigmoid_dispatch(use_rocm_aiter: bool):
    topk_func = dispatch_topk_sigmoid_func(use_rocm_aiter)

    if current_platform.is_rocm() and use_rocm_aiter:
        assert topk_func == rocm_aiter_ops.topk_sigmoid
    else:
        assert topk_func == vllm_topk_sigmoid
