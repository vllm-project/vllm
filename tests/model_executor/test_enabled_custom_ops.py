# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.model_executor.custom_op import CustomOp, op_registry
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


@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_topk_softmax_dispatch(use_rocm_aiter: bool):
    topk_func = dispatch_topk_softmax_func(use_rocm_aiter)

    if current_platform.is_rocm() and use_rocm_aiter:
        assert topk_func == rocm_aiter_ops.topk_softmax
    else:
        assert topk_func == vllm_topk_softmax


def _topk_gating_launch(
    num_tokens: int = 4,
    num_experts: int = 512,
    topk: int = 10,
    num_shared_experts: int = 0,
    weights_width: int | None = None,
    ids_width: int | None = None,
    contiguous: bool = True,
) -> dict:
    """Tensors for one softmax top-k launch, as dispatch_topk_softmax_func sees them."""
    width = topk + num_shared_experts
    weights = torch.empty(num_tokens, weights_width or width)
    ids = torch.empty(num_tokens, ids_width or width, dtype=torch.int32)
    ids = ids[:, :topk]
    if contiguous:
        gating = torch.empty(num_tokens, num_experts + num_shared_experts)
    else:
        gating = torch.empty(num_experts + num_shared_experts + 3, num_tokens).t()
    return dict(topk_weights=weights, topk_indices=ids, gating_output=gating)


@pytest.mark.parametrize(
    "launch, num_shared_experts, scoring_func, expect_gating",
    [
        pytest.param({}, 0, "", True, id="plain-softmax"),
        pytest.param({"num_tokens": 4096}, 0, "", True, id="max-tokens"),
        pytest.param({"num_tokens": 4097}, 0, "", False, id="too-many-tokens"),
        pytest.param({"contiguous": False}, 0, "", False, id="non-contiguous"),
        pytest.param({}, 0, "sigmoid", False, id="scoring-without-shared"),
        pytest.param({"num_shared_experts": 1}, 1, "sigmoid", True, id="one-shared"),
        pytest.param({"num_shared_experts": 8}, 8, "sigmoid", True, id="eight-shared"),
        pytest.param(
            {"num_shared_experts": 3}, 3, "sigmoid", False, id="unsupported-count"
        ),
        pytest.param({"num_shared_experts": 1}, 1, "", False, id="no-shared-scoring"),
        pytest.param(
            {"num_shared_experts": 1}, 1, "softmax", False, id="wrong-shared-scoring"
        ),
        pytest.param(
            {"num_shared_experts": 1, "weights_width": 10},
            1,
            "sigmoid",
            False,
            id="weights-too-narrow",
        ),
        pytest.param(
            {"num_shared_experts": 1, "ids_width": 12},
            1,
            "sigmoid",
            False,
            id="row-stride-mismatch",
        ),
        pytest.param(
            {"num_tokens": 4097, "num_shared_experts": 1},
            1,
            "sigmoid",
            False,
            id="shared-too-many-tokens",
        ),
    ],
)
@pytest.mark.parametrize("gating_enabled", [True, False])
def test_topk_softmax_dispatch_aiter_topk_gating(
    monkeypatch: pytest.MonkeyPatch,
    launch: dict,
    num_shared_experts: int,
    scoring_func: str,
    expect_gating: bool,
    gating_enabled: bool,
):
    """AITER launches use topk_gating only where it is enabled, and only when it
    supports them.

    Every other AITER launch must reach the legacy topk_softmax, so this is the
    one place the choice is made.
    """
    monkeypatch.setattr(
        rocm_aiter_ops, "is_topk_gating_enabled", lambda: gating_enabled
    )
    topk_func = dispatch_topk_softmax_func(
        True,
        **_topk_gating_launch(**launch),
        num_shared_experts=num_shared_experts,
        shared_expert_scoring_func=scoring_func,
    )
    if expect_gating and gating_enabled:
        assert topk_func == rocm_aiter_ops.topk_gating
    else:
        assert topk_func == rocm_aiter_ops.topk_softmax


@pytest.mark.parametrize(
    "arch, fused_moe_enabled, expected",
    [
        pytest.param("gfx942", True, True, id="gfx942"),
        pytest.param("gfx950", True, True, id="gfx950"),
        pytest.param("gfx90a", True, False, id="gfx90a"),
        pytest.param("gfx942", False, False, id="gfx942-aiter-moe-off"),
        pytest.param("gfx950", False, False, id="gfx950-aiter-moe-off"),
    ],
)
def test_topk_gating_enabled_by_arch(
    monkeypatch: pytest.MonkeyPatch,
    arch: str,
    fused_moe_enabled: bool,
    expected: bool,
):
    """topk_gating is on for gfx942 and gfx950 with AITER MoE enabled, nowhere else."""
    monkeypatch.setattr("vllm._aiter_ops.is_aiter_found_and_supported", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_fused_moe_enabled", lambda: fused_moe_enabled
    )
    for name in ("gfx942", "gfx950"):
        monkeypatch.setattr(f"vllm.platforms.rocm.on_{name}", lambda a=name: a == arch)
    assert bool(rocm_aiter_ops.is_topk_gating_enabled()) is expected


@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_topk_sigmoid_dispatch(use_rocm_aiter: bool):
    topk_func = dispatch_topk_sigmoid_func(use_rocm_aiter)

    if current_platform.is_rocm() and use_rocm_aiter:
        assert topk_func == rocm_aiter_ops.topk_sigmoid
    else:
        assert topk_func == vllm_topk_sigmoid
