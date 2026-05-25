# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel tests for ROCm AITER fused MoE.

This file owns the ROCm-specific fused-MoE custom-op path:
- custom-op registration and fake-tensor support
- enablement gating for ``VLLM_ROCM_USE_AITER_MOE`` and shared experts
- BF16 accuracy for the fused MoE kernel on representative shapes
- gfx950-only AITER MXFP4 W4A16 MoE support, accuracy, and determinism
- MoE-facing FP8 group-quant activation quality
- deterministic routing and representative gfx942 / gfx950 coverage

Generic fused-MoE backend selection and non-ROCm kernel coverage live in the
generic MoE test files under ``tests/kernels/moe``.

Raw ROCm AITER helper-op coverage such as ``group_fp8_quant`` and
``per_tensor_quant`` lives in ``tests/kernels/core/test_rocm_aiter_ops.py``.
This file only keeps the MoE-shaped integration angle for those helpers.
"""

import importlib
import math
import warnings
from typing import Any, NamedTuple

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.utils import _assert_deterministic
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx942, on_gfx950
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# Helpers -----------------------------------------------------------------


def _reload_envs():
    import vllm.envs as envs

    return importlib.reload(envs)


def _assert_aiter_supported() -> None:
    from vllm._aiter_ops import is_aiter_found_and_supported

    assert is_aiter_found_and_supported(), (
        "aiter is required on supported ROCm hardware for this test"
    )


def _format_observed_rate(count: int, total: int) -> str:
    return f"{count / total:.4%} ({count}/{total})"


def _format_allowed_rate(rate: float, total: int) -> str:
    allowed_count = int(rate * total)
    return f"{rate:.4%} (<= {allowed_count}/{total})"


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return torch.quantile(values, q).item()


def _assert_close_budget(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    atol: float,
    rtol: float = 0.0,
    pass_rate: float = 0.99999,
    max_violation_factor: float = 3.0,
) -> None:
    actual_f = actual.detach().float().flatten()
    expected_f = expected.detach().float().flatten()
    abs_diff = (actual_f - expected_f).abs()
    allowed = atol + rtol * expected_f.abs()

    total = abs_diff.numel()
    within = abs_diff <= allowed
    passed = int(within.sum().item())
    failed = total - passed
    allowed_fail_rate = 1.0 - pass_rate

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    p99_abs = _quantile(abs_diff, 0.99)
    p999_abs = _quantile(abs_diff, 0.999)
    worst_ratio = (abs_diff / allowed.clamp_min(1e-12)).max().item()
    max_atol = max_violation_factor * atol
    above_max_count = int((abs_diff > max_atol).sum().item())

    msg = (
        "[rocm_aiter_moe] "
        f"{label}: "
        f"pass={passed / total:.4%} ({passed}/{total}) "
        f"fail={_format_observed_rate(failed, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"atol={atol:g} "
        f"rtol={rtol:g} "
        f"abs>{max_atol:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(0.0, total)} "
        f"max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} "
        f"p99_abs={p99_abs:.6g} "
        f"p999_abs={p999_abs:.6g} "
        f"worst_ratio={worst_ratio:.6g}"
    )
    print(msg)
    if failed > 0:
        warnings.warn(msg, stacklevel=2)

    assert passed / total >= pass_rate, msg
    assert max_abs <= max_atol, msg
    assert mean_abs <= atol * 0.25, msg


def _assert_group_quant_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    preferred_rel: float = 0.06,
    pass_rate: float = 0.9999,
    max_rel: float = 0.55,
    max_fail_rate: float = 0.000005,
    mean_limit: float = 0.03,
) -> None:
    rel = (
        (actual.float() - expected.float()).abs()
        / expected.float().abs().clamp_min(1e-5)
    ).flatten()
    total = rel.numel()
    within_preferred_count = int((rel <= preferred_rel).sum().item())
    fail_count = total - within_preferred_count
    allowed_fail_rate = 1.0 - pass_rate
    above_max_count = int((rel > max_rel).sum().item())
    mean_rel = rel.mean().item()
    max_rel_err = rel.max().item()
    p99 = _quantile(rel, 0.99)
    p999 = _quantile(rel, 0.999)

    msg = (
        "[rocm_aiter_moe] "
        f"{label}: "
        f"rel<={preferred_rel:g} pass={within_preferred_count / total:.4%} "
        f"({within_preferred_count}/{total}) "
        f"fail={_format_observed_rate(fail_count, total)} "
        f"allowed_fail={_format_allowed_rate(allowed_fail_rate, total)} "
        f"rel>{max_rel:g}={_format_observed_rate(above_max_count, total)} "
        f"allowed_above_max={_format_allowed_rate(max_fail_rate, total)} "
        f"mean_rel={mean_rel:.6g} "
        f"max_rel={max_rel_err:.6g} "
        f"p99={p99:.6g} "
        f"p999={p999:.6g}"
    )
    print(msg)
    if fail_count > 0:
        warnings.warn(msg, stacklevel=2)

    assert within_preferred_count / total >= pass_rate, msg
    assert above_max_count / total <= max_fail_rate, msg
    assert mean_rel < mean_limit, msg


def ref_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Float32 mask-based MoE reference for the ROCm fused-MoE kernel."""
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_dim = w1.shape[1] // 2
    topk = topk_ids.shape[1]
    device = hidden_states.device

    hidden_states_f = hidden_states.float()
    w1_f = w1.float()
    w2_f = w2.float()

    expanded = hidden_states_f.view(num_tokens, 1, hidden_dim).expand(
        num_tokens, topk, hidden_dim
    )
    expanded = expanded.reshape(num_tokens * topk, hidden_dim)
    output = torch.zeros(
        num_tokens * topk, hidden_dim, dtype=torch.float32, device=device
    )
    flat_topk_ids = topk_ids.view(-1).long()

    for expert_idx in range(num_experts):
        expert_mask = flat_topk_ids == expert_idx
        if expert_mask.sum() == 0:
            continue
        gate_up = expanded[expert_mask] @ w1_f[expert_idx].T
        gate = gate_up[:, :intermediate_dim]
        up = gate_up[:, intermediate_dim:]
        if activation == "silu":
            act = F.silu(gate) * up
        elif activation == "gelu":
            act = F.gelu(gate) * up
        else:
            raise ValueError(f"Unknown activation: {activation}")
        output[expert_mask] = act @ w2_f[expert_idx].T

    output = output.view(num_tokens, topk, hidden_dim)
    weights = topk_weights.float().view(num_tokens, topk, 1)
    return (output * weights).sum(dim=1)


class AiterMxfp4MoeCase(NamedTuple):
    hidden_states: torch.Tensor
    w1_kernel: torch.Tensor
    w2_kernel: torch.Tensor
    w1_ref: torch.Tensor
    w2_ref: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    moe_config: Any
    quant_config: Any


def _make_topk_ids(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate distinct expert IDs per token with the same top-k shape as
    production routing."""
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    _, topk_ids = torch.topk(torch.softmax(router_logits, dim=-1), k=topk, dim=-1)
    return topk_ids.to(torch.int32)


def _shuffle_moe_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle MoE weights into the AITER CK layout used in production."""
    from vllm._aiter_ops import rocm_aiter_ops

    w1_shuffled, w2_shuffled = rocm_aiter_ops.shuffle_weights(w1, w2)
    w1_shuffled.is_shuffled = True
    w2_shuffled.is_shuffled = True
    return w1_shuffled, w2_shuffled


def _make_moe_case(
    *,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.set_default_device("cuda")
    set_random_seed(seed)

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
    }


def _make_aiter_mxfp4_moe_case(
    *,
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> AiterMxfp4MoeCase:
    from triton_kernels.numerics_details.mxfp import (
        downcast_to_mxfp,
        upcast_from_mxfp,
    )

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_quant_config,
    )

    torch.set_default_device("cuda")
    set_random_seed(seed)

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(intermediate_dim)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk)

    w1_q, w1_scale = downcast_to_mxfp(w1, torch.uint8, axis=-1)
    w2_q, w2_scale = downcast_to_mxfp(w2, torch.uint8, axis=-1)

    w1_ref = upcast_from_mxfp(w1_q, w1_scale, torch.bfloat16, axis=-1)
    w2_ref = upcast_from_mxfp(w2_q, w2_scale, torch.bfloat16, axis=-1)

    (
        w1_kernel,
        w2_kernel,
        w1_scale_kernel,
        w2_scale_kernel,
        _,
        _,
    ) = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.AITER,
        layer=torch.nn.Module(),
        w13_weight=w1_q.clone(),
        w2_weight=w2_q.clone(),
        w13_weight_scale=w1_scale.clone(),
        w2_weight_scale=w2_scale.clone(),
    )

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_dim,
        in_dtype=torch.bfloat16,
    )
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.AITER,
        w1_scale=w1_scale_kernel,
        w2_scale=w2_scale_kernel,
    )

    return AiterMxfp4MoeCase(
        hidden_states=hidden_states,
        w1_kernel=w1_kernel,
        w2_kernel=w2_kernel,
        w1_ref=w1_ref,
        w2_ref=w2_ref,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=moe_config,
        quant_config=quant_config,
    )


def _run_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation_method: int,
    quant_method: int,
) -> torch.Tensor:
    w1_shuffled, w2_shuffled = _shuffle_moe_weights(w1, w2)
    return torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=activation_method,
        quant_method=quant_method,
        doweight_stage1=False,
    )


# Custom op tests ---------------------------------------------------------


def test_aiter_fused_moe_custom_op_registered():
    """The main fused-MoE custom op should stay registered for runtime use."""
    _assert_aiter_supported()
    import vllm._aiter_ops as aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_fused_moe")
    assert callable(torch.ops.vllm.rocm_aiter_fused_moe)


def test_aiter_asm_moe_tkw1_custom_op_registered():
    """The tkw1 custom op should stay registered for FP8 apply-router-weight
    paths."""
    _assert_aiter_supported()
    import vllm._aiter_ops as aiter_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "rocm_aiter_asm_moe_tkw1")
    assert callable(torch.ops.vllm.rocm_aiter_asm_moe_tkw1)


def test_aiter_fused_moe_fake_tensor_support():
    """The fused-MoE op should preserve fake-tensor compatibility for
    torch.compile-style tracing."""
    _assert_aiter_supported()
    import vllm._aiter_ops  # noqa: F401

    num_tokens = 16
    hidden_dim = 1024
    intermediate_dim = 2048
    num_experts = 8
    topk = 2

    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = _make_topk_ids(num_tokens, num_experts, topk, device="cuda")

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_fused_moe,
        (hidden_states, w1, w2, topk_weights, topk_ids),
        kwargs={
            "expert_mask": None,
            "activation_method": 0,
            "quant_method": 0,
            "doweight_stage1": False,
        },
        test_utils=("test_faketensor",),
    )


# Env gating tests --------------------------------------------------------


@pytest.mark.parametrize(
    ("use_aiter", "use_moe", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_aiter_moe_enablement_follows_env(
    use_aiter: bool,
    use_moe: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """The fused-MoE gate should depend only on the main AITER toggle and the
    MoE-specific toggle."""
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1" if use_moe else "0")
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

        assert rocm_aiter_ops.is_fused_moe_enabled() is expected

    _reload_envs()
    aiter_ops.rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    ("use_aiter", "use_moe", "use_shared", "expected"),
    [
        (True, True, True, True),
        (True, True, False, False),
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_aiter_moe_shared_experts_enablement_follows_env(
    use_aiter: bool,
    use_moe: bool,
    use_shared: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Shared-expert fusion should only be enabled when the fused-MoE path is
    enabled too."""
    import vllm._aiter_ops as aiter_ops
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", "1" if use_moe else "0")
        mp.setenv(
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
            "1" if use_shared else "0",
        )
        _reload_envs()
        rocm_aiter_ops.refresh_env_variables()

        assert rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() is expected

    _reload_envs()
    aiter_ops.rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize("moe_padding", [True, False])
def test_aiter_moe_padding_env_var(
    moe_padding: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """The ROCm MoE padding env var should keep its exact parse contract."""
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_MOE_PADDING", "1" if moe_padding else "0")
        envs = _reload_envs()
        assert envs.VLLM_ROCM_MOE_PADDING is moe_padding

    _reload_envs()


# Enum tests --------------------------------------------------------------


def test_quant_method_enum_values():
    """The AITER quant-method bridge enum should keep its wire values."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import QuantMethod

    assert QuantMethod.NO == 0
    assert QuantMethod.PER_TENSOR == 1
    assert QuantMethod.PER_TOKEN == 2
    assert QuantMethod.BLOCK_1X32 == 3
    assert QuantMethod.BLOCK_1X128 == 4
    assert QuantMethod.BLOCK_128x128 == 5


def test_activation_method_enum_values():
    """The AITER activation-method bridge enum should keep its wire values."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
    )

    assert ActivationMethod.SILU == 0
    assert ActivationMethod.GELU == 1


# MXFP4 kernel tests ------------------------------------------------------


def test_aiter_mxfp4_quant_scheme_support_matches_gfx950():
    """AITER MXFP4 MoE support should stay gfx950-only."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        AiterExperts,
    )

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Static,
    )

    assert AiterExperts._supports_quant_scheme(kMxfp4Static, None) is on_gfx950()


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.skipif(
    not has_triton_kernels(),
    reason="triton_kernels are required for MXFP4 test weight quantization",
)
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="native FP4 dtype not available in this torch build",
)
def test_aiter_fused_moe_mi350_mxfp4_w4a16_accuracy():
    """The gfx950 AITER MXFP4 W4A16 MoE path should match the dequantized
    MXFP4 reference."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        rocm_aiter_fused_experts,
    )

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    _assert_aiter_supported()
    case = _make_aiter_mxfp4_moe_case(
        num_tokens=32,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=11,
    )
    ref_out = ref_moe_forward(
        case.hidden_states,
        case.w1_ref,
        case.w2_ref,
        case.topk_weights,
        case.topk_ids,
    )
    out = rocm_aiter_fused_experts(
        hidden_states=case.hidden_states,
        w1=case.w1_kernel,
        w2=case.w2_kernel,
        topk_weights=case.topk_weights,
        topk_ids=case.topk_ids,
        activation=MoEActivation.SILU,
        quant_config=case.quant_config,
        moe_config=case.moe_config,
        expert_map=None,
    )

    assert out.shape == case.hidden_states.shape
    _assert_close_budget(
        out.float(),
        ref_out.float(),
        label="mi350_mxfp4_w4a16_accuracy",
        atol=0.1,
        rtol=0.0,
        pass_rate=0.9999,
        max_violation_factor=2.0,
    )


@pytest.mark.skipif(not on_gfx950(), reason="gfx950 ROCm only")
@pytest.mark.skipif(
    not has_triton_kernels(),
    reason="triton_kernels are required for MXFP4 test weight quantization",
)
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="native FP4 dtype not available in this torch build",
)
def test_aiter_fused_moe_mi350_mxfp4_w4a16_determinism():
    """The gfx950 AITER MXFP4 W4A16 MoE path should stay bitwise
    deterministic."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        rocm_aiter_fused_experts,
    )

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    _assert_aiter_supported()
    case = _make_aiter_mxfp4_moe_case(
        num_tokens=8,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=13,
    )

    def run_mxfp4_moe():
        return rocm_aiter_fused_experts(
            hidden_states=case.hidden_states,
            w1=case.w1_kernel,
            w2=case.w2_kernel,
            topk_weights=case.topk_weights,
            topk_ids=case.topk_ids,
            activation=MoEActivation.SILU,
            quant_config=case.quant_config,
            moe_config=case.moe_config,
            expert_map=None,
        )

    _assert_deterministic(run_mxfp4_moe, n_runs=4)


# FP8 group-quant tests ---------------------------------------------------


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.parametrize("num_tokens,hidden_dim", [(16, 2048), (64, 4096), (128, 8192)])
def test_aiter_moe_group_fp8_quant_reconstructs_hidden_states(
    num_tokens: int,
    hidden_dim: int,
):
    """The MoE-facing FP8 group-quant helper should reconstruct hidden states
    within the expected FP8 error budget.

    The raw op's shape and standalone roundtrip contracts are covered in
    ``tests/kernels/core/test_rocm_aiter_ops.py``. This test keeps the
    representative hidden-state sizes that the fused-MoE path actually cares
    about.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()
    torch.set_default_device("cuda")
    set_random_seed(1)

    group_size = 128
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(hidden_states, group_size)

    assert x_fp8.shape == hidden_states.shape
    assert scales.shape == (num_tokens, (hidden_dim + group_size - 1) // group_size)
    assert scales.dtype == torch.float32

    scales_expanded = scales.repeat_interleave(group_size, dim=1)[:, :hidden_dim]
    dequantized = x_fp8.float() * scales_expanded

    _assert_group_quant_quality(
        dequantized,
        hidden_states,
        label=f"group_fp8_quant shape=({num_tokens}, {hidden_dim})",
    )


# Kernel accuracy tests ---------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,hidden_dim,intermediate_dim",
    [
        (16, 512, 1024),
        (128, 2048, 4096),
        (2048, 4096, 11008),
    ],
)
def test_aiter_fused_moe_bf16_accuracy(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
):
    """The ROCm AITER fused-MoE BF16 path should match the float32 reference
    on representative shapes."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=8,
        topk=2,
        seed=0,
    )
    ref_out = ref_moe_forward(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation="silu",
    )
    out = _run_fused_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
    )

    assert out.shape == (num_tokens, hidden_dim)
    _assert_close_budget(
        out.float(),
        ref_out,
        label=f"bf16_accuracy shape=({num_tokens}, {hidden_dim}, {intermediate_dim})",
        atol=0.05,
        rtol=0.0,
    )


def test_aiter_fused_moe_gelu_accuracy():
    """The GELU activation variant should stay aligned with the float32
    reference."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=32,
        hidden_dim=512,
        intermediate_dim=1024,
        num_experts=4,
        topk=2,
        seed=42,
    )
    ref_out = ref_moe_forward(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation="gelu",
    )
    out = _run_fused_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation_method=int(ActivationMethod.GELU),
        quant_method=int(QuantMethod.NO),
    )

    assert out.shape == case["hidden_states"].shape
    _assert_close_budget(
        out.float(),
        ref_out,
        label="gelu_accuracy",
        atol=0.05,
        rtol=0.0,
    )


def test_aiter_fused_moe_determinism():
    """The BF16 fused-MoE kernel should stay bitwise deterministic for the
    same inputs."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=8,
        hidden_dim=256,
        intermediate_dim=512,
        num_experts=4,
        topk=2,
        seed=2,
    )

    def run_moe():
        return _run_fused_moe(
            case["hidden_states"],
            case["w1"],
            case["w2"],
            case["topk_weights"],
            case["topk_ids"],
            activation_method=int(ActivationMethod.SILU),
            quant_method=int(QuantMethod.NO),
        )

    _assert_deterministic(run_moe, n_runs=4)


# Routed end-to-end tests -------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 8, 2),
        (16, 8, 2),
        (64, 16, 4),
    ],
)
def test_aiter_fused_moe_end_to_end(
    num_tokens: int,
    num_experts: int,
    topk: int,
):
    """The full router-logits to top-k to fused-MoE path should stay accurate."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    torch.set_default_device("cuda")
    set_random_seed(7)

    hidden_dim = 512
    intermediate_dim = 1024
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts,
        hidden_dim,
        intermediate_dim,
        dtype=torch.bfloat16,
    ) / math.sqrt(intermediate_dim)

    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    router_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, k=topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_moe_forward(hidden_states, w1, w2, topk_weights, topk_ids)
    out = _run_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
    )

    assert out.shape == (num_tokens, hidden_dim)
    assert out.dtype == torch.bfloat16
    _assert_close_budget(
        out.float(),
        ref_out,
        label=f"end_to_end tokens={num_tokens} experts={num_experts} topk={topk}",
        atol=0.05,
        rtol=0.0,
    )


# Arch-specific tests -----------------------------------------------------


@pytest.mark.skipif(
    not (on_gfx942() or on_gfx950()),
    reason="gfx942/gfx950 ROCm only",
)
def test_aiter_fused_moe_mi3xx_bf16_accuracy():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    _assert_aiter_supported()
    case = _make_moe_case(
        num_tokens=64,
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        topk=2,
        seed=42,
    )
    ref_out = ref_moe_forward(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
    )
    out = _run_fused_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["topk_weights"],
        case["topk_ids"],
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.NO),
    )

    _assert_close_budget(
        out.float(),
        ref_out,
        label=(f"mi3xx_bf16_accuracy arch={'gfx942' if on_gfx942() else 'gfx950'}"),
        atol=0.05,
        rtol=0.0,
    )


@pytest.mark.skipif(
    not (on_gfx942() or on_gfx950()),
    reason="gfx942/gfx950 ROCm only",
)
@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
def test_aiter_fused_moe_mi3xx_fp8_accuracy():
    """The MI3xx FP8 per-tensor MoE path should stay within the measured FP8
    error budget."""
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        ActivationMethod,
        QuantMethod,
    )

    from tests.kernels.moe.utils import make_test_weights
    from vllm._aiter_ops import rocm_aiter_ops

    _assert_aiter_supported()
    torch.set_default_device("cuda")
    set_random_seed(99)

    num_tokens = 32
    hidden_dim = 512
    intermediate_dim = 1024
    num_experts = 4
    topk = 2

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16) / 10
    fp8_dtype = current_platform.fp8_dtype()
    (w1_bf16, w1_fp8, w1_scale, _), (w2_bf16, w2_fp8, w2_scale, _) = make_test_weights(
        num_experts,
        intermediate_dim * 2,
        hidden_dim,
        torch.bfloat16,
        fp8_dtype,
        per_out_ch_quant=False,
    )
    _, a1_scale = rocm_aiter_ops.per_tensor_quant(hidden_states, fp8_dtype)
    router_logits = torch.randn(num_tokens, num_experts, device="cuda")
    router_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(router_probs, k=topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.to(torch.int32)

    ref_out = ref_moe_forward(
        hidden_states,
        w1_bf16,
        w2_bf16,
        topk_weights,
        topk_ids,
    )

    w1_shuffled, w2_shuffled = _shuffle_moe_weights(w1_fp8, w2_fp8)
    out = torch.ops.vllm.rocm_aiter_fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation_method=int(ActivationMethod.SILU),
        quant_method=int(QuantMethod.PER_TENSOR),
        doweight_stage1=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
    )

    assert out.shape == hidden_states.shape
    _assert_close_budget(
        out.float(),
        ref_out,
        label=(f"mi3xx_fp8_accuracy arch={'gfx942' if on_gfx942() else 'gfx950'}"),
        atol=0.02,
        rtol=0.0,
        pass_rate=1.0,
        max_violation_factor=1.5,
    )
