# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for SharedFusedMoE with routed_input_transform.

Verifies that applying routed_input_transform inside SharedFusedMoE
produces the same results as applying the transform manually outside.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer, set_random_seed


class SimpleLinear(nn.Module):
    """A simple linear transform mimicking latent projection in latent MoE."""

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device="cuda", dtype=dtype) / 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight)


class SimpleSharedExperts(nn.Module):
    """A simple 2-layer MLP mimicking shared experts."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.up = nn.Linear(
            hidden_size, intermediate_size * 2, bias=False, device="cuda", dtype=dtype
        )
        self.down = nn.Linear(
            intermediate_size, hidden_size, bias=False, device="cuda", dtype=dtype
        )
        with torch.no_grad():
            self.up.weight.div_(10)
            self.down.weight.div_(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(nn.functional.silu(gate) * up)


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    """assert_close that prints diff diagnostics on both success and failure."""
    actual_nans = int(actual.isnan().sum().item())
    expected_nans = int(expected.isnan().sum().item())
    actual_zeros = int((actual == 0).sum().item())
    expected_zeros = int((expected == 0).sum().item())
    n_total = actual.numel()

    diff = (actual - expected).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    n_exceed = int((diff > atol).sum().item())
    pct_exceed = n_exceed / n_total * 100

    print(
        f"[{label}] "
        f"shape={list(actual.shape)}, "
        f"max_diff={max_diff:.6e}, "
        f"mean_diff={mean_diff:.6e}, "
        f"exceed_atol({atol})={n_exceed}/{n_total} ({pct_exceed:.2f}%), "
        f"actual=[{actual.min().item():.4f}, {actual.max().item():.4f}], "
        f"expected=[{expected.min().item():.4f}, {expected.max().item():.4f}], "
        f"nan(actual/expected)={actual_nans}/{expected_nans}, "
        f"zeros(actual/expected)={actual_zeros}/{expected_zeros}"
    )

    assert actual_nans == 0, (
        f"{label}: actual has {actual_nans}/{n_total} NaN values "
        f"(expected has {expected_nans}). "
        f"This indicates a kernel bug, not a precision issue."
    )
    assert expected_nans == 0, (
        f"{label}: expected has {expected_nans}/{n_total} NaN values. "
        f"This indicates a kernel bug, not a precision issue."
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=(
            f"{label}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"exceed_atol({atol})={n_exceed}/{n_total} ({pct_exceed:.2f}%)"
        ),
    )


@pytest.fixture(autouse=True)
def setup_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.set_default_device("cuda")


@pytest.mark.parametrize("num_tokens", [1, 32])
@pytest.mark.parametrize("hidden_size,latent_size", [(256, 128), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
@pytest.mark.skipif(
    is_torch_equal_or_newer("2.10.0"),
    reason="Test fails with PyTorch 2.10.0 see: https://github.com/vllm-project/vllm/issues/33995",
)
def test_routed_input_transform_inside_vs_outside(
    num_tokens: int,
    hidden_size: int,
    latent_size: int,
    dtype: torch.dtype,
    use_rocm_aiter: bool,
    dist_init,
    workspace_init,
    monkeypatch,
):
    """Compare SharedFusedMoE with transform inside vs manually applying outside.
    Method A (inside): SharedFusedMoE with routed_input_transform
    Method B (outside): Manually transform, then SharedFusedMoE without transform
    """
    if current_platform.is_rocm():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_rocm_aiter else "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1" if use_rocm_aiter else "0")
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

    set_random_seed(42)

    num_experts = 8
    top_k = 2
    intermediate_size = hidden_size * 2

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()

    shared_experts = SimpleSharedExperts(hidden_size, intermediate_size, dtype)
    routed_transform = SimpleLinear(hidden_size, latent_size, dtype)

    with set_current_vllm_config(vllm_config):
        # Method A: SharedFusedMoE WITH routed_input_transform
        moe_with_transform = SharedFusedMoE(
            shared_experts=shared_experts,
            routed_input_transform=routed_transform,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=latent_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            renormalize=True,
            params_dtype=dtype,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            prefix="moe_with_transform",
        )

        # Method B: SharedFusedMoE WITHOUT routed_input_transform
        # Note: shared_experts=None because when transform is done outside,
        moe_without_transform = SharedFusedMoE(
            shared_experts=None,
            routed_input_transform=None,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=latent_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            renormalize=True,
            params_dtype=dtype,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            prefix="moe_without_transform",
        )

        # Weights are created via torch.empty (uninitialized).
        # Initialize with seeded random values for reproducibility.
        with torch.no_grad():
            moe_with_transform.w13_weight.normal_()
            moe_with_transform.w13_weight.div_(10)
            moe_with_transform.w2_weight.normal_()
            moe_with_transform.w2_weight.div_(10)
            moe_without_transform.w13_weight.copy_(moe_with_transform.w13_weight)
            moe_without_transform.w2_weight.copy_(moe_with_transform.w2_weight)

        moe_with_transform.quant_method.process_weights_after_loading(
            moe_with_transform
        )
        moe_without_transform.quant_method.process_weights_after_loading(
            moe_without_transform
        )

        hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
        router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)

        # Clone inputs so any in-place modification by Method A
        # cannot affect Method B's computation.
        hidden_states_A = hidden_states.clone()
        router_logits_A = router_logits.clone()

        with set_forward_context(None, vllm_config, num_tokens=num_tokens):
            shared_out_A, routed_out_A = moe_with_transform(
                hidden_states_A, router_logits_A
            )

            transformed_hidden = routed_transform(hidden_states)
            shared_out_B, routed_out_B = moe_without_transform(
                transformed_hidden, router_logits
            )

        expected_shared_out = shared_experts(hidden_states)

        _assert_close(
            routed_out_A,
            routed_out_B,
            atol=1e-3,
            rtol=1e-3,
            label="Routed output: transform inside vs outside",
        )
        _assert_close(
            shared_out_A,
            expected_shared_out,
            atol=1e-3,
            rtol=1e-3,
            label="Shared expert output",
        )
