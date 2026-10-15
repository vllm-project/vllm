# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel test utils."""

from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import torch
from torch._prims_common import TensorLikeType

from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.model_executor.custom_op import op_registry
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

# For now, disable "test_aot_dispatch_dynamic" since there are some
# bugs related to this test in PyTorch 2.4.
DEFAULT_OPCHECK_TEST_UTILS: tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
)

ALL_OPCHECK_TEST_UTILS: tuple[str, ...] = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def _assert_accurate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float = 0.0,
    pass_rate: float = 0.99999,
    max_violation_factor: float = 3.0,
) -> None:
    """Check numeric accuracy with pass-rate, max-error, and mean-error bounds."""
    a = actual.detach().float().flatten()
    e = expected.detach().float().flatten()

    abs_err = (a - e).abs()
    tol = atol + rtol * e.abs()

    rate = (abs_err <= tol).float().mean().item()
    assert rate >= pass_rate, (
        f"Accuracy pass rate {rate:.6f} < {pass_rate} (atol={atol}, rtol={rtol})"
    )

    max_err = abs_err.max().item()
    assert max_err <= max_violation_factor * atol, (
        f"Max absolute error {max_err:.6f} exceeds {max_violation_factor} * atol={atol}"
    )

    mean_err = abs_err.mean().item()
    assert mean_err <= atol * 0.25, (
        f"Mean absolute error {mean_err:.6f} >= atol * 0.25 = {atol * 0.25:.6f}"
    )


def _assert_deterministic(
    fn,
    *args,
    n_runs: int = 4,
    **kwargs,
) -> None:
    """Verify that repeated calls produce bitwise-identical tensor outputs."""

    def _collect(result: Any) -> list[torch.Tensor]:
        if isinstance(result, torch.Tensor):
            return [result.detach().clone()]
        if isinstance(result, (tuple, list)):
            return [t.detach().clone() for t in result if isinstance(t, torch.Tensor)]
        raise TypeError(f"Unexpected return type {type(result)}")

    reference = _collect(fn(*args, **kwargs))

    for run in range(1, n_runs):
        outputs = _collect(fn(*args, **kwargs))
        for idx, (ref, out) in enumerate(zip(reference, outputs)):
            assert torch.equal(ref, out), (
                f"Run {run}: output[{idx}] differs from run 0 "
                f"(max diff = {(out.float() - ref.float()).abs().max().item():.2e})"
            )


# Copied/modified from torch._refs.__init__.py
def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """Reference implementation of torch.allclose."""
    torch._refs._check_close_args(name="torch.allclose", a=a, b=b, rtol=rtol, atol=atol)

    return bool(
        torch.all(
            torch.isclose(
                a.double(), b.double(), rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        ).item()
    )


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable-step distance between two bf16 tensors.

    Reinterprets the bf16 bit patterns under the IEEE-754 total ordering so
    that adjacent representable values differ by exactly 1.
    """

    def key(t: torch.Tensor) -> torch.Tensor:
        u = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)

    return (key(a) - key(b)).abs()


def fp8_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable-step distance between two 8-bit fp8 tensors.

    Reinterprets the fp8 bytes under a sign-magnitude total ordering so that
    adjacent representable values differ by exactly 1. Inputs must already share
    the same fp8 encoding (e.g. both FP8_STORE_DTYPE).
    """

    def key(t: torch.Tensor) -> torch.Tensor:
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return (key(a) - key(b)).abs()


# Marlin MoE test utils


def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )


def torch_experts(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    b_bias1: torch.Tensor | None = None,
    b_bias2: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    quant_dtype: torch.dtype | None = None,
    per_act_token_quant=False,
    block_shape: list[int] | None = None,
    apply_router_weights_on_input: bool = False,
    activation: MoEActivation = MoEActivation.SILU,
) -> torch.Tensor:
    assert (
        global_num_experts == -1
        or (global_num_experts == w1.shape[0] and expert_map is None)
        or (expert_map is not None and global_num_experts == expert_map.shape[0])
    )

    if quant_dtype in [torch.float16, torch.bfloat16]:
        quant_dtype = None
    quant_input_only = quant_dtype is not None and w1_scale is None and w2_scale is None
    if quant_input_only:
        assert a1_scale is None and a2_scale is None
        assert per_act_token_quant

    M, K = a.shape
    topk = topk_ids.shape[1]

    if apply_router_weights_on_input:
        assert topk == 1
        a = a * topk_weight.to(a.dtype)

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    if a1_scale:
        assert not per_act_token_quant and block_shape is None
    a, a_scale = moe_kernel_quantize_input(
        a, a1_scale, quant_dtype, per_act_token_quant, block_shape
    )

    if quant_input_only:
        a = (a.float() * a_scale.view(-1, 1)).to(w1.dtype)

    num_experts = w1.shape[0]

    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    f32 = torch.float32

    act = op_registry[activation.custom_op_name]

    for i in range(num_experts):
        mask = topk_ids == i
        if mask.sum():
            if quant_dtype is None:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                if b_bias1 is not None:
                    tmp1 = tmp1 + b_bias1[i].view(1, -1).to(tmp1.dtype)
                tmp2 = act()(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
                if b_bias2 is not None:
                    out[mask] = out[mask] + b_bias2[i].view(1, -1).to(tmp1.dtype)
            elif quant_input_only:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = SiluAndMul()(tmp1)
                tmp2, tmp2_scale = moe_kernel_quantize_input(
                    tmp2, None, quant_dtype, per_act_token_quant
                )
                tmp2 = (tmp2.float() * tmp2_scale.view(-1, 1)).to(w2.dtype)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
            elif block_shape is not None:
                # block quantized
                assert (
                    a_scale is not None
                    and w1_scale is not None
                    and w2_scale is not None
                )
                tmp1 = native_w8a8_block_matmul(
                    a[mask], w1[i], a_scale[mask], w1_scale[i], block_shape, out.dtype
                )
                if b_bias1 is not None:
                    tmp1 = tmp1 + b_bias1[i].view(1, -1).to(tmp1.dtype)
                tmp2 = SiluAndMul()(tmp1)
                tmp2, b_scale = moe_kernel_quantize_input(
                    tmp2, a2_scale, quant_dtype, per_act_token_quant, block_shape
                )

                out[mask] = native_w8a8_block_matmul(
                    tmp2, w2[i], b_scale, w2_scale[i], block_shape, out.dtype
                )
                if b_bias2 is not None:
                    out[mask] = out[mask] + b_bias2[i].view(1, -1).to(tmp1.dtype)
            else:
                assert (
                    a_scale is not None
                    and w1_scale is not None
                    and w2_scale is not None
                )
                scales = a_scale if a_scale.numel() == 1 else a_scale[mask]

                tmp1 = a[mask].to(f32) * scales
                w1_dq = (w1[i].to(f32) * w1_scale[i]).transpose(0, 1)
                tmp1 = (tmp1 @ w1_dq).to(out.dtype)
                if b_bias1 is not None:
                    tmp1 = tmp1 + b_bias1[i].view(1, -1).to(out.dtype)

                tmp2 = act()(tmp1).to(out.dtype)

                tmp2, b_scale = moe_kernel_quantize_input(
                    tmp2, a2_scale, quant_dtype, per_act_token_quant, block_shape
                )
                assert b_scale is not None

                tmp2 = tmp2.to(f32) * b_scale
                w2_dq = (w2[i].to(f32) * w2_scale[i]).transpose(0, 1)
                out[mask] = (tmp2 @ w2_dq).to(out.dtype)
                if b_bias2 is not None:
                    out[mask] = out[mask] + b_bias2[i].view(1, -1).to(out.dtype)

    if apply_router_weights_on_input:
        return out
    else:
        return (
            (out.view(M, -1, w2.shape[1]).to(f32) * topk_weight.view(M, -1, 1))
            .sum(dim=1)
            .to(out.dtype)
        )


def torch_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    b_bias1: torch.Tensor | None = None,
    b_bias2: torch.Tensor | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    activation: MoEActivation = MoEActivation.SILU,
) -> torch.Tensor:
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    return torch_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
        global_num_experts,
        b_bias1,
        b_bias2,
        expert_map,
        activation=activation,
    )


def torch_moe_single(a, w, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    _, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.view(-1)
    for i in range(w.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = a[mask] @ w[i].transpose(0, 1)
    return (out.view(B, -1, w.shape[1])).sum(dim=1)


# A special version of op check that has a restricted default set of test_utils
# and a patched version of allclose that supports fp8 types.
def opcheck(
    op: torch._ops.OpOverload
    | torch._ops.OpOverloadPacket
    | torch._library.custom_ops.CustomOpDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    test_utils: str | Sequence[str] = ALL_OPCHECK_TEST_UTILS,
    raise_exception: bool = True,
    cond: bool = True,
) -> dict[str, str]:
    with patch("torch.allclose", new=fp8_allclose):
        return (
            torch.library.opcheck(
                op, args, kwargs, test_utils=test_utils, raise_exception=raise_exception
            )
            if cond
            else {}
        )


# For testing quantized linear kernels
def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # We treat N-dimensional group scaling as extended numpy-style broadcasting
    # in numpy simply stretches dimensions with an extent of 1 to match
    # the target shape by repeating the data along that dimension (broadcasting)
    # , we extend these semantics to say if the extent of a dimension in the
    # source shape is not 1 and does not match the target shape we repeat each
    # element along that dimension src_shape[dim] // target_shape[dim] times
    # example if we have:
    #       a = [[1, 2], and target_shape = (2, 4)
    #            [3, 4]]
    # then we would expand a to:
    #       a = [[1, 1, 2, 2],
    #            [3, 3, 4, 4]]
    # NOTE this function does not explicitly broadcast dimensions
    # with an extent of 1, since this can be done implicitly by pytorch
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    output = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output
