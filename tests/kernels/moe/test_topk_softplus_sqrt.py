# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.platforms import current_platform


def _torch_topk_softplus_sqrt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 0,
):
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores

    if hash_indices_table is not None:
        assert input_ids is not None
        if bias_vl is None or image_sentinel_lo == 0:
            topk_ids = hash_indices_table[input_ids.long()]
        else:
            image_mask = (input_ids.long() >= image_sentinel_lo) & (
                input_ids.long() < image_sentinel_lo + 5
            )
            topk_ids = hash_indices_table[input_ids.long()]
            vl_ids = (scores + bias_vl).topk(topk, dim=-1).indices
            topk_ids = torch.where(
                image_mask.unsqueeze(-1), vl_ids.to(topk_ids.dtype), topk_ids
            )
    else:
        if bias_vl is not None and image_sentinel_lo > 0:
            # Image tokens carry five consecutive in-vocab sentinel ids
            # starting at image_sentinel_lo and select experts with bias_vl
            # instead of the regular bias.
            assert input_ids is not None
            assert e_score_correction_bias is not None
            image_mask = (
                (input_ids.long() >= image_sentinel_lo)
                & (input_ids.long() < image_sentinel_lo + 5)
            ).unsqueeze(-1)
            row_bias = torch.where(
                image_mask,
                bias_vl.unsqueeze(0),
                e_score_correction_bias.unsqueeze(0),
            )
            scores_for_choice = scores + row_bias
        elif e_score_correction_bias is not None:
            scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
        else:
            scores_for_choice = scores
        # Match the fused kernel's deterministic tie-break: lower expert ids
        # win when scores are equal. torch.topk does not guarantee which tied
        # index it selects at the k-th boundary.
        topk_ids = torch.argsort(
            scores_for_choice, dim=-1, descending=True, stable=True
        )[:, :topk]

    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend selection")
@pytest.mark.parametrize("has_hash_routing", [False, True])
@pytest.mark.parametrize(
    "backend,expected_experts",
    [
        ("aiter_triton_mxfp4_bf16", "AiterW4A16ExpertsMonolithic"),
        ("aiter", "AiterExperts"),
        ("triton_unfused", "UnfusedOAITritonExperts"),
    ],
)
def test_hash_routing_backend_selection(
    dist_init,
    default_vllm_config,
    monkeypatch,
    has_hash_routing,
    backend,
    expected_experts,
):
    """Hash tables must reach backend selection; modular AITER remains eligible."""
    from dataclasses import replace

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        select_deepseek_v4_mxfp4_moe_backend,
    )
    from vllm.platforms import rocm

    # Exercise both native MX and monolithic AITER selection on any ROCm device.
    monkeypatch.setattr(rocm, "on_gfx950", lambda: True)
    monkeypatch.setattr(rocm, "on_gfx1250", lambda: False)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    default_vllm_config.kernel_config.moe_backend = "triton"
    layer = FusedMoEFactory(
        num_experts=8,
        top_k=2,
        hidden_size=256,
        intermediate_size=256,
        params_dtype=torch.bfloat16,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=torch.zeros(8),
        hash_indices_table=(
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
            if has_hash_routing
            else None
        ),
        prefix="hash_routing_selection",
    )
    config = replace(layer.routed_experts.moe_config, moe_backend=backend)

    if has_hash_routing and backend == "aiter_triton_mxfp4_bf16":
        with pytest.raises(ValueError, match="hash routing"):
            select_deepseek_v4_mxfp4_moe_backend(config)
    else:
        _, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)
        assert experts_cls.__name__ == expected_experts


def test_torch_topk_softplus_sqrt_breaks_ties_by_expert_id():
    gating_output = torch.tensor([[2.0, 1.0, 1.0, 0.0]])

    _, topk_ids = _torch_topk_softplus_sqrt(
        gating_output,
        topk=2,
        renormalize=False,
        routed_scaling_factor=1.0,
    )

    assert topk_ids.tolist() == [[0, 1]]


def test_sqrtsoftplus_bias_uses_deepseek_v4_routing_method():
    assert (
        get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=8,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=True,
        )
        == RoutingMethodType.DeepseekV4
    )
    assert (
        get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=8,
            renormalize=False,
            num_expert_group=None,
            has_e_score_bias=True,
        )
        == RoutingMethodType.Unspecified
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [128, 256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Different kernels may return the topk experts in different orders when
    # scores tie; sort by expert id before comparing.
    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("num_tokens", [1, 33, 128])
@pytest.mark.parametrize("hidden_size", [1024, 2048])
@pytest.mark.parametrize("num_experts", [256, 384, 512])
@pytest.mark.parametrize("topk", [6, 8, 16])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_hash(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    vocab_size = 1024
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    # Per-token fixed expert selection: for each vocab id pick `topk` distinct
    # experts.
    hash_indices_table = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
    ).to(device="cuda", dtype=torch.long)
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.long, device="cuda"
    )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=topk,
        renormalize=renormalize,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        routed_scaling_factor=routed_scaling_factor,
    )

    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(sorted_ref_ids, sorted_ids, atol=0, rtol=0)

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The DeepSeek V4 fast path is CUDA-only.",
)
@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "indices_type"),
    [
        (0, 256, torch.uint32),
        (17, 256, torch.uint32),
        (17, 384, torch.int64),
    ],
)
def test_dsv4_fast_topk(
    num_tokens: int,
    num_experts: int,
    indices_type: torch.dtype,
):
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=6,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=correction_bias,
    )
    topk_weights, topk_ids = dsv4_topk(
        gating_output, correction_bias, indices_type, 1.5
    )

    assert topk_ids.dtype == indices_type
    torch.testing.assert_close(topk_ids_ref.to(indices_type), topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        topk_weights_ref,
        topk_weights,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize("use_hash", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("use_padding_mask", [False, True])
@pytest.mark.parametrize("pad_with_nan", [False, True])
@pytest.mark.parametrize("num_experts", [128, 256, 384])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
def test_fused_topk_softplus_sqrt_padding(
    use_hash: bool,
    use_bias: bool,
    use_padding_mask: bool,
    pad_with_nan: bool,
    num_experts: int,
    dtype: torch.dtype,
):
    """Verify explicit padding and NaN-padded rows do not affect real rows."""
    torch.manual_seed(0)
    num_tokens = 8
    topk = 6
    indices_dtype = torch.int32

    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    # Exercise top-k cutoff ties without relying on random low-precision collisions.
    gating_output[0] = 0

    padding_rows = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    padding_rows[1::2] = True
    if pad_with_nan:
        gating_output[padding_rows] = float("nan")
    is_padding = padding_rows if use_padding_mask else None

    # A negative correction bias makes explicit pad rows look selectable unless
    # the kernel uses the is_padding guard.
    e_score_correction_bias = None
    if use_bias:
        e_score_correction_bias = (
            -torch.rand((num_experts,), dtype=torch.float32, device="cuda") - 1.0
        )

    input_ids = None
    hash_indices_table = None
    if use_hash:
        vocab_size = 64
        hash_indices_table = torch.stack(
            [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
        ).to(device="cuda", dtype=indices_dtype)
        input_ids = torch.randint(
            0, vocab_size, (num_tokens,), dtype=indices_dtype, device="cuda"
        )

    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(num_tokens, topk, dtype=indices_dtype, device="cuda")
    token_expert_indices = torch.empty(
        num_tokens, topk, dtype=torch.int32, device="cuda"
    )

    ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=True,
        routed_scaling_factor=1.0,
        e_score_correction_bias=e_score_correction_bias,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        is_padding=is_padding,
    )

    if use_padding_mask:
        pad_ids = topk_ids[padding_rows]
        pad_weights = topk_weights[padding_rows]
        assert torch.equal(pad_ids, torch.full_like(pad_ids, -1)), (
            f"Explicit pad rows should contain only -1 ids, got {pad_ids.tolist()}"
        )
        assert (pad_weights == 0).all(), (
            "Explicit pad rows should have all-zero weights, "
            f"got {pad_weights.tolist()}"
        )

    if pad_with_nan:
        nan_pad_weights = topk_weights[padding_rows]
        assert torch.isfinite(nan_pad_weights).all(), (
            f"NaN-padded rows have non-finite weights, got {nan_pad_weights.tolist()}"
        )
        assert (nan_pad_weights == 0).all(), (
            "NaN-padded rows should have all-zero weights, "
            f"got {nan_pad_weights.tolist()}"
        )

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
        e_score_correction_bias=e_score_correction_bias,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
    )

    rows_to_compare = torch.ones(num_tokens, dtype=torch.bool, device="cuda")
    if use_padding_mask or pad_with_nan:
        rows_to_compare = ~padding_rows

    sorted_ref_ids, idx_ref = topk_ids_ref[rows_to_compare].sort(dim=-1)
    sorted_ids, idx_ops = topk_ids[rows_to_compare].sort(dim=-1)
    torch.testing.assert_close(
        sorted_ref_ids, sorted_ids.to(sorted_ref_ids.dtype), atol=0, rtol=0
    )

    sorted_w_ref = topk_weights_ref[rows_to_compare].gather(1, idx_ref)
    sorted_w = topk_weights[rows_to_compare].gather(1, idx_ops)
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


def _make_mixed_input_ids(
    num_tokens: int, image_sentinel_lo: int, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """Mix of text, image-sentinel, and above-sentinel ids.

    Every third token is an image sentinel id (cycling through the five slots
    [image_sentinel_lo, image_sentinel_lo + 5)); every sixth token carries an
    id just above the sentinel block, simulating the named special tokens
    that follow it — those must route as regular text tokens.
    """
    input_ids = torch.randint(
        0, image_sentinel_lo, (num_tokens,), dtype=torch.long, device="cuda"
    )
    pos = torch.arange(num_tokens, device="cuda")
    image_rows = pos % 3 == 0
    input_ids[image_rows] = image_sentinel_lo + (pos[image_rows] // 3) % 5
    above_rows = pos % 6 == 3
    input_ids[above_rows] = image_sentinel_lo + 5
    return input_ids.to(dtype)


def _assert_topk_matches(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights_ref: torch.Tensor,
    topk_ids_ref: torch.Tensor,
) -> None:
    # Different kernels may return the topk experts in different orders when
    # scores tie; sort by expert id before comparing.
    sorted_ref_ids, idx_ref = topk_ids_ref.sort(dim=-1)
    sorted_ids, idx_ops = topk_ids.sort(dim=-1)
    torch.testing.assert_close(
        sorted_ref_ids, sorted_ids.to(sorted_ref_ids.dtype), atol=0, rtol=0
    )

    sorted_w_ref = topk_weights_ref.gather(1, idx_ref)
    sorted_w = topk_weights.gather(1, idx_ops.to(torch.long))
    torch.testing.assert_close(sorted_w_ref, sorted_w, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize(
    ("num_experts", "topk", "renormalize", "dtype"),
    [
        (256, 6, True, torch.float32),  # Triton dsv4 fast path
        (256, 8, True, torch.bfloat16),  # CUDA topk kernel
        (384, 6, False, torch.half),
        (512, 16, True, torch.float32),
    ],
)
def test_fused_topk_softplus_sqrt_bias_vl(
    num_experts: int,
    topk: int,
    renormalize: bool,
    dtype: torch.dtype,
):
    """Image sentinel tokens must select experts with bias_vl.

    Rows with ids just above the sentinel block (image_sentinel_lo + 5)
    simulate the named special tokens that follow it and must route through
    the regular bias path; the reference comparison covers them.
    """
    torch.manual_seed(0)
    num_tokens = 64
    hidden_size = 1024
    vocab_size = 128
    image_sentinel_lo = vocab_size - 8
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device="cuda"
    )
    bias_vl = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    input_ids = _make_mixed_input_ids(num_tokens, image_sentinel_lo)
    assert (input_ids == image_sentinel_lo + 5).any()

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
        e_score_correction_bias=e_score_correction_bias,
        input_ids=input_ids,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=e_score_correction_bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
        input_tokens=input_ids,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    _assert_topk_matches(topk_weights, topk_ids, topk_weights_ref, topk_ids_ref)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="This test is skipped on non-CUDA platform.",
)
@pytest.mark.parametrize(
    ("num_experts", "topk", "renormalize", "dtype", "indices_dtype"),
    [
        (256, 6, True, torch.float32, torch.long),  # specialized hash kernel
        (384, 6, True, torch.float32, torch.int32),  # specialized hash kernel
        (384, 8, False, torch.bfloat16, torch.long),  # generic hash kernel
        (512, 6, True, torch.float32, torch.long),  # generic hash kernel
    ],
)
def test_fused_topk_softplus_sqrt_hash_bias_vl(
    num_experts: int,
    topk: int,
    renormalize: bool,
    dtype: torch.dtype,
    indices_dtype: torch.dtype,
):
    """Hash MoE: text rows use tid2eid, image rows use topk(score + bias_vl).

    Sentinel ids are in-vocab; rows with ids above the sentinel block
    (image_sentinel_lo + 5) must still come straight from the hash table.
    """
    torch.manual_seed(0)
    num_tokens = 64
    hidden_size = 1024
    vocab_size = 64
    image_sentinel_lo = vocab_size - 8
    hidden_states = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    hash_indices_table = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
    ).to(device="cuda", dtype=indices_dtype)
    bias_vl = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    input_ids = _make_mixed_input_ids(num_tokens, image_sentinel_lo)
    image_rows = (input_ids >= image_sentinel_lo) & (input_ids < image_sentinel_lo + 5)
    assert image_rows.any() and (~image_rows).any()
    assert (input_ids == image_sentinel_lo + 5).any()

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=2.5,
        input_ids=input_ids,
        hash_indices_table=hash_indices_table,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    topk_weights, topk_ids = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=gating_output,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=topk,
        renormalize=renormalize,
        input_tokens=input_ids,
        hash_indices_table=hash_indices_table,
        routed_scaling_factor=2.5,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    _assert_topk_matches(topk_weights, topk_ids, topk_weights_ref, topk_ids_ref)

    # Text rows must come straight from the hash table, image rows from
    # topk(score + bias_vl).
    text_ids = hash_indices_table[input_ids[~image_rows].long()]
    torch.testing.assert_close(
        topk_ids[~image_rows].to(text_ids.dtype), text_ids, atol=0, rtol=0
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The DeepSeek V4 fast path is CUDA-only.",
)
def test_dsv4_fast_topk_bias_vl():
    torch.manual_seed(0)
    num_tokens = 33
    num_experts = 256
    vocab_size = 128
    image_sentinel_lo = vocab_size - 8
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    bias_vl = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    input_ids = _make_mixed_input_ids(num_tokens, image_sentinel_lo)

    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=6,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=correction_bias,
        input_ids=input_ids,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )
    topk_weights, topk_ids = dsv4_topk(
        gating_output,
        correction_bias,
        torch.int64,
        1.5,
        input_ids=input_ids,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    assert topk_ids.dtype == torch.int64
    torch.testing.assert_close(topk_ids_ref.to(torch.int64), topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights_ref, topk_weights, atol=2e-5, rtol=2e-5)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


@pytest.mark.skipif(not _on_gfx950(), reason="fused router gate targets gfx950")
@pytest.mark.parametrize(
    "num_tokens", [1, 4, 8, 17, 33, 64, 110, 128, 256, 768, 1280, 1536]
)
@pytest.mark.parametrize(
    ("topk", "renormalize", "has_bias", "indices_dtype"),
    [
        (1, False, True, torch.int32),
        (6, False, False, torch.int32),
        (8, True, True, torch.int64),
    ],
)
# 5120: DeepSeek-V4.1-Flash, 7168: DeepSeek-V4-Pro.
@pytest.mark.parametrize("hidden_size", [5120, 7168])
def test_rocm_fused_router_gate_matches_gate_gemm_plus_selection(
    num_tokens: int,
    topk: int,
    renormalize: bool,
    has_bias: bool,
    indices_dtype: torch.dtype,
    hidden_size: int,
) -> None:
    """Tiled gate GEMM and selection must preserve routing across token masks."""
    from vllm.model_executor.layers.fused_moe.router.rocm_fused_router_gate import (
        rocm_fused_router_gate,
    )

    torch.manual_seed(0)
    num_experts = 384
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_weight = (
        torch.randn(num_experts, hidden_size, dtype=torch.float32, device="cuda")
        * hidden_size**-0.5
    ).to(torch.bfloat16)
    correction_bias = (
        torch.randn(num_experts, dtype=torch.float32, device="cuda")
        if has_bias
        else None
    )

    gating_output = hidden_states.float() @ router_weight.float().t()
    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
        e_score_correction_bias=correction_bias,
    )

    topk_weights, topk_ids = rocm_fused_router_gate(
        hidden_states,
        router_weight,
        correction_bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=1.5,
        indices_dtype=indices_dtype,
    )

    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == indices_dtype
    torch.testing.assert_close(topk_ids_ref.to(indices_dtype), topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights_ref, topk_weights, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not _on_gfx950(), reason="fused router gate targets gfx950")
@pytest.mark.parametrize("logits_kind", ["ties", "negative_tail"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("num_tokens", [3, 256, 1536])
def test_rocm_fused_router_gate_preserves_ties_and_small_scores(
    logits_kind: str, renormalize: bool, num_tokens: int
) -> None:
    """Equal ranks favor lower ids, and negative logits retain nonzero weights."""
    from vllm.model_executor.layers.fused_moe.router.rocm_fused_router_gate import (
        rocm_fused_router_gate,
    )

    hidden_size, num_experts, topk = 7168, 384, 8
    hidden_states = torch.zeros(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    hidden_states[:, 0] = 1
    router_weight = torch.zeros(
        num_experts, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    correction_bias = None
    if logits_kind == "negative_tail":
        router_weight[:, 0] = torch.arange(num_experts, device="cuda") * 0.125 - 68
    else:
        correction_bias = torch.zeros(num_experts, device="cuda")
        correction_bias[::3] = 1

    gating_output = router_weight[:, 0].float().expand(num_tokens, -1)
    topk_weights_ref, topk_ids_ref = _torch_topk_softplus_sqrt(
        gating_output,
        topk,
        renormalize,
        routed_scaling_factor=1.5,
        e_score_correction_bias=correction_bias,
    )
    topk_weights, topk_ids = rocm_fused_router_gate(
        hidden_states,
        router_weight,
        correction_bias,
        topk,
        renormalize,
        routed_scaling_factor=1.5,
    )

    torch.testing.assert_close(topk_ids_ref, topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights_ref, topk_weights, atol=1e-9, rtol=2e-5)


@pytest.mark.skipif(not _on_gfx950(), reason="fused router gate targets gfx950")
@pytest.mark.parametrize("num_tokens", [17, 256])
def test_rocm_fused_router_gate_image_bias_and_padding(num_tokens: int) -> None:
    """Image sentinel rows rank with bias_vl; padding rows get expert -1.

    Mirrors ``topk_hash_softplus_sqrt``, which the fused gate replaces in the
    DeepSeek-V4.1 MoE runner.
    """
    from vllm.model_executor.layers.fused_moe.router.rocm_fused_router_gate import (
        rocm_fused_router_gate,
    )

    torch.manual_seed(0)
    hidden_size, num_experts, topk, sentinel_lo = 5120, 384, 6, 1000
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    router_weight = (
        torch.randn(num_experts, hidden_size, device="cuda") * hidden_size**-0.5
    ).to(torch.bfloat16)
    bias = torch.randn(num_experts, device="cuda")
    bias_vl = torch.randn(num_experts, device="cuda")
    # Every third row is an image sentinel; sentinel_lo + 5 is a regular token.
    input_ids = torch.full((num_tokens,), sentinel_lo + 5, device="cuda")
    input_ids[::3] = sentinel_lo + torch.arange(0, num_tokens, 3, device="cuda") % 5
    is_padding = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    is_padding[-3:] = True

    logits = hidden_states.float() @ router_weight.float().t()
    image = (input_ids >= sentinel_lo) & (input_ids < sentinel_lo + 5)
    ref_weights = torch.empty(num_tokens, topk, device="cuda")
    ref_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
    for mask, row_bias in ((image, bias_vl), (~image, bias)):
        w, i = _torch_topk_softplus_sqrt(
            logits[mask], topk, True, 1.5, e_score_correction_bias=row_bias
        )
        ref_weights[mask], ref_ids[mask] = w, i.to(torch.int32)
    ref_weights[is_padding], ref_ids[is_padding] = 0.0, -1

    topk_weights, topk_ids = rocm_fused_router_gate(
        hidden_states,
        router_weight,
        bias,
        topk,
        True,
        routed_scaling_factor=1.5,
        bias_vl=bias_vl,
        image_sentinel_lo=sentinel_lo,
        input_ids=input_ids,
        is_padding=is_padding,
    )

    torch.testing.assert_close(ref_ids, topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(ref_weights, topk_weights, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not _on_gfx950(), reason="fused router gate targets gfx950")
def test_rocm_fused_router_gate_empty_rows() -> None:
    from vllm.model_executor.layers.fused_moe.router.rocm_fused_router_gate import (
        rocm_fused_router_gate,
    )

    hidden_states = torch.empty(0, 7168, dtype=torch.bfloat16, device="cuda")
    router_weight = torch.empty(384, 7168, dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = rocm_fused_router_gate(
        hidden_states, router_weight, None, 6, True, indices_dtype=torch.int64
    )

    assert topk_weights.shape == topk_ids.shape == (0, 6)
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == torch.int64
    assert topk_weights.device == topk_ids.device == hidden_states.device


@pytest.mark.skipif(not _on_gfx950(), reason="fused router gate targets gfx950")
@pytest.mark.parametrize(
    "invalid_input", ["cpu", "bias_device", "bias_strided", "indices_dtype"]
)
def test_rocm_fused_router_gate_rejects_unsupported_inputs(invalid_input: str) -> None:
    """Reject unsupported storage before launch or dispatch into the fast path."""
    from vllm.model_executor.layers.fused_moe.router.rocm_fused_router_gate import (
        can_use_rocm_fused_router_gate,
        rocm_fused_router_gate,
    )

    hidden_states = torch.empty(1, 7168, dtype=torch.bfloat16, device="cuda")
    router_weight = torch.empty(384, 7168, dtype=torch.bfloat16, device="cuda")
    correction_bias = None
    indices_dtype = torch.int32
    if invalid_input == "cpu":
        hidden_states = hidden_states.cpu()
        router_weight = router_weight.cpu()
    elif invalid_input == "bias_device":
        correction_bias = torch.empty(384, dtype=torch.float32)
    elif invalid_input == "bias_strided":
        correction_bias = torch.empty(768, dtype=torch.float32, device="cuda")[::2]
    else:
        indices_dtype = torch.float32

    if invalid_input != "indices_dtype":
        assert not can_use_rocm_fused_router_gate(
            hidden_states, router_weight, correction_bias, 8
        )
    with pytest.raises(ValueError):
        rocm_fused_router_gate(
            hidden_states,
            router_weight,
            correction_bias,
            8,
            True,
            indices_dtype=indices_dtype,
        )
