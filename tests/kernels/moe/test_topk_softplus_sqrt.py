# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import importlib.util

import pytest
import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import (
    _dsv4_topk_triton,
    dsv4_topk,
    dsv4_topk_platform_supported,
    is_cutedsl_dsv4_topk_available,
)
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


def _dsv4_moe_config(monkeypatch, default_vllm_config, has_hash_routing, backend):
    from dataclasses import replace

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
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
    return layer, replace(layer.routed_experts.moe_config, moe_backend=backend)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend selection")
@pytest.mark.parametrize("has_hash_routing", [False, True])
@pytest.mark.parametrize(
    "backend,expected_experts",
    [
        ("aiter_triton_mxfp4_bf16", "AiterW4A16ExpertsMonolithic"),
        ("aiter", "AiterMxfp4Experts"),
        ("triton_unfused", "UnfusedOAITritonExperts"),
    ],
)
@override_parallel_groups(SingleRankGroup())
def test_hash_routing_backend_selection(
    default_vllm_config,
    monkeypatch,
    has_hash_routing,
    backend,
    expected_experts,
):
    """Hash tables reach backend selection and hash-capable kernels."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        select_deepseek_v4_mxfp4_moe_backend,
    )

    layer, config = _dsv4_moe_config(
        monkeypatch, default_vllm_config, has_hash_routing, backend
    )
    _, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)
    assert experts_cls is not None
    assert experts_cls.__name__ == expected_experts
    # Monolithic kernels route internally, so they read the table from the layer.
    assert (layer.routed_experts.hash_indices_table is not None) == has_hash_routing


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend selection")
@override_parallel_groups(SingleRankGroup())
def test_hash_routing_rejects_monolithic_without_support(
    default_vllm_config, monkeypatch
):
    """A monolithic kernel that does not implement hash routing is rejected."""
    from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a16_moe import (
        AiterW4A16ExpertsMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        select_deepseek_v4_mxfp4_moe_backend,
    )

    monkeypatch.setattr(
        AiterW4A16ExpertsMonolithic,
        "_supports_hash_routing",
        staticmethod(lambda: False),
    )
    _, config = _dsv4_moe_config(
        monkeypatch, default_vllm_config, True, "aiter_triton_mxfp4_bf16"
    )
    with pytest.raises(ValueError, match="hash routing"):
        select_deepseek_v4_mxfp4_moe_backend(config)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend selection")
@pytest.mark.parametrize(
    "ep_config,supported",
    [
        ({"use_ep": True, "ep_size": 8}, True),
        ({"use_ep": True, "ep_size": 8, "enable_eplb": True}, False),
        ({"use_ep": True, "ep_size": 8, "dp_size": 2}, False),
    ],
    ids=["naive_ep", "eplb", "all2all"],
)
@override_parallel_groups(SingleRankGroup())
def test_aiter_w4a16_expert_parallel_selection(
    default_vllm_config, monkeypatch, ep_config, supported
):
    """Naive EP (no all2all, no EPLB) selects the AITER W4A16 monolithic kernel,
    including on hash-routed layers."""
    from dataclasses import replace

    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        select_deepseek_v4_mxfp4_moe_backend,
    )

    _, config = _dsv4_moe_config(
        monkeypatch, default_vllm_config, True, "aiter_triton_mxfp4_bf16"
    )
    config = replace(
        config, moe_parallel_config=replace(config.moe_parallel_config, **ep_config)
    )
    if supported:
        _, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)
        assert experts_cls is not None
        assert experts_cls.__name__ == "AiterW4A16ExpertsMonolithic"
    else:
        with pytest.raises(ValueError, match="parallel config"):
            select_deepseek_v4_mxfp4_moe_backend(config)


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
    not dsv4_topk_platform_supported(),
    reason="The DeepSeek V4 fast path runs on CUDA and gfx950.",
)
@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "indices_type"),
    [
        (0, 256, torch.int32),
        (17, 256, torch.int32),
        (17, 384, torch.int64),
    ],
)
@pytest.mark.parametrize("use_padding", [False, True])
def test_dsv4_fast_topk(
    num_tokens: int,
    num_experts: int,
    indices_type: torch.dtype,
    use_padding: bool,
    monkeypatch: pytest.MonkeyPatch,
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
    is_padding = torch.arange(num_tokens, device="cuda") % 2 == 1
    if use_padding:
        monkeypatch.setattr(
            "vllm.model_executor.layers.fused_moe.router."
            "fused_topk_bias_router.get_moe_padding_mask",
            lambda _: is_padding,
        )
        gating_output[is_padding] = float("nan")
        topk_weights_ref[is_padding] = 0
        topk_ids_ref[is_padding] = -1
    topk_weights, topk_ids = fused_topk_bias(
        gating_output,
        gating_output,
        "sqrtsoftplus",
        correction_bias,
        6,
        True,
        indices_type=indices_type,
        routed_scaling_factor=1.5,
    )

    assert topk_ids.dtype == indices_type
    torch.testing.assert_close(topk_ids_ref.to(indices_type), topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        topk_weights_ref,
        topk_weights,
        atol=2e-5,
        rtol=2e-5,
    )

    if use_padding and num_tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            weights, ids = dsv4_topk(
                gating_output, correction_bias, indices_type, 1.5, is_padding=is_padding
            )
        is_padding.fill_(True)
        graph.replay()
        assert (ids == -1).all()
        assert (weights == 0).all()


@pytest.mark.skipif(
    not (current_platform.is_rocm() and dsv4_topk_platform_supported()),
    reason="ROCm AITER MoE path on gfx950",
)
@pytest.mark.parametrize("num_tokens", [1, 48, 224, 4096])
@pytest.mark.parametrize("use_padding", [False, True])
def test_dsv4_fast_topk_with_aiter_fused_moe(
    num_tokens: int,
    use_padding: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.router import (
        fused_topk_bias_router,
    )

    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    calls = []

    def dsv4_topk_spy(*args, **kwargs):
        calls.append(1)
        return dsv4_topk(*args, **kwargs)

    monkeypatch.setattr(fused_topk_bias_router, "dsv4_topk", dsv4_topk_spy)

    torch.manual_seed(0)
    num_experts = 384
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
    if use_padding:
        is_padding = torch.arange(num_tokens, device="cuda") % 2 == 1
        monkeypatch.setattr(
            fused_topk_bias_router, "get_moe_padding_mask", lambda _: is_padding
        )
        gating_output[is_padding] = float("nan")
        topk_weights_ref[is_padding] = 0
        topk_ids_ref[is_padding] = -1

    topk_weights, topk_ids = fused_topk_bias(
        gating_output,
        gating_output,
        "sqrtsoftplus",
        correction_bias,
        6,
        True,
        routed_scaling_factor=1.5,
    )

    assert calls, "the AITER MoE path did not take dsv4_topk"
    torch.testing.assert_close(topk_ids_ref, topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights_ref, topk_weights, atol=2e-5, rtol=2e-5)


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
    not dsv4_topk_platform_supported(),
    reason="The DeepSeek V4 fast path runs on CUDA and gfx950.",
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


@pytest.mark.skipif(
    not is_cutedsl_dsv4_topk_available(),
    reason="CuTe-DSL dsv4_topk requires CUDA SM80+ with cutlass.cute.",
)
@pytest.mark.parametrize("num_tokens", [17, 1027])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("has_padding", [False, True])
@pytest.mark.parametrize("vl_input_dtype", [None, torch.int32, torch.int64])
def test_dsv4_topk_cutedsl_matches_triton_bit_exact(
    num_tokens: int,
    num_experts: int,
    indices_dtype: torch.dtype,
    has_padding: bool,
    vl_input_dtype: torch.dtype | None,
):
    """Verify CuTe-DSL dsv4_topk matches Triton bit-for-bit across both warp paths."""
    torch.manual_seed(42)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    correction_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    # Edge cases: ties, -0.0/+0.0 ties, softplus >20 cutoff, zero sum, NaN.
    gating_output[0] = 0.0
    gating_output[1] = -0.0
    gating_output[1, ::2] = 0.0
    gating_output[2] = torch.linspace(19.0, 25.0, num_experts, device="cuda")
    gating_output[3] = -100.0
    gating_output[4, ::3] = float("nan")

    is_padding = (
        (torch.arange(num_tokens, device="cuda") % 3 == 2) if has_padding else None
    )
    if is_padding is not None:
        gating_output[is_padding] = float("nan")

    input_ids = None
    bias_vl = None
    image_sentinel_lo = 0
    if vl_input_dtype is not None:
        image_sentinel_lo = 120
        input_ids = _make_mixed_input_ids(num_tokens, image_sentinel_lo).to(
            vl_input_dtype
        )
        bias_vl = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    ref_w, ref_ids = _dsv4_topk_triton(
        gating_output,
        correction_bias,
        indices_dtype,
        1.75,
        is_padding=is_padding,
        input_ids=input_ids,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk_cutedsl import (
        dsv4_topk_cutedsl,
    )

    out_w, out_ids = dsv4_topk_cutedsl(
        gating_output,
        correction_bias,
        indices_dtype,
        1.75,
        is_padding=is_padding,
        input_ids=input_ids,
        bias_vl=bias_vl,
        image_sentinel_lo=image_sentinel_lo,
    )

    assert out_ids.dtype == indices_dtype
    torch.testing.assert_close(out_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(out_w, ref_w, atol=0, rtol=0)


@pytest.mark.skipif(
    not is_cutedsl_dsv4_topk_available(),
    reason="CuTe-DSL dsv4_topk requires CUDA SM80+ with cutlass.cute.",
)
@pytest.mark.parametrize("num_tokens", [17, 1027])
@pytest.mark.parametrize("num_experts", [256, 384])
def test_dsv4_topk_cutedsl_ties_and_neg_inf_parity(
    num_tokens: int,
    num_experts: int,
):
    """Verify final-score ties (across all 4 warps), +/-0.0 ties, and -inf bias."""
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk_cutedsl import (
        dsv4_topk_cutedsl,
    )

    torch.manual_seed(7)
    # 1. Exact ties on final (score + bias) spanning all 4 warps (> 6 tied).
    logits = torch.full(
        (num_tokens, num_experts), -20.0, dtype=torch.float32, device="cuda"
    )
    bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
    tied_ids = [5, 37, 101, 133, 197, 229, 250]
    logits[0, tied_ids] = 2.0
    # Signed-zero tie on final score + bias: softplus(-100) == 0 -> sqrt == 0.
    logits[1] = -100.0
    bias_zero_tie = torch.full((num_experts,), -1.0, dtype=torch.float32, device="cuda")
    bias_zero_tie[tied_ids] = 0.0
    bias_zero_tie[tied_ids[1::2]] = -0.0

    for row_logits, row_bias, expected_row0 in [
        (logits, bias, tied_ids[:6]),
        (logits[1:2].repeat(num_tokens, 1), bias_zero_tie, tied_ids[:6]),
        # All -inf bias: selects 6 distinct lowest expert IDs 0..5.
        (
            torch.zeros((num_tokens, num_experts), dtype=torch.float32, device="cuda"),
            torch.full(
                (num_experts,), -float("inf"), dtype=torch.float32, device="cuda"
            ),
            [0, 1, 2, 3, 4, 5],
        ),
    ]:
        ref_w, ref_ids = _dsv4_topk_triton(row_logits, row_bias, torch.int32, 1.5)
        out_w, out_ids = dsv4_topk_cutedsl(row_logits, row_bias, torch.int32, 1.5)
        torch.testing.assert_close(out_ids, ref_ids, atol=0, rtol=0)
        torch.testing.assert_close(out_w, ref_w, atol=0, rtol=0)
        assert out_ids[0].tolist() == expected_row0

    # Partial -inf bias (< 6 finite experts): finite winners first, then lowest IDs.
    partial_bias = torch.full(
        (num_experts,), -float("inf"), dtype=torch.float32, device="cuda"
    )
    partial_bias[200] = 5.0
    partial_bias[100] = 3.0
    zero_logits = torch.zeros(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    ref_w, ref_ids = _dsv4_topk_triton(zero_logits, partial_bias, torch.int32, 1.5)
    out_w, out_ids = dsv4_topk_cutedsl(zero_logits, partial_bias, torch.int32, 1.5)
    torch.testing.assert_close(out_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(out_w, ref_w, atol=0, rtol=0)
    assert out_ids[0].tolist() == [200, 100, 0, 1, 2, 3]


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The DeepSeek V4 fast path is CUDA-only.",
)
def test_dsv4_topk_direct_entry_unsupported_dtypes_fallback():
    """Direct dsv4_topk calls with unsupported dtypes fall back safely to Triton."""
    torch.manual_seed(0)
    num_tokens, num_experts = 8, 384
    logits_f32 = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    # int16 indices (2 bytes/elem) must not enter the 4-byte CuTe-DSL writer.
    out_w, out_ids = dsv4_topk(logits_f32, bias, torch.int16, 1.5)
    ref_w, ref_ids = _dsv4_topk_triton(logits_f32, bias, torch.int16, 1.5)
    assert out_ids.dtype == torch.int16
    torch.testing.assert_close(out_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(out_w, ref_w, atol=0, rtol=0)

    # bf16 logits (2 bytes/elem) must not enter the 4-byte CuTe-DSL reader.
    logits_bf16 = logits_f32.to(torch.bfloat16)
    out_w2, out_ids2 = dsv4_topk(logits_bf16, bias, torch.int32, 1.5)
    ref_w2, ref_ids2 = _dsv4_topk_triton(logits_bf16, bias, torch.int32, 1.5)
    torch.testing.assert_close(out_ids2, ref_ids2, atol=0, rtol=0)
    torch.testing.assert_close(out_w2, ref_w2, atol=0, rtol=0)


def _is_ll_bf16_pdl_replay_available() -> bool:
    if not (
        is_cutedsl_dsv4_topk_available() and current_platform.has_device_capability(90)
    ):
        return False
    if importlib.util.find_spec("quack") is None:
        return False
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_available,
    )

    return is_ll_bf16_available()


@pytest.mark.skipif(
    not is_cutedsl_dsv4_topk_available(),
    reason="CuTe-DSL dsv4_topk requires CUDA SM80+ with cutlass.cute.",
)
def test_dsv4_topk_cutedsl_accelerator_stream():
    """Verify torch.accelerator.set_stream ordering and CUDA-graph buffer replay."""
    device = torch.device("cuda", 0)
    num_experts = 384
    bias = torch.zeros(num_experts, dtype=torch.float32, device=device)

    # 1. Stream switch via torch.accelerator.set_stream (bypassing torch.cuda).
    alt_stream = torch.Stream(device=device)
    prev_stream = torch.accelerator.current_stream(device)
    logits = torch.full((8, num_experts), -10.0, dtype=torch.float32, device=device)
    torch.accelerator.synchronize()
    try:
        torch.accelerator.set_stream(alt_stream)
        logits[:, 10:16] = 10.0
        out_w, out_ids = dsv4_topk(logits, bias, torch.int32, 1.5)
    finally:
        torch.accelerator.set_stream(prev_stream)
    alt_stream.synchronize()
    expected = torch.arange(10, 16, dtype=torch.int32, device=device).expand(8, 6)
    torch.testing.assert_close(out_ids, expected, atol=0, rtol=0)
    assert out_w.isfinite().all()

    # 2. SM80+ CUDA-graph capture and buffer-reuse replay on both 4-warp and
    #    packed single-warp paths.
    for num_tokens in (8, 1027):
        logits_buf = torch.full(
            (num_tokens, num_experts), -10.0, dtype=torch.float32, device=device
        )
        logits_buf[:, 0:6] = 10.0
        dsv4_topk(logits_buf, bias, torch.int32, 1.5)
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_w, g_ids = dsv4_topk(logits_buf, bias, torch.int32, 1.5)

        exp_a = torch.arange(0, 6, dtype=torch.int32, device=device).expand(
            num_tokens, 6
        )
        exp_b = torch.arange(200, 206, dtype=torch.int32, device=device).expand(
            num_tokens, 6
        )
        for step in range(4):
            logits_buf.fill_(-10.0)
            if step % 2 == 0:
                logits_buf[:, 0:6] = 10.0
                exp = exp_a
            else:
                logits_buf[:, 200:206] = 10.0
                exp = exp_b
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(g_ids, exp, atol=0, rtol=0)
            assert g_w.isfinite().all()


@pytest.mark.skipif(
    not _is_ll_bf16_pdl_replay_available(),
    reason=(
        "PDL ll_bf16_gemm -> dsv4_topk replay requires CUDA SM90+ with "
        "cutlass.cute and quack."
    ),
)
def test_dsv4_topk_cutedsl_pdl_gemm_replay():
    """Verify SM90+ PDL CUDA-graph replay after ll_bf16 split-K GEMM."""
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm,
    )

    device = torch.device("cuda", 0)
    num_tokens, num_experts, k_dim = 8, 384, 5120
    bias = torch.zeros(num_experts, dtype=torch.float32, device=device)
    w_gate = torch.zeros((num_experts, k_dim), dtype=torch.bfloat16, device=device)
    w_gate[0:6, 0] = 4.0
    w_gate[200:206, 1] = 4.0
    x_buf = torch.zeros((num_tokens, k_dim), dtype=torch.bfloat16, device=device)

    # Warmup before graph capture.
    x_buf[:, 0] = 1.0
    warm_logits = ll_bf16_gemm(x_buf, w_gate, output_dtype=torch.float32)
    dsv4_topk(warm_logits, bias, torch.int32, 1.5)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Captured output buffer address is reused across all replays.
        g_logits = ll_bf16_gemm(x_buf, w_gate, output_dtype=torch.float32)
        g_w, g_ids = dsv4_topk(g_logits, bias, torch.int32, 1.5)

    exp_a = torch.arange(0, 6, dtype=torch.int32, device=device).expand(num_tokens, 6)
    exp_b = torch.arange(200, 206, dtype=torch.int32, device=device).expand(
        num_tokens, 6
    )
    for step in range(8):
        x_buf.zero_()
        if step % 2 == 0:
            x_buf[:, 0] = 1.0
            exp = exp_a
        else:
            x_buf[:, 1] = 1.0
            exp = exp_b
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(g_ids, exp, atol=0, rtol=0)
        assert g_w.isfinite().all()
