# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This is a test for the AITER ops.
# It tests if the AITER ops are
# 1. correctly registered as custom ops
# 2. correctly defined the relationship between
#    implementation and fake function
# 3. can be used with torch.compile
# This file will be skipped if AITER is not installed
# and the platform is not ROCm.

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

# this import statement is needed to ensure the ops are registered
import vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe  # noqa: F401

# need to import once to ensure the ops are registered
# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

if not aiter_available:
    pytest.skip("These tests require AITER to run.", allow_module_level=True)


def test_rocm_aiter_biased_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, "rocm_aiter_biased_grouped_topk")

    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_biased_grouped_topk)


def test_rocm_aiter_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, "rocm_aiter_grouped_topk")

    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_grouped_topk)


def test_rocm_aiter_biased_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    # Create test tensors
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scale_factor = 1.0

    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device="cuda")
    e_score_correction_bias = torch.randn(
        (expert,), dtype=torch.bfloat16, device="cuda"
    )

    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    # Define a function that uses the op
    def biased_grouped_topk_fn(
        gating_output, e_score_correction_bias, topk_weights, topk_ids
    ):
        return torch.ops.vllm.rocm_aiter_biased_grouped_topk(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scale_factor,
        )

    # Verify the op's fake implementation
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_biased_grouped_topk,
        (gating_output, e_score_correction_bias, topk_weights, topk_ids),
        kwargs={
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
            "need_renorm": renormalize,
            "routed_scaling_factor": scale_factor,
        },
        test_utils=("test_faketensor"),
    )

    # Compile the function with appropriate settings
    compiled_fn = torch.compile(
        biased_grouped_topk_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    topk_weights_original = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_original = torch.empty((token, topk), dtype=torch.int32, device=device)

    topk_weights_compiled = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_compiled = torch.empty((token, topk), dtype=torch.int32, device=device)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    biased_grouped_topk_fn(
        gating_output, e_score_correction_bias, topk_weights_original, topk_ids_original
    )
    compiled_fn(
        gating_output, e_score_correction_bias, topk_weights_compiled, topk_ids_compiled
    )

    # Sort the results for comparison since the order might not be deterministic
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1, indices_original)

    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1, indices_compiled)

    # Verify results match
    assert torch.allclose(
        topk_weights_original, topk_weights_compiled, rtol=1e-2, atol=1e-2
    )
    assert torch.allclose(topk_ids_original, topk_ids_compiled)


def test_rocm_aiter_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    # Create test tensors
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scoring_func = "softmax"
    scale_factor = 1.0

    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device="cuda")

    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    # Define a function that uses the op
    def grouped_topk_fn(gating_output, topk_weights, topk_ids, scoring_func):
        return torch.ops.vllm.rocm_aiter_grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
            scale_factor,
        )

    # Verify the op's fake implementation
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_grouped_topk,
        (gating_output, topk_weights, topk_ids),
        kwargs={
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
            "need_renorm": renormalize,
            "scoring_func": scoring_func,
            "routed_scaling_factor": scale_factor,
        },
        test_utils=("test_faketensor"),
    )

    # Compile the function with appropriate settings
    compiled_fn = torch.compile(
        grouped_topk_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    topk_weights_original = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_original = torch.empty((token, topk), dtype=torch.int32, device=device)

    topk_weights_compiled = torch.empty(
        (token, topk), dtype=torch.float32, device=device
    )
    topk_ids_compiled = torch.empty((token, topk), dtype=torch.int32, device=device)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    grouped_topk_fn(
        gating_output, topk_weights_original, topk_ids_original, scoring_func
    )
    compiled_fn(gating_output, topk_weights_compiled, topk_ids_compiled, scoring_func)

    # Sort the results for comparison since the order might not be deterministic
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1, indices_original)

    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1, indices_compiled)

    # Verify results match
    assert torch.allclose(
        topk_weights_original, topk_weights_compiled, rtol=1e-2, atol=1e-2
    )
    assert torch.allclose(topk_ids_original, topk_ids_compiled)


def test_rocm_aiter_topk_gating_custom_op_registration():
    assert callable(torch.ops.vllm.rocm_aiter_topk_gating)


def _sort_routing(weights: torch.Tensor, ids: torch.Tensor):
    ids_sorted, order = torch.sort(ids, dim=-1)
    weights_sorted = torch.gather(weights, 1, order)
    return weights_sorted, ids_sorted


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
@pytest.mark.parametrize("num_experts", [256, 512])
@pytest.mark.parametrize("topk", [8, 10])
@pytest.mark.parametrize("renormalize", [True, False])
def test_rocm_aiter_topk_gating_matches_softmax_reference(
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
):
    """Numerical check vs softmax + topk, including Qwen3.8 (E=512, k=10)."""
    torch.manual_seed(0)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.bfloat16, device="cuda"
    )
    scores = torch.softmax(gating_output.float(), dim=-1)
    cutoff = torch.topk(scores, k=topk, dim=-1).values[:, -1:]

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    torch.ops.vllm.rocm_aiter_topk_gating(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        0,
        "",
    )

    # BF16 logits can tie at the top-k boundary, so a different tied expert is
    # valid. Check selection against the cutoff and weights against the scores
    # of the experts the kernel actually selected.
    assert ((topk_ids >= 0) & (topk_ids < num_experts)).all()
    sorted_ids = torch.sort(topk_ids, dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    selected_scores = torch.gather(scores, 1, topk_ids.long())
    assert (selected_scores >= cutoff).all()
    expected_weights = selected_scores
    if renormalize:
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topk_weights, expected_weights, atol=2e-2, rtol=2e-2)


def test_rocm_aiter_topk_gating_shared_expert_suffix():
    """Fused shared expert: softmax top-k on the prefix, sigmoid appended.

    Matches the Qwen3.8 FSE launch: 512 routed experts, top-10, one shared
    column, weights wide enough for the suffix, and the shared id prefilled.
    """
    torch.manual_seed(0)
    num_tokens, num_experts, topk, num_shared = 4, 512, 10, 1
    routed = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16, device="cuda")
    shared = torch.randn((num_tokens, num_shared), dtype=torch.bfloat16, device="cuda")
    shared[0, 0] = 40
    gating_output = torch.cat([routed, shared], dim=-1).contiguous()
    width = topk + num_shared
    shared_ids = torch.arange(
        num_experts, num_experts + num_shared, device="cuda", dtype=torch.int32
    )
    topk_ids = torch.full((num_tokens, width), -1, dtype=torch.int32, device="cuda")
    topk_ids[:, topk:] = shared_ids
    topk_weights = torch.empty((num_tokens, width), dtype=torch.float32, device="cuda")
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    torch.ops.vllm.rocm_aiter_topk_gating(
        topk_weights,
        topk_ids[:, :topk],
        token_expert_indices,
        gating_output,
        True,
        num_shared,
        "sigmoid",
    )

    scores = torch.softmax(routed.float(), dim=-1)
    cutoff = torch.topk(scores, k=topk, dim=-1).values[:, -1:]
    routed_ids = topk_ids[:, :topk]
    assert ((routed_ids >= 0) & (routed_ids < num_experts)).all()
    sorted_ids = torch.sort(routed_ids, dim=-1).values
    assert (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()
    selected = torch.gather(scores, 1, routed_ids.long())
    assert (selected >= cutoff).all()
    expected = selected / selected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topk_weights[:, :topk], expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        topk_weights[:, topk:],
        torch.sigmoid(shared.float()),
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.equal(topk_ids[:, topk:], shared_ids.expand_as(topk_ids[:, topk:]))


def test_rocm_aiter_topk_gating_torch_compile_compatibility():
    token, expert, topk, renormalize = 4, 512, 10, True
    gating_output = torch.randn((token, expert), dtype=torch.bfloat16, device="cuda")
    device = gating_output.device

    def gating_fn(gating_output, topk_weights, topk_ids, token_expert_indices):
        return torch.ops.vllm.rocm_aiter_topk_gating(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
            0,
            "",
        )

    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_topk_gating,
        (
            torch.empty((token, topk), dtype=torch.float32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
            gating_output,
        ),
        kwargs={
            "renormalize": renormalize,
            "num_shared_experts": 0,
            "shared_expert_scoring_func": "",
        },
        test_utils=("test_faketensor"),
    )

    compiled_fn = torch.compile(
        gating_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    def alloc():
        return (
            torch.empty((token, topk), dtype=torch.float32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
            torch.empty((token, topk), dtype=torch.int32, device=device),
        )

    w0, i0, t0 = alloc()
    w1, i1, t1 = alloc()
    gating_fn(gating_output, w0, i0, t0)
    compiled_fn(gating_output, w1, i1, t1)
    w0, i0 = _sort_routing(w0, i0)
    w1, i1 = _sort_routing(w1, i1)
    assert torch.allclose(w0, w1, rtol=1e-2, atol=1e-2)
    assert torch.equal(i0, i1)


# ---------------------------------------------------------------------------
# Triton sigmoid+bias top-k for the case where the grouping stage eliminates
# nothing (topk_group >= num_expert_group). The kernel documents itself as an
# in-place equivalent of aiter.biased_grouped_topk for that case, so every
# test below compares it against exactly that, through the same wrapper
# production uses.
# ---------------------------------------------------------------------------

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.router.rocm_sigmoid_bias_flat_topk import (  # noqa: E402, E501
    sigmoid_bias_flat_topk,
)
from vllm.utils.torch_utils import set_random_seed  # noqa: E402


def _routing_inputs(token, expert, topk, gating_dtype, bias_dtype, seed=0):
    set_random_seed(seed)
    gating = torch.randn(token, expert, device="cuda", dtype=torch.float32).to(
        gating_dtype
    )
    bias = (
        torch.randn(expert, device="cuda", dtype=torch.float32)
        .to(bias_dtype)
        .contiguous()
    )
    weights = torch.empty(token, topk, dtype=torch.float32, device="cuda")
    ids = torch.empty(token, topk, dtype=torch.int32, device="cuda")
    return gating, bias, weights, ids


def _aiter_reference(gating, bias, topk, renormalize, scale_factor, groups=1):
    weights = torch.empty(
        gating.shape[0], topk, dtype=torch.float32, device=gating.device
    )
    ids = torch.empty(gating.shape[0], topk, dtype=torch.int32, device=gating.device)
    b = bias if bias.dtype == gating.dtype else bias.to(gating.dtype)
    rocm_aiter_ops.biased_grouped_topk(
        gating,
        b,
        weights,
        ids,
        groups,
        groups,
        renormalize,
        routed_scaling_factor=scale_factor,
    )
    return weights, ids


def _assert_matches_aiter(
    token,
    expert,
    topk,
    gating_dtype,
    bias_dtype,
    renormalize,
    scale_factor=1.0,
    seed=0,
    groups=1,
):
    gating, bias, weights, ids = _routing_inputs(
        token, expert, topk, gating_dtype, bias_dtype, seed=seed
    )
    ref_weights, ref_ids = _aiter_reference(
        gating, bias, topk, renormalize, scale_factor, groups=groups
    )
    sigmoid_bias_flat_topk(gating, bias, weights, ids, renormalize, scale_factor)
    torch.accelerator.synchronize()

    assert torch.equal(
        torch.sort(ref_ids, dim=-1).values, torch.sort(ids, dim=-1).values
    )
    torch.testing.assert_close(ref_weights, weights, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "token,expert,topk",
    [
        (1, 256, 8),
        (8, 384, 8),
        (32, 256, 8),
        (128, 384, 8),
        (6, 64, 6),  # Moonlight-16B-A3B / Kimi-VL-A3B
        (3, 257, 8),  # prime expert count, one past a power of two
        (7, 131, 5),  # prime experts, prime topk, odd token count
        (13, 509, 7),  # largest prime under the expert bound, odd topk
        (1, 2, 1),  # degenerate minimum
        (5, 65, 3),  # one past a power of two, odd topk
    ],
)
@pytest.mark.parametrize("gating_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [True, False])
def test_flat_topk_matches_aiter(token, expert, topk, gating_dtype, renormalize):
    """Single declared group: the plain degenerate case."""
    _assert_matches_aiter(
        token,
        expert,
        topk,
        gating_dtype,
        gating_dtype,
        renormalize,
        seed=token + expert,
    )


@pytest.mark.parametrize("groups", [1, 2, 4, 8])
@pytest.mark.parametrize("renormalize", [True, False])
def test_flat_topk_matches_aiter_with_uneliminating_groups(groups, renormalize):
    """topk_group == num_expert_group > 1 selects every group, so the masked
    top-k degenerates to a plain top-k. FusedTopKBiasRouter reaches AITER this
    way: it synthesises a divisor that fits an AITER kernel rather than using
    the model's own group count, so this is a real production shape, not a
    synthetic one."""
    _assert_matches_aiter(
        64,
        64,
        6,
        torch.bfloat16,
        torch.float32,
        renormalize,
        seed=groups * 17,
        groups=groups,
    )


@pytest.mark.parametrize("token,expert,topk", [(8, 384, 8), (128, 384, 8), (7, 131, 5)])
@pytest.mark.parametrize("renormalize", [True, False])
def test_flat_topk_casts_bias_like_aiter(token, expert, topk, renormalize):
    """A bf16 gating output next to an fp32 correction bias is the common
    router layout, and rocm_aiter_ops.biased_grouped_topk casts the bias to the
    gating dtype before dispatching."""
    _assert_matches_aiter(
        token,
        expert,
        topk,
        torch.bfloat16,
        torch.float32,
        renormalize,
        seed=token * 3,
    )


@pytest.mark.parametrize("scale_factor", [1.0, 2.446, 2.827])
def test_flat_topk_honours_routed_scaling(scale_factor):
    """Moonlight-16B-A3B uses 2.446 and Kimi-K2 2.827."""
    _assert_matches_aiter(
        8,
        64,
        6,
        torch.bfloat16,
        torch.float32,
        True,
        scale_factor=scale_factor,
        seed=11,
    )


@pytest.mark.parametrize(
    "groups,topk_group,mode_name,substituted",
    [
        (1, 1, "FULL", True),
        (2, 2, "FULL", True),
        (8, 4, "FULL", False),  # a group is eliminated -> AITER
        (1, 1, "NONE", False),  # eager -> AITER
    ],
)
def test_biased_grouped_topk_substitutes_only_when_expected(
    monkeypatch, groups, topk_group, mode_name, substituted
):
    """End of the wiring: rocm_aiter_ops.biased_grouped_topk is the single
    call site both routers reach, so this asserts the substitution happens
    there and only for the intended shapes."""
    import vllm.forward_context as fc
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.fused_moe.router import (
        rocm_sigmoid_bias_flat_topk as K,
    )

    class _Ctx:
        def __init__(self, mode):
            self.cudagraph_runtime_mode = mode

    mode = getattr(CUDAGraphMode, mode_name)
    monkeypatch.setattr(fc, "get_forward_context", lambda: _Ctx(mode))

    calls: list[int] = []
    orig = K.sigmoid_bias_flat_topk

    def counted(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(K, "sigmoid_bias_flat_topk", counted)

    gating, bias, weights, ids = _routing_inputs(
        8, 64, 6, torch.bfloat16, torch.float32
    )
    rocm_aiter_ops.biased_grouped_topk(
        gating,
        bias,
        weights,
        ids,
        groups,
        topk_group,
        True,
        routed_scaling_factor=1.0,
    )
    torch.accelerator.synchronize()
    assert bool(calls) is substituted
    assert torch.isfinite(weights).all()
