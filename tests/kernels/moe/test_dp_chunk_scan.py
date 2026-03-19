# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the torch.scan-based DP chunking path in DefaultMoERunner
(forward_impl_scan) introduced in vllm issue #31985.

Test organisation
-----------------
Unit tests (no multi-GPU, no DeepEP):
  - test_can_use_scan_respects_flag -> _can_use_scan() honours env var
  - test_staged_tensor_shape -> staging produces (num_chunks, max_tokens, H)
  - test_staged_tensor_content -> staging preserves data, zeros padding
  - test_output_trim -> trimming from stacked → original shape

Integration tests (require 2 GPUs + deep_ep):
  - test_scan_matches_loop_ll_mode -> scan output numerically equals Python loop
  - test_scan_matches_loop_ll_mode_chunked -> same but with
    num_tokens > max_tokens_per_rank (multiple scan iterations)
"""

import unittest.mock as mock

import pytest
import torch

from vllm.utils.math_utils import cdiv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_staged(src: torch.Tensor, num_chunks: int, max_tokens: int) -> torch.Tensor:
    """Replicate the staging logic from forward_impl_scan for testing."""
    feat_dim = src.size(-1)
    buf = src.new_zeros(num_chunks, max_tokens, feat_dim)
    total_padded = num_chunks * max_tokens
    copy_len = min(src.size(0), total_padded)
    buf.view(-1, feat_dim)[:copy_len].copy_(src[:copy_len], non_blocking=False)
    return buf


def _trim_output(stacked: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Replicate the output-trimming logic from forward_impl_scan."""
    return stacked.reshape(-1, stacked.size(-1))[:num_tokens]


# ---------------------------------------------------------------------------
# Unit tests: no GPU / multi-GPU required
# ---------------------------------------------------------------------------


def test_can_use_scan_respects_flag():
    """
    _can_use_scan() must return False when VLLM_MOE_USE_SCAN_CHUNKING is not set
    and True when it is set (given a DeepEP-LL compatible runner config).
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
        DefaultMoERunner,
    )

    # Build a minimal parallel config that looks like DeepEP-LL.
    # use_deepep_ll_kernels is a computed property: requires use_ep=True and
    # all2all_backend == "deepep_low_latency".
    pc = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=2,
        dp_rank=0,
        ep_size=2,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="deepep_low_latency",
        enable_eplb=False,
    )
    assert pc.use_deepep_ll_kernels  # sanity check
    moe_config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=64,
        num_local_experts=4,
        num_logical_experts=8,
        moe_parallel_config=pc,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
    )

    # Build a minimal mock runner without a real layer / quant method
    runner = mock.MagicMock(spec=DefaultMoERunner)
    runner.moe_config = moe_config
    runner.enable_dbo = False
    # quant_method.is_monolithic = False
    qm = mock.MagicMock()
    qm.is_monolithic = False
    runner.quant_method = qm

    # Bind the real _can_use_scan method to the mock
    runner._can_use_scan = DefaultMoERunner._can_use_scan.__get__(runner)

    import vllm.envs as envs

    with mock.patch.object(envs, "VLLM_MOE_USE_SCAN_CHUNKING", False):
        assert runner._can_use_scan() is False

    with mock.patch.object(envs, "VLLM_MOE_USE_SCAN_CHUNKING", True):
        assert runner._can_use_scan() is True

    # DBO disables scan
    runner.enable_dbo = True
    with mock.patch.object(envs, "VLLM_MOE_USE_SCAN_CHUNKING", True):
        assert runner._can_use_scan() is False
    runner.enable_dbo = False

    # Monolithic quant method disables scan
    qm.is_monolithic = True
    with mock.patch.object(envs, "VLLM_MOE_USE_SCAN_CHUNKING", True):
        assert runner._can_use_scan() is False
    qm.is_monolithic = False

    # Non-DeepEP-LL kernel (MORI) disables scan
    pc2 = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=2,
        dp_rank=0,
        ep_size=2,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="mori",
        enable_eplb=False,
    )
    assert pc2.use_mori_kernels and not pc2.use_deepep_ll_kernels  # sanity check
    runner.moe_config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=64,
        num_local_experts=4,
        num_logical_experts=8,
        moe_parallel_config=pc2,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
    )
    with mock.patch.object(envs, "VLLM_MOE_USE_SCAN_CHUNKING", True):
        assert runner._can_use_scan() is False


@pytest.mark.parametrize(
    "num_tokens,max_tokens",
    [
        (64, 64),  # exactly one chunk, no padding
        (100, 64),  # two chunks, padding in second
        (1, 64),  # single token
        (128, 64),  # exactly two full chunks
        (200, 64),  # four chunks, last partial
    ],
)
def test_staged_tensor_shape(num_tokens: int, max_tokens: int):
    H = 256
    src = torch.randn(num_tokens, H)
    num_chunks = cdiv(num_tokens, max_tokens)

    staged = _make_staged(src, num_chunks, max_tokens)

    assert staged.shape == (num_chunks, max_tokens, H), (
        f"Expected ({num_chunks}, {max_tokens}, {H}), got {staged.shape}"
    )


@pytest.mark.parametrize(
    "num_tokens,max_tokens",
    [
        (64, 64),
        (100, 64),
        (128, 64),
        (5, 64),
    ],
)
def test_staged_tensor_content(num_tokens: int, max_tokens: int):
    """Staging must preserve the original tokens and zero-fill padding."""
    H = 128
    src = torch.arange(num_tokens * H, dtype=torch.float32).view(num_tokens, H)
    num_chunks = cdiv(num_tokens, max_tokens)

    staged = _make_staged(src, num_chunks, max_tokens)
    flat = staged.view(-1, H)

    # Original tokens are preserved
    torch.testing.assert_close(flat[:num_tokens], src)

    # Padding tokens (if any) are zero
    total_padded = num_chunks * max_tokens
    if total_padded > num_tokens:
        assert flat[num_tokens:total_padded].abs().sum().item() == 0.0


@pytest.mark.parametrize(
    "num_tokens,max_tokens",
    [
        (64, 64),
        (100, 64),
        (1, 64),
        (200, 64),
    ],
)
def test_output_trim(num_tokens: int, max_tokens: int):
    """Output trim from stacked (num_chunks, max_tokens, H) → (num_tokens, H)."""
    H = 64
    num_chunks = cdiv(num_tokens, max_tokens)
    stacked = torch.randn(num_chunks, max_tokens, H)

    trimmed = _trim_output(stacked, num_tokens)

    assert trimmed.shape == (num_tokens, H)
    # First num_tokens rows must match the flattened stacked tensor
    torch.testing.assert_close(trimmed, stacked.reshape(-1, H)[:num_tokens])


# ---------------------------------------------------------------------------
# Integration tests: require 2 GPUs + DeepEP
# ---------------------------------------------------------------------------

try:
    from tests.utils import multi_gpu_test
    from vllm.utils.import_utils import has_deep_ep
except ImportError:
    has_deep_ep = lambda: False  # noqa: E731
    multi_gpu_test = lambda **_: (lambda f: pytest.mark.skip(reason="no test utils")(f))  # noqa: E731

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep(),
    reason="Requires deep_ep kernels",
)

if has_deep_ep():
    from tests.kernels.moe.parallel_utils import (
        ProcessGroupInfo,
        parallel_launch,
    )
    from tests.kernels.moe.test_deepep_moe import (
        MAX_TOKENS_PER_RANK,
        TestConfig,
        TestTensors,
        make_modular_kernel,
        make_weights,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel


def _deep_ep_moe_scan(
    pgi: "ProcessGroupInfo",
    dp_size: int,
    test_tensors: "TestTensors",
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    num_experts: int,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
) -> torch.Tensor:
    """
    Scan-based forward analogous to deep_ep_moe_impl() but uses torch.scan
    instead of a Python for-loop.  Tests that torch.scan with DeepEP-LL
    communication ops produces results numerically equal to the loop version.
    """
    from torch._higher_order_ops.scan import scan as torch_scan

    num_local_experts = w1.size(0)
    total_num_tokens = test_tensors.rank_tokens.size(0)
    H = test_tensors.rank_tokens.size(1)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    q_dtype = torch.float8_e4m3fn if is_quantized else None

    def build_expert_map() -> torch.Tensor:
        em = torch.full((num_experts,), fill_value=-1, dtype=torch.int32)
        s = pgi.rank * num_local_experts
        em[s : s + num_local_experts] = torch.arange(
            num_local_experts, dtype=torch.int32
        )
        return em.to(device=torch.accelerator.current_device_index())

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    quant_config = FusedMoEQuantConfig.make(
        q_dtype,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=per_act_token_quant,
        a1_scale=test_tensors.rank_token_scales,
    )
    # One shared kernel instance for all scan iterations
    mk: FusedMoEKernel = make_modular_kernel(
        pg,
        pgi,
        True,  # low_latency_mode
        H,
        dp_size,
        num_experts,
        num_local_experts,
        q_dtype,
        use_fp8_dispatch,
        quant_config,
    )
    expert_map = build_expert_map()

    # ------------------------------------------------------------------
    # Pre-stage: (total_num_tokens, .) -> (num_chunks, max_tokens, .)
    # ------------------------------------------------------------------
    max_tokens = MAX_TOKENS_PER_RANK
    num_chunks = cdiv(total_num_tokens, max_tokens)

    def stage(src: torch.Tensor) -> torch.Tensor:
        feat = src.size(-1)
        buf = src.new_zeros(num_chunks, max_tokens, feat)
        copy_len = min(src.size(0), num_chunks * max_tokens)
        buf.view(-1, feat)[:copy_len].copy_(src[:copy_len], non_blocking=True)
        return buf

    chunked_h = stage(test_tensors.rank_tokens)  # (C, T, H)
    chunked_topk = stage(test_tensors.topk)  # (C, T, topk)
    chunked_w = stage(test_tensors.topk_weights)  # (C, T, topk)

    dummy_carry = test_tensors.rank_tokens.new_zeros(1)

    def chunk_fn(
        carry: torch.Tensor,
        xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_h, chunk_topk, chunk_w = xs
        out = mk.apply(
            hidden_states=chunk_h,
            w1=w1,
            w2=w2,
            topk_weights=chunk_w,
            topk_ids=chunk_topk,
            activation=MoEActivation.SILU,
            global_num_experts=num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
        )
        return carry, out

    _, stacked = torch_scan(
        chunk_fn,
        init=dummy_carry,
        xs=(chunked_h, chunked_topk, chunked_w),
        dim=0,
    )

    return stacked.reshape(-1, H)[:total_num_tokens]


def _worker_scan_vs_loop(
    pgi: "ProcessGroupInfo",
    dp_size: int,
    config: "TestConfig",
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    use_fp8_dispatch: bool,
    per_act_token_quant: bool,
):
    """Per-rank worker: compare loop output vs scan output for DeepEP-LL."""
    from tests.kernels.moe.test_deepep_moe import deep_ep_moe_impl
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device(f"cuda:{pgi.local_rank}")
    init_workspace_manager(device)

    is_quantized = w1.dtype == torch.float8_e4m3fn
    w1 = w1.to(device=device)
    w2 = w2.to(device=device)
    if is_quantized:
        assert w1_scale is not None
        assert w2_scale is not None
        w1_scale = w1_scale.to(device=device)
        w2_scale = w2_scale.to(device=device)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    test_tensors = TestTensors.make(config, low_latency_mode=True)

    num_local_experts = config.num_experts // pgi.world_size
    e_start = num_local_experts * pgi.rank
    w1_ep = w1[e_start : e_start + num_local_experts]
    w2_ep = w2[e_start : e_start + num_local_experts]
    w1_scale_ep = (
        w1_scale[e_start : e_start + num_local_experts]
        if w1_scale is not None
        else None
    )
    w2_scale_ep = (
        w2_scale[e_start : e_start + num_local_experts]
        if w2_scale is not None
        else None
    )

    with set_current_vllm_config(VllmConfig()):
        # Reference: Python for-loop
        loop_out = deep_ep_moe_impl(
            pg,
            pgi,
            True,  # low_latency_mode
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
            config.num_experts,
            use_fp8_dispatch,
            per_act_token_quant,
        )

        # Scan: torch.scan
        scan_out = _deep_ep_moe_scan(
            pgi,
            dp_size,
            test_tensors,
            w1_ep,
            w2_ep,
            w1_scale_ep,
            w2_scale_ep,
            config.num_experts,
            use_fp8_dispatch,
            per_act_token_quant,
        )

    torch.testing.assert_close(
        loop_out,
        scan_out,
        atol=1e-3,
        rtol=1e-3,
        msg="Scan output does not match Python-loop output",
    )


# m values that require 2 chunks (> MAX_TOKENS_PER_RANK=64) test multi-iteration scan
_SCAN_MNKs = [
    (1, 128, 2560),  # single token, one chunk
    (64, 128, 2560),  # exactly one full chunk
    (65, 128, 2560),  # one full chunk + 1 overflow → 2 scan iterations
    (128, 128, 2560),  # exactly two full chunks
]


@pytest.mark.parametrize("m,n,k", _SCAN_MNKs)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep
def test_scan_matches_loop_ll_mode(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    workspace_init,
):
    """
    For DeepEP low-latency mode, verify that the torch.scan-based chunked
    forward produces the same output as the Python for-loop implementation.

    When m > MAX_TOKENS_PER_RANK, multiple scan iterations are exercised,
    testing that the DeepEP communication ops work correctly inside torch.scan.
    """
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(42)
    num_experts = 8
    topk = 2
    world_size = 2
    dp_size = 1

    config = TestConfig(dtype=dtype, topk=topk, m=m, k=k, n=n, num_experts=num_experts)
    w1, w2, w1_scale, w2_scale = make_weights(num_experts, n, k, dtype)

    parallel_launch(
        world_size,
        _worker_scan_vs_loop,
        dp_size,
        config,
        w1,
        w2,
        w1_scale,
        w2_scale,
        False,  # use_fp8_dispatch
        False,  # per_act_token_quant
    )


@pytest.mark.parametrize("m,n,k", [(65, 128, 2560), (128, 128, 2560)])
@multi_gpu_test(num_gpus=2)
@requires_deep_ep
def test_scan_matches_loop_ll_mode_fp8_dispatch(
    m: int,
    n: int,
    k: int,
    workspace_init,
):
    """Same as test_scan_matches_loop_ll_mode but with FP8 dispatch."""
    from vllm.utils.torch_utils import set_random_seed

    set_random_seed(42)
    num_experts = 8
    topk = 2
    world_size = 2
    dp_size = 1

    config = TestConfig(
        dtype=torch.float8_e4m3fn, topk=topk, m=m, k=k, n=n, num_experts=num_experts
    )
    w1, w2, w1_scale, w2_scale = make_weights(num_experts, n, k, torch.float8_e4m3fn)

    parallel_launch(
        world_size,
        _worker_scan_vs_loop,
        dp_size,
        config,
        w1,
        w2,
        w1_scale,
        w2_scale,
        True,  # use_fp8_dispatch
        False,  # per_act_token_quant
    )
