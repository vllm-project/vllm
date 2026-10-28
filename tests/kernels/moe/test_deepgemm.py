# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-test DeepGEMM FP8 and FP4 kernels (no DeepEP).
Compare DeepGEMM path against the Triton fallback inside vLLM's fused_experts.
"""

import ctypes
import importlib
import math

import pytest
import torch

# vLLM fused-expert reference (Triton fallback + DeepGEMM option)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_moe_permute,
    ep_gather,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    _fp8_workspace_shape,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_supported,
    per_block_cast_to_fp8,
)

BLOCK_SIZE = [128, 128]


@pytest.fixture
def non_batch_invariant(monkeypatch):
    """Batch-invariant MoE rejects these block-FP8 scale layouts by design."""
    from vllm import envs

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setitem(vars(envs), "VLLM_BATCH_INVARIANT", False)


@pytest.mark.parametrize("tokens", [0, 7, 259])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_cuda_stable_expert_permutation_keeps_codes_scales_and_padding(
    tokens, mapped, index_dtype
):
    """Stable token/slot grouping handles chunk tails and unowned routes."""
    from vllm.model_executor.kernels.batch_invariant import stable_expert_permute

    experts, topk, width, alignment = 4, 6, 256, 16
    generator = torch.Generator().manual_seed(913)
    codes = torch.randint(-4, 5, (tokens, width), generator=generator).to(
        torch.float8_e4m3fn
    )
    scale = torch.ldexp(
        torch.ones(tokens, 2), torch.randint(-2, 3, (tokens, 2), generator=generator)
    )
    routes = torch.randint(
        -1, 10 if mapped else 6, (tokens, topk), generator=generator
    ).to(index_dtype)
    if tokens == 7:
        routes.fill_(0)
        routes[0, -1] = -1
        routes[-1, -1] = 8 if mapped else experts
    expert_map = (
        torch.tensor([0, -1, 1, -1, 2, -1, 3, -1], dtype=torch.int32)
        if mapped
        else None
    )
    routed = torch.full(routes.shape, -1, dtype=torch.int64)
    valid = (routes >= 0) & (routes < (8 if mapped else experts))
    if expert_map is not None:
        routed[valid] = expert_map[routes[valid].long()].long()
    else:
        routed[valid] = routes[valid].long()
    bound = tokens * topk + min(tokens * topk, experts) * (alignment - 1)
    capacity = (bound + alignment - 1) // alignment * alignment
    expected = torch.zeros(capacity, width, dtype=torch.uint8)
    expected_scale = torch.ones(capacity, 2)
    expected_layout = torch.full((capacity,), -1, dtype=torch.int32)
    expected_inverse = torch.full(routes.shape, -1, dtype=torch.int32)
    start = 0
    for expert in range(experts):
        pairs = (routed == expert).nonzero()
        for rank, (token, slot) in enumerate(pairs.tolist()):
            row = start + rank
            expected[row] = codes[token].view(torch.uint8)
            expected_scale[row] = scale[token]
            expected_layout[row] = expert
            expected_inverse[token, slot] = row
        start += (len(pairs) + alignment - 1) // alignment * alignment
    a = torch.empty(tokens, width * 2, device="cuda", dtype=codes.dtype)[:, ::2]
    a.copy_(codes)
    sf = scale.cuda().T.contiguous().T
    gpu_routes = routes.cuda().T.contiguous().T
    output = torch.empty(capacity, width, device="cuda", dtype=codes.dtype)
    output.view(torch.uint8).fill_(127)
    actual, actual_scale, layout, inverse = stable_expert_permute(
        a,
        sf,
        gpu_routes,
        experts,
        capacity,
        alignment,
        expert_map=expert_map.cuda() if expert_map is not None else None,
        output=output,
    )
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual.view(torch.uint8).cpu(), expected)
    assert torch.equal(actual_scale.cpu(), expected_scale)
    assert torch.equal(layout.cpu(), expected_layout)
    assert torch.equal(inverse.cpu(), expected_inverse)


@pytest.mark.parametrize("ep", [4, 8])
@pytest.mark.parametrize("width", [7, 128, 5120])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("arm", ["fma-midpoint", "slot-order"])
@torch.inference_mode()
def test_cuda_ordered_expert_slots_preserve_fma_and_ownership(
    ep, width, index_dtype, arm
):
    """Local assembly excludes poisoned inverse rows before ordered FP32 FMA.

    This tests slot kernels, not the numerical behavior of an EP collective.
    """
    from vllm.model_executor.kernels.batch_invariant import (
        gather_expert_slots,
        ordered_topk_combine,
    )

    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    tokens, topk, experts = 3, 6, 8
    routes = torch.tensor([0, 2, 4, 6, 1, 3], dtype=index_dtype).repeat(tokens, 1)
    weights = torch.ones(tokens, topk)
    values = torch.zeros(tokens, topk, width, dtype=torch.bfloat16)
    if arm == "slot-order":
        values[:, 0], values[:, 1], values[:, 2] = 2**30, 1, -(2**30)
    else:
        values[:, 0], values[:, 1] = -(1 + 2**-7), 1 + 2**-7
        weights[:, 1] = 1 + 2**-23
    values[:, 3] = (torch.arange(width) % 7 - 3) / 16
    routes[::2, 0], routes[0, 5], routes[1, 5] = -1, -1, experts
    invalid = (routes < 0) | (routes >= experts)
    weights[invalid] = float("nan")
    values[invalid] = float("nan")
    expected = torch.empty(tokens, width)
    for token in range(tokens):
        for column in range(width):
            total = 0.0
            for slot in range(topk):
                if not invalid[token, slot]:
                    total = fma(
                        values[token, slot, column], weights[token, slot], total
                    )
            expected[token, column] = total
    expected = expected.bfloat16()
    gpu_routes, gpu_weights = routes.cuda(), weights.cuda()
    total_slots = torch.zeros(tokens, topk, width, device="cuda")
    for rank in range(ep):
        lo, hi = rank * (experts // ep), (rank + 1) * (experts // ep)
        owned = (routes >= lo) & (routes < hi)
        pairs = owned.nonzero()
        rows = values[pairs[:, 0], pairs[:, 1]].cuda()
        inverse = torch.full(routes.shape, -999999, dtype=torch.int32)
        inverse[pairs[:, 0], pairs[:, 1]] = torch.arange(len(pairs), dtype=torch.int32)
        expert_map = torch.full((experts,), -1, device="cuda", dtype=torch.int32)
        expert_map[lo:hi] = torch.arange(hi - lo, device="cuda", dtype=torch.int32)
        slots = gather_expert_slots(rows, gpu_routes, inverse.cuda(), expert_map)
        target = values.float().masked_fill(~owned.unsqueeze(-1), 0)
        assert torch.equal(slots.cpu(), target)
        total_slots += slots
    actual = ordered_topk_combine(
        total_slots,
        gpu_weights,
        torch.bfloat16,
        routes=gpu_routes,
        global_num_experts=experts,
    )
    assert torch.equal(actual.cpu(), expected)


def test_ep_gather_uses_64_bit_row_offsets():
    hidden_size = 4096
    source_row = 1 << 18
    input_tensor = torch.empty(
        (source_row + 1, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    input_tensor[source_row].fill_(1)
    output = torch.empty((1, hidden_size), device="cuda", dtype=torch.bfloat16)

    ep_gather(
        input_tensor=input_tensor,
        recv_topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        recv_topk_weight=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        input_index=torch.full((1, 1), source_row, device="cuda", dtype=torch.int32),
        expert_map=None,
        output_tensor=output,
    )

    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


def test_ep_gather_uses_64_bit_output_row_offsets():
    hidden_size = 6144
    output_row = (1 << 31) // hidden_size + 1
    num_rows = output_row + 1
    output = torch.empty(
        (num_rows, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    ep_gather(
        input_tensor=torch.ones((1, hidden_size), device="cuda", dtype=torch.bfloat16),
        recv_topk_ids=torch.zeros((num_rows, 1), device="cuda", dtype=torch.int64),
        recv_topk_weight=torch.ones((num_rows, 1), device="cuda", dtype=torch.float32),
        input_index=torch.zeros((num_rows, 1), device="cuda", dtype=torch.int32),
        expert_map=None,
        output_tensor=output,
    )

    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)


@pytest.mark.parametrize("workspace_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_columns", [2048, 6144, 6145])
def test_fp8_workspace_shape(workspace_dtype, num_columns):
    num_rows = 17
    shape = _fp8_workspace_shape(num_rows, num_columns, workspace_dtype)

    allocated_bytes = math.prod(shape) * workspace_dtype.itemsize
    required_bytes = num_rows * num_columns * torch.float8_e4m3fn.itemsize

    assert allocated_bytes >= required_bytes
    assert allocated_bytes - required_bytes < num_rows * workspace_dtype.itemsize


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_moe_permute_initializes_padding_scales(workspace_init):
    from vllm import envs

    hidden_states = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    activations, scales = per_token_group_quant_fp8(
        hidden_states,
        group_size=128,
        use_ue8m0=True,
    )
    topk_ids = torch.tensor([[0], [1]], device="cuda", dtype=torch.int64)

    _, permuted_scales, expert_ids, _, _ = deepgemm_moe_permute(
        aq=activations,
        aq_scale=scales,
        topk_ids=topk_ids,
        local_num_experts=2,
        expert_map=None,
        expert_tokens_meta=None,
    )

    padding = expert_ids < 0
    assert padding.any()
    # The batch-invariant permutation zeroes padding rows and gives them unit
    # scales.
    torch.testing.assert_close(
        permuted_scales[padding],
        torch.full_like(
            permuted_scales[padding], 1.0 if envs.VLLM_BATCH_INVARIANT else 0.0
        ),
        rtol=0,
        atol=0,
    )


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
):
    """Generate (w1, w2) expert weights and their per-block scale tensors
    in FP8 block-quantized format.

      w1 shape: (E, 2N, K)
      w2 shape: (E, K, N)
    """
    dtype = torch.bfloat16
    fp8_max, fp8_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )

    # bf16 reference weights
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) / 10
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) / 10
    w1_bf16.clamp_(fp8_min, fp8_max)
    w2_bf16.clamp_(fp8_min, fp8_max)

    block_n, block_k = block_size
    n_tiles_w1 = math.ceil((2 * n) / block_n)
    k_tiles_w1 = math.ceil(k / block_k)
    n_tiles_w2 = math.ceil(k / block_n)
    k_tiles_w2 = math.ceil(n / block_k)

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
    w1_s = torch.empty(e, n_tiles_w1, k_tiles_w1, device="cuda", dtype=torch.float32)
    w2_s = torch.empty(e, n_tiles_w2, k_tiles_w2, device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(
            w1_bf16[i], block_size=block_size, use_ue8m0=True
        )
        w2[i], w2_s[i] = per_block_cast_to_fp8(
            w2_bf16[i], block_size=block_size, use_ue8m0=True
        )

    return w1, w2, w1_s, w2_s


@pytest.mark.parametrize("num_tokens", [16, 129, 512])
@pytest.mark.parametrize("with_metadata", [False, True])
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
@pytest.mark.usefixtures("non_batch_invariant")
def test_block_fp8_padding_on_graph_replay(num_tokens, with_metadata, monkeypatch):
    """Both FP8 GEMMs consume only live rows as routing changes on replay."""
    from unittest.mock import Mock

    import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as impl
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    if not (
        impl.current_platform.is_device_capability_family(90)
        or (
            impl.current_platform.is_device_capability_family(100)
            and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        )
    ):
        pytest.skip("Prefix-sum FP8 dispatch requires SM90 or SM10.x")

    torch.manual_seed(0)
    e, n, k, topk = 8, 1024, 512, 2
    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16) / 10
    aq, scales = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    w1, w2, s1, s2 = make_block_quant_fp8_weights(e, n, k, BLOCK_SIZE)
    quant = fp8_w8a8_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=BLOCK_SIZE)
    experts = impl.DeepGemmExperts(make_dummy_moe_config(), quant)
    ids = torch.zeros(num_tokens, topk, device="cuda", dtype=torch.int64)
    weights = torch.full((num_tokens, topk), 0.5, device="cuda")
    counts = torch.zeros(e, device="cuda", dtype=torch.int32)
    metadata = mk.ExpertTokensMetadata(counts, None) if with_metadata else None
    shapes = experts.workspace_shapes(
        num_tokens, 2 * n, k, topk, e, e, metadata, MoEActivation.SILU
    )

    def route(mode):
        ids[:, 0] = torch.arange(num_tokens, device="cuda") % e
        ids[:, 1] = (ids[:, 0] + 1) % e
        if mode == 1:
            ids.fill_(-1)
            ids[:1, 0] = e - 1
        elif mode == 2:
            ids.fill_(-1)
        counts.copy_(torch.bincount(ids[ids >= 0], minlength=e).int())

    route(0)
    variants = []
    for dense in (True, False):
        ws1, ws2, output = [
            torch.empty(shape, device="cuda", dtype=torch.bfloat16) for shape in shapes
        ]

        def invoke(output=output, ws1=ws1, ws2=ws2):
            experts.apply(
                output,
                aq,
                w1,
                w2,
                weights,
                ids,
                MoEActivation.SILU,
                e,
                None,
                scales,
                None,
                ws1,
                ws2,
                metadata,
                False,
            )

        with monkeypatch.context() as mp:
            if dense:
                platform = Mock(wraps=impl.current_platform)
                platform.is_device_capability_family.return_value = False
                mp.setattr(impl, "current_platform", platform)
            gemm = impl.m_grouped_fp8_gemm_nt_contiguous
            calls: list[int] = []

            def check_layout(*args, dense=dense, calls=calls, gemm=gemm, **kwargs):
                assert kwargs.get("use_psum_layout", False) == (not dense)
                calls.append(args[3].numel())
                return gemm(*args, **kwargs)

            mp.setattr(impl, "m_grouped_fp8_gemm_nt_contiguous", check_layout)
            invoke()
            assert len(calls) == 2
            if not dense:
                assert calls == [e, e]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                invoke()
        variants.append((graph, output, ws1, ws2))

    for mode in (0, 1, 2, 0):
        route(mode)
        for graph, _, ws1, ws2 in variants:
            ws1.fill_(float("nan"))
            ws2.fill_(float("nan"))
            graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(variants[0][1], variants[1][1], rtol=0, atol=0)
        assert torch.isfinite(variants[1][1]).all()


@pytest.mark.parametrize(("n", "k"), [(2048, 512), (512, 1024)])
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_prefix_sum_leaves_unused_capacity_untouched(n, k):
    """Both projection shapes skip unused tiles as counts change on replay."""
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import (
        m_grouped_fp8_gemm_nt_contiguous,
        mk_alignment_scope,
    )

    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        pytest.skip("Requires SM90 or SM10.x")

    capacity, experts, alignment = 512, 4, 128
    x = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16) / 10
    aq, scales = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    w, _, ws, _ = make_block_quant_fp8_weights(experts, n // 2, k, BLOCK_SIZE)
    ends = torch.tensor([0, 1, 128, 129], device="cuda", dtype=torch.int32)
    output = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)

    def invoke():
        with mk_alignment_scope(alignment):
            m_grouped_fp8_gemm_nt_contiguous(
                (aq, scales), (w, ws), output, ends, use_psum_layout=True
            )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for bounds, allocated_end in (([0, 1, 128, 129], 256), ([0] * 4, 0)):
        ends.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int32))
        output.fill_(91)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.all(output[allocated_end:] == 91)
        if allocated_end:
            assert torch.all(output[[0, 128]] != 91)


def run_single_case(m, n, k, topk, num_experts, block_size):
    """Run one (M,N,K) configuration on a single GPU and assert DeepGEMM ==
    Triton baseline within tolerance.
    """
    tokens_bf16 = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])

    # expert weight tensors
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(num_experts, n, k, block_size)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        block_shape=block_size,
    )
    moe_config = make_dummy_moe_config()

    deep_gemm_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=TritonOrDeepGemmExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # triton reference
    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
    )

    # DeepGemm
    out_deepgemm = deep_gemm_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 0.001, f"Diff exceeded 1%: {diff}"


# Note: N <= 512 will disable the deepgemm path due to performance issues.
MNKs = [
    (1024, 768, 128),
    (2048, 768, 512),
    (512, 1024, 1024),
    (4096, 4096, 1024),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize(("m", "n", "k"), MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
@pytest.mark.usefixtures("non_batch_invariant")
def test_deepgemm_vs_triton(m, n, k, topk, num_experts, monkeypatch, workspace_init):
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmExperts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmExperts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmExperts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmExperts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
            block_size=BLOCK_SIZE,
        )

        # ensure that the DeepGEMM path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )


# ---------------------------------------------------------------------------
# FP4 weight tests (DeepGEMM m_grouped_fp8_fp4_gemm_nt_contiguous)
# ---------------------------------------------------------------------------


def make_mxfp4_weights(
    e: int,
    n: int,
    k: int,
):
    """Generate (w1, w2) expert weights in MXFP4 packed format with float32 scales,
    plus BF16 reference weights for validation.

      w1 shape: (E, 2N, K//2) uint8    — packed FP4
      w2 shape: (E, K, N//2)  uint8    — packed FP4
      w1_s shape: (E, 2N, K//32) float32  — per-row block-32 scales
      w2_s shape: (E, K, N//32)  float32  — per-row block-32 scales
      w1_bf16: (E, 2N, K)   — original BF16 for reference
      w2_bf16: (E, K, N)    — original BF16 for reference
    """
    from vllm.utils.deep_gemm import _import_deep_gemm

    module = _import_deep_gemm()
    assert module is not None, "DeepGEMM is required for FP4 test operands"
    per_token_cast_to_fp4 = importlib.import_module(
        f"{module.__name__}.utils.math"
    ).per_token_cast_to_fp4

    dtype = torch.bfloat16
    gran_k = 32  # MXFP4 block size

    # bf16 reference weights — scale by 1/sqrt(dim) for numerical stability
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) * (k**-0.5)
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) * (n**-0.5)

    # Quantize per-expert to FP4
    w1 = torch.empty(e, 2 * n, k // 2, device="cuda", dtype=torch.uint8)
    w2 = torch.empty(e, k, n // 2, device="cuda", dtype=torch.uint8)
    w1_s = torch.empty(
        e, 2 * n, math.ceil(k / gran_k), device="cuda", dtype=torch.float32
    )
    w2_s = torch.empty(e, k, math.ceil(n / gran_k), device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_token_cast_to_fp4(
            w1_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )
        w2[i], w2_s[i] = per_token_cast_to_fp4(
            w2_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )

    return w1, w2, w1_s, w2_s, w1_bf16, w2_bf16


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        (dict(dp_size=4, ep_size=4, use_ep=True), True),
        (dict(dp_size=2, ep_size=4, use_ep=True), False),
        (
            dict(
                dp_size=4, ep_size=4, use_ep=True, all2all_backend="deepep_low_latency"
            ),
            False,
        ),
        (dict(sp_size=2), False),
        (dict(enable_eplb=True), False),
    ],
)
def test_mxfp4_bi_backend_selection_limits(monkeypatch, overrides, expected):
    """Production selection rejects communication paths without BI evidence."""
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm import envs
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kMxfp4Static,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(
        deep_gemm_moe,
        "current_platform",
        SimpleNamespace(is_device_capability_family=lambda family: family == 100),
    )
    cls = deep_gemm_moe.DeepGemmFP4Experts
    monkeypatch.setattr(cls, "_supports_current_device", staticmethod(lambda: True))
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        all2all_backend="allgather_reducescatter",
    )
    parallel = replace(parallel, **overrides)
    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=6,
        hidden_dim=5120,
        intermediate_size=2304,
        num_local_experts=8 // parallel.ep_size,
        num_logical_experts=8,
        moe_parallel_config=parallel,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )
    assert envs.VLLM_BATCH_INVARIANT
    supported, reason = mk.FusedMoEExperts.is_supported_config(
        cls,
        config,
        kMxfp4Static,
        kFp8Dynamic128Sym,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported == expected, reason
    if not expected:
        assert reason is not None and "parallel config" in reason


@torch.inference_mode()
def _mxfp4_bi_ep_worker(group, vllm_config, cpu_group, clamp):
    from dataclasses import replace

    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.determinism.batch_invariant import (
        enable_batch_invariant_mode,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(group.device)
    enable_batch_invariant_mode()
    torch.manual_seed(1907)
    experts, intermediate, hidden, topk = 8, 2304, 5120, 6
    w1, w2, s1, s2, _, _ = make_mxfp4_weights(experts, intermediate, hidden)
    scales = _pack_deepgemm_mxfp4_scales(w1, w2, s1, s2)
    per_rank = experts // group.world_size
    first = group.rank * per_rank
    owned = slice(first, first + per_rank)
    expert_map = torch.full((experts,), -1, device=group.device, dtype=torch.int32)
    expert_map[owned] = torch.arange(per_rank, device=group.device, dtype=torch.int32)
    base = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
    )
    parallel = FusedMoEParallelConfig.make(
        1, 1, group.world_size, 1, vllm_config.parallel_config
    )
    sharded = replace(base, num_local_experts=per_rank, moe_parallel_config=parallel)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kMxfp4Static,
    )

    supported, reason = mk.FusedMoEExperts.is_supported_config(
        DeepGemmFP4Experts,
        sharded,
        kMxfp4Static,
        kFp8Dynamic128Sym,
        mk.FusedMoEActivationFormat.Standard,
    )
    assert supported, reason

    def kernel(config, sf1, sf2):
        activation = FusedMoEQuantDesc(torch.float8_e4m3fn, GroupShape(128, 128))
        quant = FusedMoEQuantConfig(
            _a1=activation,
            _a2=activation,
            _w1=FusedMoEQuantDesc("mxfp4", scale=sf1),
            _w2=FusedMoEQuantDesc("mxfp4", scale=sf2),
            gemm1_clamp_limit=clamp,
        )
        return mk.FusedMoEKernel(
            prepare_finalize=maybe_make_prepare_finalize(
                config, quant, allow_new_interface=True
            ),
            fused_experts=DeepGemmFP4Experts(config, quant),
        )

    local = kernel(base, *scales)
    distributed = kernel(sharded, *(s[owned] for s in scales))
    # Empty input ranks and an entirely idle expert rank exercise collective
    # participation independently of expert ownership and token counts.
    for sizes, active in (([17, 5, 0, 3], experts), ([1, 0, 9, 4], 6)):
        total = sum(sizes)
        values = torch.randn(total, hidden, device=group.device, dtype=torch.bfloat16)
        values[::3] *= 32
        ids = torch.stack(
            [torch.randperm(active, device=group.device)[:topk] for _ in range(total)]
        ).int()
        scores = torch.rand(total, topk, device=group.device)
        begin = sum(sizes[: group.rank])
        rows = slice(begin, begin + sizes[group.rank])
        kwargs = dict(
            activation=MoEActivation.SILU,
            global_num_experts=experts,
            apply_router_weight_on_input=False,
        )
        expected = local.apply(
            hidden_states=values,
            w1=w1,
            w2=w2,
            topk_weights=scores,
            topk_ids=ids,
            expert_map=None,
            **kwargs,
        )[rows].clone()
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=sizes[group.rank],
            num_tokens_across_dp=torch.tensor(sizes, device="cpu", dtype=torch.int32),
        ):
            metadata = get_forward_context().dp_metadata
            assert metadata is not None
            with metadata.sp_local_sizes(sequence_parallel_size=1):
                actual = distributed.apply(
                    hidden_states=values[rows].contiguous(),
                    w1=w1[owned].contiguous(),
                    w2=w2[owned].contiguous(),
                    topk_weights=scores[rows].contiguous(),
                    topk_ids=ids[rows].contiguous(),
                    expert_map=expert_map,
                    **kwargs,
                )
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected), (group.rank, sizes, clamp)


@torch.inference_mode()
def test_mxfp4_bi_rows_independent_of_batch_and_alignment(monkeypatch, workspace_init):
    """BI pads experts to DeepGEMM's per-call BLOCK_M, which varies with M; a
    row's MoE output must not change with M, the alignment, or its batch-mates."""
    from vllm.model_executor.layers.fused_moe import deep_gemm_utils
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(2311)
    experts, intermediate, hidden, topk, total = 16, 2304, 5120, 6, 2048
    w1, w2, s1, s2, _, _ = make_mxfp4_weights(experts, intermediate, hidden)
    sf1, sf2 = _pack_deepgemm_mxfp4_scales(w1, w2, s1, s2)
    activation = FusedMoEQuantDesc(torch.float8_e4m3fn, GroupShape(128, 128))
    quant = FusedMoEQuantConfig(
        _a1=activation,
        _a2=activation,
        _w1=FusedMoEQuantDesc("mxfp4", scale=sf1),
        _w2=FusedMoEQuantDesc("mxfp4", scale=sf2),
    )
    config = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        max_num_tokens=total,
    )
    kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            config, quant, allow_new_interface=True
        ),
        fused_experts=DeepGemmFP4Experts(config, quant),
    )
    values = torch.randn(total, hidden, device="cuda", dtype=torch.bfloat16)
    values[::3] *= 32
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:topk] for _ in range(total)]
    ).int()
    ids[1:64] = ids[0]
    scores = torch.rand(total, topk, device="cuda")
    default = get_mk_alignment_for_contiguous_layout()
    block_m = {"value": None}

    def capped():
        return [block_m["value"] or default[0], default[1]]

    monkeypatch.setattr(deep_gemm_moe, "get_mk_alignment_for_contiguous_layout", capped)
    monkeypatch.setattr(
        deep_gemm_utils, "get_mk_alignment_for_contiguous_layout", capped
    )

    def run(rows):
        out = kernel.apply(
            hidden_states=values[rows].contiguous(),
            w1=w1,
            w2=w2,
            topk_weights=scores[rows].contiguous(),
            topk_ids=ids[rows].contiguous(),
            global_num_experts=experts,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )
        assert isinstance(out, torch.Tensor)
        return out.view(torch.int16)

    alone = torch.cat([run(slice(i, i + 1)) for i in range(4)])
    for value in (None, 16, 64, 128):
        block_m["value"] = value
        for m in (1, 4, 6, 37, 384, total):
            rows = torch.cat(
                [torch.arange(4), torch.randperm(total - 4)[: max(m - 4, 0)] + 4]
            )[:m].cuda()
            assert torch.equal(run(rows)[: min(m, 4)], alone[: min(m, 4)]), (value, m)


def test_bi_ordered_combine_requires_pynccl():
    """Without PyNccl the rank-order combine would sum rows it never received."""
    from types import SimpleNamespace

    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    group = SimpleNamespace(
        rank_in_group=0,
        world_size=2,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(disabled=True)),
    )
    with pytest.raises(RuntimeError, match="PyNccl"):
        AgRsAll2AllManager._ordered_combine(torch.zeros(2, 4), group, [1, 1])


@pytest.mark.parametrize("clamp", [None, 10.0])
def test_mxfp4_bi_ep_ag_rs_matches_unsharded(clamp):
    """Real EP4 dispatch/expert/combine preserves the single-rank BF16 result."""
    from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
        parallel_launch_with_config,
    )
    from vllm.config import ParallelConfig, VllmConfig

    assert torch.accelerator.device_count() >= 4, "EP evidence requires four GPUs"
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=4,
            data_parallel_size_local=4,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        )
    )
    parallel_launch_with_config(
        4, _mxfp4_bi_ep_worker, config, {"VLLM_BATCH_INVARIANT": "1"}, clamp
    )


def _bf16_moe_reference(x, w1, w2, topk_weights, topk_ids):
    """BF16 token-loop MoE reference for correctness testing."""
    import torch.nn.functional as F

    num_tokens, hidden_size = x.shape
    intermediate = w1.shape[1] // 2
    top_k = topk_ids.shape[1]

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=x.device)
    for t in range(num_tokens):
        for kk in range(top_k):
            e = topk_ids[t, kk].item()
            w = topk_weights[t, kk].item()
            fc1 = x[t : t + 1].float() @ w1[e].float().T
            linear = fc1[:, :intermediate]
            gate = fc1[:, intermediate:]
            act = F.silu(gate) * linear
            fc2 = act @ w2[e].float().T
            output[t] += w * fc2[0]
    return output.to(torch.bfloat16)


def run_single_fp4_case(m, n, k, topk, num_experts):
    """Run one (M,N,K) configuration with FP4 weights on DeepGEMM and assert
    DeepGEMM FP4 == BF16 reference within tolerance.
    """
    tokens_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)

    # FP4 expert weight tensors + BF16 originals for reference
    w1, w2, w1_s, w2_s, w1_bf16, w2_bf16 = make_mxfp4_weights(num_experts, n, k)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
    )
    from vllm.platforms import current_platform

    _fp8_dtype = current_platform.fp8_dtype()
    _block_shape = GroupShape(128, 128)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _a2=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )
    moe_config = make_dummy_moe_config()

    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )

    deep_gemm_fp4_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=DeepGemmFP4Experts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # DeepGEMM FP4 path
    out_deepgemm_fp4 = deep_gemm_fp4_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    # BF16 reference using the same original weights
    out_ref = _bf16_moe_reference(tokens_bf16, w1_bf16, w2_bf16, topk_weights, topk_ids)

    # FP4 vs BF16 reference: quantization error from FP4 weights + FP8 activations
    diff = calc_diff(out_deepgemm_fp4, out_ref)
    assert diff < 0.05, f"FP4 diff exceeded 5%: {diff}"


# DeepSeek V4 dims: H=4096, I=2048, so N=2*I=4096, K=H=4096.
# FP4 quantization with block_k=32 needs large K for good accuracy.
FP4_MNKs = [
    (128, 4096, 4096),  # DeepSeek V4 shape
    (256, 2048, 2048),  # Half-size variant
    (128, 384, 3584),  # Kimi-K3 TP8 latent-MoE shape
]

FP4_TOPKS = [2]
FP4_NUM_EXPERTS = [8]


@pytest.mark.parametrize(("m", "n", "k"), FP4_MNKs)
@pytest.mark.parametrize("topk", FP4_TOPKS)
@pytest.mark.parametrize("num_experts", FP4_NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_fp4_vs_triton(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    pytest.importorskip("deep_gemm.utils.math")
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmFP4Experts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmFP4Experts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmFP4Experts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmFP4Experts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_fp4_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
        )

        # ensure that the DeepGEMM FP4 path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM FP4 path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )
