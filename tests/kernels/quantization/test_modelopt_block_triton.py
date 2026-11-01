# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton ModelOpt IQ2_XXS / Q8_0 kernels against a pure-torch GGML reference."""

import hashlib

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.experts import (
    modelopt_block_triton_moe as moe_mod,
)
from vllm.model_executor.layers.fused_moe.experts.modelopt_block_triton_moe import (
    ModelOptBlockTritonExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
    _IQ2XXS_GRID,
    _linear_config,
    dequant_min_m,
    dequantize_blocks,
    iq2_lut,
    launch_linear,
    repack_blocks,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="needs NVIDIA SM80+",
)

BLOCK = {"iq2_xxs": (256, 66), "q8_0": (32, 34)}
# One K per codec: 4096 = 16 IQ2_XXS blocks; 1056 = 33 Q8_0 blocks (odd K / 32).
K_FOR = {"iq2_xxs": 4096, "q8_0": 1056}


@pytest.fixture(autouse=True)
def _seed():
    set_random_seed(0)


def ref_decode(raw: torch.Tensor, codec: str) -> torch.Tensor:
    """GGML reference: [..., NB, block_bytes] uint8 -> [..., NB * block_size] fp32."""
    lead = raw.shape[:-2]
    blocks = raw.reshape(-1, raw.shape[-1])
    d = blocks[:, :2].contiguous().view(torch.float16).float()
    if codec == "q8_0":
        q = blocks[:, 2:].contiguous().view(torch.int8).float()
        return (d * q).reshape(*lead, -1)
    dev = raw.device
    grid = torch.tensor(_IQ2XXS_GRID, dtype=torch.int64, device=dev)
    grid = ((grid[:, None] >> (8 * torch.arange(8, device=dev))) & 0xFF).float()
    words = blocks[:, 2:].contiguous().view(torch.int32).reshape(-1, 8, 2)
    idx, aux = words[..., 0], words[..., 1]
    scale = d * (0.5 + ((aux >> 28) & 0xF).float()) * 0.25
    g = torch.arange(4, device=dev)
    gidx = (idx[..., None] >> (8 * g)) & 0xFF
    sidx = (aux[..., None] >> (7 * g)) & 0x7F
    parity = sidx ^ (sidx >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    signs = sidx | ((parity & 1) << 7)
    neg = (signs[..., None] >> torch.arange(8, device=dev)) & 1
    vals = grid[gidx.long()] * (1 - 2 * neg).float()
    return (vals * scale[..., None, None]).reshape(*lead, -1)


def make_raw(lead, k, codec, seed, extreme=False):
    """Random ggml blocks with finite scales; `extreme` adds 0, -d, 3e-6, 60000."""
    size, nbytes = BLOCK[codec]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (*lead, k // size)
    raw = torch.randint(
        0, 256, (*shape, nbytes), dtype=torch.uint8, device="cuda", generator=gen
    )
    d = (torch.randn(shape, device="cuda", generator=gen) * 2e-3).half()
    if extreme:
        d.view(-1)[0::7] = 0.0
        d.view(-1)[1::11] = -d.view(-1)[1::11].abs()
        d.view(-1)[2::13] = 3e-6
        d.view(-1)[3::17] = 60000.0
    raw[..., :2] = d.unsqueeze(-1).view(torch.uint8)
    return raw


def linear(x, raw, codec):
    qweight, scales = repack_blocks(raw, codec)
    lut = iq2_lut(raw.device) if codec == "iq2_xxs" else None
    return torch.ops.vllm.modelopt_block_linear(x, qweight, scales, lut, codec)


def assert_close_to_cublas(out, x, w):
    """Our error vs the exact product must be within 1.5x of cuBLAS's in x.dtype."""
    want = x.double() @ w.double().T
    cublas = (x @ w.to(x.dtype).T).double()
    ours = (out.double() - want).abs().max()
    assert ours <= 1.5 * (cublas - want).abs().max() + 1e-6, ours


def test_iq2_grid_matches_ggml():
    grid = torch.tensor(_IQ2XXS_GRID, dtype=torch.int64)
    assert (
        hashlib.sha256(grid.numpy().tobytes()).hexdigest()
        == "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"
    )
    quants = pytest.importorskip("gguf.quants")
    quants.IQ2_XXS.init_grid()
    expected = torch.from_numpy(quants.IQ2_XXS.grid.reshape(256, 8).astype("int64"))
    assert torch.equal((grid[:, None] >> (8 * torch.arange(8))) & 0xFF, expected)


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
def test_decode_bit_exact(codec):
    """Both decodes, table gather (dequant) and 2-bit codes (GEMM), are exact."""
    K = K_FOR[codec]
    raw = make_raw((264,), K, codec, seed=1, extreme=True)
    qweight, scales = repack_blocks(raw, codec)
    lut = iq2_lut(raw.device) if codec == "iq2_xxs" else None
    want = ref_decode(raw, codec).bfloat16()
    assert torch.equal(dequantize_blocks(qweight, scales, lut, codec), want)
    eye = torch.eye(K, device="cuda", dtype=torch.bfloat16)
    gemm = {"BM": 64, "BN": 64, "BK": 32, "SPLIT": 1, "num_warps": 4, "num_stages": 2}
    assert torch.equal(launch_linear(eye, qweight, scales, lut, codec, gemm).T, want)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize(
    "M", [1, 2, 3, 4, 8, 15, 16, 17, 33, 64, 65, 128, 255, 256, 1000, 2048]
)
def test_linear_matches_reference(codec, M, dtype):
    raw = make_raw((1032,), K_FOR[codec], codec, seed=M)
    x = torch.randn(M, K_FOR[codec], device="cuda", dtype=dtype)
    assert_close_to_cublas(linear(x, raw, codec), x, ref_decode(raw, codec))


def test_linear_odd_k_steps():
    """An odd K step count splits too: shared_down has 5376 = 21 * 256 steps."""
    assert _linear_config(1, 4096, 5376, "iq2_xxs", 132, 227 * 1024, 9)["SPLIT"] == 7
    raw = make_raw((136,), 5376, "iq2_xxs", seed=3)
    x = torch.randn(1, 5376, device="cuda", dtype=torch.bfloat16)
    assert_close_to_cublas(linear(x, raw, "iq2_xxs"), x, ref_decode(raw, "iq2_xxs"))


@pytest.mark.parametrize("N,K", [(256, 4096), (1024, 4096), (4096, 4096), (4096, 1024)])
@pytest.mark.parametrize("extreme", [False, True])
def test_q8_decode_projection_single_reduction(N, K, extreme):
    """Single-row Q8 projections keep cuBLAS-level error through graph replay."""
    raw = make_raw((N,), K, "q8_0", seed=31, extreme=extreme)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    want = ref_decode(raw, "q8_0")
    linear(x, raw, "q8_0")
    graph = torch.cuda.CUDAGraph()
    qw, scales = repack_blocks(raw, "q8_0")
    with torch.cuda.graph(graph):
        out = torch.ops.vllm.modelopt_block_linear(x, qw, scales, None, "q8_0")
    graph.replay()
    assert_close_to_cublas(out, x, want)


@pytest.mark.parametrize("M", [1, 4, 7, 8, 16, 65, 128])
def test_iq2_dense_cache_preserves_dispatch_and_outputs(M):
    """Reuse exact BF16 weights only where the uncached op dequantizes them."""
    raw = make_raw((136,), 5376, "iq2_xxs", seed=29)
    qw, scales = repack_blocks(raw, "iq2_xxs")
    lut = iq2_lut(raw.device)
    cached = dequantize_blocks(qw, scales, lut, "iq2_xxs")
    min_m = dequant_min_m("iq2_xxs", torch.accelerator.current_device_index())
    if min_m > M:
        cached.fill_(float("nan"))
    x = torch.randn(M, 5376, device="cuda", dtype=torch.bfloat16)
    want = torch.ops.vllm.modelopt_block_linear(x, qw, scales, lut, "iq2_xxs")
    got = torch.ops.vllm.modelopt_block_linear(x, qw, scales, lut, "iq2_xxs", cached)
    assert torch.equal(got, want)


@pytest.mark.parametrize("M", [1, 17])
def test_linear_offset_view(M):
    """Unaligned activation pointers take the same kernels and stay correct."""
    K = K_FOR["iq2_xxs"]
    raw = make_raw((520,), K, "iq2_xxs", seed=3)
    big = torch.randn(M * K + 8, device="cuda", dtype=torch.bfloat16)
    x = big[3 : 3 + M * K].view(M, K)
    assert_close_to_cublas(linear(x, raw, "iq2_xxs"), x, ref_decode(raw, "iq2_xxs"))


def test_linear_int64_offsets():
    """M * N > 2**31 (lm_head with large prefills) must not overflow offsets."""
    if torch.accelerator.get_memory_info()[0] < 12 << 30:
        pytest.skip("needs 12 GiB of free GPU memory")
    M, N, K = 16400, 131072, 4096
    raw = make_raw((N,), K, "q8_0", seed=4)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    tail = linear(x, raw, "q8_0")[-8:].float()
    w = ref_decode(raw, "q8_0")
    want = x[-8:].float() @ w.T
    cublas = (x[-8:] @ w.bfloat16().T).float()
    assert (tail - want).abs().max() <= 1.5 * (cublas - want).abs().max() + 1e-4


def _kernel_names(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.accelerator.synchronize()
    return {e.key for e in prof.key_averages() if "_kernel" in e.key}


def test_linear_compile_does_not_freeze_dispatch():
    """Traced once at large M with guards dropped, as vLLM does, M=1 still
    takes the M=1 kernel (review finding B2)."""
    raw = make_raw((520,), 1024, "iq2_xxs", seed=5)
    qweight, scales = repack_blocks(raw, "iq2_xxs")
    lut = iq2_lut(raw.device)

    def f(x):
        return torch.ops.vllm.modelopt_block_linear(x, qweight, scales, lut, "iq2_xxs")

    compiled = torch.compile(
        f,
        dynamic=True,
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )
    compiled(torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16))
    x = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
    assert _kernel_names(lambda: compiled(x)) == _kernel_names(lambda: f(x))
    assert_close_to_cublas(compiled(x), x, ref_decode(raw, "iq2_xxs"))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("M", [1, 4, 17])
def test_linear_cuda_graph_replay(M, cached):
    raw = make_raw((520,), 1024, "iq2_xxs", seed=6)
    qweight, scales = repack_blocks(raw, "iq2_xxs")
    lut = iq2_lut(raw.device)
    weight = dequantize_blocks(qweight, scales, lut, "iq2_xxs") if cached else None
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    torch.ops.vllm.modelopt_block_linear(x, qweight, scales, lut, "iq2_xxs", weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.vllm.modelopt_block_linear(
            x, qweight, scales, lut, "iq2_xxs", weight
        )
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        allocated = torch.accelerator.memory_allocated()
        graph.replay()
        assert torch.accelerator.memory_allocated() == allocated
        assert_close_to_cublas(out, x, ref_decode(raw, "iq2_xxs"))


def ref_moe(x, w1, w2, topk_weights, topk_ids, activation, dtype):
    """Unfused MoE with every intermediate rounded to `dtype`."""
    x, out = x.to(dtype), torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for e in range(w1.shape[0]):
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        h = x[token] @ w1[e].to(dtype).T
        if activation == MoEActivation.SILU:
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = F.relu(h).square()
        y = (h @ w2[e].to(dtype).T) * topk_weights[token, slot, None].to(dtype)
        out.index_add_(0, token, y.double())
    return out


def assert_moe_close_to_unfused(out, x, w1, w2, topk_weights, topk_ids, activation):
    """Our error vs the exact MoE must be within 2x of an unfused MoE's in x.dtype.

    Over random inputs the ratio has a median of ~0.9 and a tail to ~1.6; a
    routing or weighting bug is orders of magnitude off.

    Plus one ulp of the output scale: the fused contract keeps c1/c2 in the
    model dtype, so anything under a ulp of the largest output cannot show up in
    the result. Real Bolt expert weights need this slack at decode sizes (2-4x
    the bare 2x bar at M=4) while matching a torch simulation of the same bf16
    buffer chain exactly, so the slack is the buffers, not the kernels.
    """
    want = ref_moe(x, w1, w2, topk_weights, topk_ids, activation, torch.float64)
    unfused = ref_moe(x, w1, w2, topk_weights, topk_ids, activation, x.dtype)
    ours = (out.double() - want).abs().max()
    ulp = want.abs().max().double() * torch.finfo(x.dtype).eps / 2
    assert ours <= 2 * (unfused - want).abs().max() + ulp + 1e-6, ours


def make_moe(codec, activation, E=16, K=1024, inter=512, top_k=2):
    gated = activation == MoEActivation.SILU
    raw1 = make_raw((E, (2 if gated else 1) * inter), K, codec, seed=7)
    raw2 = make_raw((E, K), inter, codec, seed=8)
    w1, s1 = repack_blocks(raw1, codec)
    w2, s2 = repack_blocks(raw2, codec)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(dtype=codec, scale=s1),
        _w2=FusedMoEQuantDesc(dtype=codec, scale=s2),
    )
    moe_config = make_dummy_moe_config(
        num_experts=E,
        experts_per_token=top_k,
        hidden_dim=K,
        intermediate_size=inter,
        activation=activation,
    )
    lut = iq2_lut(w1.device) if codec == "iq2_xxs" else None
    kernel = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        ModelOptBlockTritonExperts(moe_config, quant_config, codec, lut),
    )

    def run(x, topk_weights, topk_ids):
        return kernel.apply(
            x, w1, w2, topk_weights, topk_ids, activation=activation,
            global_num_experts=E, expert_map=None, apply_router_weight_on_input=False,
        )  # fmt: skip

    return run, ref_decode(raw1, codec), ref_decode(raw2, codec)


def make_routing(M, E, top_k):
    topk_ids = torch.rand(M, E, device="cuda").topk(top_k, dim=-1).indices
    topk_weights = torch.rand(M, top_k, device="cuda").softmax(dim=-1)
    return topk_weights, topk_ids.to(torch.int32)


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("top_k", [1, 9])
@pytest.mark.parametrize("fused_relu2", [False, True])
def test_moe_gemv_direct_routing_matches_aligned(codec, M, top_k, fused_relu2):
    """Skipping token sorting preserves every routed slot, including repeats."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
        invoke_block_moe_gemv_kernel,
        moe_gemv_config,
    )

    E, N, K = 16, 520, 1024
    raw = make_raw((E, N), K, codec, seed=23)
    qw, scales = repack_blocks(raw, codec)
    lut = iq2_lut(raw.device) if codec == "iq2_xxs" else None
    # Repeated experts and descending IDs make identity/sorted-index confusion
    # observable. top_k=1 exercises GEMM2's already expanded input rows.
    ids = (torch.arange(M * 9, device="cuda") % 5).flip(0).int().view(M, 9)
    weights = torch.rand((M, 9), device="cuda")
    x = torch.randn(M * 9 // top_k, K, device="cuda", dtype=torch.bfloat16)
    want = torch.empty((M, 9, N), device="cuda", dtype=torch.bfloat16)
    got = torch.empty_like(want)
    cfg = moe_gemv_config(M, E, 9, codec)
    routing = moe_align_block_size(ids, 1, E, None)
    for out, route, relu2 in (
        (want, routing, False),
        (got, (None, ids, None), fused_relu2),
    ):
        sorted_ids, expert_ids, post_padded = route
        invoke_block_moe_gemv_kernel(
            x,
            qw,
            scales,
            lut,
            out,
            weights,
            sorted_ids,
            expert_ids,
            post_padded,
            True,
            top_k,
            cfg,
            codec,
            apply_relu2=relu2,
        )
    if fused_relu2:
        want = torch.square(torch.relu(want))
    assert torch.equal(got, want)


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("activation", [MoEActivation.RELU2_NO_MUL, MoEActivation.SILU])
@pytest.mark.parametrize("top_k", [2, 9])
@pytest.mark.parametrize("M", [1, 7, 64, 333])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_matches_reference(codec, activation, top_k, M, dtype, workspace_init):
    run, w1, w2 = make_moe(codec, activation, top_k=top_k)
    x = torch.randn(M, 1024, device="cuda", dtype=dtype)
    topk_weights, topk_ids = make_routing(M, 16, top_k)
    got = run(x, topk_weights, topk_ids)
    assert_moe_close_to_unfused(got, x, w1, w2, topk_weights, topk_ids, activation)


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("M", [1, 8])
def test_moe_paths_agree(codec, M, workspace_init, monkeypatch):
    """Both MoE paths on the same routing, not only both against the reference.

    The runtime gate picks one per call, so the two must agree to the rounding
    of the bf16 output: a path-only bug can still hide under the loose
    reference bar in `test_moe_matches_reference`.
    """
    run, w1, w2 = make_moe(codec, MoEActivation.RELU2_NO_MUL, top_k=9)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    topk_weights, topk_ids = make_routing(M, 16, 9)
    assert moe_mod.moe_gemv_config(M, 16, 9, codec) is not None  # gate is on
    gemv = run(x, topk_weights, topk_ids)
    monkeypatch.setattr(moe_mod, "moe_gemv_config", lambda *a: None)
    dot = run(x, topk_weights, topk_ids)
    # Same products, different fp32 tile order, and the output is a bf16 sum of
    # cancelling terms, so compare in the max-abs norm the reference bar uses:
    # the gap must stay under what an unfused bf16 MoE already shows (measured
    # at 40-50% of it, i.e. one bf16 ulp of the output scale). Relative error
    # is useless here - near-zero outputs reach 6x.
    args = (x, w1, w2, topk_weights, topk_ids, MoEActivation.RELU2_NO_MUL)
    want = ref_moe(*args, torch.float64)
    bar = 2 * (ref_moe(*args, torch.bfloat16) - want).abs().max()
    gap = (gemv.double() - dot.double()).abs().max()
    assert gap <= bar + 1e-6, (gap.item(), bar.item())


@pytest.mark.parametrize("M", [1, 8, 65])  # 1, 8 -> GEMV path; 65 -> the tl.dot path
def test_moe_cuda_graph_new_routing(M, workspace_init):
    """Routing changes every step when serving; replays must follow it."""
    run, w1, w2 = make_moe("iq2_xxs", MoEActivation.RELU2_NO_MUL, top_k=9)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    topk_weights, topk_ids = make_routing(M, 16, 9)
    run(x, topk_weights, topk_ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(x, topk_weights, topk_ids)
    for _ in range(2):
        new_weights, new_ids = make_routing(M, 16, 9)
        x.copy_(torch.randn_like(x))
        topk_weights.copy_(new_weights)
        topk_ids.copy_(new_ids)
        allocated = torch.accelerator.memory_allocated()
        graph.replay()
        assert torch.accelerator.memory_allocated() == allocated
        assert_moe_close_to_unfused(
            out, x, w1, w2, topk_weights, topk_ids, MoEActivation.RELU2_NO_MUL
        )


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_q8_embedding_exact(id_dtype, dtype):
    raw = make_raw((300,), 2048, "q8_0", seed=9, extreme=True)
    qweight, scales = repack_blocks(raw, "q8_0")
    ids = torch.tensor([5, 5, 299, 0, 17, 3, 5], device="cuda", dtype=id_dtype)
    out = torch.ops.vllm.modelopt_block_embedding(ids, qweight, scales, dtype)
    assert torch.equal(out, ref_decode(raw[ids.long()], "q8_0").to(dtype))


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("M", [16, 32])
@pytest.mark.parametrize("activation", [MoEActivation.RELU2_NO_MUL, MoEActivation.SILU])
def test_moe_singleton_tiles_follow_routing_under_graph(
    codec, M, activation, workspace_init, monkeypatch
):
    """A graph follows singleton and shared-expert routes without dropping rows."""
    monkeypatch.setattr(moe_mod, "moe_gemv_config", lambda *a: None)  # tl.dot path
    run, w1, w2 = make_moe(codec, activation, E=128, top_k=2)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((M, 2), device="cuda").softmax(-1)
    ids = torch.arange(M * 2, device="cuda", dtype=torch.int32).view(M, 2)
    run(x, weights, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run(x, weights, ids)
    for routing in (
        torch.arange(2, device="cuda", dtype=torch.int32).expand(M, 2),
        torch.arange(M * 2, device="cuda", dtype=torch.int32).view(M, 2).flip(0),
    ):
        ids.copy_(routing)
        graph.replay()
        assert_moe_close_to_unfused(out, x, w1, w2, weights, ids, activation)
