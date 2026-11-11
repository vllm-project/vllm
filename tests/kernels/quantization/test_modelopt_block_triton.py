# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton ModelOpt IQ2_XXS / Q8_0 kernels against a pure-torch GGML reference."""

import hashlib
import struct
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
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
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization import (
    modelopt_block_quant_triton as triton_mod,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
    _IQ2XXS_GRID,
    dequantize_blocks,
    invoke_block_moe_gemv_kernel,
    iq2_lut,
    launch_linear,
    moe_gemv_config,
    repack_blocks,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed

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
    # sha256 of gguf-py's `IQ2_XXS.grid` packed the same way (gguf 0.19.0).
    grid = torch.tensor(_IQ2XXS_GRID, dtype=torch.int64)
    assert (
        hashlib.sha256(grid.numpy().tobytes()).hexdigest()
        == "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"
    )


def iq2_block(d, groups):
    """One 66-byte IQ2_XXS block from (grid indices, sign indices, s) per 32
    weights; missing groups are all-zero bytes."""
    data = bytearray(struct.pack("<e", d))
    for grid, signs, s in groups + [((0,) * 4, (0,) * 4, 0)] * (8 - len(groups)):
        aux = s << 28 | sum(sign << (7 * i) for i, sign in enumerate(signs))
        data += bytes(grid) + struct.pack("<I", aux)
    return torch.tensor(list(data), dtype=torch.uint8, device="cuda")


# A weight is d * (0.5 + s) / 4 * grid_byte * sign. Grid entry 0 is eight 8s,
# entry 1 is 43 then seven 8s, entry 255 is bytes 08 19 08 08 19 2B 2B 2B. A
# sign index sets the negative lanes; bit 7 makes the popcount even.
HAND_BLOCK = [
    ((0, 0, 0, 0), (0, 0, 0, 0), 0),
    ((0, 0, 0, 0), (0, 0, 0, 0), 1),
    ((0, 0, 0, 0), (1, 1, 1, 1), 0),
    ((0, 0, 0, 0), (1, 1, 1, 1), 2),
    ((1, 255, 0, 0), (0, 0, 0, 0), 0),
    ((0, 0, 0, 0), (127, 127, 127, 127), 15),
    ((0, 0, 0, 0), (0, 0, 0, 0), 15),
]
HAND_VALUES = (
    [1.0] * 32
    + [3.0] * 32
    + [-1.0, 1, 1, 1, 1, 1, 1, -1] * 4
    + [-5.0, 5, 5, 5, 5, 5, 5, -5] * 4
    + [5.375, 1, 1, 1, 1, 1, 1, 1]
    + [1.0, 3.125, 1, 1, 3.125, 5.375, 5.375, 5.375]
    + [1.0] * 16
    + [-31.0] * 32
    + [31.0] * 32
    + [1.0] * 32
)


def test_iq2_hand_built_block():
    raw = torch.stack([iq2_block(1.0, HAND_BLOCK), iq2_block(-2.0, HAND_BLOCK)])
    raw = raw.view(2, 1, 66)
    row = torch.tensor(HAND_VALUES, device="cuda")
    want = torch.stack([row, -2 * row])
    assert torch.equal(ref_decode(raw, "iq2_xxs"), want)
    qweight, scales = repack_blocks(raw, "iq2_xxs")
    lut = iq2_lut(raw.device)
    got = dequantize_blocks(qweight, scales, lut, "iq2_xxs")
    assert torch.equal(got, want.bfloat16())
    # x = ones gives the row sums: 32 + 96 + 16 + 80 + 53.75 - 992 + 992 + 32.
    x = torch.ones(1, 256, device="cuda", dtype=torch.float16)
    assert linear(x, raw, "iq2_xxs").tolist() == [[309.75, -619.5]]


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


# Bolt's dense layers at TP=1 as (codec, N, K). shared_down has an odd count of
# 256-wide K steps (21), which takes the non-power-of-two split-K.
BOLT_LINEAR = {
    "qkv_proj": ("q8_0", 4608, 4096),
    "fc2_latent_proj": ("q8_0", 4096, 1024),
    "shared_up": ("iq2_xxs", 5376, 4096),
    "shared_down": ("iq2_xxs", 4096, 5376),
}


@pytest.mark.parametrize("M", [1, 8, 64, 512])
@pytest.mark.parametrize("layer", BOLT_LINEAR)
def test_linear_bolt_shapes(layer, M):
    codec, N, K = BOLT_LINEAR[layer]
    raw = make_raw((N,), K, codec, seed=M)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    assert_close_to_cublas(linear(x, raw, codec), x, ref_decode(raw, codec))


@pytest.mark.parametrize("M", [1, 64, 512])
def test_shared_expert_error_does_not_pile_up(M):
    """Bolt's shared expert, up -> relu2 -> down, stays within 1.5x of the same
    chain on cuBLAS: per-layer errors must not compound."""
    raw_up = make_raw((5376,), 4096, "iq2_xxs", seed=11)
    raw_down = make_raw((4096,), 5376, "iq2_xxs", seed=12)
    up, down = ref_decode(raw_up, "iq2_xxs"), ref_decode(raw_down, "iq2_xxs")
    x = torch.randn(M, 4096, device="cuda", dtype=torch.bfloat16)
    ours = linear(F.relu(linear(x, raw_up, "iq2_xxs")).square(), raw_down, "iq2_xxs")
    cublas = F.relu(x @ up.bfloat16().T).square() @ down.bfloat16().T
    exact = F.relu(x.double() @ up.double().T).square() @ down.double().T
    err = (ours.double() - exact).abs().max()
    assert err <= 1.5 * (cublas.double() - exact).abs().max(), err


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "N,K",
    [(256, 4096), (1024, 4096), (2304, 4096), (4096, 4096), (4096, 1024), (4608, 4096)],
)
def test_q8_single_row_hopper_shapes(N, K, dtype):
    """Hopper runs these single-row Q8_0 shapes as one wide-K GEMV without
    split-K (`_linear_config`); it must keep cuBLAS-level error under replay."""
    raw = make_raw((N,), K, "q8_0", seed=31, extreme=dtype == torch.bfloat16)
    x = torch.randn(1, K, device="cuda", dtype=dtype)
    want = ref_decode(raw, "q8_0")
    linear(x, raw, "q8_0")
    graph = torch.cuda.CUDAGraph()
    qw, scales = repack_blocks(raw, "q8_0")
    with torch.cuda.graph(graph):
        out = torch.ops.vllm.modelopt_block_linear(x, qw, scales, None, "q8_0")
    graph.replay()
    assert_close_to_cublas(out, x, want)


@pytest.mark.parametrize("M", [1, 17])
def test_linear_offset_view(M):
    """Unaligned activation pointers take the same kernels and stay correct."""
    K = K_FOR["iq2_xxs"]
    raw = make_raw((520,), K, "iq2_xxs", seed=3)
    big = torch.randn(M * K + 8, device="cuda", dtype=torch.bfloat16)
    x = big[3 : 3 + M * K].view(M, K)
    assert_close_to_cublas(linear(x, raw, "iq2_xxs"), x, ref_decode(raw, "iq2_xxs"))


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("M", [1, 17, 512])
def test_linear_non_finite_scales_stay_in_their_rows(codec, M):
    """Inf / nan block scales poison only their own output column, as they
    would with cuBLAS on the dequantized weights."""
    K = K_FOR[codec]
    raw = make_raw((1032,), K, codec, seed=13)
    bad = torch.tensor([float("inf"), -float("inf"), float("nan")], device="cuda")
    raw[[5, 100, 1031], [0, 7, -1], :2] = bad.half().view(torch.uint8).view(3, 2)
    w = ref_decode(raw, codec)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out = linear(x, raw, codec)
    finite = torch.isfinite(w).all(1)
    assert torch.equal(torch.isfinite(out), finite.expand(M, -1))
    assert_close_to_cublas(out[:, finite], x, w[finite])


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
    takes the M=1 kernel."""
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


@pytest.mark.parametrize("M", [1, 4, 17])
def test_linear_cuda_graph_replay(M):
    raw = make_raw((520,), 1024, "iq2_xxs", seed=6)
    qweight, scales = repack_blocks(raw, "iq2_xxs")
    lut = iq2_lut(raw.device)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    torch.ops.vllm.modelopt_block_linear(x, qweight, scales, lut, "iq2_xxs")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.vllm.modelopt_block_linear(x, qweight, scales, lut, "iq2_xxs")
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        allocated = torch.accelerator.memory_allocated()
        graph.replay()
        assert torch.accelerator.memory_allocated() == allocated
        assert_close_to_cublas(out, x, ref_decode(raw, "iq2_xxs"))


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_q8_embedding_exact(id_dtype, dtype):
    raw = make_raw((300,), 2048, "q8_0", seed=9, extreme=True)
    qweight, scales = repack_blocks(raw, "q8_0")
    ids = torch.tensor([5, 5, 299, 0, 17, 3, 5], device="cuda", dtype=id_dtype)
    out = torch.ops.vllm.modelopt_block_embedding(ids, qweight, scales, dtype)
    assert torch.equal(out, ref_decode(raw[ids.long()], "q8_0").to(dtype))


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


def make_moe(codec, activation, E=16, K=1024, inter=512, top_k=2, raw=None):
    gated = activation == MoEActivation.SILU
    raw1, raw2 = raw or (
        make_raw((E, (2 if gated else 1) * inter), K, codec, seed=7),
        make_raw((E, K), inter, codec, seed=8),
    )
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

    def decode(raw):  # one expert at a time bounds ref_decode's temporaries
        return torch.stack([ref_decode(r, codec) for r in raw])

    return run, decode(raw1), decode(raw2)


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


@pytest.mark.parametrize("gemv", [True, False])
def test_moe_hand_built(gemv, workspace_init, monkeypatch):
    """Two experts of constant IQ2_XXS blocks (every weight = d), relu2.

    Expert 0: W1 = W2 = 1/256. Expert 1: W1 = 1/128, W2 = -1/256. With K = 256:
    token 0 = ones on (e0, e1) with weights (0.75, 0.25):
        0.75 * relu2(1) * 1 + 0.25 * relu2(2) * -1 = 0.75 - 1 = -0.25
    token 1 = twos on (e1, e0) with weights (0.5, 0.5):
        0.5 * relu2(4) * -1 + 0.5 * relu2(2) * 1 = -8 + 2 = -6
    """
    if not gemv:
        monkeypatch.setattr(moe_mod, "moe_gemv_config", lambda *a: None)

    def expert_weights(*ds):
        return torch.stack([iq2_block(d, []).expand(256, 1, 66) for d in ds])

    raw = (expert_weights(2**-8, 2**-7), expert_weights(2**-8, -(2**-8)))
    run, _, _ = make_moe(
        "iq2_xxs", MoEActivation.RELU2_NO_MUL, E=2, K=256, inter=256, raw=raw
    )
    x = torch.tensor([[1.0], [2.0]], device="cuda").expand(2, 256).bfloat16()
    weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]], device="cuda")
    ids = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int32)
    want = torch.tensor([[-0.25], [-6.0]], device="cuda").expand(2, 256)
    assert torch.equal(run(x.contiguous(), weights, ids), want.bfloat16())


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


@pytest.mark.parametrize("M", [1, 8, 64, 512])
def test_moe_bolt_shape(M, workspace_init):
    """Bolt's most common routed-expert layer: 240 experts of 2048 x 1024, top 9."""
    act = MoEActivation.RELU2_NO_MUL
    run, w1, w2 = make_moe("iq2_xxs", act, E=240, inter=2048, top_k=9)
    x = torch.randn(M, 1024, device="cuda", dtype=torch.bfloat16)
    topk_weights, topk_ids = make_routing(M, 240, 9)
    got = run(x, topk_weights, topk_ids)
    assert_moe_close_to_unfused(got, x, w1, w2, topk_weights, topk_ids, act)


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


@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("M", [16, 32])
@pytest.mark.parametrize("activation", [MoEActivation.RELU2_NO_MUL, MoEActivation.SILU])
def test_moe_singleton_tiles_follow_routing_under_graph(
    codec, M, activation, workspace_init, monkeypatch
):
    """With the decode tile, one-row tiles take `fused_moe_kernel`'s GEMV branch
    and fuller tiles the tensor-core one (M=16 also skips the token sort); a
    captured graph must follow routing that moves between them."""
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


def test_batch_invariant_rows(workspace_init, monkeypatch):
    """With VLLM_BATCH_INVARIANT=1 a token's output does not depend on M.

    Covers the packed GEMM (forced by DEQUANT_MAX_NUMEL = 0) and the MoE; the
    dequantized path's matmul is made invariant by vLLM's own mode.
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(triton_mod, "DEQUANT_MAX_NUMEL", 0)
    raw = make_raw((1032,), 1024, "iq2_xxs", seed=14)
    run, _, _ = make_moe("iq2_xxs", MoEActivation.RELU2_NO_MUL, top_k=9)
    x = torch.randn(300, 1024, device="cuda", dtype=torch.bfloat16)
    topk_weights, topk_ids = make_routing(300, 16, 9)
    dense = linear(x[:1], raw, "iq2_xxs")
    moe = run(x[:1], topk_weights[:1], topk_ids[:1])
    for M in (5, 37, 300):
        assert torch.equal(linear(x[:M], raw, "iq2_xxs")[:1], dense)
        assert torch.equal(run(x[:M], topk_weights[:M], topk_ids[:M])[:1], moe)


@pytest.mark.parametrize(
    "b12x,backend,codec,dtype,expected",
    [
        (True, "auto", "iq2_xxs", torch.bfloat16, "b12x"),
        (True, "triton", "iq2_xxs", torch.bfloat16, "triton"),
        (True, "auto", "iq2_xxs", torch.float16, "triton"),
        (True, "b12x", "q8_0", torch.float16, "error"),
        (False, "auto", "iq2_xxs", torch.bfloat16, "triton"),
        (False, "auto", "q8_0", torch.float16, "triton"),
        (False, "triton", "q8_0", torch.bfloat16, "triton"),
        (False, "auto", "q8_0", torch.float32, "error"),
        (False, "b12x", "iq2_xxs", torch.bfloat16, "error"),
        (False, "marlin", "iq2_xxs", torch.bfloat16, "error"),
        (False, "auto", "iq2_xs", torch.bfloat16, "error"),
    ],
)
def test_backend_selection(monkeypatch, b12x, backend, codec, dtype, expected):
    """b12x stays the SM12x BF16 default; elsewhere IQ2_XXS / Q8_0 take Triton,
    and unsupported requests fail at construction."""
    import vllm.utils.b12x as b12x_utils
    from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
    from vllm.model_executor.layers.quantization.modelopt_block_quant import (
        ModelOptBlockQuantLinearMethod,
        ModelOptBlockQuantMoEMethod,
    )

    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: b12x)
    monkeypatch.setattr(
        b12x_utils,
        "get_b12x_blockscaled",
        lambda: (
            SimpleNamespace(BlockQuantLinearWeight=object, is_supported=lambda: True)
            if b12x
            else None
        ),
    )
    moe_config = make_dummy_moe_config(hidden_dim=256, intermediate_size=512)
    moe_config.moe_backend = backend
    moe_config.in_dtype = dtype
    builders = (
        lambda: ModelOptBlockQuantMoEMethod(moe_config, codec=codec),
        lambda: ModelOptBlockQuantLinearMethod(codec),
    )
    vllm_config = VllmConfig(kernel_config=KernelConfig(linear_backend=backend))
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(dtype):
        for build in builders:
            if expected == "error":
                with pytest.raises(ValueError, match=f"ModelOpt {codec.upper()}"):
                    build()
            else:
                assert build().use_triton == (expected == "triton")
