# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the RDNA3/RDNA3.5 (gfx11) W4A8 MoE decode kernel.

``moe_w4a8_decode_rdna3`` consumes the TritonWNA16Experts weight tensors
(uint8 codes, optional nibble-packed zero points) and quantizes activations
to int8 per 32 values. It is checked against:
  - a torch reference that applies the same activation quantization
    (tight tolerance), and
  - the Triton WNA16 ``fused_experts`` path on the same tensors (loose
    tolerance, int8 activation error only), which also pins the layout.

Run `pytest tests/kernels/quantization/test_rdna3_moe_w4a8.py`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 W4A8 MoE kernel is ROCm-only", allow_module_level=True)

from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_w4a8_moe import (  # noqa: E402
    RDNA3_W4A8_MAX_TOKENS,
    Rdna3W4A8Experts,
    moe_w4a8_workspace_bytes,
    rdna3_w4a8_moe_kernel_available,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (  # noqa: E402
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    fused_experts,
)

pytestmark = pytest.mark.skipif(
    not rdna3_w4a8_moe_kernel_available(),
    reason="Requires a gfx11 GPU and the moe_w4a8_decode_rdna3 op",
)

device = "cuda"

# (E, K=hidden, N=intermediate, top_k, group_size). E is capped; the kernel
# is expert-count independent.
SHAPES = [
    pytest.param(16, 2048, 512, 8, 128, id="qwen3.6-35b-a3b-g128"),
    pytest.param(16, 2048, 768, 8, 32, id="qwen3-30b-a3b-g32"),
    pytest.param(8, 256, 128, 2, 64, id="small-g64"),
]
NUM_TOKENS = [1, 2, 4, 8]


def _pack_k(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] codes in [0, 16) -> [..., K/2] uint8, low nibble = even k."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _pack_rows(zp: torch.Tensor) -> torch.Tensor:
    """[E, R, G] zero points -> [E, R/2, G] uint8, low nibble = even row."""
    return zp[:, 0::2] | (zp[:, 1::2] << 4)


def _make_expert_weights(E, rows, cols, group, dtype, with_zp, gen):
    codes = torch.randint(
        0, 16, (E, rows, cols), dtype=torch.uint8, device=device, generator=gen
    )
    scale = (
        torch.rand((E, rows, cols // group), device=device, generator=gen) * 0.02
        + 0.005
    ).to(dtype)
    if with_zp:
        zp = torch.randint(
            0,
            16,
            (E, rows, cols // group),
            dtype=torch.uint8,
            device=device,
            generator=gen,
        )
    else:
        zp = torch.full((E, rows, cols // group), 8, dtype=torch.uint8, device=device)
    deq = (
        codes.float() - zp.repeat_interleave(group, -1).float()
    ) * scale.float().repeat_interleave(group, -1)
    packed_zp = _pack_rows(zp) if with_zp else None
    return _pack_k(codes), scale, packed_zp, deq


def _quant_act_ref(x: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 fake-quant per 32 values, as the kernel does."""
    shape = x.shape
    g = x.reshape(*shape[:-1], shape[-1] // 32, 32)
    scale = (g.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(g / scale).clamp(-127, 127)
    return (q * scale).reshape(shape)


def _reference(x, topk_ids, topk_w, w13_deq, w2_deq, quant_act=True):
    E, two_n, _ = w13_deq.shape
    N = two_n // 2
    xf = x.float()
    xq = _quant_act_ref(xf) if quant_act else xf
    out = torch.zeros(x.size(0), w2_deq.size(1), device=device)
    for m in range(x.size(0)):
        for s in range(topk_ids.size(1)):
            e = int(topk_ids[m, s])
            if e < 0 or e >= E:
                continue
            gu = w13_deq[e] @ xq[m]
            h = F.silu(gu[:N]) * gu[N:]
            hq = _quant_act_ref(h) if quant_act else h
            out[m] += topk_w[m, s] * (w2_deq[e] @ hq)
    return out


def _routing(M, E, top_k, gen):
    ids = torch.stack(
        [torch.randperm(E, device=device, generator=gen)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    logits = torch.randn(M, top_k, device=device, generator=gen)
    return ids, torch.softmax(logits, dim=-1)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _setup(E, K, N, top_k, group, dtype, with_zp, M, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    w13, w13_s, w13_zp, w13_deq = _make_expert_weights(
        E, 2 * N, K, group, dtype, with_zp, gen
    )
    w2, w2_s, w2_zp, w2_deq = _make_expert_weights(E, K, N, group, dtype, with_zp, gen)
    x = torch.randn(M, K, device=device, generator=gen).to(dtype)
    ids, wts = _routing(M, E, top_k, gen)
    return x, ids, wts, (w13, w13_s, w13_zp, w2, w2_s, w2_zp), (w13_deq, w2_deq)


def _workspace(M, K, N, top_k):
    nbytes = moe_w4a8_workspace_bytes(M, K, N, top_k)
    return torch.empty(nbytes, dtype=torch.uint8, device=device)


def _run(x, ids, wts, w, expert_map=None, workspace=None):
    out = torch.empty_like(x)
    w13 = w[0]
    if workspace is None:
        workspace = _workspace(x.size(0), x.size(1), w13.size(1) // 2, ids.size(1))
    ops.moe_w4a8_decode_rdna3(out, x, ids, wts, expert_map, workspace, *w)
    return out


@pytest.mark.parametrize("E,K,N,top_k,group", SHAPES)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_zp", [False, True])
def test_matches_w4a8_reference(E, K, N, top_k, group, M, dtype, with_zp):
    x, ids, wts, w, deq = _setup(E, K, N, top_k, group, dtype, with_zp, M)
    out = _run(x, ids, wts, w)
    ref = _reference(x, ids, wts, *deq)
    assert _rel_err(out, ref) < 1e-2


@pytest.mark.parametrize("E,K,N,top_k,group", SHAPES)
@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("with_zp", [False, True])
def test_close_to_triton_wna16(E, K, N, top_k, group, M, with_zp):
    """Same weight tensors through the Triton W4A16 path."""
    dtype = torch.bfloat16
    x, ids, wts, w, _ = _setup(E, K, N, top_k, group, dtype, with_zp, M)
    w13, w13_s, w13_zp, w2, w2_s, w2_zp = w
    out = _run(x, ids, wts, w)
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w13_s,
        w2_scale=w2_s,
        w1_zp=w13_zp,
        w2_zp=w2_zp,
        block_shape=[0, group],
    )
    ref = fused_experts(x, w13, w2, wts, ids, quant_config=quant_config)
    assert _rel_err(out, ref) < 5e-2


def test_masked_slots_are_skipped():
    E, K, N, top_k, group = 8, 256, 128, 4, 64
    x, ids, wts, w, deq = _setup(E, K, N, top_k, group, torch.bfloat16, True, 4)
    ids[0, 1] = -1
    ids[2, :] = -1
    ids[3, 3] = E
    out = _run(x, ids, wts, w)
    ref = _reference(x, ids, wts, *deq)
    assert torch.all(out[2] == 0)
    assert _rel_err(out, ref) < 1e-2


def test_repeated_expert_and_zero_input():
    E, K, N, top_k, group = 8, 256, 128, 4, 64
    x, ids, wts, w, deq = _setup(E, K, N, top_k, group, torch.bfloat16, False, 2)
    ids[:] = 3
    out = _run(x, ids, wts, w)
    assert _rel_err(out, _reference(x, ids, wts, *deq)) < 1e-2

    zero = torch.zeros_like(x)
    assert torch.all(_run(zero, ids, wts, w) == 0)


def test_deterministic():
    x, ids, wts, w, _ = _setup(16, 2048, 512, 8, 128, torch.bfloat16, False, 4)
    first = _run(x, ids, wts, w)
    for _ in range(3):
        assert torch.equal(_run(x, ids, wts, w), first)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("wts_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_routing_dtypes(ids_dtype, wts_dtype):
    """topk_ids and topk_weights are read in the router's dtypes."""
    E, K, N, top_k, group = 16, 512, 256, 4, 64
    x, ids, wts, w, deq = _setup(E, K, N, top_k, group, torch.bfloat16, True, 4)
    wts = wts.to(wts_dtype)
    out = _run(x, ids.to(ids_dtype), wts, w)
    # Reference with the same (rounded) routing weights.
    assert _rel_err(out, _reference(x, ids, wts.float(), *deq)) < 1e-2


def test_expert_map():
    """Global ids go through expert_map; experts of other ranks (-1) are
    skipped."""
    E_global, K, N, top_k, group = 16, 256, 128, 4, 64
    x, ids, wts, w, deq = _setup(E_global, K, N, top_k, group, torch.bfloat16, True, 6)
    local = torch.arange(E_global, device=device) % 2 == 0  # this rank: even ids
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=device)
    expert_map[local] = torch.arange(int(local.sum()), dtype=torch.int32, device=device)
    sel = local.nonzero().flatten()
    w_local = (
        w[0][sel],
        w[1][sel],
        w[2][sel] if w[2] is not None else None,
        w[3][sel],
        w[4][sel],
        w[5][sel] if w[5] is not None else None,
    )
    out = _run(x, ids, wts, w_local, expert_map=expert_map)
    masked = torch.where(local[ids.long()], ids, torch.full_like(ids, -1))
    assert _rel_err(out, _reference(x, masked, wts, *deq)) < 1e-2


def test_workspace_too_small_is_rejected():
    E, K, N, top_k, group = 8, 256, 128, 4, 64
    x, ids, wts, w, _ = _setup(E, K, N, top_k, group, torch.bfloat16, False, 2)
    small = _workspace(2, K, N, top_k)[:-16]
    with pytest.raises(RuntimeError, match="workspace"):
        _run(x, ids, wts, w, workspace=small)


# ----------------------------------------------- through Rdna3W4A8Experts


def _experts(cls, E, K, N, top_k, w, num_experts=None):
    w13, w13_s, w13_zp, w2, w2_s, w2_zp = w
    moe_config = make_dummy_moe_config(
        num_experts=num_experts or E,
        num_local_experts=E,
        experts_per_token=top_k,
        hidden_dim=K,
        intermediate_size=N,
        in_dtype=torch.bfloat16,
    )
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w13_s,
        w2_scale=w2_s,
        w1_zp=w13_zp,
        w2_zp=w2_zp,
        block_shape=[0, K // w13_s.size(2)],
    )
    return cls(moe_config, quant_config)


def _apply(experts, x, w, ids, wts, expert_map=None, global_num_experts=-1):
    w13, w2 = w[0], w[3]
    M, K = x.shape
    ws13_shape, ws2_shape, out_shape = experts.workspace_shapes(
        M, w13.size(1), K, ids.size(1), global_num_experts, w13.size(0), None,
        MoEActivation.SILU,
    )  # fmt: skip
    ws13 = torch.empty(ws13_shape, dtype=x.dtype, device=device)
    ws2 = torch.empty(ws2_shape, dtype=x.dtype, device=device)
    out = torch.zeros(out_shape, dtype=x.dtype, device=device)
    experts.apply(
        out, x, w13, w2, wts, ids, MoEActivation.SILU, global_num_experts,
        expert_map, None, None, ws13, ws2, None, False,
    )  # fmt: skip
    return out


@pytest.fixture
def vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.parametrize("M", [1, 4, RDNA3_W4A8_MAX_TOKENS])
def test_apply_matches_triton(vllm_config, M):
    E, K, N, top_k, group = 16, 512, 256, 4, 64
    x, ids, wts, w, _ = _setup(E, K, N, top_k, group, torch.bfloat16, True, M)
    out = _apply(_experts(Rdna3W4A8Experts, E, K, N, top_k, w), x, w, ids, wts)
    ref = _apply(_experts(TritonWNA16Experts, E, K, N, top_k, w), x, w, ids, wts)
    assert _rel_err(out, ref) < 5e-2


def test_apply_falls_back_to_triton_above_max_tokens(vllm_config):
    E, K, N, top_k, group = 16, 512, 256, 4, 64
    M = RDNA3_W4A8_MAX_TOKENS + 1
    x, ids, wts, w, _ = _setup(E, K, N, top_k, group, torch.bfloat16, True, M)
    out = _apply(_experts(Rdna3W4A8Experts, E, K, N, top_k, w), x, w, ids, wts)
    ref = _apply(_experts(TritonWNA16Experts, E, K, N, top_k, w), x, w, ids, wts)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_apply_expert_map(vllm_config):
    """EP: global ids are remapped and remote experts (-1) contribute 0, as in
    the Triton path."""
    E_global, K, N, top_k, group = 16, 512, 256, 4, 64
    x, ids, wts, w, _ = _setup(E_global, K, N, top_k, group, torch.bfloat16, True, 4)
    local = torch.arange(E_global, device=device) >= E_global // 2  # upper half
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=device)
    expert_map[local] = torch.arange(int(local.sum()), dtype=torch.int32, device=device)
    sel = local.nonzero().flatten()
    w_local = tuple(t[sel] if t is not None else None for t in w)
    E = int(local.sum())
    kwargs = dict(expert_map=expert_map, global_num_experts=E_global)
    out = _apply(
        _experts(Rdna3W4A8Experts, E, K, N, top_k, w_local, E_global),
        x, w_local, ids, wts, **kwargs,
    )  # fmt: skip
    ref = _apply(
        _experts(TritonWNA16Experts, E, K, N, top_k, w_local, E_global),
        x, w_local, ids, wts, **kwargs,
    )  # fmt: skip
    assert _rel_err(out, ref) < 5e-2


def test_apply_fused_shared_expert(vllm_config):
    """A fused shared expert is an extra row of w13/w2 (w1.shape[0] > routed
    experts) that every token selects."""
    E_routed, K, N, top_k, group = 16, 512, 256, 4, 64
    x, ids, wts, w, _ = _setup(
        E_routed + 1, K, N, top_k, group, torch.bfloat16, True, 4
    )
    ids = ids % E_routed
    ids[:, -1] = E_routed  # the shared expert slot
    E = E_routed + 1
    out = _apply(_experts(Rdna3W4A8Experts, E, K, N, top_k, w), x, w, ids, wts)
    ref = _apply(_experts(TritonWNA16Experts, E, K, N, top_k, w), x, w, ids, wts)
    assert _rel_err(out, ref) < 5e-2
