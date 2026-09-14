# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics for the DSV4.1 MXFP8 Triton SIMT skinny GEMM and the wo_a chain."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    swizzle_mxfp8_scale,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
from vllm.models.deepseek_v4_1.nvidia import low_latency_gemm as dsv41_gemm
from vllm.models.deepseek_v4_1.nvidia.ops.mxfp8_skinny_gemm import (
    Mxfp8SimtGemmConfig,
    mxfp8_simt_gemm,
    swizzle_wo_a_packed_scale,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported

SIMT_MXFP8_CELLS = [
    (n, k, m, config)
    for (n, k), spec in dsv41_gemm.DSV41_PROJECTIONS_SM100.items()
    if spec.mxfp8
    for m, config in spec.simt_configs
]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("SIMT skinny GEMM requires CUDA")


def _require_sm100_and_deep_gemm() -> None:
    _require_cuda()
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("wo_a chain comparison requires SM100")
    if not is_deep_gemm_supported():
        pytest.skip("DeepGEMM is not available")


def _dequant_reference(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    m, k = q.shape
    descale = torch.exp2(sf.to(torch.float32) - 127.0)
    return (q.to(torch.float32).view(m, k // 32, 32) * descale.unsqueeze(-1)).view(m, k)


def _pack_ue8m0(linear: torch.Tensor) -> torch.Tensor:
    """(R, C) uint8 -> (R, C//4) int32; byte j of element c is linear[:, 4c+j]."""
    r, c = linear.shape
    shifts = torch.arange(4, device=linear.device) * 8
    packed = (linear.to(torch.int64).view(r, c // 4, 4) << shifts).sum(-1)
    return packed.to(torch.int32)


def _rel_rmse(output: torch.Tensor, reference: torch.Tensor) -> float:
    delta = output.float() - reference.float()
    return (
        delta.square().mean().sqrt()
        / reference.float().square().mean().sqrt().clamp_min(1e-30)
    ).item()


@pytest.mark.parametrize("n,k,m,config", SIMT_MXFP8_CELLS)
@pytest.mark.parametrize("scale", [1e-5, 0.1, 1.0, 10.0, 100.0])
def test_dsv41_simt_mxfp8_matches_quantized_reference(
    n: int,
    k: int,
    m: int,
    config: Mxfp8SimtGemmConfig,
    scale: float,
) -> None:
    """SIMT vs per-32 E8M0 quantize + fp32 reference GEMM, mirroring round_mx."""
    _require_cuda()
    torch.manual_seed(42 + m)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * scale
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    q_w, lin_w = _mxfp8_e4m3_quantize_torch(weight)
    sw = swizzle_mxfp8_scale(lin_w, n, k)

    output = mxfp8_simt_gemm(x, q_w, sw, None, config)

    assert output.dtype == torch.bfloat16
    q_x, lin_x = _mxfp8_e4m3_quantize_torch(x)
    reference = _dequant_reference(q_x, lin_x) @ _dequant_reference(q_w, lin_w).t()
    # Round the fp32 reference to bf16 as well, isolating the kernel's math
    # from the (shared) bf16 output rounding.
    assert _rel_rmse(output, reference.to(torch.bfloat16)) < 1e-3


def test_wo_a_packed_scale_swizzle_matches_manual_unpack() -> None:
    torch.manual_seed(0)
    linear = torch.randint(1, 254, (1024, 128), dtype=torch.uint8)
    swizzled = swizzle_wo_a_packed_scale(_pack_ue8m0(linear))

    assert torch.equal(swizzled, swizzle_mxfp8_scale(linear, 1024, 4096))


def _patch_active(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dsv41_gemm, "_ACTIVE", True)
    monkeypatch.setattr(
        dsv41_gemm.current_platform,
        "is_device_capability",
        lambda cc: cc == (10, 0),
    )


def _mx_layer(weight: torch.Tensor) -> SimpleNamespace:
    """(N, K) BF16 weight -> fake MXFP8 layer with col-major fp8 and sw scale."""
    q, lin = _mxfp8_e4m3_quantize_torch(weight)
    return SimpleNamespace(
        weight=q.t(),
        weight_scale=swizzle_mxfp8_scale(lin, weight.shape[0], weight.shape[1]),
        weight_deq=_dequant_reference(q, lin),
    )


def test_dsv41_fused_wqa_wkv_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    _patch_active(monkeypatch)
    torch.manual_seed(42)
    weight = torch.randn(1792, 5120, dtype=torch.bfloat16, device="cuda")
    layer = _mx_layer(weight)
    x = torch.randn(1, 5120, dtype=torch.bfloat16, device="cuda")

    output = dsv41_gemm.try_fused_wqa_wkv_gemm(layer, x)

    assert output is not None
    q_x, lin_x = _mxfp8_e4m3_quantize_torch(x)
    reference = _dequant_reference(q_x, lin_x) @ layer.weight_deq.t()
    assert _rel_rmse(output, reference.to(torch.bfloat16)) < 1e-3
    # Only M=1 measured for this cell.
    x2 = torch.randn(2, 5120, dtype=torch.bfloat16, device="cuda")
    assert dsv41_gemm.try_fused_wqa_wkv_gemm(layer, x2) is None


class _FakeWoB(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        layer = _mx_layer(weight)
        self.weight = layer.weight
        self.weight_scale = layer.weight_scale
        self.weight_deq = layer.weight_deq
        self.bias = None
        self.reduce_results = False
        self.tp_size = 1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q, s = _mxfp8_e4m3_quantize_torch(z)
        return (_dequant_reference(q, s) @ self.weight_deq.t()).to(torch.bfloat16)


class _FakeWoA(nn.Module):
    """wo_a in the production DeepGEMM BMM layout: 3D fp8 weight plus the
    MN-major TMA-packed per-row UE8M0 scales from
    ``deepgemm_post_process_fp8_weight_block``."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
        )

        q, lin = _mxfp8_e4m3_quantize_torch(weight)
        ws = torch.exp2(lin.to(torch.float32) - 127.0)
        self.weight, self.weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=q,
            ws=ws,
            quant_block_shape=(1, 32),
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=1,
        )


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_dsv41_wo_b_cell(monkeypatch: pytest.MonkeyPatch, num_tokens: int) -> None:
    _require_cuda()
    _patch_active(monkeypatch)
    torch.manual_seed(42 + num_tokens)
    weight = torch.randn(5120, 1024, dtype=torch.bfloat16, device="cuda")
    wo_b = _FakeWoB(weight)
    z = torch.randn(num_tokens, 1024, dtype=torch.bfloat16, device="cuda")

    output = dsv41_gemm.try_wo_b_gemm(wo_b, z, einsum_recipe=(1, 1, 32))

    assert output is not None
    reference = wo_b(z)
    assert _rel_rmse(output, reference) < 1e-3


def test_dsv41_wo_b_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    _patch_active(monkeypatch)
    torch.manual_seed(42)
    wo_b = _FakeWoB(torch.randn(5120, 1024, dtype=torch.bfloat16, device="cuda"))
    z = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")

    # SM90-style recipe never dispatches.
    assert dsv41_gemm.try_wo_b_gemm(wo_b, z, einsum_recipe=(1, 128, 128)) is None
    # Only M in {1, 2} measured.
    z4 = torch.randn(4, 1024, dtype=torch.bfloat16, device="cuda")
    assert dsv41_gemm.try_wo_b_gemm(wo_b, z4, einsum_recipe=(1, 1, 32)) is None


_O_PROJ_SHAPES = dict(
    n_groups=1,
    heads_per_group=8,
    nope_dim=448,
    rope_dim=64,
    o_lora_rank=1024,
    einsum_recipe=(1, 1, 32),
    tma_aligned_scales=True,
)


def _cos_sin_cache(max_pos: int) -> torch.Tensor:
    torch.manual_seed(42)
    angles = torch.rand(max_pos, 32, dtype=torch.float32, device="cuda") * 6.283
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


@pytest.mark.parametrize("num_tokens", [1, 2, 4])
def test_dsv41_wo_a_chain_matches_baseline(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int
) -> None:
    _require_sm100_and_deep_gemm()
    torch.manual_seed(7 + num_tokens)
    o = torch.randn(num_tokens, 8, 512, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(num_tokens, device="cuda")
    cos_sin_cache = _cos_sin_cache(2 * num_tokens)
    wo_a = _FakeWoA(torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda"))
    wo_b = _FakeWoB(torch.randn(5120, 1024, dtype=torch.bfloat16, device="cuda"))

    monkeypatch.setattr(dsv41_gemm, "_ACTIVE", False)
    baseline = deep_gemm_fp8_o_proj(
        o, positions, cos_sin_cache, wo_a, wo_b, **_O_PROJ_SHAPES
    )

    _patch_active(monkeypatch)
    holder = nn.Module()
    holder.wo_a = wo_a
    dsv41_gemm.prepare_dsv41_wo_a_scales(holder)
    assert isinstance(wo_a._dsv41_wo_a_scale, torch.Tensor)
    candidate = deep_gemm_fp8_o_proj(
        o, positions, cos_sin_cache, wo_a, wo_b, **_O_PROJ_SHAPES
    )

    cosine = torch.nn.functional.cosine_similarity(
        candidate.float().flatten(), baseline.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_dsv41_wo_a_chain_m8_falls_back_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M=8 is outside the table: active dispatch must run the original path."""
    _require_sm100_and_deep_gemm()
    torch.manual_seed(8)
    o = torch.randn(8, 8, 512, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(8, device="cuda")
    cos_sin_cache = _cos_sin_cache(16)
    wo_a = _FakeWoA(torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda"))
    wo_b = _FakeWoB(torch.randn(5120, 1024, dtype=torch.bfloat16, device="cuda"))

    monkeypatch.setattr(dsv41_gemm, "_ACTIVE", False)
    baseline = deep_gemm_fp8_o_proj(
        o, positions, cos_sin_cache, wo_a, wo_b, **_O_PROJ_SHAPES
    )

    _patch_active(monkeypatch)
    holder = nn.Module()
    holder.wo_a = wo_a
    dsv41_gemm.prepare_dsv41_wo_a_scales(holder)
    candidate = deep_gemm_fp8_o_proj(
        o, positions, cos_sin_cache, wo_a, wo_b, **_O_PROJ_SHAPES
    )

    assert torch.equal(candidate, baseline)


def test_dsv41_wo_a_chain_shape_mismatch_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off-shape wo_a (e.g. v4.0) must not dispatch even when activated."""
    _patch_active(monkeypatch)
    wo_a = SimpleNamespace(
        weight=torch.zeros(1, 1024, 8192).to(torch.float8_e4m3fn),
        _dsv41_wo_a_scale=torch.zeros(1, dtype=torch.uint8),
    )
    o = torch.randn(1, 8, 512, dtype=torch.bfloat16)
    positions = torch.arange(1)
    output = dsv41_gemm.try_wo_a_chain_gemm(
        o,
        positions,
        torch.zeros(16, 64, dtype=torch.float32),
        wo_a,
        **_O_PROJ_SHAPES,
    )
    assert output is None
    # Missing pre-rearranged scale buffer: no dispatch.
    wo_a_missing = SimpleNamespace(
        weight=torch.zeros(1, 1024, 4096).to(torch.float8_e4m3fn)
    )
    assert (
        dsv41_gemm.try_wo_a_chain_gemm(
            o,
            positions,
            torch.zeros(16, 64, dtype=torch.float32),
            wo_a_missing,
            **_O_PROJ_SHAPES,
        )
        is None
    )
