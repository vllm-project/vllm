# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the tensor-descriptor (TD) load path in
fused_moe_kernel_gptq_awq.

Covers use_int4_w4a16 (packed-nibble unpack via tl.interleave), the only weight
layout with a TD path in this kernel, gated on VLLM_TRITON_USE_TD. Every test
forces TD on and skips where TD cannot run; standalone pointer-path coverage
lives in tests/kernels/moe/test_moe.py::test_fused_moe_wn16, over a wider shape
sweep.

The _matches_pointer tests compare against the pointer path rather than the fp32
reference, which is tight enough to catch a subtly wrong nibble interleave. The
K- and N-tail cases target the two places the paths legitimately read different
B values, and are XPU-only because the launcher keeps TD off unaligned K
elsewhere.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    should_moe_wna16_use_cuda,
)
from vllm.model_executor.layers.fused_moe.utils import (
    TD_MIN_GATHER_ROWS,
    moe_use_td_hw_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl

DEVICE = "xpu" if current_platform.is_xpu() else "cuda"
_HAS_TL_MAKE_DESC = hasattr(tl, "make_tensor_descriptor")
_TD_SKIP_REASON = (
    "TD path needs tl.make_tensor_descriptor (Triton >= 3.6) and hardware "
    "whose A-gather compiles: XPU, or NVIDIA Blackwell (sm100+), since "
    "tile::gather4 (tcgen05/TMEM) is rejected by ptxas on Hopper and earlier"
)


def _td_unsupported() -> bool:
    return not _HAS_TL_MAKE_DESC or not moe_use_td_hw_supported()


def _skip_if_diverted_to_cuda_kernel(
    m: int, e: int, topk: int, group_size: int
) -> None:
    """Skip shapes that never reach this Triton kernel on CUDA.

    ``fused_experts_impl`` consults ``should_moe_wna16_use_cuda()`` per GEMM and
    hands the launch to ``invoke_fused_moe_wna16_cuda_kernel`` when it holds, so
    neither the TD nor the pointer path of ``fused_moe_kernel_gptq_awq`` runs and
    the comparison would pass with the whole TD branch deleted. Both GEMMs see the
    same ``num_valid_tokens`` (GEMM1: ``m * topk`` with ``top_k=topk``; GEMM2:
    ``A.size(0) == m * topk`` with ``top_k=1``), so one check covers both. The
    predicate requires ``is_cuda()``, so this never fires on XPU.
    """
    if should_moe_wna16_use_cuda(
        num_valid_tokens=m * topk, group_size=group_size, num_experts=e, bit=4
    ):
        pytest.skip(
            "should_moe_wna16_use_cuda() diverts both GEMMs of this shape to the "
            "CUDA WNA16 kernel, so fused_moe_kernel_gptq_awq is never launched"
        )


# One bf16 ULP at the magnitudes this kernel produces (2^-13 for values around
# 0.03), doubled to leave headroom over a single-element rounding disagreement.
_ONE_ULP_ATOL = 2.5e-4


@pytest.fixture(scope="module")
def vllm_config():
    return VllmConfig()


# m=1 is kept on purpose, but it is XPU-only coverage, and the reason differs
# per platform:
#   * On XPU the launcher gates TD off for a single-row A (the M == 1 check in
#     invoke_fused_moe_wna16_triton_kernel) while the second GEMM still runs with
#     M = m * topk and does take TD -- so the case exercises the TD path and
#     guards the gate against regressing into wrong output rather than merely
#     slower output.
#   * On CUDA it exercises neither path of this kernel: with m == 1 the ratio
#     m * topk / e is at most 1, so should_moe_wna16_use_cuda() always holds for
#     the group sizes GPTQ/AWQ actually ship, and both GEMMs are diverted to
#     invoke_fused_moe_wna16_cuda_kernel. _skip_if_diverted_to_cuda_kernel()
#     skips it there instead of letting it pass for the wrong reason.
WN16_MNK = [
    (1, 128, 128),
    (32, 2048, 128),
    (222, 2048, 1024),
]
NUM_EXPERTS = [8]
TOP_KS = [2]
GROUP_SIZES = [128]
HAS_ZP = [True, False]

# A skip keyed off production dispatch could silently swallow the whole sweep if
# that predicate is ever retuned, so pin the premise: everything above m=1 must
# still reach this kernel. Trivially true off CUDA, load-bearing on CUDA.
assert not any(
    should_moe_wna16_use_cuda(
        num_valid_tokens=m * TOP_KS[0],
        group_size=GROUP_SIZES[0],
        num_experts=NUM_EXPERTS[0],
        bit=4,
    )
    for m, _, _ in WN16_MNK
    if m > 1
), "only the m=1 shape may be diverted to the CUDA WNA16 kernel"


def fused_moe(
    hidden_states,
    w1,
    w2,
    score,
    topk,
    renormalize=False,
    quant_config=None,
    global_num_experts=-1,
    expert_map=None,
):
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score.float(), topk, renormalize
    )
    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        quant_config=quant_config,
    )


def torch_moe(a, w1, w2, score, topk):
    """Pure-PyTorch MoE reference for correctness validation.

    Implements fused MoE with SiLU+Mul activation and expert routing.
    Used as reference to validate Triton kernel outputs.
    """
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    m, k = a.shape
    a_rep = a.view(m, -1, k).repeat(1, topk, 1).reshape(-1, k)
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_flat = topk_ids.view(-1)
    act = SiluAndMul()
    for i in range(w1.shape[0]):
        mask = topk_flat == i
        if mask.sum():
            tmp = a_rep[mask] @ w1[i].transpose(0, 1)
            tmp = act(tmp)
            out[mask] = tmp @ w2[i].transpose(0, 1)

    return (
        (out.view(m, -1, w2.shape[1]).to(torch.float32) * topk_weight.view(m, -1, 1))
        .sum(dim=1)
        .to(out.dtype)
    )


def _prepare_quantized_weights(e, n, k, group_size, has_zp, device, dtype):
    """Prepare int4 quantized MoE weights with scales and zero-points.

    int4 only: it is the sole layout of this kernel with a TD path, so there is
    nothing for an int8 branch here to compare against.

    Returns: (w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp)
    """
    w1 = torch.randn((e, 2 * n, k), device=device, dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device=device, dtype=dtype) / 10

    pack_factor = 2
    quant_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8

    w1_ref = w1.clone()
    w2_ref = w2.clone()
    w1_qw = torch.empty((e, 2 * n, k // pack_factor), device=device, dtype=torch.uint8)
    w2_qw = torch.empty((e, k, n // pack_factor), device=device, dtype=torch.uint8)
    w1_sc = torch.empty((e, 2 * n, k // group_size), device=device, dtype=dtype)
    w2_sc = torch.empty((e, k, n // group_size), device=device, dtype=dtype)

    w1_zp = torch.empty(
        (e, 2 * n // pack_factor, k // group_size), device=device, dtype=torch.uint8
    )
    w2_zp = torch.empty(
        (e, k // pack_factor, n // group_size), device=device, dtype=torch.uint8
    )

    for i in range(e * 2):
        expert_id = i % e
        if i // e == 0:
            w, w_ref_arr, w_qw_arr, w_sc_arr, w_zp_arr = w1, w1_ref, w1_qw, w1_sc, w1_zp
        else:
            w, w_ref_arr, w_qw_arr, w_sc_arr, w_zp_arr = w2, w2_ref, w2_qw, w2_sc, w2_zp

        weight, qweight, scales, qzeros = quantize_weights(
            w[expert_id].T, quant_type, group_size, has_zp, False
        )
        weight = weight.T
        qweight = qweight.T.contiguous().to(torch.uint8)
        scales = scales.T

        if has_zp:
            qzeros = qzeros.T.contiguous().to(torch.uint8)

        qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
        if has_zp:
            qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

        w_ref_arr[expert_id] = weight
        w_qw_arr[expert_id] = qweight
        w_sc_arr[expert_id] = scales
        if has_zp:
            w_zp_arr[expert_id] = qzeros

    return w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp


def _assert_td_matches_pointer(td_output, pointer_output):
    """Compare the TD and pointer paths at one bf16 ULP.

    Both accumulate in fp32 and round to bf16 at slightly different points, so a
    single-element 1-ULP disagreement (2^-13 at the magnitudes here) is expected
    rather than a defect; observed on Blackwell with TD the value closer to the
    fp32 reference. Anything larger is a real divergence -- fault injection at
    4 ULP fails, as does 1% of elements off by 5%.
    """
    torch.testing.assert_close(td_output, pointer_output, atol=_ONE_ULP_ATOL, rtol=1e-4)


def _build_quant_config(w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size):
    return int4_w4a16_moe_quant_config(
        w1_scale=w1_sc,
        w2_scale=w2_sc,
        w1_zp=w1_zp if has_zp else None,
        w2_zp=w2_zp if has_zp else None,
        block_shape=[0, group_size],
    )


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
def test_fused_moe_wn16_use_td(
    m, n, k, e, topk, group_size, has_zp, monkeypatch, vllm_config
):
    """TD-path correctness vs the PyTorch reference.

    TD-on only: the TD-off leg would duplicate
    tests/kernels/moe/test_moe.py::test_fused_moe_wn16, which already covers
    the pointer path against the same reference over a wider shape sweep.
    """
    _skip_if_diverted_to_cuda_kernel(m, e, topk, group_size)
    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1")
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = (
        _prepare_quantized_weights(e, n, k, group_size, has_zp, DEVICE, dtype)
    )
    quant_config = _build_quant_config(w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size)

    with set_current_vllm_config(vllm_config):
        triton_output = fused_moe(
            a,
            w1_qw,
            w2_qw,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )
        torch_output = torch_moe(a, w1_ref, w2_ref, score, topk)

    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("m,n,k", WN16_MNK)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("has_zp", HAS_ZP)
def test_fused_moe_wn16_td_matches_pointer(
    m, n, k, e, topk, group_size, has_zp, monkeypatch, vllm_config
):
    """Direct TD-vs-pointer-path comparison on identical inputs.

    Tighter than the fp32-reference tolerance check above (atol=2e-2), which
    is loose enough to miss a subtly wrong nibble interleave -- a swapped
    low/high nibble would often still land within that tolerance for random
    weights. The two Triton paths should agree much more closely than either
    agrees with the fp32 reference.
    """
    _skip_if_diverted_to_cuda_kernel(m, e, topk, group_size)
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    _, _, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = _prepare_quantized_weights(
        e, n, k, group_size, has_zp, DEVICE, dtype
    )
    quant_config = _build_quant_config(w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size)

    def run(use_td: bool) -> torch.Tensor:
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with set_current_vllm_config(vllm_config):
            return fused_moe(
                a,
                w1_qw,
                w2_qw,
                score,
                topk,
                renormalize=False,
                global_num_experts=e,
                quant_config=quant_config,
            )

    pointer_output = run(use_td=False)
    td_output = run(use_td=True)

    _assert_td_matches_pointer(td_output, pointer_output)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("has_zp", HAS_ZP)
def test_fused_moe_wn16_td_k_tail_matches_pointer(has_zp, monkeypatch, vllm_config):
    """K-tail (block_k_diviable=False) case: forces the automatic
    tensor-descriptor zero-fill, compared bit-exact against the pointer
    path's explicit K-mask.

    Bypasses get_moe_wna16_block_config's auto block-size selection (which
    always keeps K block-aligned for the group_size/BLOCK_SIZE_K combinations
    it picks) via override_config, forcing a BLOCK_SIZE_K that does not
    divide K.

    As with the N-tail test, the kernel's ``K`` is ``A.size(1)``, not this
    ``k``: GEMM1 runs with ``K = k`` and GEMM2 with ``K = n`` (its A is the
    intermediate activation). Here the tail lands in GEMM1 (96 % 64 == 32),
    which is enough to exercise the descriptor's zero-fill; asserted below so
    the premise cannot silently decay.

    XPU-only, for the same reason the N-tail test is: off XPU the launcher
    disables TD for exactly the unaligned K this test constructs.
    """
    m, n, k = 33, 512, 96
    e, topk, group_size = 8, 2, 32
    # k % group_size == 0 (96 % 32 == 0) keeps the scale-tensor shape valid.
    forced_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    assert k % forced_config["BLOCK_SIZE_K"] != 0, "GEMM1 (K=k) must have a K tail"
    # The unaligned K that makes this test meaningful is the same condition the
    # launcher bails out on off XPU, which would leave GEMM1 running
    # pointer-vs-pointer while only GEMM2 (aligned, so no tail) took TD.
    if not current_platform.is_xpu():
        pytest.skip(
            "TD is disabled off-XPU for unaligned K, so the K-tail leg would "
            "compare the pointer path against itself"
        )
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    _, _, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = _prepare_quantized_weights(
        e, n, k, group_size, has_zp, DEVICE, dtype
    )
    quant_config = _build_quant_config(w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size)

    def run(use_td: bool) -> torch.Tensor:
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with set_current_vllm_config(vllm_config), override_config(forced_config):
            return fused_moe(
                a,
                w1_qw,
                w2_qw,
                score,
                topk,
                renormalize=False,
                global_num_experts=e,
                quant_config=quant_config,
            )

    pointer_output = run(use_td=False)
    td_output = run(use_td=True)

    _assert_td_matches_pointer(td_output, pointer_output)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("has_zp", HAS_ZP)
def test_fused_moe_wn16_td_n_tail_matches_pointer(has_zp, monkeypatch, vllm_config):
    """N-tail: the pointer path wraps tail lanes with ``% N`` while TD gets
    zero-fill, reconciled only by the ``offs_cn < N`` store mask.

    The kernel's N is ``B.size(1)``, so GEMM1 runs at ``N = 2n`` and GEMM2 at
    ``N = k``; both are asserted below to keep a tail.
    """
    m, n, k = 33, 48, 48
    e, topk, group_size = 8, 2, 16
    forced_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    block_n = forced_config["BLOCK_SIZE_N"]
    assert (2 * n) % block_n != 0, "GEMM1 (N=2n) must have an N tail"
    assert k % block_n != 0, "GEMM2 (N=k) must have an N tail"
    # These shapes also leave a K tail, which off XPU trips the K-alignment
    # bail-out and disables TD for both GEMMs -- the comparison would then be
    # pointer-vs-pointer and pass for the wrong reason. Skip instead.
    if not current_platform.is_xpu() and k % forced_config["BLOCK_SIZE_K"] != 0:
        pytest.skip(
            "TD is disabled off-XPU for unaligned K, so this would compare the "
            "pointer path against itself"
        )
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)

    _, _, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = _prepare_quantized_weights(
        e, n, k, group_size, has_zp, DEVICE, dtype
    )
    quant_config = _build_quant_config(w1_sc, w2_sc, w1_zp, w2_zp, has_zp, group_size)

    def run(use_td: bool) -> torch.Tensor:
        monkeypatch.setenv("VLLM_TRITON_USE_TD", "1" if use_td else "0")
        with set_current_vllm_config(vllm_config), override_config(forced_config):
            return fused_moe(
                a,
                w1_qw,
                w2_qw,
                score,
                topk,
                renormalize=False,
                global_num_experts=e,
                quant_config=quant_config,
            )

    _assert_td_matches_pointer(run(use_td=True), run(use_td=False))


# Shape shared by the two fallback tests. m=33 with topk=2 over 8 experts keeps
# the launch clear of the should_moe_wna16_use_cuda diversion on CUDA; the
# kernel's K is 128 for GEMM1 and 512 (= n) for GEMM2.
_FALLBACK_MNK = (33, 512, 128)
_FALLBACK_E_TOPK_GROUP = (8, 2, 32)
assert not should_moe_wna16_use_cuda(
    num_valid_tokens=_FALLBACK_MNK[0] * _FALLBACK_E_TOPK_GROUP[1],
    group_size=_FALLBACK_E_TOPK_GROUP[2],
    num_experts=_FALLBACK_E_TOPK_GROUP[0],
    bit=4,
), "fallback shapes must reach this kernel, not the CUDA WNA16 one"


def _assert_td_falls_back(forced_config, monkeypatch, vllm_config):
    """Force TD on with a block config it cannot serve, and require the launch to
    survive on the pointer path instead of aborting.

    Reaching the comparison at all is the regression check; the tolerance mirrors
    the fp32-reference comparison used elsewhere in this file.
    """
    m, n, k = _FALLBACK_MNK
    e, topk, group_size = _FALLBACK_E_TOPK_GROUP
    dtype = torch.bfloat16
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 10
    score = torch.randn((m, e), device=DEVICE, dtype=dtype)
    w1_ref, w2_ref, w1_qw, w2_qw, w1_sc, w2_sc, w1_zp, w2_zp = (
        _prepare_quantized_weights(e, n, k, group_size, False, DEVICE, dtype)
    )
    quant_config = _build_quant_config(w1_sc, w2_sc, w1_zp, w2_zp, False, group_size)

    monkeypatch.setenv("VLLM_TRITON_USE_TD", "1")
    with set_current_vllm_config(vllm_config), override_config(forced_config):
        out = fused_moe(
            a,
            w1_qw,
            w2_qw,
            score,
            topk,
            renormalize=False,
            global_num_experts=e,
            quant_config=quant_config,
        )
        ref = torch_moe(a, w1_ref, w2_ref, score, topk)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
@pytest.mark.parametrize("block_size_m", [2, 4])
def test_td_skipped_below_min_gather_rows(block_size_m, monkeypatch, vllm_config):
    """A BLOCK_SIZE_M below the gather minimum must fall back, not abort.

    tensor_descriptor.gather() asserts at least TD_MIN_GATHER_ROWS rows, so a
    smaller tile used to kill the launch outright ("descriptor gather must have
    at least 8 rows"); reproduced on B200. Two config sources reach it: the
    override_config used here (honoured verbatim by try_get_optimal_moe_config,
    so this arrives through fused_experts_impl) and get_default_config's
    use_moe_wna16_cuda branch, min(16, next_power_of_2(M)), which is the
    production path via TritonWNA16Experts.apply. Every other test here forces
    a block config >= 8, which is why a green suite still shipped the crash.
    """
    assert block_size_m < TD_MIN_GATHER_ROWS, "premise: must be below the minimum"
    _assert_td_falls_back(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        },
        monkeypatch,
        vllm_config,
    )


@pytest.mark.skipif(_td_unsupported(), reason=_TD_SKIP_REASON)
def test_td_skipped_below_min_block_k(monkeypatch, vllm_config):
    """A BLOCK_SIZE_K below 32 must fall back, not abort.

    The int4 B descriptor is built at byte granularity, so its innermost block
    dim is BLOCK_SIZE_K // 2 -- below 16 bytes, which make_tensor_descriptor
    rejects. Only override_config can produce such a config
    (get_moe_wna16_block_config returns 32 or 64), which is why this needs a
    forced config rather than a shape.
    """
    forced_config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    }
    # Both GEMMs must stay K-aligned, or the earlier unaligned-K bail-out would
    # disable TD off XPU and mask the gate under test.
    _, n, k = _FALLBACK_MNK
    assert all(K % forced_config["BLOCK_SIZE_K"] == 0 for K in (k, n)), (
        "premise: neither GEMM may hit the unaligned-K bail-out first"
    )
    _assert_td_falls_back(forced_config, monkeypatch, vllm_config)
