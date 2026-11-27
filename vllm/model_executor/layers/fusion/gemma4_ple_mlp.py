# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 per-layer-input (PLE) branch GEMMs as Triton kernels.

  gate = x @ Wg^T                         [T, 256]   (K = hidden = 2560)
  h    = gelu_tanh(gate) * pli            pli: this layer's [T, 256] slice of the per-layer inputs
  out  = h @ Wp^T                         [T, 2560]

``gemma4_ple_gate_act`` fuses the activation and the product into the epilogue of the gate GEMM
(the unfused chain runs a separate gelu*mul kernel over the [T, 256] intermediate). The gate is rounded
to bf16 before the activation, as the unfused chain does, and the activation and product run in fp32
with a single bf16 rounding of the result, as the compiled unfused kernel does.

``gemma4_ple_proj`` is the projection GEMM. At small T (decode) a Triton tile that spreads the 2560
output columns over the compute units beats hipBLASLt's tile; above 8192 tokens it calls the
regular bf16 GEMM.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# (BM, BN, BK, num_warps, num_stages, GROUP_M) keyed by the largest token count of the bucket
_GATE_CONFIGS: list[tuple[int, tuple[int, int, int, int, int, int]]] = [
    (128, (16, 16, 256, 4, 3, 1)),
    (256, (16, 32, 256, 2, 3, 1)),
    (512, (64, 16, 256, 4, 3, 4)),
    (1024, (32, 64, 256, 2, 3, 4)),
    (2048, (64, 64, 128, 4, 3, 4)),
    (4096, (64, 64, 128, 4, 2, 4)),
    (8192, (128, 64, 64, 4, 2, 16)),
    (1 << 30, (128, 128, 64, 4, 2, 16)),
]
_PROJ_CONFIGS: list[tuple[int, tuple[int, int, int, int, int, int]]] = [
    (32, (16, 32, 128, 2, 3, 1)),
    (64, (32, 64, 128, 2, 3, 1)),
    (128, (64, 64, 128, 4, 3, 1)),
    (256, (64, 64, 64, 4, 3, 1)),
    (512, (64, 64, 64, 4, 2, 8)),
    (1024, (64, 128, 64, 4, 2, 8)),
    (2048, (128, 128, 64, 4, 2, 8)),
    (4096, (128, 128, 64, 8, 2, 8)),
    (8192, (128, 128, 64, 4, 2, 8)),
]


def _pick(table, m: int):
    for limit, cfg in table:
        if m <= limit:
            return cfg
    return None


@triton.jit
def _ple_gemm_kernel(
    a_ptr,
    w_ptr,
    pli_ptr,
    c_ptr,
    M,
    stride_a,
    stride_pli,
    stride_c,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    EPI: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n: tl.constexpr = N // BN
    width = GM * num_n
    first_m = (pid // width) * GM
    gsz = tl.minimum(num_m - first_m, GM)
    pid_m = first_m + (pid % width) % gsz
    pid_n = (pid % width) // gsz
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    row = offs_m[:, None].to(tl.int64)
    a_ptrs = a_ptr + row * stride_a + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K // BK):
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        w_ptrs += BK
    if EPI:
        g = acc.to(tl.bfloat16).to(tl.float32)
        u = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        act = g * tl.sigmoid(2.0 * u)
        p = tl.load(
            pli_ptr + row * stride_pli + offs_n[None, :], mask=mask_m[:, None], other=0.0
        ).to(tl.float32)
        out = (act * p).to(tl.bfloat16)
    else:
        out = acc.to(tl.bfloat16)
    tl.store(c_ptr + row * stride_c + offs_n[None, :], out, mask=mask_m[:, None])


def _launch(a, w, pli, epi: bool, cfg) -> torch.Tensor:
    bm, bn, bk, nw, ns, gm = cfg
    m, k = a.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    if m:
        _ple_gemm_kernel[(triton.cdiv(m, bm) * (n // bn),)](
            a,
            w,
            pli if pli is not None else a,
            out,
            m,
            a.stride(0),
            pli.stride(0) if pli is not None else 0,
            out.stride(0),
            N=n,
            K=k,
            BM=bm,
            BN=bn,
            BK=bk,
            GM=gm,
            EPI=epi,
            num_warps=nw,
            num_stages=ns,
        )
    return out


def supports(hidden: int, ple: int) -> bool:
    return ple == 256 and hidden % 256 == 0


def gate_act(x: torch.Tensor, w_gate: torch.Tensor, pli: torch.Tensor) -> torch.Tensor:
    return _launch(x, w_gate, pli, True, _pick(_GATE_CONFIGS, x.shape[0]))


def proj(h: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    cfg = _pick(_PROJ_CONFIGS, h.shape[0])
    if cfg is None:
        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_tgemm_enabled():
            from aiter.tuned_gemm import tgemm

            return tgemm.mm(h, w_proj)
        return torch.nn.functional.linear(h, w_proj)
    return _launch(h, w_proj, None, False, cfg)


def _gate_act_impl(x: torch.Tensor, w_gate: torch.Tensor, pli: torch.Tensor) -> torch.Tensor:
    logger.info_once("gemma4_ple_gate_act kernel running (first call)")
    return gate_act(x, w_gate, pli)


def _gate_act_fake(x: torch.Tensor, w_gate: torch.Tensor, pli: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], w_gate.shape[0]), dtype=x.dtype, device=x.device)


def _proj_impl(h: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    logger.info_once("gemma4_ple_proj kernel running (first call)")
    return proj(h, w_proj)


def _proj_fake(h: torch.Tensor, w_proj: torch.Tensor) -> torch.Tensor:
    return torch.empty((h.shape[0], w_proj.shape[0]), dtype=h.dtype, device=h.device)


direct_register_custom_op(
    op_name="gemma4_ple_gate_act",
    op_func=_gate_act_impl,
    mutates_args=[],
    fake_impl=_gate_act_fake,
)
direct_register_custom_op(
    op_name="gemma4_ple_proj",
    op_func=_proj_impl,
    mutates_args=[],
    fake_impl=_proj_fake,
)
