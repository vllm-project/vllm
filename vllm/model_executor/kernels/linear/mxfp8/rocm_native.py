# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP8 dense linear for AMD CDNA4 (gfx950) via Triton ``tl.dot_scaled``.

Reuses the shared fused activation quant (``mxfp8_e4m3_quantize``) and adds a
graph-tuned, M-bucketed ``dot_scaled`` GEMM (``_select_cfg``) — the GEMM tile
selection is the speedup vs the upstream 2-bucket launcher. Math is unchanged vs
the upstream kernel (fp32 accumulate, tol 6e-2).
"""

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _mxfp8_dot_scaled_linear(
    x: torch.Tensor,  # [M, K] bf16/fp16
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    # Shared fused activation quant (already the upstream ROCm path), then a
    # graph-tuned, M-bucketed dot_scaled GEMM. Tiles are pipelined (num_stages>=2,
    # larger BLOCK_K) and shape-adaptive: large-K prefill uses BLOCK_K=256; short-K
    # (K=768) widens N. The tile selection (_select_cfg) is the speedup here.
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _select_cfg(M, N, K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mxfp8_linear_kernel[grid](
        x_q,
        x_scale,
        w,
        w_scale,
        out,
        M,
        N,
        K,
        x_q.stride(0),
        x_q.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _select_cfg(M, N, K):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) — graph-tuned on gfx950.

    Occupancy- and shape-aware: keyed on the LOCAL (M, N, K), so it adapts to the
    TP-sharded shapes (e.g. MiniMax-M3 TP=4 vs TP=8, where local N and K differ).
    BLOCK_K must divide K (the K-loop is unmasked), so every BLOCK_K below is guarded
    to be K-divisible (served K: 384/768/1024/2048/6144).
    """
    if M <= 64:
        # decode (M in {1,32,64}): tiny-M GEMV is weight-BW + GPU-OCCUPANCY bound. The
        # lever is NARROW BLOCK_N=16 (maximize N-tiles so more CUs stream the weight in
        # parallel) + LARGE BLOCK_K (fewer K-iters, bigger coalesced weight loads).
        # Tuned by CUDA-graph replay latency. Optimal at both TP=4 and TP=8.
        if K % 1024 == 0:  # K=2048, 6144 -> graph-best 16x16x1024 (all M)
            return 16, 16, 1024, 2, 2
        if K % 512 == 0:
            return 16, 16, 512, 2, 3
        if K % 256 == 0:  # K=768 (shared_down) -> graph-best 16x32x256
            return 16, 32, 256, 4, 3
        return 16, 32, 128, 4, 3
    # mid-M (65..256) on SMALL local-N: still occupancy-bound (a 64x64 tile makes too
    # few N-tiles), so the narrow-BLOCK_N decode-style tile fills the CUs better.
    # N<=1536 covers the real fused-qkv local N at TP=8: q heads shard but the GQA KV
    # (4) + sparse-indexer (4) heads are < TP=8, so vLLM replicates them to 1/rank ->
    # N = 1024 + 4*128 = 1536 (not 2560/2=1280). For the wider 1280<N<=1536 band the
    # narrow tile only wins up to M=128 (at M=256 the 64x64 tile is better), so cap it
    # there; N<=1280 keeps the narrow tile through M=256. TP=4 qkv N=2560 is unchanged.
    if (M <= 256 and N <= 1280) or (M <= 128 and N <= 1536):
        if K % 1024 == 0:
            return 16, 16, 1024, 2, 2
        if K % 512 == 0:
            return 16, 16, 512, 2, 3
        if K % 256 == 0:
            return 16, 16, 256, 2, 3
        return 16, 16, 128, 2, 3
    # right-sized launch grid (host-side), used to gate the tall 256-BLOCK_M tile.
    occ = triton.cdiv(M, 256) * triton.cdiv(N, 128)
    if K <= 1024:  # short-K (shared_down K=384/768; TP=8 o_proj K=1024)
        if M <= 256:
            return (64, 64, 256, 8, 2) if K % 256 == 0 else (64, 64, 128, 8, 2)
        # large prefill. (The former 256x128x256 tall tile was faster only on triton
        # 3.6; on triton 3.7 its large BLOCK_M register/LDS footprint spills or hits
        # "out of resources", so use 128x128x256 -- within the known-good footprint.)
        if M >= 4096 and K >= 1024 and K % 256 == 0 and occ >= 256:
            return 128, 128, 256, 8, 3
        return 128, 128, 128, 8, 3
    # large-K (K >= 2048). BLOCK_K is K-divisibility-guarded (the K-loop is unmasked):
    # served large-K is 2048/6144 (%256==0), but fall back to 128 (always divides, since
    # the entry requires K%128==0) for any other K to stay correct.
    if M <= 256:  # conc~128 decode + small prefill chunk: occupancy tile
        if K % 512 == 0:
            return 64, 64, 512, 8, 2
        return (64, 64, 256, 8, 2) if K % 256 == 0 else (64, 64, 128, 8, 2)
    if M <= 1024:  # medium prefill chunk: BN=64 keeps small-N occupied
        return (128, 64, 256, 8, 3) if K % 256 == 0 else (128, 64, 128, 8, 3)
    # large prefill (M > 1024). The previously graph-tuned tall 256x128x256 and deep
    # 128x128x512 tiles won only on triton 3.6; on triton 3.7 their larger BLOCK_M /
    # BLOCK_K register+LDS footprint spills (or hits "out of resources" on stricter
    # ROCm/triton builds). The 128x128x256 tile is equal-or-faster on triton 3.7 (the
    # M=4096,N=2048,K=6144 shape: ~104us vs the 256x128x256 tile's ~184us), within ~5%
    # on 3.6, and inside the footprint of the tiles used elsewhere in this selector.
    # Covers the qkv-class local N=1536 (TP=8 qkv / TP=4 shared_gate_up) and the deep-K
    # / very-large-M shapes.
    if K % 256 == 0 and (1280 < N <= 1536 or (occ >= 128 and (K >= 4096 or M >= 4096))):
        return 128, 128, 256, 8, 3
    # small local-N (e.g. TP=8 shared_gate_up N=768): a 64-wide BLOCK_N doubles the
    # N-tile count -> better CU fill than 128x128 at this mid-large M (~1.4x there).
    if N <= 1024 and K % 256 == 0:
        return 128, 64, 256, 8, 3
    return (128, 128, 256, 8, 2) if K % 256 == 0 else (128, 256, 128, 8, 3)


class RocmDotScaledMxfp8LinearKernel(Mxfp8LinearKernel):
    """Native CDNA4 (gfx950) MXFP8 linear via Triton ``tl.dot_scaled``."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        # supports_mx() == gfx95x (CDNA4 native microscaling hardware). On other
        # archs dot_scaled would upcast to BF16, so the kernel selector falls
        # through to the BF16 emulation (hipBLASLt) path instead.
        if not current_platform.supports_mx():
            return False, "native MX requires CDNA4 (gfx95x)"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K] fp8
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )
        out_shape = (*x.shape[:-1], layer.weight.shape[0])
        x2d = x.reshape(-1, x.shape[-1])
        if x2d.shape[-1] % 128 == 0:
            out = _mxfp8_dot_scaled_linear(x2d, layer.weight, layer.weight_scale)
        else:
            # dot_scaled tiling needs K % 128 == 0; dequantize fallback otherwise.
            w_bf16 = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)
            out = torch.nn.functional.linear(x2d, w_bf16).to(x.dtype)
        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
