# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP8 linear GEMM for AMD CDNA4 (gfx950) via Triton ``tl.dot_scaled``.

Consumes the FP8 E4M3 weights + E8M0 block scales directly (no dequant-to-BF16);
activations are MXFP8-quantized per token (inside the GEMM for small M), or
arrive pre-quantized as a ``QuantizedActivation`` with row-major (not swizzled)
E8M0 scales. Uses the CDNA4 hardware microscaling matrix cores. ``dot_scaled``
tiles K by 128; a weight whose K is not a multiple of that is dequantized to
BF16 once at load and served by a plain linear instead, since ``can_implement``
does not filter on K.
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig
from .rocm_block32_gemm import BLOCK_ROWS, rocm_mxfp8_block32_gemm

_FP8_MAX = float(torch.finfo(MXFP8_VALUE_DTYPE).max)
_TINY = float(torch.finfo(torch.float32).tiny)


@triton.jit
def _quant_mxfp8_tile(x, FP8_MAX: tl.constexpr, TINY: tl.constexpr):
    """MXFP8-quantize a [BLOCK_M, BLOCK_K] tile: bit-exact with
    ``mxfp8_e4m3_quantize`` (see ``_mxfp8_quant_triton_kernel`` for why the
    scale is applied as a reciprocal)."""
    BM: tl.constexpr = x.shape[0]
    BK: tl.constexpr = x.shape[1]
    xb = tl.reshape(x.to(tl.float32), (BM, BK // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(xb), axis=2), TINY)
    sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    rescale = tl.exp2(127.0 - sb)
    xq = tl.reshape((xb * rescale[:, :, None]).to(tl.float8e4nv), (BM, BK))
    return xq, sb.to(tl.uint8)


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
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
    QUANT_X: tl.constexpr,
    QUANT_ROWS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Rows of x that are loaded: all BLOCK_M, or with QUANT_X the first
    # QUANT_ROWS (a power of 2 >= the rows this tile stores).
    offs_xm = pid_m * BLOCK_M + tl.arange(0, QUANT_ROWS)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    xm_mask = offs_xm < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        if QUANT_X:
            # x is the unquantized activation; every N-tile quantizes its own
            # copy, which is cheaper than a separate quant launch at small M.
            xf = tl.load(x_ptrs, mask=xm_mask[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
            x, xs = _quant_mxfp8_tile(xf, FP8_MAX, TINY)
            if QUANT_ROWS < BLOCK_M:
                # BLOCK_M is the smallest MFMA tile; repeating the quantized
                # rows fills it, and rows >= M are never stored.
                REPS: tl.constexpr = BLOCK_M // QUANT_ROWS
                x = tl.reshape(
                    tl.broadcast_to(x[None, :, :], (REPS, QUANT_ROWS, BLOCK_K)),
                    (BLOCK_M, BLOCK_K),
                )
                xs = tl.reshape(
                    tl.broadcast_to(xs[None, :, :], (REPS, QUANT_ROWS, BLOCK_K // 32)),
                    (BLOCK_M, BLOCK_K // 32),
                )
        else:
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
            xs_ptrs += (BLOCK_K // 32) * stride_xsk
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _launch(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> None:
    """``out = x @ w.T``. ``x_scale`` None means ``x`` is the unquantized
    activation and is MXFP8-quantized inside the kernel."""
    M, K = x.shape
    N = w.shape[0]
    quant_x = x_scale is None
    xs = x if x_scale is None else x_scale  # Unused by the kernel if quant_x.
    # Quantizing only the rows that can be stored keeps the fused quant's cost
    # (serial per N-tile) from scaling with BLOCK_M at M < BLOCK_M. At
    # BLOCK_K=128 that cost is small and repeating the rows costs more.
    if quant_x and block_k >= 256:
        quant_rows = min(block_m, triton.next_power_of_2(M))
    else:
        quant_rows = block_m
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _mxfp8_linear_kernel[grid](
        x,
        xs,
        w,
        w_scale,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        xs.stride(0),
        xs.stride(1),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        QUANT_X=quant_x,
        QUANT_ROWS=quant_rows,
        FP8_MAX=_FP8_MAX,
        TINY=_TINY,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# Past this many rows, quantizing the activation once per N-tile inside the
# GEMM costs more than one separate quant launch.
_FUSED_QUANT_MAX_M = 16


def _select_fused_quant_cfg(M: int, N: int, K: int):
    """Tile config with the activation quant fused (M <= 16), or None to
    quantize separately.
    """
    if K == 2048 and M <= 4:
        return 16, 32, 512, 4, 3
    if K > 1024:
        return None
    if K == 1024:
        return 16, 32, 512, 4, 3
    return 16, 32, 256 if K % 256 == 0 else 128, 4, 3


def _mxfp8_dot_scaled_linear(
    x: torch.Tensor,  # [M, K] bf16/fp16
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    if (
        0 < M <= _FUSED_QUANT_MAX_M
        and x.dtype == torch.bfloat16
        and not envs.VLLM_BATCH_INVARIANT
    ):
        cfg = _select_fused_quant_cfg(M, N, K)
        if cfg is not None:
            out = torch.empty((M, N), dtype=x.dtype, device=x.device)
            _launch(x, None, w, w_scale, out, *cfg)
            return out
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    return _mxfp8_dot_scaled_gemm(x_q, x_scale, w, w_scale, x.dtype)


def _mxfp8_dot_scaled_gemm(
    x_q: torch.Tensor,  # [M, K] fp8 e4m3
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0), row-major
    w: torch.Tensor,  # [N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = x_q.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=out_dtype, device=x_q.device)
    _launch(x_q, x_scale, w, w_scale, out, *_select_cfg(M, N, K))
    return out


# gfx950's async-copy LDS (default in the triton 3.8 this branch pins) leaves no
# room for num_stages=3 at BLOCK_K=256; use 2 there.
_BK256_STAGES = 3
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950

    if on_gfx950():
        _BK256_STAGES = 2


def _select_cfg(M, N, K):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) — graph-tuned on gfx950.

    The M-bucketed, shape-adaptive tile selection here is the speedup over the
    upstream 2-bucket launcher. Tiles are pipelined (num_stages>=2, larger BLOCK_K)
    and occupancy- and shape-aware: keyed on the LOCAL (M, N, K), so it adapts to the
    TP-sharded shapes (e.g. MiniMax-M3 TP=4 vs TP=8, where local N and K differ) —
    large-K prefill uses BLOCK_K=256; short-K (K=768) widens N. BLOCK_K must divide K
    (the K-loop is unmasked), so every BLOCK_K below is guarded to be K-divisible
    (served K: 384/768/1024/2048/6144).
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
            return 128, 128, 256, 8, _BK256_STAGES
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
        return 128, 128, 256, 8, _BK256_STAGES
    # small local-N (e.g. TP=8 shared_gate_up N=768): a 64-wide BLOCK_N doubles the
    # N-tile count -> better CU fill than 128x128 at this mid-large M (~1.4x there).
    if N <= 1024 and K % 256 == 0:
        return 128, 64, 256, 8, 3
    return (128, 128, 256, 8, 2) if K % 256 == 0 else (128, 256, 128, 8, 3)


_DOT_SCALED_K_ALIGN = 128


def _as_block32_scale(weight_scale: torch.Tensor) -> torch.Tensor | None:
    """The [N / 32, K / 32] block scales, if every 32 rows share their scales."""
    N = weight_scale.shape[0]
    if N % BLOCK_ROWS != 0:
        return None
    blocks = weight_scale.view(N // BLOCK_ROWS, BLOCK_ROWS, -1)
    if not torch.equal(blocks, blocks[:, :1].expand_as(blocks)):
        return None
    return blocks[:, 0].contiguous()


class RocmDotScaledMxfp8LinearKernel(Mxfp8LinearKernel):
    """Native CDNA4 (gfx950) MXFP8 linear via Triton ``tl.dot_scaled``."""

    supports_pre_processed_weights = True

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
        # rocm_mxfp8_block32_gemm's split-K varies with M.
        block_scale = (
            None if envs.VLLM_BATCH_INVARIANT else _as_block32_scale(weight_scale)
        )
        if block_scale is not None:
            # Checkpoints with 32x32 scale blocks (DeepSeek V4/V4.1) get them
            # back from the loader expanded per row; keep the blocks and use
            # the block-scaled GEMM, which also handles any K % 32 == 0.
            weight_scale = block_scale
        elif K % _DOT_SCALED_K_ALIGN != 0:
            weight = dequant_mxfp8_to_bf16(weight.contiguous(), weight_scale)
        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", weight_scale)

    def input_quant_key(self) -> QuantKey:
        # Row-major [M, K // 32] E8M0 scales, unlike FlashInfer's swizzled ones
        # under the same key: producers must also check the kernel type.
        return kMxfp8Dynamic

    @staticmethod
    def supports_quantized_input(layer: torch.nn.Module) -> bool:
        """Whether ``apply_weights`` reads a ``QuantizedActivation`` for
        ``layer`` as is (fp8 weights), rather than dequantizing it back."""
        return layer.weight.element_size() == 1 and (
            layer.weight_scale.shape[0] != layer.weight.shape[0]
            or layer.weight.shape[1] % _DOT_SCALED_K_ALIGN == 0
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None and not self.supports_quantized_input(layer):
            # BF16 weight (K % 128 != 0): serve the dequantized activation.
            x = dequant_mxfp8_to_bf16(qa.data, qa.scale).to(qa.orig_dtype)
            x, qa = x.reshape(qa.orig_shape), None
        if qa is not None:
            out_dtype, in_shape = qa.orig_dtype, qa.orig_shape
            x_q, x_scale = qa.data.reshape(-1, in_shape[-1]), qa.scale
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, in_shape = x.dtype, x.shape
            x2d = x.reshape(-1, in_shape[-1])
        out_shape = (*in_shape[:-1], layer.weight.shape[0])
        if layer.weight_scale.shape[0] != layer.weight.shape[0]:
            # One scale row per 32 weight rows (see process_weights_after_loading).
            if qa is None:
                x_q, x_scale = mxfp8_e4m3_quantize(x2d)
            out = rocm_mxfp8_block32_gemm(
                x_q, x_scale, layer.weight, layer.weight_scale, out_dtype
            )
        elif qa is not None:
            out = _mxfp8_dot_scaled_gemm(
                x_q, x_scale, layer.weight, layer.weight_scale, out_dtype
            )
        elif layer.weight.element_size() >= 2:
            out = torch.nn.functional.linear(x2d, layer.weight.to(out_dtype))
        elif x2d.shape[-1] % _DOT_SCALED_K_ALIGN == 0:
            out = _mxfp8_dot_scaled_linear(x2d, layer.weight, layer.weight_scale)
        else:
            # process_weights_after_loading dequantizes these weights once, so
            # this runs only if it did not; dot_scaled would be invalid here.
            w_bf16 = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)
            out = torch.nn.functional.linear(x2d, w_bf16).to(out_dtype)
        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
