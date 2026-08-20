# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)

_cutedsl_available: bool | None = None

_DEFAULT_CONFIG = (16, 2, 4)
# SM100f-tuned configs: (tile_n, num_stages, num_dma_warps).
_SM100F_TUNED_CONFIGS: dict[
    tuple[int, int], dict[int, tuple[int, int, int]]
] = {
    #(512, 4096): {M: (16, 2, 2) for M in range(1, 11)},
    #(1024, 8192): {
    #    **{M: (16, 3, 2) for M in range(1, 12)},
    #    **{M: (16, 2, 2) for M in range(12, 17)},
    #},
    #(2048, 4096): {M: (16, 2, 2) for M in range(1, 33)},
    #(4096, 1024): {M: (8, 3, 4) for M in range(1, 33)},
    #(4096, 1536): {16: (16, 3, 4)},
}
# Shapes that remain faster than DeepGEMM through M=32 on SM100f.
_SM100F_M32_SHAPES = {
    (4096, 1536),
    (4096, 1024),
    (2048, 4096),
    (512, 4096),
}


# Called once per process.
def is_available() -> bool:
    global _cutedsl_available
    if _cutedsl_available is not None:
        return _cutedsl_available
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401

        _cutedsl_available = True
    except ImportError:
        _cutedsl_available = False
        logger.info(
            "cuteDSL (CUTLASS Python) not available, ll_fp8_block_gemm disabled"
        )
    return _cutedsl_available


class LLFp8BlockGemm:
    """Compile and launch the shape-dynamic low-latency FP8 GEMM."""

    @dataclass(frozen=True, slots=True)
    class CompileKey:
        tile_n: int = 16
        tile_k: int = 256  # bf16-view elements, equivalent to 512 FP8 elements
        num_stages: int = 2
        num_dma_warps: int = 4
        has_k_tail: bool = False

    def __init__(self) -> None:
        # Fully shape-dynamic; keyed only by the kernel configuration.
        self._compiled_cache: dict[LLFp8BlockGemm.CompileKey, Any] = {}

    @staticmethod
    def dispatch(*, M: int, K: int, N: int) -> CompileKey:
        tile_n, num_stages, num_dma_warps = _SM100F_TUNED_CONFIGS.get(
            (K, N), {}
        ).get(M, _DEFAULT_CONFIG)
        return LLFp8BlockGemm.CompileKey(
            tile_n=tile_n,
            num_stages=num_stages,
            num_dma_warps=num_dma_warps,
            has_k_tail=K % 512 != 0,
        )

    def get_warmup_keys(self, *, shapes: Iterable[tuple[int, int]]) -> list[CompileKey]:
        keys = []
        for K, N in shapes:
            max_m = 32 if (K, N) in _SM100F_M32_SHAPES else 16
            keys.extend(self.dispatch(M=M, K=K, N=N) for M in range(1, max_m + 1))
        return list(dict.fromkeys(keys))

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled_cache:
            return

        cute, _ = _cute()
        from cutlass import BFloat16, Int32

        from ._ll_fp8_block_warpspecialized import LLFp8BlockGemm as CuteGemm

        m = cute.sym_int()
        n = cute.sym_int(divisibility=8)
        k_divisibility = 64 if compile_key.has_k_tail else 256
        k = cute.sym_int(divisibility=k_divisibility)
        scale_k = cute.sym_int()
        scale_stride_a = cute.sym_int64(divisibility=4)
        # Share shape symbols with strides to preserve compact dynamic layouts.
        a = cute.runtime.make_fake_tensor(
            BFloat16, (m, k), stride=(k, 1), assumed_align=16
        )
        b = cute.runtime.make_fake_tensor(
            BFloat16, (n, k), stride=(k, 1), assumed_align=16
        )
        output = cute.runtime.make_fake_tensor(
            BFloat16, (m, n), stride=(n, 1), assumed_align=16
        )
        a_scale = cute.runtime.make_fake_tensor(
            Int32, (m, scale_k), stride=(1, scale_stride_a), assumed_align=4
        )
        b_scale = cute.runtime.make_fake_tensor(
            Int32, (n, scale_k), stride=(1, n), assumed_align=4
        )
        gemm = CuteGemm(
            tile_n=compile_key.tile_n,
            tile_k=compile_key.tile_k,
            num_stages=compile_key.num_stages,
            num_dma_warps=compile_key.num_dma_warps,
            use_pdl=_use_pdl(),
            has_k_tail=compile_key.has_k_tail,
        )
        self._compiled_cache[compile_key] = cute.compile(
            gemm,
            a,
            b,
            output,
            a_scale,
            b_scale,
            _stream(),
            options="--enable-tvm-ffi",
        )
        logger.debug("Compiled ll_fp8_block_gemm: %s", compile_key)

    def warmup(self, *, shapes: Iterable[tuple[int, int]]) -> None:
        for compile_key in self.get_warmup_keys(shapes=shapes):
            self.compile(compile_key)

    @staticmethod
    def _validate_inputs(
        q_input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[int, int, int]:
        if q_input.dim() != 2 or weight.dim() != 2:
            raise ValueError("q_input and weight must be 2D tensors")
        if q_input.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("q_input and weight must have dtype=float8_e4m3fn")
        if input_scale.dtype != torch.int32 or weight_scale.dtype != torch.int32:
            raise ValueError("input_scale and weight_scale must have dtype=int32")
        if output.dtype != torch.bfloat16:
            raise ValueError("ll_fp8_block_gemm requires output dtype=bfloat16")

        tensors = (q_input, input_scale, weight, weight_scale, output)
        if any(tensor.device.type != "cuda" for tensor in tensors):
            raise ValueError("all tensors must have device_type=cuda")
        if any(tensor.device != q_input.device for tensor in tensors[1:]):
            raise ValueError("all tensors must be on the same CUDA device")

        M, K = q_input.shape
        N, weight_k = weight.shape
        if weight_k != K:
            raise ValueError("q_input and weight must have matching K dimensions")
        if not 0 < M <= 32:
            raise ValueError("ll_fp8_block_gemm requires M to be in [1, 32]")
        if K <= 0 or K % 128 != 0:
            raise ValueError("ll_fp8_block_gemm requires K to be divisible by 128")
        if N <= 0 or N % 8 != 0:
            raise ValueError("ll_fp8_block_gemm requires N to be divisible by 8")
        if output.shape != (M, N):
            raise ValueError("output must have shape (M, N)")

        packed_k = (K + 511) // 512
        if input_scale.shape != (M, packed_k):
            raise ValueError("input_scale must have shape (M, ceil(K / 512))")
        if weight_scale.shape != (N, packed_k):
            raise ValueError("weight_scale must have shape (N, ceil(K / 512))")
        if input_scale.stride(0) != 1 or input_scale.stride(1) < M:
            raise ValueError("input_scale must use the packed column-major layout")
        if weight_scale.stride(0) != 1 or weight_scale.stride(1) < N:
            raise ValueError("weight_scale must use the packed column-major layout")
        if not q_input.is_contiguous() or not weight.is_contiguous():
            raise ValueError("q_input and weight must be contiguous row-major tensors")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous row-major")
        return M, K, N

    def __call__(
        self,
        q_input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        M, K, N = self._validate_inputs(
            q_input, input_scale, weight, weight_scale, output
        )
        compile_key = self.dispatch(M=M, K=K, N=N)
        self.compile(compile_key)
        self._compiled_cache[compile_key](
            q_input.view(torch.bfloat16),
            weight.view(torch.bfloat16),
            output,
            input_scale,
            weight_scale,
            _stream(),
        )


_cute_ctx = None


def _cute():
    global _cute_ctx
    if _cute_ctx is not None:
        return _cute_ctx
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream

    _cute_ctx = (cute, CUstream)
    return _cute_ctx


def _stream():
    _, CUstream = _cute()
    from vllm.utils.torch_utils import current_stream

    return CUstream(current_stream().cuda_stream)


def _use_pdl() -> bool:
    return current_platform.is_arch_support_pdl()


ll_fp8_block_gemm_kernel = LLFp8BlockGemm()


def _ll_fp8_block_gemm(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    ll_fp8_block_gemm_kernel(q_input, input_scale, weight, weight_scale, output)


class LLFp8BlockScaledMMKernel(DeepGemmFp8BlockScaledMMKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        if not is_available():
            return False, "CuTe DSL not available"
        if not current_platform.is_device_capability_family(100):
            return False, "requires the SM100 family"

        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        if not is_deep_gemm_e8m0_used():
            return False, "requires packed UE8M0 scales"
        if (
            config.activation_quant_key.dtype != torch.float8_e4m3fn
            or config.weight_quant_key.dtype != torch.float8_e4m3fn
        ):
            return False, "supports only E4M3 FP8 inputs and weights"

        can_base, reason = super().can_implement(config)
        if not can_base:
            return False, reason
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
        )
        torch.ops.vllm.ll_fp8_block_dispatch_op(
            A, As, B, Bs, output, self.use_deep_gemm_e8m0
        )
        return output


# ── Custom op (opaque to torch.compile) ───────────────────────────────


def _ll_fp8_block_dispatch(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    if weight.ndim != 2:
        from vllm.utils.deep_gemm import fp8_gemm_nt

        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )
        return
    M = q_input.shape[0]
    K_fp8 = q_input.shape[1]
    N = weight.shape[0]
    if M <= 16 and K_fp8 <= 4096 and K_fp8 % 256 == 0 and N <= 4096:
    # Keep automatic selection inside the measured SM100f crossover envelope.
    #max_work = 1 << (26 if N <= 4096 else 24)
    #max_m = 32 if (K_fp8, N) in _SM100F_M32_SHAPES else 16
    #if (
    #    use_deep_gemm_e8m0
    #    and 0 < M <= max_m
    #    and K_fp8 % 128 == 0
    #    and N % 8 == 0
    #    and K_fp8 <= 65536
    #    and N <= 8192
    #    and K_fp8 * N <= max_work
    #):
        _ll_fp8_block_gemm(q_input, input_scale, weight, weight_scale, output)
    else:
        from vllm.utils.deep_gemm import fp8_gemm_nt

        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )


def _ll_fp8_block_dispatch_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    return None


direct_register_custom_op(
    "ll_fp8_block_dispatch_op",
    _ll_fp8_block_dispatch,
    mutates_args=["output"],
    fake_impl=_ll_fp8_block_dispatch_fake,
)
