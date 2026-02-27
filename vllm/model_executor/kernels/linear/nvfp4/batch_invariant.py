# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.batch_invariant import (
    _compute_pid,
    _matmul_launch_metadata,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.platform_utils import num_compute_units

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


@triton.jit
def _unswizzle_scale(
    scale_raw,
    TILE_ROWS: tl.constexpr,
    TILE_SCALE_COLS: tl.constexpr,
):
    """Un-swizzle NVFP4 block scales from hardware-interleaved 128x4 layout
    to standard 2D row-major layout expected by tl.dot_scaled.

    The swizzled layout stores a (M, K_s) scale tensor as:
        (M//128, K_s//4, 32, 4, 4)
    The standard layout is:
        (M//128, 4, 32, K_s//4, 4)
    The inverse permutation (0, 3, 2, 1, 4) recovers the standard layout.
    """
    return (
        scale_raw.reshape(TILE_ROWS // 128, TILE_SCALE_COLS // 4, 32, 4, 4)
        .trans(0, 3, 2, 1, 4)
        .reshape(TILE_ROWS, TILE_SCALE_COLS)
    )


@triton.jit
def _maybe_widen(val, LARGE: tl.constexpr):
    """Promote *val* to int64 when LARGE (tensor exceeds int32 indexing)."""
    if LARGE:
        return val.to(tl.int64)
    return val


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_nvfp4_kernel_persistent(
    a_ptr,  # (M, K // 2), packed fp4
    a_scale_ptr,  # (M_padded, K_s_padded), fp8 block scales (swizzled)
    b_ptr,  # (N, K // 2), packed fp4
    b_scale_ptr,  # (N_padded, K_s_padded), fp8 block scales (swizzled)
    c_ptr,  # (M, N), output
    alpha_ptr,  # scalar fp32 global alpha
    bias_ptr,  # (N,), optional bias vector
    M,
    N,
    K,  # K in fp4 elements
    stride_am,
    stride_ak,
    stride_ask,
    stride_bm,
    stride_bk,
    stride_bsk,
    stride_cm,
    stride_cn,
    a_scale_cols_total,
    b_scale_cols_total,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    K_BYTES: tl.constexpr = BLOCK_SIZE_K // 2
    SCALE_K_TILE: tl.constexpr = BLOCK_SIZE_K // 16
    SCALE_K_TILES: tl.constexpr = SCALE_K_TILE // 4
    SCALE_M_TILES: tl.constexpr = BLOCK_SIZE_M // 128
    SCALE_N_TILES: tl.constexpr = BLOCK_SIZE_N // 128

    k_bytes = K // 2
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, k_bytes],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_SIZE_M, K_BYTES],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, k_bytes],
        strides=[stride_bm, stride_bk],
        block_shape=[BLOCK_SIZE_N, K_BYTES],
        padding_option="zero",
    )

    # Swizzled scale storage is logically 5D:
    # [m_tiles, k_tiles, 32, 4, 4] in row-major.
    # Reshape the trailing 32*16 = 512 elements into (2, 256) so the
    # innermost TMA dimension is 256 bytes, avoiding many small messages.
    a_scale_desc = tl.make_tensor_descriptor(
        a_scale_ptr,
        shape=[1, tl.cdiv(M, 128), a_scale_cols_total // 4, 2, 256],
        strides=[
            tl.cdiv(M, 128) * (a_scale_cols_total // 4) * 512 * stride_ask,
            (a_scale_cols_total // 4) * 512 * stride_ask,
            512 * stride_ask,
            256 * stride_ask,
            stride_ask,
        ],
        block_shape=[1, SCALE_M_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )
    b_scale_desc = tl.make_tensor_descriptor(
        b_scale_ptr,
        shape=[1, tl.cdiv(N, 128), b_scale_cols_total // 4, 2, 256],
        strides=[
            tl.cdiv(N, 128) * (b_scale_cols_total // 4) * 512 * stride_bsk,
            (b_scale_cols_total // 4) * 512 * stride_bsk,
            512 * stride_bsk,
            256 * stride_bsk,
            stride_bsk,
        ],
        block_shape=[1, SCALE_N_TILES, SCALE_K_TILES, 2, 256],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        padding_option="zero",
    )

    alpha = tl.load(alpha_ptr).to(tl.float32)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N

        # Scale offsets use block-aligned indices (scale tensors are pre-padded
        # to multiples of 128 rows and SCALE_K_TILE columns, so no masking).
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            k_start_bytes = _maybe_widen(ki * K_BYTES, A_LARGE or B_LARGE)
            a_m_start = _maybe_widen(start_m, A_LARGE)
            b_n_start = _maybe_widen(start_n, B_LARGE)

            a = a_desc.load([a_m_start, k_start_bytes])
            b = b_desc.load([b_n_start, k_start_bytes]).T

            scale_tile_k = _maybe_widen(ki * SCALE_K_TILES, A_LARGE or B_LARGE)
            scale_tile_m = _maybe_widen(start_m // 128, A_LARGE)
            scale_tile_n = _maybe_widen(start_n // 128, B_LARGE)

            a_scale_raw = a_scale_desc.load([0, scale_tile_m, scale_tile_k, 0, 0])
            b_scale_raw = b_scale_desc.load([0, scale_tile_n, scale_tile_k, 0, 0])

            a_scale = _unswizzle_scale(
                a_scale_raw.reshape(BLOCK_SIZE_M, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_M,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )
            b_scale = _unswizzle_scale(
                b_scale_raw.reshape(BLOCK_SIZE_N, SCALE_K_TILE),
                TILE_ROWS=BLOCK_SIZE_N,
                TILE_SCALE_COLS=SCALE_K_TILE,
            )

            accumulator = tl.dot_scaled(
                a,
                a_scale,
                "e2m1",
                b,
                b_scale,
                "e2m1",
                accumulator,
            )

        accumulator *= alpha
        if HAS_BIAS:
            bias_offsets = start_n + tl.arange(0, BLOCK_SIZE_N)
            bias_mask = bias_offsets < N
            bias_vec = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
            accumulator += bias_vec[None, :].to(tl.float32)
        c = accumulator.to(c_ptr.dtype.element_ty)
        c_desc.store([start_m, start_n], c)


def matmul_nvfp4_persistent(
    a_fp4: torch.Tensor,
    b_fp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 256

    M = a_fp4.shape[0]
    N = b_fp4.shape[0]
    K = a_fp4.shape[1] * 2
    NUM_SMS = num_compute_units(a_fp4.device.index)

    c = torch.empty((M, N), device=a_fp4.device, dtype=output_dtype)

    def grid(meta):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, meta["BLOCK_SIZE_M"])
                * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            ),
        )

    matmul_nvfp4_kernel_persistent[grid](
        a_fp4,
        a_scale,
        b_fp4,
        b_scale,
        c,
        alpha,
        bias,
        M,
        N,
        K,
        a_fp4.stride(0),
        a_fp4.stride(1),
        a_scale.stride(1),
        b_fp4.stride(0),
        b_fp4.stride(1),
        b_scale.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.shape[1],
        b_scale.shape[1],
        NUM_SMS=NUM_SMS,
        A_LARGE=a_fp4.numel() > 2**31,
        B_LARGE=b_fp4.numel() > 2**31,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        num_stages=2,
        num_warps=8,
    )
    return c


def linear_batch_invariant_nvfp4(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    output_size: int,
    bias: torch.Tensor | None = None,
    *,
    weights_padding_cols: int = 0,
    quant_backend: str = "cutlass",
) -> torch.Tensor:
    """
    Deterministic Blackwell NVFP4 linear path using Triton tl.dot_scaled.
    """
    assert hasattr(tl, "dot_scaled")
    assert weight.dtype == torch.uint8 and weight_scale.dtype == torch.float8_e4m3fn
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert alpha.dtype == torch.float32

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1]).contiguous()

    # Lazily import to avoid circular import with vllm._custom_ops.
    from vllm._custom_ops import scaled_fp4_quant

    x_fp4, x_blockscale = scaled_fp4_quant(
        input_2d,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
        backend=quant_backend,
    )
    if weights_padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()

    assert x_fp4.dtype == torch.uint8
    assert x_blockscale.dtype == torch.float8_e4m3fn
    assert x_fp4.shape[1] == weight.shape[1]

    out_2d = matmul_nvfp4_persistent(
        a_fp4=x_fp4,
        b_fp4=weight,
        a_scale=x_blockscale,
        b_scale=weight_scale,
        alpha=alpha,
        output_dtype=input.dtype,
        bias=bias,
    )
    out_2d = slice_nvfp4_output(out_2d, output_size)

    return out_2d.reshape(*original_shape[:-1], output_size)


class BatchInvariantNvFp4LinearKernel(NvFp4LinearKernel):
    """Deterministic Blackwell NVFP4 linear using Triton tl.dot_scaled.

    Uses the CUTLASS-compatible packed weight/scale layout but delegates
    the actual GEMM to the batch-invariant Triton persistent kernel so
    that results are independent of batch composition.
    """

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        set_triton_allocator(current_platform.current_device())

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.has_device_capability(100):
            return False, "Batch-invariant NVFP4 requires Blackwell (sm100+)"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return linear_batch_invariant_nvfp4(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_global_scale_inv=layer.input_global_scale_inv,
            alpha=layer.alpha,
            output_size=layer.output_size_per_partition,
            bias=bias,
            weights_padding_cols=getattr(layer, "weights_padding_cols", 0),
            quant_backend="cutlass",
        )
