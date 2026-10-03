# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of NVFP4 GEMM (FP4 x FP4 -> BF16/FP16).

Uses tl.dot_scaled for block-scaled FP4 matrix multiplication on Blackwell
(SM100 and SM120). This is a reference implementation for benchmarking
against the CUTLASS NVFP4 GEMM in benchmark_nvfp4_gemm.py (issue #21014);
it is not a dispatch backend and nothing in vllm/ calls it.

Data format (NVFP4):
  - FP4 values: E2M1 format, 2 packed per uint8 byte along K dimension
  - Block scales: float8_e4m3fn, 1 scale per 16 FP4 elements
  - Global scale: float32 scalar
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize_on_alignment=["alpha_ptr"])
def _triton_nvfp4_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointers to block scales
    a_scale_ptr,
    b_scale_ptr,
    # Pointer to the global alpha = 1/(global_scale_a * global_scale_b),
    # a single float32 on the device (read in the kernel, no host sync)
    alpha_ptr,
    # Matrix dimensions
    M,
    N,
    K,  # logical K (unpacked)
    # Strides (in elements, accounting for packing)
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Scale strides
    stride_a_scale_m,
    stride_b_scale_n,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # NVFP4 block scale group size (elements per scale)
    VEC_SIZE: tl.constexpr,
    # True when K is a multiple of BLOCK_K, so no K step needs a mask
    EVEN_K: tl.constexpr,
    # Rows of output tiles per band in the launch order (see below)
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton kernel for NVFP4 GEMM: C = alpha * (A @ B^T).

    A is [M, K//2] uint8 (packed FP4, row-major)
    B is [N, K//2] uint8 (packed FP4, row-major, weight format)
    C is [M, N] output (bf16/fp16)
    a_scale is [M, K//VEC_SIZE] float8_e4m3fn (linear layout)
    b_scale is [N, K//VEC_SIZE] float8_e4m3fn (linear layout)

    Computes C = alpha * dot_scaled(A, B^T) where dot_scaled handles
    block-scale dequantization internally via tensor core instructions.
    """
    # Map the program id to an output tile in bands of GROUP_SIZE_M tile rows:
    # go down the rows of one band, step one column right, repeat across all
    # columns, then move to the next band. The programs running at the same
    # time then share a few row strips of A and a few column strips of B, so
    # both stay in L2 instead of A being re-read once per tile column.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    # The last band can have fewer than GROUP_SIZE_M tile rows.
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row and column indices used for loads. Indices past M or N wrap back
    # into range, so every load reads a real row of A or B. The rows they
    # produce are thrown away by the masked store at the end.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # K is packed: 2 FP4 values per uint8 byte
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    offs_k = tl.arange(0, BLOCK_K_PACKED)

    # Scale offsets: one scale per VEC_SIZE elements along K
    SCALE_K: tl.constexpr = BLOCK_K // VEC_SIZE
    offs_scale_k = tl.arange(0, SCALE_K)

    # Initialize pointers for A [M, K//2] and B^T [K//2, N]
    # A is row-major: a[m, k] = a_ptr + m * stride_am + k * stride_ak
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B is [N, K//2] row-major, but we want B^T [K//2, N]
    # b^T[k, n] = b[n, k] = b_ptr + n * stride_bn + k * stride_bk
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Scale pointers
    # a_scale [M, K//VEC_SIZE]: a_scale[m, g] = a_scale_ptr + m * stride + g
    a_scale_ptrs = (
        a_scale_ptr + offs_m[:, None] * stride_a_scale_m + offs_scale_k[None, :]
    )
    # b_scale [N, K//VEC_SIZE]: b_scale[n, g] = b_scale_ptr + n * stride + g
    b_scale_ptrs = (
        b_scale_ptr + offs_n[:, None] * stride_b_scale_n + offs_scale_k[None, :]
    )

    # Accumulator in float32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_iter in tl.range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            # Every K step is a full tile
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            scale_a = tl.load(a_scale_ptrs)
            scale_b = tl.load(b_scale_ptrs)
        else:
            # The last K step may be partial. Positions past the end of a row
            # load as zero, for both the packed values and their scales. The
            # scales must be masked too: a stray scale byte can be NaN, and
            # NaN times a zeroed value is still NaN.
            k_left = K - k_iter * BLOCK_K  # logical elements left in the row
            k_packed_ok = offs_k < k_left // 2
            k_scale_ok = offs_scale_k < k_left // VEC_SIZE
            a = tl.load(a_ptrs, mask=k_packed_ok[None, :], other=0)
            b = tl.load(b_ptrs, mask=k_packed_ok[:, None], other=0)
            scale_a = tl.load(a_scale_ptrs, mask=k_scale_ok[None, :], other=0.0)
            scale_b = tl.load(b_scale_ptrs, mask=k_scale_ok[None, :], other=0.0)

        # Block-scaled FP4 dot product
        # tl.dot_scaled handles: dequant(a, scale_a) @ dequant(b, scale_b)
        accumulator = tl.dot_scaled(
            a,
            scale_a,
            "e2m1",
            b,
            scale_b,
            "e2m1",
            accumulator,
            lhs_k_pack=True,
            rhs_k_pack=True,
        )

        # Advance pointers
        a_ptrs += BLOCK_K_PACKED * stride_ak
        b_ptrs += BLOCK_K_PACKED * stride_bk
        a_scale_ptrs += SCALE_K
        b_scale_ptrs += SCALE_K

    # Apply global scale: C = alpha * accumulator
    alpha = tl.load(alpha_ptr)
    c = (accumulator * alpha).to(c_ptr.dtype.element_ty)

    # Store output with bounds checking
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # int64 row offset: stride_cm * row passes 2**31 once M * N does.
    c_ptrs = (
        c_ptr + offs_cm[:, None].to(tl.int64) * stride_cm + offs_cn[None, :] * stride_cn
    )
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


# Tile configs for autotuning, chosen by a two-pass sweep on an RTX 5090 (SM120):
# 571 configs that fit, then the 143 closest to the best re-timed at every M,
# over 10 GEMM shapes (the Llama-3.1-8B and Llama-3.3-70B linear layers at TP1,
# plus two with K not a multiple of 128) at 11 M values from 1 to 16384.
# Autotuning over this list reaches a geometric mean of 0.971 of the best config
# per case, and at least 0.909 in every case. Small BLOCK_M entries serve
# decode-sized M. Configs that do not fit a GPU's shared memory are pruned per
# device by _prune_configs. Not yet swept on SM100 (B200), where only the entries
# with BLOCK_M >= 128 compile (see nvfp4_gemm_min_block_m).
NVFP4_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 32},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 32},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_SIZE_M": 1},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_SIZE_M": 1},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_SIZE_M": 1},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_SIZE_M": 8},
        num_warps=4,
        num_stages=4,
    ),
]


def nvfp4_gemm_smem_bytes(bm: int, bn: int, bk: int, stages: int) -> int:
    """Estimate the shared memory one config needs for its pipelined loads.

    Triton's pipeliner gives each pipelined load num_stages - 1 buffers. This
    is an estimate, not the compiler's count: it leaves out epilogue
    layout-conversion scratch. It only skips configs that clearly cannot fit;
    a config the compiler finds too large raises OutOfResources, which the
    autotuner scores as inf.

    Args:
        bm: BLOCK_M.
        bn: BLOCK_N.
        bk: BLOCK_K, in unpacked FP4 elements.
        stages: num_stages.

    Returns:
        Bytes of shared memory: packed A and B tiles (two FP4 values per
        byte) plus one e4m3 scale per 16 values, per buffer, times
        max(num_stages - 1, 1).

    """
    per_stage = bm * bk // 2 + bn * bk // 2 + (bm + bn) * (bk // 16)
    return max(stages - 1, 1) * per_stage


def nvfp4_gemm_min_block_m(device_index: int) -> int:
    """Return the smallest BLOCK_M that this kernel compiles for on a GPU.

    On SM100 (B200) with Triton 3.7.1, BLOCK_M of 16, 32 and 64 fail to compile
    ("PassManager::run failed") for every BLOCK_N, BLOCK_K, num_warps and
    num_stages tried, while 128 and 256 compile and match the reference. SM120
    compiles BLOCK_M of 16 and up. Other architectures are not measured and get
    the SM120 bound.

    Args:
        device_index: CUDA device index of the output tensor.

    Returns:
        128 on compute capability 10.x, otherwise 16.

    """
    capability = current_platform.get_device_capability(device_index)
    if capability is not None and capability.major == 10:
        return 128
    return 16


def _prune_configs(configs, named_args, **kwargs):
    """Drop configs that cannot fit this GPU or that are far larger than M/N.

    Keeping BLOCK_M no larger than the smallest power of two >= M (at least the
    GPU's minimum BLOCK_M) avoids tuning 128-row tiles for M=1 where smaller
    tiles compile, and likewise for N. Configs below the GPU's minimum BLOCK_M
    are dropped, since they do not compile there. If nothing is left, the
    config with the smallest tile that meets the minimum is kept so a launch
    always has a candidate.
    """
    m, n = named_args["M"], named_args["N"]
    # Shared memory of the GPU the output tensor lives on.
    props = triton.runtime.driver.active.utils.get_device_properties(
        named_args["c_ptr"].device.index
    )
    max_smem = props["max_shared_mem"]
    min_bm = nvfp4_gemm_min_block_m(named_args["c_ptr"].device.index)
    m_cap = max(min_bm, triton.next_power_of_2(m))
    n_cap = max(16, triton.next_power_of_2(n))
    kept = []
    for cfg in configs:
        kw = cfg.kwargs
        fits = (
            nvfp4_gemm_smem_bytes(
                kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"], cfg.num_stages
            )
            <= max_smem
        )
        if fits and min_bm <= kw["BLOCK_M"] <= m_cap and kw["BLOCK_N"] <= n_cap:
            kept.append(cfg)
    if kept:
        return kept
    compilable = [c for c in configs if c.kwargs["BLOCK_M"] >= min_bm] or configs
    # Smallest tile first (M, then N), then the least shared memory.
    return [
        min(
            compilable,
            key=lambda c: (
                c.kwargs["BLOCK_M"],
                c.kwargs["BLOCK_N"],
                nvfp4_gemm_smem_bytes(
                    c.kwargs["BLOCK_M"],
                    c.kwargs["BLOCK_N"],
                    c.kwargs["BLOCK_K"],
                    c.num_stages,
                ),
            ),
        )
    ]


# EVEN_K depends on the chosen BLOCK_K, so it is derived per config.
_nvfp4_gemm_kernel = triton.heuristics(
    {"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0}
)(_triton_nvfp4_gemm_kernel)

_nvfp4_gemm_kernel_autotuned = triton.autotune(
    configs=NVFP4_GEMM_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)(_nvfp4_gemm_kernel)


def triton_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: float | torch.Tensor,
    out_dtype: torch.dtype,
    config: dict | None = None,
) -> torch.Tensor:
    """Triton-based NVFP4 GEMM: C = alpha * dequant(A) @ dequant(B)^T.

    Args:
        a:       Packed FP4 activations [M, K//2] uint8
        b:       Packed FP4 weights    [N, K//2] uint8
        a_scale: Block scales for A    [M, K//16] float8_e4m3fn (linear layout)
        b_scale: Block scales for B    [N, K//16] float8_e4m3fn (linear layout)
        alpha:   Global scale = 1 / (global_scale_a * global_scale_b)
        out_dtype: Output dtype (torch.bfloat16 or torch.float16)
        config:  Optional fixed tile config with keys BLOCK_M, BLOCK_N,
            BLOCK_K, GROUP_SIZE_M, num_warps and num_stages. When None
            (the default), the config is autotuned over NVFP4_GEMM_CONFIGS
            for each (M, N, K). A fixed config is for sweeps and tests.

    Returns:
        C: [M, N] tensor in out_dtype

    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.dtype == torch.uint8 and b.dtype == torch.uint8

    M, K_packed = a.shape
    N, K_packed_b = b.shape
    assert K_packed == K_packed_b, (
        f"K dimension mismatch: A has {K_packed}, B has {K_packed_b}"
    )
    K = K_packed * 2  # Logical K (unpacked)

    # Validate scale shapes
    VEC_SIZE = 16  # NVFP4 block scale group size
    # A partial scale group would get no scale and drop up to 15 K values.
    assert K % VEC_SIZE == 0, f"K={K} must be a multiple of {VEC_SIZE}"
    assert a_scale.shape == (M, K // VEC_SIZE), (
        f"a_scale shape {a_scale.shape} != expected ({M}, {K // VEC_SIZE})"
    )
    assert b_scale.shape == (N, K // VEC_SIZE), (
        f"b_scale shape {b_scale.shape} != expected ({N}, {K // VEC_SIZE})"
    )
    # The kernel reads each row of scales with unit stride along K.
    assert a_scale.shape[1] <= 1 or a_scale.stride(1) == 1, "a_scale K stride != 1"
    assert b_scale.shape[1] <= 1 or b_scale.stride(1) == 1, "b_scale K stride != 1"

    # The kernel reads alpha from device memory, so no path here syncs the
    # host and the call can be captured in a CUDA graph. A CUDA tensor is
    # passed through, so a graph replay reads its current value. A float or
    # CPU tensor is written by a device fill (torch.tensor(x, device=...) is a
    # blocking host-to-device copy), so under capture it is fixed at capture.
    if isinstance(alpha, torch.Tensor):
        assert alpha.numel() == 1, f"alpha must have one element, got {alpha.numel()}"
    if isinstance(alpha, torch.Tensor) and alpha.is_cuda:
        alpha_t = alpha.to(device=a.device, dtype=torch.float32)
    else:
        alpha_t = torch.full((), float(alpha), device=a.device, dtype=torch.float32)

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # One program per output tile; the tile size comes from the config.
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    # Strides for A [M, K//2] — row-major
    stride_am = a.stride(0)
    stride_ak = a.stride(1)

    # Strides for B [N, K//2] — row-major
    # For B^T access: b^T[k, n] = b[n, k]
    stride_bn = b.stride(0)  # stride to move along N (rows of B)
    stride_bk = b.stride(1)  # stride to move along K (cols of B)

    # Scale strides
    stride_a_scale_m = a_scale.stride(0)
    stride_b_scale_n = b_scale.stride(0)

    args = (
        a,
        b,
        c,
        a_scale,
        b_scale,
        alpha_t,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        stride_a_scale_m,
        stride_b_scale_n,
    )
    if config is None:
        _nvfp4_gemm_kernel_autotuned[grid](*args, VEC_SIZE=VEC_SIZE)
    else:
        assert config["BLOCK_K"] % (2 * VEC_SIZE) == 0, (
            f"BLOCK_K={config['BLOCK_K']} must be a multiple of {2 * VEC_SIZE}"
        )
        min_bm = nvfp4_gemm_min_block_m(a.device.index)
        if config["BLOCK_M"] < min_bm:
            raise ValueError(
                f"BLOCK_M={config['BLOCK_M']} does not compile on this GPU; "
                f"use BLOCK_M >= {min_bm}"
            )
        _nvfp4_gemm_kernel[grid](*args, VEC_SIZE=VEC_SIZE, **config)

    return c
