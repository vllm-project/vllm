# Scale preshuffling on AMD GPUs
#
# Similar to NVIDIA GPUs, on AMD GPUs with CDNA4 architecture, scaled MFMA instructions natively
# support scaled matrix multiplication. Since it only supports OCP microscaling formats each
# scale is an 8-bit value that scales 32 elements from A or B operand tensors.
# Scales are stored as 8-bit tensors. Since MFMA instructions are warp-level instructions, that
# means that each thread provides a fixed set of operand values to MFMA instructions.
#
# For example, in an MFMA instruction with shape 16x16x128:
# - 4 threads contribute elements along the K dimension.
# - 16 threads contribute elements along the M or N dimension.
#
# From the perspective of the scales tensor, even if the K dimension is stored contiguously in
# shared memory, each thread sees its elements along K dim as strided due to interleaving with
# other threads. This striding limits the ability to load scale values using vectorized memory
# access.
#
# Our goal is to reorganize the scale tensor so that:
# 1. Each thread stores the 4 scale values it needs for 4 MFMA ops in contiguous memory.
# 2. Continuous threads access contiguous memory locations improving global memory coalescing when
# bypassing LDS, which is especially beneficial for "skinny" matmuls.
#
# We consider two MFMA cases: one with non-K dimension 16, and one with 32.
# In both, the minimum tile size for preshuffling is 32x32x256.
# For example, for a 32x256 operand tile, the corresponding scale tensor has shape 32x8,
# where each scale covers 32 elements along the K dimension.
#
# Each thread holds one scale per MFMA operation. We pack the 4 scale values
# (for 4 different MFMA ops) next to each other in memory.
#
# Case 1: mfma_scaled_16x16x128
#
# Packing order: mfma_op_0, mfma_op_2, mfma_op_1, mfma_op_3
#
#            K = 128       K = 128
#        +------------+ +------------+
#    M=16|  MFMA op 0 | |  MFMA op 1 |
#        +------------+ +------------+
#    M=16|  MFMA op 2 | |  MFMA op 3 |
#        +------------+ +------------+
#
# Case 2: mfma_scaled_32x32x64
#
# Packing order: mfma_op_0, mfma_op_1, mfma_op_2, mfma_op_3
#
#            K=64     K=64     K=64     K=64
#        +--------+ +--------+ +--------+ +--------+
#    M=32| op 0   | | op 1   | | op 2   | | op 3   |
#        +--------+ +--------+ +--------+ +--------+

import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def supports_block_scaling():
    return (is_cuda() and torch.cuda.get_device_capability()[0] == 10) or is_hip_cdna4()


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.bfloat16
    elif output_type == 3:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.bfloat16:
        dtype_dst = 2
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 3
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    # Similar to Hopper's wgmma symmetric fp8 instruction, the RHS is expected
    # to be in col-major layout for Blackwell's tcgen05.mma when using fp4 operands.
    # To conform to the expected semantics of tl.dot_scaled, (M, K) x (K, N),
    # the data is generated in col-major layout, packed along K for fp4, and then
    # logically transposed. Note that if one operand is of fp8 precision, unlike Hopper,
    # Blackwell supports both row-major and col-major layouts for the RHS matrix.
    # For the mixed-precision case, the fp4 RHS can be either in row or col-major layout.
    # But for performance reason, it is recommended to use col-major layout. If TMA is used
    # for the fp4 RHS operand load in mixed-precision dot, as in this tutorial, it must be
    # in col-major layout.
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        # Pack two fp4 elements per byte along K
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]
    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # Use 5D TMA descriptor [1, rep_m, rep_k, 2, 256] with uint8 elements.
    # With 256 elements we better utilize the L2 and don't require the TMA
    # engine to emit many small messages (16B) messages as with 32x16xu8.
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]
    # a_scale = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    # b_scale = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale = a_scale.view(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.view(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    reference = None
    if compute_reference:
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            # [M // 128, K // VEC_SIZE // 4, 32, 16]
            # [M // 128, K // VEC_SIZE // 4, 32, 4, 4]
            # [N // 128, 4, 32, K // VEC_SIZE // 4, 4]
            packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)


    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    return a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference


def validate_block_scaled(M, N, K, block_scale_type="nvfp4"):
    a_desc, a_scale, b_desc, b_scale, rep_m, rep_n, rep_k, configs, reference = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=True)
    output = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float32, M, N, K, rep_m, rep_n, rep_k, configs)
    print(f"{torch.max(torch.abs(output - reference))=}")
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type})")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    a_desc, a_scale, b_desc, b_scale, rep_m, rep_n, rep_k, configs, _ = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=False)
    
    # Warmup
    for _ in range(10):
        _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)

    # Combined proton profiling and CUDA event timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    proton.deactivate(0)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / reps
    flops = 2.0 * M * N * K
    tflops = (flops / avg_time_ms) / 1e9  # TFLOPs/s
    
    print(f"Done benchmarking: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOP/s")
    
    return {"K": K, "format": block_scale_type, "time_ms": avg_time_ms, "tflops": tflops}


def bench_bf16(K, reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}, BF16 Matmul")

    A = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    B = torch.randn(K, N, device="cuda").to(torch.bfloat16)

    # Warmup
    for _ in range(10):
        torch.matmul(A, B)
    
    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(reps):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / reps
    flops = 2.0 * M * N * K
    tflops = (flops / avg_time_ms) / 1e9  # TFLOPs/s
    
    print(f"BF16 Matmul: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOP/s")
    print("Done benchmarking BF16 Matmul")
    
    # Store results for unified report
    return {"K": K, "format": "bf16", "time_ms": avg_time_ms, "tflops": tflops}


def bench_fp8(K, reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}, FP8 Matmul")

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(N, K, device="cuda").t()

    def _to_float8(x, dtype=torch.float8_e4m3fn):
        finfo = torch.finfo(dtype)
        # Calculate the scale as dtype max divided by absmax
        scale = finfo.max / x.abs().max().clamp(min=1e-12)
        # scale and clamp the tensor to bring it to
        # the representative range of float8 data type
        # (as default cast is unsaturated)
        x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
        # Return both float8 data and the inverse scale (as float),
        # as both required as inputs to torch._scaled_mm
        return x_scl_sat.to(dtype), scale.float().reciprocal()

    A_fp8, A_scale_inv = _to_float8(A)
    B_fp8, B_scale_inv = _to_float8(B)

    # Warmup
    for _ in range(10):
        torch._scaled_mm(A_fp8, B_fp8, out_dtype=torch.float16,
                              scale_a=A_scale_inv, scale_b=B_scale_inv)
    
    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(reps):
        C = torch._scaled_mm(A_fp8, B_fp8, out_dtype=torch.float16,
                              scale_a=A_scale_inv, scale_b=B_scale_inv)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / reps
    flops = 2.0 * M * N * K
    tflops = (flops / avg_time_ms) / 1e9  # TFLOPs/s
    
    print(f"FP8 Matmul: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOP/s")
    print("Done benchmarking FP8 Matmul")
    
    # Store results for unified report
    return {"K": K, "format": "fp8", "time_ms": avg_time_ms, "tflops": tflops}


def show_profile(profile_name, mx_results=None, bf16_results=None, fp8_results=None):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)
    
    # Display unified performance comparison table
    if mx_results or bf16_results or fp8_results:
        # Determine the MX format name from the first result
        mx_format_name = "MX"
        if mx_results and len(mx_results) > 0:
            mx_format_name = mx_results[0]['format'].upper()
        
        mx_header = f"{mx_format_name} TFLOP/s"
        
        print("\n" + "="*110)
        print("PERFORMANCE COMPARISON (CUDA Event Timing)")
        print("="*110)
        print(f"{'K':<10} {'BF16 TFLOP/s':<20} {'FP8 TFLOP/s':<20} {mx_header:<20} {'vs BF16':<15} {'vs FP8':<15}")
        print("-"*110)
        
        # Create a dict to group results by K for easy comparison
        results_by_k = {}
        
        if bf16_results:
            for result in bf16_results:
                k = result['K']
                if k not in results_by_k:
                    results_by_k[k] = {}
                results_by_k[k]['bf16'] = result
        
        if fp8_results:
            for result in fp8_results:
                k = result['K']
                if k not in results_by_k:
                    results_by_k[k] = {}
                results_by_k[k]['fp8'] = result
        
        if mx_results:
            for result in mx_results:
                k = result['K']
                if k not in results_by_k:
                    results_by_k[k] = {}
                fmt = result['format']
                results_by_k[k][fmt] = result
        
        # Print results organized by K
        for k in sorted(results_by_k.keys()):
            k_results = results_by_k[k]
            
            bf16_tflops = k_results.get('bf16', {}).get('tflops', None)
            fp8_tflops = k_results.get('fp8', {}).get('tflops', None)
            
            # Print MX format results
            for fmt in sorted(k_results.keys()):
                if fmt in ['bf16', 'fp8']:
                    continue
                mx_res = k_results[fmt]
                bf16_str = f"{bf16_tflops:.2f}" if bf16_tflops else "N/A"
                fp8_str = f"{fp8_tflops:.2f}" if fp8_tflops else "N/A"
                speedup_bf16 = f"{mx_res['tflops']/bf16_tflops:.2f}x" if bf16_tflops else "N/A"
                speedup_fp8 = f"{mx_res['tflops']/fp8_tflops:.2f}x" if fp8_tflops else "N/A"
                print(f"{k:<10} {bf16_str:<20} {fp8_str:<20} {mx_res['tflops']:<20.2f} {speedup_bf16:<15} {speedup_fp8:<15}")
        
        print("="*110)


@triton.jit
def block_scaled_matmul_kernel_cdna4(a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K, stride_am, stride_ak,
                                     stride_bk, stride_bn, stride_ck, stride_cm, stride_cn, stride_asm, stride_ask,
                                     stride_bsn, stride_bsk,
                                     # Meta-parameters
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                     mfma_nonkdim: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # Create pointers for first block of A and B input matrices
    # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Create pointers for the first block of A and B scales
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, (BLOCK_N // 32))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # B scales are N x K even though B operand is K x N.
    b_scale_ptrs = (b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk)
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, (BLOCK_M // 32))) % M
    a_scale_ptrs = (a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # Here we "undo" the shuffle done in global memory (shuffle_scales_cdna4 function).
        if mfma_nonkdim == 32:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        elif mfma_nonkdim == 16:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # Advance the ptrs to the next K block.
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


def shuffle_scales_cdna4(scales: torch.Tensor, mfma_nonkdim: int):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape

    if mfma_nonkdim == 32:
        scales_shuffled = scales_shuffled.view(sm // 32, 32, sn // 8, 4, 2, 1)
        scales_shuffled = scales_shuffled.permute(0, 2, 4, 1, 3, 5).contiguous()
    elif mfma_nonkdim == 16:
        scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()

    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def initialize_block_scaled_amd(M, N, K, mfma_nonkdim):

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    configs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "num_stages": 2,
        "num_warps": 8,
        "mfma_nonkdim": mfma_nonkdim,
    }

    torch.manual_seed(5)

    x = MXFP4Tensor(size=(M, K), device="cuda").random()
    w = MXFP4Tensor(size=(N, K), device="cuda").random()

    x_scales = torch.randint(124, 128, (K // 32, M), dtype=torch.uint8, device="cuda")
    w_scales = torch.randint(124, 128, (K // 32, N), dtype=torch.uint8, device="cuda")
    x_scales = x_scales.T
    w_scales = w_scales.T
    x_scales_shuffled = shuffle_scales_cdna4(x_scales, configs["mfma_nonkdim"])
    w_scales_shuffled = shuffle_scales_cdna4(w_scales, configs["mfma_nonkdim"])

    return (
        x,
        w,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        configs,
    )


def validate_block_scaled_amd(M, N, K, block_scale_type="mxfp4", mfma_nonkdim=16):

    def e8m0_to_f32(x):
        x_f32 = 2**((x - 127).to(torch.float32))
        x_f32[x_f32 == 128] = float("nan")
        return x_f32

    def run_torch(x, w, x_scales, w_scales, dtype):
        # First convert the x and w inputs to f32.
        x_f32 = x.to(torch.float32)
        w_f32 = w.to(torch.float32)
        # Next convert the e8m0 scales to f32.
        x_scales = x_scales.repeat_interleave(32, dim=1).to(torch.float32)
        x_scales_f32 = e8m0_to_f32(x_scales)
        x_f32 = x_f32 * x_scales_f32
        w_scales = w_scales.repeat_interleave(32, dim=1).to(torch.float32)
        w_scales_f32 = e8m0_to_f32(w_scales)
        w_f32 = w_f32 * w_scales_f32
        return torch.mm(x_f32, w_f32.T).to(dtype)

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton, configs = \
    initialize_block_scaled_amd(M, N, K, mfma_nonkdim)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    triton_out = torch.empty((M, N), device=x.device)
    triton_out = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)
    triton_out = triton_out.to(torch.float32)

    torch_out = run_torch(x_mxfp4, w_mxfp4, x_scales, w_scales, torch.float32)
    torch.testing.assert_close(torch_out, triton_out)
    print(f"✅ (pass {block_scale_type}, mfma_nonk_dim {mfma_nonkdim})")


def block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs):
    M, K = x.shape
    N, K = w.shape
    w = w.T
    triton_out = torch.empty((M, N), device=x.device)

    kernel_kwargs = {}
    kernel_kwargs["matrix_instr_nonkdim"] = configs["mfma_nonkdim"]

    BLOCK_M = configs["BLOCK_M"]
    BLOCK_N = configs["BLOCK_N"]

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    triton_out = torch.empty((M, N), device="cuda")

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel_cdna4[grid](x, w, triton_out, x_scales_triton, w_scales_triton, M, N, K, x.stride(0),
                                           x.stride(1), w.stride(0), w.stride(1), 0, triton_out.stride(0),
                                           triton_out.stride(1), x_scales_triton.stride(0), x_scales_triton.stride(1),
                                           w_scales_triton.stride(0), w_scales_triton.stride(1), BLOCK_M, BLOCK_N,
                                           configs["BLOCK_K"], configs["mfma_nonkdim"], num_warps=configs["num_warps"],
                                           num_stages=configs["num_stages"], **kernel_kwargs)
    triton_out = triton_out.to(torch.float32)

    return triton_out


def bench_block_scaled_amd(K, block_scale_type="mxfp4", reps=10, mfma_nonkdim=16):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton, configs = \
    initialize_block_scaled_amd(M, N, K, mfma_nonkdim)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    # Warmup
    for _ in range(10):
        _ = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)

    # Combined proton profiling and CUDA event timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)
    proton.deactivate(0)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / reps
    flops = 2.0 * M * N * K
    tflops = (flops / avg_time_ms) / 1e9  # TFLOPs/s
    
    print(f"Done benchmarking: {avg_time_ms:.3f} ms, {tflops:.2f} TFLOP/s")
    
    return {"K": K, "format": f"{block_scale_type}_mfma{mfma_nonkdim}", "time_ms": avg_time_ms, "tflops": tflops}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true", default=True)
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="mxfp8")
    parser.add_argument("--reps", type=int, default=10000)
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(42)

        if is_cuda():
            validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format)
        elif is_hip_cdna4():
            assert args.format == "mxfp4", "AMD tutorial only supports mxpf4 format currently"
            validate_block_scaled_amd(8192, 8192, 8192, block_scale_type=args.format, mfma_nonkdim=16)
            validate_block_scaled_amd(8192, 8192, 8192, block_scale_type=args.format, mfma_nonkdim=32)

        if args.bench:
            bf16_results = []
            fp8_results = []
            mx_results = []
            proton.start("block_scaled_matmul", hook="triton")
            proton.deactivate(0)  # Skip argument creation
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                if is_cuda():
                    mx_result = bench_block_scaled(K, reps=args.reps, block_scale_type=args.format)
                    mx_results.append(mx_result)
                elif is_hip_cdna4():
                    mx_result_16 = bench_block_scaled_amd(K, reps=args.reps, block_scale_type=args.format, mfma_nonkdim=16)
                    mx_result_32 = bench_block_scaled_amd(K, reps=args.reps, block_scale_type=args.format, mfma_nonkdim=32)
                    mx_results.extend([mx_result_16, mx_result_32])
                bf16_results.append(bench_bf16(K, reps=args.reps))
                fp8_results.append(bench_fp8(K, reps=args.reps))
            proton.finalize()
            show_profile("block_scaled_matmul", mx_results=mx_results, bf16_results=bf16_results, fp8_results=fp8_results)