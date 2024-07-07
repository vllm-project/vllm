"""This file is used for /tests and /benchmarks"""
import random
from typing import Optional

import numpy
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.format_24 import (
    mask_creator, sparse_semi_structured_from_dense_cutlass)
from vllm.model_executor.layers.quantization.utils.marlin_24_perms import (
    marlin_24_perm, marlin_24_scale_perm, marlin_24_scale_perm_single)
from vllm.model_executor.layers.quantization.utils.marlin_perms import (
    marlin_perm, marlin_scale_perm, marlin_scale_perm_single)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_pack_factor, quantize_weights, sort_weights)
from vllm.platforms import current_platform
from vllm.utils import print_warning_once

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

GPTQ_MARLIN_SUPPORTED_NUM_BITS = [4, 8]
GPTQ_MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_SUPPORTED_SYM = [True]


def is_marlin_supported():
    capability = current_platform.get_device_capability()
    return capability[0] >= 8


def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n, )

    output = ops.fp8_marlin_gemm(
        a=reshaped_x,
        b_q_weight=weight,
        b_scales=weight_scale,
        workspace=workspace,
        num_bits=8,
        size_m=reshaped_x.shape[0],
        size_n=size_n,
        size_k=size_k,
    )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def prepare_fp8_layer_for_marlin(layer: torch.nn.Module) -> None:
    print_warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads.")

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition

    device = layer.weight.device

    # WEIGHTS
    # Repack weights to gptq format (packed int32 elements)
    packed_gptq_qweight = pack_fp8_to_int32(layer.weight)

    # Repack weights to marlin format
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=packed_gptq_qweight,
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Currently Marlin doesn't support per-tensor scales, so we
    # expand it to channelwise
    scales = layer.weight_scale.repeat(1, part_size_n).to(
        layer.orig_dtype).to(device)
    # Permute scales
    num_bits = 8
    marlin_scales = marlin_permute_scales(
        s=scales,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=-1,
        scale_perm=marlin_scale_perm[num_bits],
        scale_perm_single=marlin_scale_perm_single[num_bits])
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)

    # Allocate marlin workspace
    max_workspace_size = (part_size_n //
                          GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL
    workspace = torch.zeros(max_workspace_size,
                            dtype=torch.int,
                            device=device,
                            requires_grad=False)

    layer.workspace = workspace


def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_packed = numpy.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=numpy.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(numpy.int32)).to(orig_device)

    return q_packed


def marlin_permute_scales(s, size_k, size_n, group_size, scale_perm,
                          scale_perm_single):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
    act_order: bool,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = quantize_weights(w, num_bits, group_size,
                                                       act_order)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits,
                                marlin_perm[num_bits])
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size,
                                     marlin_scale_perm[num_bits],
                                     marlin_scale_perm_single[num_bits])

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def inject_24(w, size_k, size_n):
    assert w.shape == (size_k, size_n)

    mask = mask_creator(w.t()).t().cuda().bool()

    return (mask * w).contiguous(), mask.contiguous()


def check_24(w, num_rows_to_sample=50, _verbose=False):
    BLOCK_SIZE = 4
    MAX_NON_ZEROS = 2

    w = w.t().contiguous()

    print("check_24: w.shape = {}".format(w.shape))

    num_rows, num_cols = w.shape
    sampled_row_idxs = random.choices(range(num_rows), k=num_rows_to_sample)
    if _verbose:
        print(f"Sampled row idxs = {sampled_row_idxs}")

    total_segments = 0
    non_24_segments = 0
    for i in sampled_row_idxs:
        for j in range(0, num_cols - BLOCK_SIZE, BLOCK_SIZE):
            total_segments += 1
            block = w[i, j:j + BLOCK_SIZE]
            num_nonzero = torch.count_nonzero(block)
            if num_nonzero > MAX_NON_ZEROS:
                print("i = {} j = {} block = {}".format(i, j, block))
                non_24_segments += 1

    print(f"{non_24_segments} / {total_segments} do not have 2:4 structure.")


def compress_quantized_24_weight(q_24, size_k, size_n, num_bits):
    assert q_24.shape == (size_k, size_n)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp = q_24 - zp

    # Compress
    q_24_no_zp = q_24_no_zp.t().contiguous()
    q_24_no_zp_comp, meta = sparse_semi_structured_from_dense_cutlass(
        q_24_no_zp)
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()

    # Restore zp
    q_24_comp = q_24_no_zp_comp + zp

    # Resize meta to its actual shape (without moving any data)
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    return q_24_comp, meta


def marlin_24_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Inject 2:4 sparsity
    w_24, mask_24 = inject_24(w, size_k, size_n)

    # Quantize
    w_24_ref, q_w_24, s, g_idx, rand_perm = quantize_weights(w_24,
                                                             num_bits,
                                                             group_size,
                                                             act_order=False)

    # Compress quantized weight
    q_w_24_comp, meta = compress_quantized_24_weight(q_w_24, size_k, size_n,
                                                     num_bits)
    size_k_comp = size_k // 2

    # Reformat to marlin
    marlin_24_q_w_comp = marlin_weights(q_w_24_comp, size_k_comp, size_n,
                                        num_bits, marlin_24_perm[num_bits])
    marlin_24_s = marlin_permute_scales(s, size_k, size_n, group_size,
                                        marlin_24_scale_perm[num_bits],
                                        marlin_24_scale_perm_single[num_bits])

    # Create result
    res_list = [w_24_ref, marlin_24_q_w_comp, meta, marlin_24_s]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")


def pack_fp8_to_int32(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.shape[0] % 4 == 0

    # Reshape to prepare for packing
    reshaped = fp8_tensor.reshape(-1, 4, *fp8_tensor.shape[1:])

    # Convert fp8 to uint8 (byte) representation
    byte_tensor = reshaped.view(torch.uint8)

    # Pack 4 uint8 values into one int32
    packed = (byte_tensor[:, 0].to(torch.int32) |
              (byte_tensor[:, 1].to(torch.int32) << 8) |
              (byte_tensor[:, 2].to(torch.int32) << 16) |
              (byte_tensor[:, 3].to(torch.int32) << 24))

    return packed.view(fp8_tensor.shape[0] // 4,
                       *fp8_tensor.shape[1:]).contiguous()
