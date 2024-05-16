"""This file is used for /tests and /benchmarks"""
import numpy
import torch

from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_TILE)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_pack_factor, quantize_weights, sort_weights)

__cuda_arch = torch.cuda.get_device_capability()


def is_marlin_supported():
    return __cuda_arch[0] >= 8


# Precompute permutations for Marlin weight and scale shuffling # noqa: E501
#
# Marlin works on [16,64] tiles. The goal of the permutations is to reorder the weight data so that it is compatible noqa: # noqa: E501
# with the tensor-core format that is described here:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type # noqa: E501
#
# As a result of this reordering, the vector loads inside the kernel will get the data as it is needed for tensor-core # noqa: E501
# (without the need to use ldmatrix instructions) # noqa: E501
def _get_perms(num_bits):
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = numpy.array(perm_list)

    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm = {}
_scale_perm = {}
_scale_perm_single = {}
for num_bits in [4, 8]:
    perm, scale_perm, scale_perm_single = _get_perms(num_bits)
    _perm[num_bits] = perm
    _scale_perm[num_bits] = scale_perm
    _scale_perm_single[num_bits] = scale_perm_single


def marlin_permute_weights(q_w,
                           size_k,
                           size_n,
                           num_bits,
                           tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape(
        (-1, _perm[num_bits].numel()))[:, _perm[num_bits]].reshape(q_w.shape)

    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, num_bits)

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


def marlin_permute_scales(s, size_k, size_n, group_size, num_bits):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(_scale_perm[num_bits])))[:,
                                                        _scale_perm[num_bits]]
    else:
        s = s.reshape(
            (-1,
             len(_scale_perm_single[num_bits])))[:,
                                                 _scale_perm_single[num_bits]]
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
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size, num_bits)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


class MarlinWorkspace:

    def __init__(self, out_features):
        assert (out_features % GPTQ_MARLIN_MIN_THREAD_N == 0), (
            "out_features = {} is undivisible by GPTQ_MARLIN_MIN_THREAD_N = {}"
            .format(out_features, GPTQ_MARLIN_MIN_THREAD_N))

        max_workspace_size = ((out_features // GPTQ_MARLIN_MIN_THREAD_N) *
                              GPTQ_MARLIN_MAX_PARALLEL)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")
