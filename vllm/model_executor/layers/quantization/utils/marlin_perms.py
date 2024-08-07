"""This file is used for /tests and /benchmarks"""
import numpy
import torch


# Precompute permutations for Marlin weight and scale shuffling # noqa: E501
#
# Marlin works on [16,64] tiles. The goal of the permutations is to reorder the weight data so that it is compatible noqa: # noqa: E501
# with the tensor-core format that is described here:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type # noqa: E501
#
# As a result of this reordering, the vector loads inside the kernel will get the data as it is needed for tensor-core # noqa: E501
# (without the need to use ldmatrix instructions) # noqa: E501
def get_perms(num_bits):
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


marlin_perm = {}
marlin_scale_perm = {}
marlin_scale_perm_single = {}
for num_bits in [4, 8]:
    perm, scale_perm, scale_perm_single = get_perms(num_bits)
    marlin_perm[num_bits] = perm
    marlin_scale_perm[num_bits] = scale_perm
    marlin_scale_perm_single[num_bits] = scale_perm_single
