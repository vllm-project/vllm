# SPDX-License-Identifier: Apache-2.0
from itertools import product

import numpy as np


class Blocked:

    def __init__(self, arr: np.ndarray, sizes, index_map):
        assert len(arr.shape) == len(sizes)
        self.arr = arr
        self.sizes = sizes
        self.index_map = index_map

    def get_block(self, idxs):
        return eval(f"self.arr[{self._get_index_str(idxs)}]")

    def set_block(self, idxs, val):
        exec(f"self.arr[{self._get_index_str(idxs)}] = val")

    def _get_index_str(self, idxs):
        return ", ".join(f"{idx*self.sizes[i]}:{(idx+1)*self.sizes[i]}"
                         for i, idx in enumerate(self.index_map(*idxs)))


# np.random.seed(4)
T, D, L, N = 128, 3072, 16, 8

D1 = 8
D2 = D // D1

bT = 1
bL = 16
bD1 = 8
bD2 = 128
bD = bD1 * bD2

inputs = np.random.randn(T, D)
loras = np.random.randn(1, L, D)

print("ref1", (inputs @ loras.squeeze(0).T).sum())

inputs_1 = inputs.reshape((T, D1, D2))
loras_1 = loras.reshape((1, L, D1, D2))

print("ref2", np.einsum("tdD,ondD->tn", inputs_1, loras_1).sum())


def fast_bgmv(inputs, loras):
    out = np.zeros((T, L))
    grid = (T // bT, L // bL, D1 // bD1, D2 // bD2)
    x_b = Blocked(inputs, (bT, bD1, bD2), lambda i, j, k1, k2: (i, k1, k2))
    l_b = Blocked(loras, (1, bL, bD1, bD2), lambda i, j, k1, k2:
                  (0, j, k1, k2))
    out_b = Blocked(out, (bT, bL), lambda i, j, k1, k2: (i, j))
    acc_ref = np.zeros((bT, bL))

    for idxs in product(*list(map(range, grid))):
        x_ref = x_b.get_block(idxs)
        l_ref = l_b.get_block(idxs)

        if idxs[2] == 0 and idxs[3] == 0:
            acc_ref = np.zeros_like(acc_ref)

        acc_ref += (x_ref * l_ref[0]).sum(-1).sum(-1)

        if idxs[2] == grid[2] - 1 and idxs[3] == grid[3] - 1:
            out_b.set_block(idxs, acc_ref)
    return out


def slow_bgmv(inputs, loras):
    out = np.zeros((T, L))
    grid = (T // bT, L // bL, D // bD)
    x_b = Blocked(inputs, (bT, bD), lambda i, j, k: (i, k))
    l_b = Blocked(loras, (1, bL, bD), lambda i, j, k: (0, j, k))
    out_b = Blocked(out, (bT, bL), lambda i, j, k: (i, j))
    acc_ref = np.zeros((bT, bL))

    for idxs in product(*list(map(range, grid))):
        x_ref = x_b.get_block(idxs)
        l_ref = l_b.get_block(idxs)

        if idxs[2] == 0:
            acc_ref = np.zeros_like(acc_ref)

        acc_ref += x_ref @ l_ref[0].T

        if idxs[2] == grid[2] - 1:
            out_b.set_block(idxs, acc_ref)
    return out


# result = np.zeros((5, 64))

# for t in range(5):

#     result[t] = (x1[t] * l1[:]).sum(axis=1).sum(axis=1)
#     # for n in range(64):
#         # print((x1[t] * l1[n]).shape)
#         # result[t, n] = (x1[t] * l1[n]).sum()

print("test slow", slow_bgmv(inputs, loras).sum())
print("test fast", fast_bgmv(inputs_1, loras_1).sum())
