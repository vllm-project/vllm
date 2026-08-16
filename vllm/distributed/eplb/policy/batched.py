# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from .default import DefaultEplbPolicy


class BatchedEplbPolicy(DefaultEplbPolicy):
    """DeepSeek EPLB policy with packing vectorized across layers."""

    @classmethod
    def balanced_packing(
        cls, weight: np.ndarray, num_packs: int
    ) -> tuple[np.ndarray, np.ndarray]:
        num_layers, num_groups = weight.shape
        assert num_groups % num_packs == 0
        groups_per_pack = num_groups // num_packs

        if groups_per_pack == 1:
            pack_index = np.tile(np.arange(num_groups, dtype=np.int64), (num_layers, 1))
            return pack_index, np.zeros_like(pack_index)

        indices = np.argsort(-weight, axis=-1)
        pack_index = np.full((num_layers, num_groups), -1, dtype=np.int64)
        rank_in_pack = np.full_like(pack_index, -1)
        pack_weights = np.zeros((num_layers, num_packs), dtype=np.float64)
        pack_items = np.zeros((num_layers, num_packs), dtype=np.int64)

        if num_layers == 1:
            return super().balanced_packing(weight, num_packs)

        layers = np.arange(num_layers)
        for group_idx in range(num_groups):
            groups = indices[:, group_idx]
            packs = np.argmin(pack_weights, axis=1)

            pack_index[layers, groups] = packs
            rank_in_pack[layers, groups] = pack_items[layers, packs]
            pack_weights[layers, packs] += weight[layers, groups]
            pack_items[layers, packs] += 1
            full = pack_items[layers, packs] == groups_per_pack
            pack_weights[layers[full], packs[full]] = np.inf

        return pack_index, rank_in_pack
