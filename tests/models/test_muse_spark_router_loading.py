# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.muse_spark.amd.model as amd_model
import vllm.models.muse_spark.nvidia.model as nvidia_model
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)

MULTIPLIER = 0.25


class _RouterModel(torch.nn.Module):
    """One Muse Spark MoE block's router, loaded through its real weight loader."""

    def __init__(self, model_module):
        super().__init__()
        moe = model_module.MuseSparkMoE.__new__(model_module.MuseSparkMoE)
        torch.nn.Module.__init__(moe)
        moe.router = torch.nn.Linear(4, 2, bias=False)
        moe.router.weight_loader = ReplicatedLinear.weight_loader.__get__(moe.router)
        moe.router.weight.weight_loader = moe._load_router_weight
        moe.router_output_multiplier = MULTIPLIER
        # Only the AMD block reads these.
        moe._router_multiplier_folded = False
        moe.moe_router = SimpleNamespace(router_output_multiplier=MULTIPLIER)
        self.moe = moe

    def load_weights(self, weights):
        loaded_names = set()
        for name, loaded_weight in weights:
            weight = self.moe.router.weight
            weight.weight_loader(weight, loaded_weight)
            loaded_names.add(name)
        # The AMD block folds the multiplier in post-load processing.
        fold = getattr(self.moe, "fold_router_output_multiplier", None)
        if fold is not None:
            fold()
        return loaded_names


def _patch(values, indices=None) -> CheckpointWeightPatch:
    return CheckpointWeightPatch(
        name="router.weight",
        shape=(2, 4),
        dtype=torch.float32,
        values=torch.tensor(values, dtype=torch.float32),
        indices=None if indices is None else torch.tensor(indices, dtype=torch.int32),
    )


@pytest.mark.parametrize("model_module", [nvidia_model, amd_model])
def test_sparse_router_patch_scales_only_the_patched_elements(model_module):
    model = _RouterModel(model_module).requires_grad_(False)
    checkpoint = torch.arange(1.0, 9.0)
    load_checkpoint_weight_patches(model, [_patch(checkpoint.tolist())])
    weight = model.moe.router.weight
    torch.testing.assert_close(weight.flatten(), checkpoint * MULTIPLIER)

    checkpoint[3] = 40.0
    load_checkpoint_weight_patches(model, [_patch([40.0], [3])])
    torch.testing.assert_close(weight.flatten(), checkpoint * MULTIPLIER)

    # A dense reload after the sparse patch still folds the multiplier once.
    checkpoint = torch.arange(11.0, 19.0)
    load_checkpoint_weight_patches(model, [_patch(checkpoint.tolist())])
    torch.testing.assert_close(weight.flatten(), checkpoint * MULTIPLIER)
