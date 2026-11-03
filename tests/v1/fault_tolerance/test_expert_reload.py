# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.sentinel.expert_reload import _placement_weight_loader


def test_reload_every_local_replica_before_deferred_processing():
    """Reload recovered placement and every local replica using physical IDs."""
    weights = torch.full((3,), -1.0)
    pending = []

    def buffer(param, loaded_weight, expert_id, return_success=False):
        pending.append((param, loaded_weight, expert_id))
        return True if return_success else None

    load = _placement_weight_loader(buffer, {2: [4, 5], 0: [6]}, 4, 8)
    assert load(weights, torch.tensor(20.0), 2, True)
    assert load(weights, torch.tensor(10.0), 0, True)
    assert not load(weights, torch.tensor(99.0), 6, True)
    assert not load(weights, torch.tensor(99.0), 1, True)
    assert torch.equal(weights, torch.full((3,), -1.0))
    for param, value, physical_id in pending:
        param[physical_id - 4].copy_(value)
    assert torch.equal(weights, torch.tensor([20.0, 20.0, 10.0]))


def test_preserve_global_activation_scales_for_nonlocal_experts():
    """Preserve global activation scales even when no weight slots are local."""
    scales = torch.zeros(8)

    def load_scale(param, loaded_weight, weight_name, expert_id, return_success=False):
        param[expert_id].copy_(loaded_weight)
        return True if return_success else None

    load = _placement_weight_loader(load_scale, {}, 4, 8, use_global_sf=True)
    for expert_id in range(8):
        assert load(
            scales, torch.tensor(float(expert_id)), "input_scale", expert_id, True
        )
    assert torch.equal(scales, torch.arange(8, dtype=scales.dtype))
