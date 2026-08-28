# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts


def test_kernel_buffer_registration_reuses_captured_storage() -> None:
    layer = nn.Module()
    owner = SimpleNamespace()
    original = torch.tensor([1.0, 2.0])

    registered = FusedMoEExperts._register_persistent_buffer(
        owner, layer, "scale", original
    )
    pointer = registered.data_ptr()
    replacement = FusedMoEExperts._register_persistent_buffer(
        owner, layer, "scale", torch.tensor([3.0, 4.0])
    )

    assert replacement.data_ptr() == pointer
    assert torch.equal(replacement, torch.tensor([3.0, 4.0]))
    assert "_moe_scale" not in layer.state_dict()

    restored = torch.tensor([5.0, 6.0])
    layer._buffers["_moe_scale"] = restored
    FusedMoEExperts.rebind_sleep_buffers(owner, layer)
    assert owner.scale is restored
