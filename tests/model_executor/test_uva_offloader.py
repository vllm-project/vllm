# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.model_executor.offloader.uva import UVAOffloader


class _CopyRecorder:
    def __init__(self):
        self.calls: list[tuple[torch.device, bool]] = []

    def to(self, device: torch.device, non_blocking: bool):
        self.calls.append((device, non_blocking))
        return self


class _StateDictRaises(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.register_buffer("persistent_buffer", torch.ones(1))
        self.register_buffer("temporary_buffer", torch.ones(1), persistent=False)

    def state_dict(self, *args, **kwargs):
        raise AssertionError("state_dict should not be called")


def test_named_state_tensors_does_not_call_state_dict():
    module = _StateDictRaises()

    state = dict(UVAOffloader._named_state_tensors(module))

    assert set(state) == {"weight", "persistent_buffer"}


def test_move_state_to_device_uses_requested_non_blocking():
    tensor = _CopyRecorder()
    device = torch.device("cuda")

    state = UVAOffloader._move_state_to_device(
        [("weight", tensor)],
        device,
        non_blocking=False,
    )

    assert state == {"weight": tensor}
    assert tensor.calls == [(device, False)]
