# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.offloader.uva import UVAOffloader


class _FakeCpuData:
    def __init__(self):
        self.pin_memory_calls = 0

    def to(self, *, device):
        assert device == "cpu"
        return self

    def pin_memory(self):
        self.pin_memory_calls += 1
        return self

    def numel(self):
        return 1

    def element_size(self):
        return 4


class _FakeParameter:
    def __init__(self):
        self.device = torch.device("cuda")
        self.data = _FakeCpuData()


class _FakeModule:
    def __init__(self):
        self.parameter = _FakeParameter()

    def parameters(self):
        yield self.parameter

    def named_parameters(self):
        yield "weight", self.parameter

    def forward(self):
        pass


@pytest.mark.parametrize("uva_offloading, expected_pin_calls", [(True, 0), (False, 1)])
def test_uva_offloader_only_pins_fallback_cpu_weights(
    monkeypatch, uva_offloading, expected_pin_calls
):
    offloader = object.__new__(UVAOffloader)
    offloader.cpu_offload_max_bytes = 1024
    offloader.cpu_offload_bytes = 0
    offloader.cpu_offload_params = set()
    offloader.pin_memory = True
    offloader.uva_offloading = uva_offloading

    if uva_offloading:
        monkeypatch.setattr(
            "vllm.model_executor.offloader.uva.get_accelerator_view_from_cpu_tensor",
            lambda tensor: tensor,
        )

    module = _FakeModule()
    offloader._maybe_offload_to_cpu(module)

    assert module.parameter.data.pin_memory_calls == expected_pin_calls
