# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts
from vllm.model_executor.layers.quantization.humming import HummingMoEMethod
from vllm.model_executor.models.conformer_encoder import RelPositionalEncoding


def test_static_model_tensor_is_nonpersistent_buffer() -> None:
    module = RelPositionalEncoding(d_model=8, max_len=4)

    assert dict(module.named_buffers())["pe"] is module.pe
    assert "pe" not in module.state_dict()


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


def test_weight_wake_hooks_reset_module_and_quant_state() -> None:
    from vllm.v1.worker.gpu_worker import _run_post_weights_wake_up_hooks

    events: list[str] = []

    class QuantMethod:
        def post_weights_wake_up(self, layer: nn.Module) -> None:
            events.append("quant")

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = QuantMethod()

        def post_weights_wake_up(self) -> None:
            events.append("module")

    model = nn.Sequential(Layer())
    _run_post_weights_wake_up_hooks(model)

    assert events == ["module", "quant"]


def test_humming_moe_can_be_prepared_for_repeated_reload() -> None:
    method = object.__new__(HummingMoEMethod)
    method.processed = True

    method.prepare_for_reload(SimpleNamespace())

    assert not method.processed
