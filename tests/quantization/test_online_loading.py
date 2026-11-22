# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaders that quantize each arriving tensor (direct reload of online quant)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.online import loading


class _Experts(torch.nn.Module):
    """Local experts 0 and 1; records what the checkpoint loaders receive."""

    def __init__(self):
        super().__init__()
        self.params_dtype = torch.bfloat16
        self.moe_config = SimpleNamespace(is_act_and_mul=True)
        self.layer_name = "experts"
        self.loaded: dict[tuple[str, str, int], torch.Tensor] = {}
        for p in ("w13_", "w2_"):
            for n in ("weight", "weight_scale", "weight_scale_2"):
                param = torch.nn.Parameter(torch.empty(0, device="cuda"), False)
                param.weight_loader = self._load
                self.register_parameter(p + n, param)
        self.method = SimpleNamespace(process_weights_after_loading=lambda _: None)

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id if expert_id in (0, 1) else -1

    def _load(self, param, loaded_weight, weight_name, shard_id, expert_id, **_):
        self.loaded[weight_name, shard_id, expert_id] = loaded_weight.clone()
        return True

    def send(self, shard_id, expert_id, value):
        # keyword call, as RoutedExperts.load_weights makes it
        return self.w13_weight.weight_loader(
            param=self.w13_weight,
            loaded_weight=torch.full((2, 16), value, dtype=torch.bfloat16),
            weight_name="w13_weight",
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )


@pytest.fixture
def experts(monkeypatch):
    def quantize(w):  # values, block scales, global scale = amax
        return w, w, w.abs().amax().reshape(1)

    monkeypatch.setattr(loading, "_quantize_moe_weight_to_nvfp4", quantize)
    layer = _Experts()
    loading._nvfp4_on_load(layer, layer.method)
    return layer


class TestNvfp4Pairing:
    @pytest.mark.parametrize(
        "sends,expected",
        [
            ([("w1", 1.0), ("w3", 3.0)], {"w1": 1.0, "w3": 3.0}),
            ([("w3", 3.0), ("w1", 1.0)], {"w1": 1.0, "w3": 3.0}),
            ([("w1", 9.0), ("w1", 1.0), ("w3", 3.0)], {"w1": 1.0, "w3": 3.0}),
        ],
        ids=["w1-first", "w3-first", "stale-half-replaced"],
    )
    def test_shared_global_scale(self, experts, sends, expected):
        for shard_id, value in sends:
            assert experts.send(shard_id, 0, value)
        for shard_id, value in expected.items():
            assert experts.loaded["w13_weight", shard_id, 0].eq(value).all()
            scale_2 = experts.loaded["w13_weight_scale_2", shard_id, 0]
            assert scale_2.item() == max(expected.values())
        experts.method.process_weights_after_loading(experts)

    def test_other_rank_expert_is_not_buffered(self, experts):
        assert experts.send("w1", 5, 1.0) is False
        assert not experts.loaded
        experts.method.process_weights_after_loading(experts)

    def test_lone_half_fails_post_load(self, experts):
        experts.send("w1", 0, 1.0)
        with pytest.raises(RuntimeError, match=r"experts \[0\] got only one"):
            experts.method.process_weights_after_loading(experts)
        experts.method.process_weights_after_loading(experts)  # state cleared


def test_quantizer_gets_contiguous_tensor():
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(torch.empty(4, 6, dtype=torch.int8), False)
    layer.weight_scale = torch.nn.Parameter(torch.empty(()), False)

    def quantize(w):
        assert w.is_contiguous()
        return w.to(torch.int8), w.float().abs().amax()

    loading.quantize_on_load(layer, "weight", quantize, "weight_scale")
    transposed = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4).t()
    layer.weight.weight_loader(layer.weight, transposed)
    assert torch.equal(layer.weight, transposed.to(torch.int8))
    assert layer.weight_scale.item() == 23


def test_fp8_per_channel_checkpoint_method_uses_channel_key(
    default_vllm_config,
) -> None:
    """Direct reload of fp8_per_channel must pass kFp8StaticChannelSym to ModelOpt."""
    from tests.single_rank_group import SingleRankGroup, override_parallel_groups
    from vllm.config import ModelConfig
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype

    default_vllm_config.model_config = ModelConfig(dtype="bfloat16")
    spec = QuantSpec(weight=kFp8StaticChannelSym, activation=kFp8DynamicTokenSym)
    with (
        override_parallel_groups(SingleRankGroup()),
        set_default_torch_dtype(torch.bfloat16),
    ):
        layer = ReplicatedLinear(32, 32, bias=False, params_dtype=torch.bfloat16)
        method = loading.checkpoint_method(layer, spec, kFp8DynamicTokenSym)
        assert isinstance(method, ModelOptLinearMethod)
        assert method.spec.weight == kFp8StaticChannelSym
        assert method.spec.activation == kFp8DynamicTokenSym

        if torch.cuda.is_available() and current_platform.supports_fp8():
            with torch.device("cuda"):
                method.create_weights(layer, 32, [32], 32, 32, torch.bfloat16)
                w_bf16 = torch.randn(32, 32, dtype=torch.bfloat16, device="cuda")
                layer.weight.weight_loader(layer.weight, w_bf16)
            assert layer.weight.dtype in (
                torch.float8_e4m3fn,
                current_platform.fp8_dtype(),
            )
            assert layer.weight_scale.shape == (32,)
            assert layer.weight_scale.dtype == torch.float32
            assert torch.isfinite(layer.weight_scale).all()
            assert (layer.weight_scale > 0).all()
