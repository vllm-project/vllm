# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loaders that quantize each arriving tensor (direct reload of online quant)."""

from types import SimpleNamespace

import pytest
import torch

from tests.single_rank_group import SingleRankGroup, override_parallel_groups
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


def test_layer_writing_outside_its_loaders_keeps_the_online_method():
    # e.g. gpt-oss experts: quantizing on load would miss those writes
    layer = SimpleNamespace(writes_outside_weight_loaders=True)
    assert loading.checkpoint_method(layer, spec=None, default=None) is None


@override_parallel_groups(SingleRankGroup())
def test_model_writing_outside_its_loaders_keeps_the_online_method(
    default_vllm_config,
):
    # Step3TextModel.load_weights writes q/k/v with `narrow().copy_`: a
    # quantize-on-load checkpoint method would hold unscaled FP8 casts
    from vllm.config import WeightTransferConfig
    from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.utils import quant_utils as qu
    from vllm.model_executor.models.step3_text import _Step3QKVProj

    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, quantization=None
    )
    default_vllm_config.weight_transfer_config = WeightTransferConfig()
    spec = QuantSpec(weight=qu.kFp8StaticTensorSym)
    config = OnlineQuantizationConfig(QuantizationConfigArgs(linear=spec))
    layer = _Step3QKVProj(64, 96, bias=False, quant_config=config)
    assert isinstance(layer.quant_method, OnlineLinearBase)


@pytest.fixture
def tp2(monkeypatch):
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 2
    )


@pytest.mark.parametrize(
    "sizes,fits",
    [
        # a replicated layer picks its method before it sets its partition sizes
        ({"input_size": 200, "output_size": 300}, True),
        (
            {
                "input_size": 256,
                "output_size": 384,
                "input_size_per_partition": 256,
                "output_partition_sizes": [192],  # not a multiple of 128
            },
            False,
        ),
    ],
    ids=["replicated", "unaligned-tp-shard"],
)
def test_block_method_fits_only_aligned_shards(tp2, sizes, fits):
    layer = SimpleNamespace(tp_size=2, **sizes)
    assert loading._block_shapes_fit(layer) is fits


@pytest.mark.parametrize(
    "ispp,fits",
    [(256, True), (192, True), (48, False)],
    ids=["aligned", "refined-scales", "unrefinable"],
)
def test_block_moe_method_fits_aligned_or_refinable_shards(tp2, ispp, fits):
    # e.g. Qwen3-30B-A3B at TP4 (192) refines its scale blocks to 64x64
    moe = SimpleNamespace(
        intermediate_size_per_partition=ispp, tp_size=2, hidden_dim=2048
    )
    assert loading._moe_block_shapes_fit(moe) is fits
