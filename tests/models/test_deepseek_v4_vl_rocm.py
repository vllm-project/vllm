# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


def test_rocm_packed_kv_cache_auto_uses_ds_mla_layout() -> None:
    from vllm.config import CacheConfig
    from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype

    cache_config = CacheConfig()

    resolved_dtype, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        use_fp8_ds_mla_layout=True,
        kv_cache_dtype=cache_config.cache_dtype,
        cache_config=cache_config,
    )

    assert resolved_dtype == "fp8_ds_mla"
    assert torch_dtype is torch.uint8
    assert cache_config.cache_dtype == "fp8_ds_mla"


def test_rocm_packed_kv_cache_rejects_unquantized_dtype() -> None:
    from vllm.config import CacheConfig
    from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype

    cache_config = CacheConfig(cache_dtype="bfloat16")

    with pytest.raises(ValueError, match="only supports fp8 kv-cache"):
        _resolve_dsv4_kv_cache_dtype(
            use_fp8_ds_mla_layout=True,
            kv_cache_dtype=cache_config.cache_dtype,
            cache_config=cache_config,
        )


def test_vl_mapper_preserves_rocm_weight_mapping() -> None:
    from vllm.models.deepseek_v4.amd.model import _make_deepseek_v4_weights_mapper
    from vllm.models.deepseek_v4.common.vl_model import (
        _make_deepseek_v4_vl_weights_mapper,
    )

    text_mapper = _make_deepseek_v4_weights_mapper("fp4", fuse_shared_experts=True)
    mapper = _make_deepseek_v4_vl_weights_mapper(text_mapper, image_enabled=True)

    assert mapper._map_name("layers.3.attn.wq_a.input_scale") == (
        "language_model.model.layers.3.attn.wq_a.input_scale_2"
    )
    assert mapper._map_name("layers.3.ffn.shared_experts.w2.weight") == (
        "language_model.model.layers.3.ffn.shared_experts.w2.weight"
    )
    assert mapper._map_name("head.weight") == "language_model.lm_head.weight"


def test_rocm_moe_wires_vision_routing_on_hash_and_regular_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v4.amd import model as rocm_model
    from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

    captured: list[dict] = []

    class FakeGate(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

    def fake_factory(**kwargs):
        captured.append(kwargs)
        return nn.Identity()

    monkeypatch.setattr(rocm_model, "GateLinear", FakeGate)
    monkeypatch.setattr(rocm_model, "FusedMoEFactory", fake_factory)
    monkeypatch.setattr(rocm_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rocm_model, "get_tensor_model_parallel_rank", lambda: 0)

    config = SimpleNamespace(
        hidden_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        swiglu_limit=None,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        num_hash_layers=1,
        vocab_size=32,
        topk_method="noaux_tc",
        vision_n_layers=1,
        n_shared_experts=None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config), quant_config=None
    )

    hash_moe = rocm_model.DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")
    regular_moe = rocm_model.DeepseekV4MoE(vllm_config, prefix="model.layers.1.ffn")

    assert hash_moe.gate.tid2eid is not None
    assert regular_moe.gate.tid2eid is None
    for moe, factory_kwargs in zip((hash_moe, regular_moe), captured, strict=True):
        assert moe.gate.e_score_correction_bias is not None
        assert moe.gate.bias_vl is not None
        assert factory_kwargs["bias_vl"] is moe.gate.bias_vl
        assert factory_kwargs["image_sentinel_lo"] == IMAGE_SENTINEL_BASE_ID


def test_rocm_mtp_forwards_input_ids_for_vision_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v4.amd import mtp as rocm_mtp

    hidden_size = 4
    hc_mult = 2

    class FakeNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = 1e-6

    class FakeMTPBlock(nn.Module):
        use_fused_mhc = False

        def __init__(self) -> None:
            super().__init__()
            self.seen_input_ids: torch.Tensor | None = None

        def forward(
            self,
            *,
            positions: torch.Tensor,
            x: torch.Tensor,
            input_ids: torch.Tensor | None,
        ):
            self.seen_input_ids = input_ids
            return x, None, None, None

    def passthrough_mtp_input(
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        *args,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return inputs_embeds, previous_hidden_states

    monkeypatch.setattr(rocm_mtp, "fused_mtp_input_rmsnorm", passthrough_mtp_input)

    layer = object.__new__(rocm_mtp.DeepSeekV4MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(hidden_size=hidden_size)
    layer.hc_mult = hc_mult
    layer.enorm = FakeNorm()
    layer.hnorm = FakeNorm()
    layer.e_proj = nn.Identity()
    layer.h_proj = nn.Identity()
    layer.mtp_block = FakeMTPBlock()

    input_ids = torch.tensor([11, 12])
    positions = torch.tensor([3, 4])
    inputs_embeds = torch.arange(8, dtype=torch.float32).view(2, hidden_size)
    previous_hidden_states = torch.arange(16, dtype=torch.float32).view(2, -1)

    output = layer(
        input_ids,
        positions,
        previous_hidden_states,
        inputs_embeds,
    )

    assert layer.mtp_block.seen_input_ids is input_ids
    expected = previous_hidden_states.view(2, hc_mult, hidden_size)
    expected = expected + inputs_embeds.unsqueeze(-2)
    torch.testing.assert_close(output, expected.flatten(1))


def test_rocm_compute_logits_local_skips_gather() -> None:
    from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM

    calls: list[tuple[nn.Module, torch.Tensor, bool]] = []

    def logits_processor(
        lm_head: nn.Module, hidden_states: torch.Tensor, *, skip_gather: bool = False
    ) -> torch.Tensor:
        calls.append((lm_head, hidden_states, skip_gather))
        return hidden_states + 1

    model = object.__new__(DeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = nn.Identity()
    model.logits_processor = logits_processor
    hidden_states = torch.tensor([4.0])

    result = model.compute_logits_local(hidden_states)

    assert torch.equal(result, torch.tensor([5.0]))
    assert calls == [(model.lm_head, hidden_states, True)]


class _FakeLanguageModel(nn.Module):
    finalizes_weights_during_load = False

    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []

    def process_weights_after_loading(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1


def test_vl_wrapper_streams_then_delegates_finalization() -> None:
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    def interleaved_weights():
        yield "language_model.tensor_a", torch.tensor([1.0])
        assert model.language_model.tensor_a.item() == 1.0
        yield "vision.tensor_b", torch.tensor([2.0])
        assert model.vision.tensor_b.item() == 2.0
        yield "language_model.tensor_c", torch.tensor([3.0])

    loaded = model.load_weights(interleaved_weights())

    assert loaded == {
        "language_model.tensor_a",
        "vision.tensor_b",
        "language_model.tensor_c",
    }
    assert model.language_model.finalized_values == []

    model.process_weights_after_loading()

    assert model.language_model.finalized_values == [(1.0, 3.0)]
    assert torch.equal(
        model.compute_logits_local(torch.tensor([4.0])), torch.tensor([5.0])
    )
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(1.0, 3.0)]


class _FakeFinalizingLanguageModel(_FakeLanguageModel):
    finalizes_weights_during_load = True

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def load_weights(self, weights) -> set[str]:
        self.load_calls += 1
        loaded = set()
        for name, value in weights:
            getattr(self, name).data.copy_(value)
            loaded.add(name)
        self.process_weights_after_loading()
        return loaded


def test_vl_wrapper_groups_child_that_finalizes_during_load() -> None:
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = _FakeFinalizingLanguageModel()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    loaded = model.load_weights(
        iter(
            (
                ("language_model.tensor_a", torch.tensor([1.0])),
                ("vision.tensor_b", torch.tensor([2.0])),
                ("language_model.tensor_c", torch.tensor([3.0])),
            )
        )
    )

    assert loaded == {
        "language_model.tensor_a",
        "vision.tensor_b",
        "language_model.tensor_c",
    }
    assert model.language_model.load_calls == 1
    assert model.language_model.finalized_values == [(1.0, 3.0)]

    # The framework's later model-level hook must not double-finalize a child
    # which already completed this work in load_weights.
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(1.0, 3.0)]


def test_vl_wrapper_dummy_load_delegates_finalization() -> None:
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = _FakeFinalizingLanguageModel()

    # DummyModelLoader bypasses model.load_weights(), so no finalized marker
    # exists and the framework-level hook must still delegate to the child.
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(0.0, 0.0)]
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(0.0, 0.0)]
