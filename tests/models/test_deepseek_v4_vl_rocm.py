# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig
from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


@pytest.mark.parametrize("hash_routing", [False, True], ids=["regular", "hash"])
@pytest.mark.parametrize("vision", [False, True], ids=["text", "vision"])
@torch.inference_mode()
def test_rocm_moe_routing_and_shared_experts_match_reference(
    hash_routing, vision, monkeypatch, default_vllm_config, dist_init
) -> None:
    """AMD V4 routing, clamped experts and shared output agree with PyTorch."""
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v4.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0")
    default_vllm_config.kernel_config.moe_backend = "triton"
    config = SimpleNamespace(
        hidden_size=512,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        swiglu_limit=0.05,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        num_hash_layers=int(hash_routing),
        vocab_size=IMAGE_SENTINEL_BASE_ID + 8,
        topk_method="noaux_tc",
        vision_n_layers=int(vision),
        n_shared_experts=1,
        hidden_act="silu",
    )
    torch.manual_seed(42)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        moe = DeepseekV4MoE(
            SimpleNamespace(
                model_config=SimpleNamespace(hf_config=config), quant_config=None
            ),
            prefix="model.layers.0.ffn",
        )
        for parameter in moe.parameters():
            if parameter.is_floating_point():
                parameter.normal_(std=0.04)
        hidden = torch.randn(17, config.hidden_size) * 0.1
        input_ids = torch.arange(17, dtype=torch.int32)
        if vision:
            # Include both ends of the five-token sentinel interval and the
            # first ordinary token above it.
            input_ids[1:4] = torch.tensor(
                [
                    IMAGE_SENTINEL_BASE_ID,
                    IMAGE_SENTINEL_BASE_ID + 4,
                    IMAGE_SENTINEL_BASE_ID + 5,
                ],
                dtype=torch.int32,
            )
        if moe.gate.tid2eid is not None:
            rows = torch.arange(config.vocab_size)
            moe.gate.tid2eid.copy_(torch.stack((rows % 8, (rows + 3) % 8), dim=1))
        if moe.gate.e_score_correction_bias is not None:
            moe.gate.e_score_correction_bias.copy_(torch.arange(8) * 0.1)
        if moe.gate.bias_vl is not None:
            moe.gate.bias_vl.copy_(torch.arange(8) * -0.1)

    scores = torch.nn.functional.softplus(
        hidden.float() @ moe.gate.weight.float().T
    ).sqrt()
    bias = moe.gate.e_score_correction_bias
    selection_scores = scores if bias is None else scores + bias
    expected_ids = selection_scores.topk(2, dim=-1).indices
    if hash_routing:
        expected_ids = moe.gate.tid2eid[input_ids.long()].long()
    if vision:
        image_mask = (input_ids >= IMAGE_SENTINEL_BASE_ID) & (
            input_ids < IMAGE_SENTINEL_BASE_ID + 5
        )
        expected_ids[image_mask] = (
            (scores + moe.gate.bias_vl).topk(2, dim=-1).indices[image_mask]
        )
    expected_weights = scores.gather(1, expected_ids)
    expected_weights *= config.routed_scaling_factor / expected_weights.sum(
        dim=-1, keepdim=True
    )
    router_logits, _ = moe.gate(hidden)
    weights, ids = moe.experts.router.select_experts(
        hidden, router_logits, topk_indices_dtype=torch.int32, input_ids=input_ids
    )
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)

    def expert_reference(x, gate_up_weight, down_weight):
        gate, up = (
            torch.nn.functional.linear(x, gate_up_weight).float().chunk(2, dim=-1)
        )
        gate = gate.clamp(max=config.swiglu_limit)
        up = up.clamp(-config.swiglu_limit, config.swiglu_limit)
        activated = (torch.nn.functional.silu(gate) * up).to(x.dtype)
        return torch.nn.functional.linear(activated, down_weight).float()

    routed = moe.experts.routed_experts
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for expert_id in range(config.n_routed_experts):
        token, slot = torch.where(expected_ids == expert_id)
        expected[token] += (
            expert_reference(
                hidden[token], routed.w13_weight[expert_id], routed.w2_weight[expert_id]
            )
            * expected_weights[token, slot, None]
        )
    shared = moe.shared_experts
    expected += expert_reference(
        hidden, shared.gate_up_proj.weight, shared.down_proj.weight
    )
    moe.experts._quant_method.process_weights_after_loading(routed)
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda:0"))
    with set_forward_context(None, default_vllm_config, num_tokens=hidden.shape[0]):
        actual = moe(hidden, input_ids)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=5e-5)


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
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=ParallelConfig(),
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


def test_rocm_fse_rejects_data_parallel_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v4.amd.model import _fuse_shared_experts_enabled

    config = SimpleNamespace(n_shared_experts=1)

    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    with pytest.raises(ValueError, match="data parallelism"):
        _fuse_shared_experts_enabled(config, ParallelConfig(data_parallel_size=2))

    # Expert parallelism already disables FSE: no raise, FSE stays off.
    assert not _fuse_shared_experts_enabled(
        config, ParallelConfig(data_parallel_size=2, enable_expert_parallel=True)
    )
    assert _fuse_shared_experts_enabled(config, ParallelConfig())


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

    monkeypatch.setattr(
        rocm_mtp, "_FUSED_MTP_INPUT_RMSNORM_KERNEL", passthrough_mtp_input
    )

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
