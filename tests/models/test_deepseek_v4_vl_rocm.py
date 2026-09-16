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


def _preshuffled_fp8_linear(
    holder: str = "quant_method", weight_shape: tuple[int, int] = (256, 128)
) -> nn.Module:
    pytest.importorskip("aiter")
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
        AiterPreshuffledFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
    )

    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        weight_shape=weight_shape,
    )
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.randn(*weight_shape, device="cuda").to(current_platform.fp8_dtype()),
        requires_grad=False,
    )
    layer.register_parameter(
        "weight_scale" if holder == "scheme" else "weight_scale_inv",
        nn.Parameter(
            torch.ones(*(dim // 128 for dim in weight_shape), device="cuda"),
            requires_grad=False,
        ),
    )
    setattr(
        layer,
        holder,
        SimpleNamespace(fp8_linear=AiterPreshuffledFp8BlockScaledMMKernel(config)),
    )
    return layer


@pytest.mark.parametrize("holder", ["quant_method", "scheme"])
def test_rocm_wo_a_keeps_row_major_weights(holder: str, default_vllm_config) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    layer = _preshuffled_fp8_linear(holder)
    layer.is_bmm = True
    original = layer.weight.detach().clone()
    kernel = getattr(layer, holder).fp8_linear
    kernel.process_weights_after_loading(layer)

    actual = _get_cached_wo_a_bf16(layer, 2, 128, 128)
    torch.testing.assert_close(actual, original.to(torch.bfloat16).view(2, 128, 128))


@pytest.mark.parametrize("holder", ["quant_method", "scheme"])
def test_rocm_gateup_shuffles_once_and_down_proj_keeps_preshuffled_backend(
    holder: str, monkeypatch: pytest.MonkeyPatch, default_vllm_config
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v4.amd import model as rocm_model

    gate_up, down = _preshuffled_fp8_linear(holder), _preshuffled_fp8_linear(holder)
    original_gate_up = gate_up.weight.detach().clone()
    original_down = down.weight.detach().clone()
    monkeypatch.setattr(
        rocm_model, "MergedColumnParallelLinear", lambda *a, **k: gate_up
    )
    monkeypatch.setattr(rocm_model, "RowParallelLinear", lambda *a, **k: down)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    model = rocm_model.DeepseekV4MLP(128, 128, "silu")
    for linear in (gate_up, down):
        getattr(linear, holder).fp8_linear.process_weights_after_loading(linear)
    model.prepare_gateup_preshuffle()

    for linear, original in ((gate_up, original_gate_up), (down, original_down)):
        expected = rocm_aiter_ops.shuffle_weight(original, layout=(16, 16))
        torch.testing.assert_close(
            linear.weight.view(torch.uint8), expected.view(torch.uint8)
        )


def test_rocm_fused_qkv_quant_matches_preshuffled_gemm_scale_layout(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterMLAAttention,
        apply_pre_quantized_block_scaled_mm,
    )

    if not rocm_aiter_ops.is_blockscale_bpreshuffle_tuned(1024, 256):
        pytest.skip("Requires a tuned AITER preshuffled FP8 GEMM")
    linear = _preshuffled_fp8_linear(weight_shape=(1024, 256))
    original = linear.weight.detach().clone()
    linear.quant_method.fp8_linear.process_weights_after_loading(linear)
    attention = DeepseekV4ROCMAiterMLAAttention.__new__(DeepseekV4ROCMAiterMLAAttention)
    nn.Module.__init__(attention)
    attention.wq_b = linear
    attention.indexer = None
    attention.q_lora_rank, attention.head_dim = 256, 128
    attention.eps = 1e-5
    attention.q_norm = SimpleNamespace(
        weight=torch.ones(256, dtype=torch.bfloat16, device="cuda")
    )
    attention.kv_norm = SimpleNamespace(
        weight=torch.ones(128, dtype=torch.bfloat16, device="cuda")
    )
    monkeypatch.setattr(rocm_aiter_ops, "is_linear_fp8_enabled", lambda: True)
    # Multiple rows and groups with different ranges expose byte-order errors.
    inputs = torch.randn(16, 384, dtype=torch.bfloat16, device="cuda")
    inputs[:, :128] *= 4
    quantized, scales, _ = attention._split_qkv_and_norm(inputs)
    assert scales is not None
    output = apply_pre_quantized_block_scaled_mm(linear, quantized, scales)
    q = inputs[:, :256].float()
    normalized = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + attention.eps)
    reference_scales = normalized.view(16, 2, 128).abs().amax(dim=-1)
    reference_scales /= torch.finfo(current_platform.fp8_dtype()).max
    dequantized = quantized.float() * reference_scales.repeat_interleave(128, dim=1)
    expected = dequantized @ original.float().T
    torch.testing.assert_close(output.float(), expected, atol=0.125, rtol=0.01)


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
