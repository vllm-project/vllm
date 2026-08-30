# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.models.hy_v4.amd.attention import HYV4MLAAttention
from vllm.models.hy_v4.amd.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
)
from vllm.models.hy_v4.amd.mtp import (
    HYV4MTP,
    HYV4MultiTokenPredictor,
    HYV4MultiTokenPredictorLayer,
)
from vllm.models.hy_v4.amd.rocm import (
    HYV4ROCMAiterMLASparseBackend,
    HYV4ROCMAiterMLASparseImpl,
)
from vllm.models.hy_v4.nvidia.model import (
    HYV4ForCausalLM as NvidiaHYV4ForCausalLM,
)
from vllm.models.hy_v4.nvidia.mtp import HYV4MTP as NvidiaHYV4MTP
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseImpl,
)


class _FakeLmHead:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.quant_method = UnquantizedEmbeddingMethod()
        self.shard_indices = type(
            "ShardIndices",
            (),
            {"num_org_vocab_padding": 0, "org_vocab_start_index": 0},
        )()
        self.tp_size = 1


class _FakeSharedHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.head = _FakeLmHead(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _FakeMTPPredictorLayer(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.shared_head = _FakeSharedHead(weight)


class _FakeLogitsProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip_gather_calls: list[bool] = []

    def forward(
        self,
        lm_head: _FakeLmHead,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor:
        self.skip_gather_calls.append(skip_gather)
        return torch.nn.functional.linear(
            hidden_states,
            lm_head.weight,
            embedding_bias,
        )

    def get_top_tokens(
        self, lm_head: _FakeLmHead, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self(lm_head, hidden_states).argmax(dim=-1)


def _make_mtp_predictor(weights: list[torch.Tensor]) -> HYV4MultiTokenPredictor:
    predictor = HYV4MultiTokenPredictor.__new__(HYV4MultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.mtp_start_layer_idx = 80
    predictor.num_mtp_layers = len(weights)
    predictor.spec_step_idx = 0
    predictor.layers = nn.ModuleDict(
        {
            str(predictor.mtp_start_layer_idx + idx): _FakeMTPPredictorLayer(weight)
            for idx, weight in enumerate(weights)
        }
    )
    predictor.logits_processor = _FakeLogitsProcessor()
    return predictor


def _make_mtp(weights: list[torch.Tensor]) -> HYV4MTP:
    mtp = HYV4MTP.__new__(HYV4MTP)
    nn.Module.__init__(mtp)
    mtp.model = _make_mtp_predictor(weights)
    return mtp


def test_rocm_model_and_mtp_use_amd_attention() -> None:
    assert HYV4DecoderLayer.attention_cls is HYV4MLAAttention
    assert HYV4Model.decoder_layer_cls is HYV4DecoderLayer
    assert HYV4ForCausalLM.model_cls is HYV4Model
    assert HYV4MultiTokenPredictorLayer.decoder_layer_cls is HYV4DecoderLayer
    assert HYV4MultiTokenPredictor.predictor_layer_cls is HYV4MultiTokenPredictorLayer
    assert HYV4MTP.predictor_cls is HYV4MultiTokenPredictor


@pytest.mark.parametrize("soft_cap", [None, 2.0])
def test_rocm_target_local_logits_match_gathered_logits(
    soft_cap: float | None,
) -> None:
    hidden_states = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    weight = torch.tensor([[4.0, 0.0], [0.0, 3.0], [-1.0, -1.0]])
    model = HYV4ForCausalLM.__new__(HYV4ForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = _FakeLmHead(weight)
    model.logits_processor = _FakeLogitsProcessor()
    model.config = SimpleNamespace(
        soft_logits_capping=soft_cap is not None,
        soft_logits_capping_logits=soft_cap,
    )

    local_logits = model.compute_logits_local(hidden_states)
    gathered_logits = model.compute_logits(hidden_states)

    torch.testing.assert_close(local_logits, gathered_logits)
    assert model.logits_processor.skip_gather_calls == [True, False]


def test_rocm_target_exposes_local_logits_only_on_amd() -> None:
    assert "compute_logits_local" in HYV4ForCausalLM.__dict__
    assert "compute_logits_local" not in NvidiaHYV4ForCausalLM.__dict__


def test_rocm_target_fused_shared_expert_rewrites_checkpoint_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HYV4ForCausalLM.__new__(HYV4ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.is_fused_shared_expert_enabled = True
    model.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
    captured: list[tuple[str, torch.Tensor]] = []

    def fake_load_weights(self, weights):
        captured.extend(weights)
        return {name for name, _ in captured}

    monkeypatch.setattr(NvidiaHYV4ForCausalLM, "load_weights", fake_load_weights)
    weight = torch.arange(6).view(2, 3)
    loaded = model.load_weights(
        iter(
            [
                (
                    "model.layers.1.mlp.shared_experts.gate_proj.weight",
                    weight,
                )
            ]
        )
    )

    expected_name = "model.layers.1.mlp.experts.2.gate_proj.weight"
    assert loaded == {expected_name}
    assert len(captured) == 1
    assert captured[0][0] == expected_name
    assert torch.equal(captured[0][1], weight)


def test_rocm_mtp_local_argmax_uses_current_layer(default_vllm_config) -> None:
    hidden_states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = [
        torch.tensor([[4.0, 0.0], [0.0, 3.0], [-1.0, -1.0]]),
        torch.tensor([[0.0, 5.0], [6.0, 0.0], [-1.0, -1.0]]),
    ]
    mtp = _make_mtp(weights)

    mtp.model.spec_step_idx = 0
    step_zero = mtp.get_top_tokens(hidden_states)
    expected_zero = mtp.model.compute_logits(hidden_states).argmax(dim=-1)

    mtp.model.spec_step_idx = 1
    step_one = mtp.get_top_tokens(hidden_states)
    expected_one = mtp.model.compute_logits(hidden_states).argmax(dim=-1)

    assert torch.equal(step_zero, expected_zero)
    assert torch.equal(step_one, expected_one)
    assert not torch.equal(step_zero, step_one)


def test_rocm_mtp_exposes_local_argmax_only_on_amd() -> None:
    assert "get_top_tokens" in HYV4MTP.__dict__
    assert "get_top_tokens" not in NvidiaHYV4MTP.__dict__


def test_rocm_mtp_fused_shared_expert_rewrites_checkpoint_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mtp = HYV4MTP.__new__(HYV4MTP)
    nn.Module.__init__(mtp)
    mtp.model = nn.Module()
    mtp.model.is_fused_shared_expert_enabled = True
    mtp.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
    captured: list[tuple[str, torch.Tensor]] = []

    def fake_load_weights(self, weights):
        captured.extend(weights)
        return {name for name, _ in captured}

    monkeypatch.setattr(NvidiaHYV4MTP, "load_weights", fake_load_weights)
    scale = torch.arange(3, dtype=torch.uint8).view(1, 3)
    loaded = mtp.load_weights(
        iter(
            [
                (
                    "model.mtp_layers.0.mlp.shared_experts.down_proj.weight_scale",
                    scale,
                )
            ]
        )
    )

    expected_name = "model.mtp_layers.0.mlp.experts.2.down_proj.weight_scale"
    assert loaded == {expected_name}
    assert len(captured) == 1
    assert captured[0][0] == expected_name
    assert torch.equal(captured[0][1], scale)


@pytest.mark.skipif(
    getattr(torch.version, "hip", None) is None or not torch.accelerator.is_available(),
    reason="requires ROCm",
)
def test_rocm_mtp_local_argmax_supports_graph_replay(default_vllm_config) -> None:
    weights = [torch.randn(32, 16, device="cuda")]
    mtp = _make_mtp(weights)
    mtp.model.logits_processor = LogitsProcessor(weights[0].shape[0])
    mtp.model.logits_processor.head_dtype = weights[0].dtype
    hidden_states = torch.randn(4, 16, device="cuda")

    mtp.get_top_tokens(hidden_states)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_tokens = mtp.get_top_tokens(hidden_states)

    hidden_states.copy_(torch.randn_like(hidden_states))
    graph.replay()
    torch.accelerator.synchronize()

    expected = mtp.model.compute_logits(hidden_states).argmax(dim=-1)
    assert torch.equal(graph_tokens, expected)


def test_rocm_sparse_backend_preserves_name_and_supports_sink() -> None:
    assert HYV4ROCMAiterMLASparseBackend.get_name() == "ROCM_AITER_MLA_SPARSE"
    assert HYV4ROCMAiterMLASparseBackend.supports_sink()
    assert HYV4ROCMAiterMLASparseBackend.get_impl_cls() is HYV4ROCMAiterMLASparseImpl


def test_rocm_sparse_impl_reuses_validated_base_initialization() -> None:
    assert HYV4ROCMAiterMLASparseImpl.__init__ is ROCMAiterMLASparseImpl.__init__
