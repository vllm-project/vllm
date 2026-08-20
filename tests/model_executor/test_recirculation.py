# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.gemma4 import Gemma4Model
from vllm.model_executor.models.glm4_moe import Glm4MoeModel
from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteModel
from vllm.model_executor.models.gpt_oss import GptOssModel
from vllm.model_executor.models.interfaces import supports_recirculation
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
from vllm.model_executor.models.llama4 import Llama4Model
from vllm.model_executor.models.mimo_v2 import MiMoV2Model
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Model
from vllm.model_executor.models.mistral import MistralModel
from vllm.model_executor.models.mixtral import MixtralModel
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
)
from vllm.model_executor.models.recirculation import (
    RecirculationConfig,
    RecirculationDecoderMixin,
)
from vllm.model_executor.models.step3p5 import Step3p5Model
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3Model

pytestmark = pytest.mark.skip_global_cleanup


class _AdditiveLayer(nn.Module):
    def __init__(self, layer_idx: int, calls: list[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.calls = calls

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(self.layer_idx)
        residual = hidden_states if residual is None else hidden_states + residual
        return torch.full_like(residual, self.layer_idx + 1), residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        assert residual is not None
        return hidden_states + residual, None


def _make_llama_model(
    monkeypatch: pytest.MonkeyPatch,
    model_type: type[LlamaModel] = LlamaModel,
) -> tuple[LlamaModel, list[int]]:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(
        "vllm.model_executor.models.llama.get_pp_group", lambda: pp_group
    )
    calls: list[int] = []
    model = cast(LlamaModel, object.__new__(model_type))
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.layers = nn.ModuleList([_AdditiveLayer(i, calls) for i in range(3)])
    model.norm = _FinalNorm()
    model.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
        wavefront=True,
    )
    return model, calls


def test_llama_uses_shared_wavefront_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_llama_model(monkeypatch)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_llama_top_level_advertises_engine_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _make_llama_model(monkeypatch)
    causal_lm = LlamaForCausalLM.__new__(LlamaForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model

    assert supports_recirculation(causal_lm)


def test_mistral_delegates_to_shared_wavefront_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_llama_model(monkeypatch, MistralModel)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
        t_cond=None,
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_unvalidated_llama_subclass_does_not_inherit_adapter() -> None:
    class UnvalidatedLlamaModel(LlamaModel):
        pass

    model = UnvalidatedLlamaModel.__new__(UnvalidatedLlamaModel)

    assert not model.has_recirculation_adapter()


def test_engine_capability_rejects_incomplete_forward() -> None:
    class IncompleteModel:
        supports_recirculation = True

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            return input_ids

    assert not supports_recirculation(IncompleteModel())


@pytest.mark.parametrize(
    ("model_type", "adapter", "wavefront"),
    [
        (DeepseekV2Model, "deepseek_moe", False),
        (Gemma4Model, "gemma4", True),
        (Glm4MoeModel, "glm4_moe", True),
        (Glm4MoeLiteModel, "glm4_moe_lite", False),
        (GptOssModel, "gpt_oss_moe", False),
        (Llama4Model, "llama4_moe", True),
        (MiniMaxM2Model, "minimax_m2_moe", True),
        (MiniMaxM3Model, "minimax_m3_sparse_moe", False),
        (MiMoV2Model, "mimo_v2_moe", True),
        (MixtralModel, "mixtral", True),
        (Qwen2Model, "qwen2", True),
        (Qwen3Model, "qwen3", True),
        (Qwen3MoeModel, "qwen3_moe", True),
        (Qwen3NextModel, "qwen3_next_hybrid", False),
        (Qwen3_5Model, "qwen3_5_hybrid", False),
        (Step3p5Model, "step3p5_moe", True),
    ],
)
def test_reviewed_family_capabilities(
    model_type: type[RecirculationDecoderMixin],
    adapter: str,
    wavefront: bool,
) -> None:
    model = cast(RecirculationDecoderMixin, object.__new__(model_type))
    if isinstance(model, Gemma4Model):
        model.hidden_size_per_layer_input = 0

    capabilities = model.get_recirculation_capabilities()

    assert capabilities is not None
    assert capabilities.adapter == adapter
    assert capabilities.serial
    assert capabilities.wavefront is wavefront


def test_gemma4_per_layer_embeddings_are_serial_only() -> None:
    model = cast(Gemma4Model, object.__new__(Gemma4Model))
    model.hidden_size_per_layer_input = 16

    capabilities = model.get_recirculation_capabilities()

    assert capabilities is not None
    assert capabilities.serial
    assert not capabilities.wavefront


def test_nested_text_config_is_used_for_recirculation() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=4,
        recirculation_config={
            "source_layer": 2,
            "destination_layer": 1,
            "alpha": 0.1,
        },
    )
    wrapper_config = SimpleNamespace(get_text_config=lambda: text_config)

    config = RecirculationConfig.from_hf_config(wrapper_config)

    assert config is not None
    assert config.source_layer == 2
    assert config.destination_layer == 1


def test_qwen_next_restores_active_gdn_state_before_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix = "model.layers.0.linear_attn"
            self.kv_cache = (
                torch.arange(12, dtype=torch.float32).reshape(3, 4),
                torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
            )

    layer = cast(Qwen3NextDecoderLayer, object.__new__(Qwen3NextDecoderLayer))
    nn.Module.__init__(layer)
    layer.linear_attn = FakeLinearAttention()
    model = cast(Qwen3NextModel, object.__new__(Qwen3NextModel))
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([layer])
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=torch.tensor([2, 0], dtype=torch.int64),
        num_prefills=1,
        num_decodes=1,
    )
    context = SimpleNamespace(attn_metadata={"model.layers.0.linear_attn": metadata})
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_forward_context",
        lambda: context,
    )

    snapshot = model._capture_recirculation_layer_state(0)
    untouched = tuple(state[1].clone() for state in layer.linear_attn.kv_cache)
    for state in layer.linear_attn.kv_cache:
        state.add_(100)
    model._restore_recirculation_layer_state(0, snapshot)

    assert snapshot is not None
    state_indices, captured = snapshot
    for cache, saved, untouched_row in zip(
        layer.linear_attn.kv_cache, captured, untouched
    ):
        torch.testing.assert_close(cache.index_select(0, state_indices), saved)
        torch.testing.assert_close(cache[1], untouched_row + 100)
