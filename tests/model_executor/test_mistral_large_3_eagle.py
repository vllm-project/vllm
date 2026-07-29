# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config.compilation import CompilationMode
from vllm.model_executor.models import deepseek_v2 as deepseek_mod
from vllm.model_executor.models import mistral_large_3_eagle as eagle_mod


class DummyPPGroup:
    world_size = 1
    is_first_rank = True
    is_last_rank = True


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, *args, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        return torch.zeros(
            (*input_ids.shape, self.hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )


class DummyLinear(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.zeros(
            (*x.shape[:-1], self.out_features),
            dtype=x.dtype,
            device=x.device,
        )


class DummyNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states, residual=None):
        return hidden_states, residual


class DummyDecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
        return hidden_states, residual


def make_vllm_config(
    *, model_type="mistral3", qk_nope_head_dim=128, qk_rope_head_dim=64
):
    hf_config = SimpleNamespace(
        model_type=model_type,
        first_k_dense_replace=0,
        vocab_size=32000,
        hidden_size=16,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.fixture(autouse=True)
def patch_heavy_modules(monkeypatch):
    monkeypatch.setattr(eagle_mod, "get_pp_group", lambda: DummyPPGroup())
    monkeypatch.setattr(deepseek_mod, "get_pp_group", lambda: DummyPPGroup())

    monkeypatch.setattr(eagle_mod, "VocabParallelEmbedding", DummyEmbedding)
    monkeypatch.setattr(eagle_mod, "RowParallelLinear", DummyLinear)
    monkeypatch.setattr(eagle_mod, "RMSNorm", DummyNorm)
    monkeypatch.setattr(eagle_mod, "DeepseekV2DecoderLayer", DummyDecoderLayer)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("model_type", "qk_nope_head_dim", "qk_rope_head_dim", "expected_use_mha"),
    [
        # MLA-style config: should not use MHA.
        ("mistral3", 128, 64, False),
        # No MLA dims: should use MHA, matching DeepseekV2Model.__init__ logic.
        ("mistral3", 0, 0, True),
        # DeepSeek model type always uses MHA by the parent logic.
        ("deepseek", 128, 64, True),
    ],
)
def test_eagle_mistral_large3_initializes_deepseek_runtime_attrs(
    model_type,
    qk_nope_head_dim,
    qk_rope_head_dim,
    expected_use_mha,
):
    vllm_config = make_vllm_config(
        model_type=model_type,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    model = eagle_mod.EagleMistralLarge3Model(vllm_config=vllm_config)

    assert model.aux_hidden_state_layers == ()
    assert model.use_mha is expected_use_mha

    # Add this if your fix also copies num_redundant_experts from
    # DeepseekV2Model.__init__.
    assert model.num_redundant_experts == 0


@pytest.mark.cpu_test
def test_eagle_mistral_large3_forward_reuses_deepseek_parent_forward():
    vllm_config = make_vllm_config()
    model = eagle_mod.EagleMistralLarge3Model(vllm_config=vllm_config)

    input_ids = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[0, 1, 2]])
    hidden_states = torch.zeros((1, 3, 16))

    output = model(input_ids, positions, hidden_states)

    assert isinstance(output, torch.Tensor)
    assert output.shape == hidden_states.shape
