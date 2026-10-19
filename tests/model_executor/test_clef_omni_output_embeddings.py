# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU loading and lexical lookup for Clef-Omni's output embedding table."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor import parameter
from vllm.model_executor.layers import vocab_parallel_embedding as vocab
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.models import qwen3_moe as qwen
from vllm.model_executor.models import qwen3_omni_moe_thinker as omni
from vllm.model_executor.models.clef_omni import ClefOmniForJointSchema
from vllm.utils.torch_utils import set_default_torch_dtype


def _quant_config():
    # The checkpoint excludes input embeddings but stores its output table as
    # BF16 too, without an lm_head ignore entry. Keep that distinction here.
    config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "config_groups": {
            "group_0": {
                "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"],
            }
        },
        "ignore": [
            "thinker.model.embed_tokens",
            "thinker.model.layers.0.mlp.gate",
            "thinker.model.layers.0.self_attn*",
            "thinker.audio_tower*",
            "thinker.visual*",
        ],
    }
    result = modelopt.ModelOptNvFp4Config.from_config(config)
    result.apply_vllm_mapper(
        omni.Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper()
    )
    result.packed_modules_mapping = (
        omni.Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping
    )
    return result


@pytest.fixture
def make_model(monkeypatch):
    # Decoder/tower construction, distributed metadata, and GPU kernel choice
    # are outside these CPU loading tests. The actual Clef constructor, inner
    # Omni constructor, ModelOpt allocators, weight loaders and mapper execute.
    monkeypatch.setattr(
        qwen, "make_layers", lambda *a, **kw: (0, 0, torch.nn.ModuleList())
    )
    monkeypatch.setattr(modelopt, "select_linear_kernel", lambda *a, **kw: object())
    monkeypatch.setattr(modelopt, "expose_input_quant_key", lambda *a, **kw: None)

    def make(
        *, quantized=True, hidden_size=32, vocab_size=65, rank=0, tp_size=1, tied=False
    ):
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            parameter, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
        monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            vocab, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
        config = VllmConfig(
            compilation_config=CompilationConfig(mode=CompilationMode.NONE)
        )
        head_config = dict(
            hidden_size=hidden_size,
            width=16,
            routing_layers=1,
            layers=1,
            heads=2,
            feedforward=32,
        )
        text = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=tied,
            mlp_only_layers=[],
            joint_head_config=head_config,
        )
        config.model_config = SimpleNamespace(
            hf_config=text,
            hf_text_config=text,
            dtype=torch.bfloat16,
            head_dtype=None,
            model="local-checkpoint",
            revision="pinned",
        )
        config.quant_config = _quant_config() if quantized else None
        config.scheduler_config.enable_chunked_prefill = False
        config.scheduler_config.async_scheduling = False

        def init_towers(self, *, vllm_config, prefix=""):
            torch.nn.Module.__init__(self)
            self.quant_config = vllm_config.quant_config
            self.language_model = omni.Qwen3MoeLLMForCausalLM(
                vllm_config=vllm_config, prefix="language_model"
            )

        monkeypatch.setattr(
            omni.Qwen3OmniMoeThinkerForConditionalGeneration, "__init__", init_towers
        )
        with set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16):
            model = ClefOmniForJointSchema(vllm_config=config)
        return model, config

    return make


def _tables(vocab_size, hidden_size):
    cells = torch.arange(vocab_size * hidden_size).reshape(vocab_size, hidden_size)
    return (cells % 17).to(torch.bfloat16), (100 + cells % 19).to(torch.bfloat16)


def _load_tables(model, input_table, output_table):
    return model.load_weights(
        [
            ("thinker.model.embed_tokens.weight", input_table),
            ("thinker.lm_head.weight", output_table),
        ]
    )


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("hidden_size", [32, 2048])
def test_output_table_loads_and_lexical_uses_output_values(
    make_model, quantized, hidden_size
):
    model, config = make_model(quantized=quantized, hidden_size=hidden_size)
    input_table, output_table = _tables(65, hidden_size)
    names = _load_tables(model, input_table, output_table)
    assert names == {
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    }
    table = model.language_model.lm_head
    assert isinstance(table, vocab.VocabParallelEmbedding)
    assert isinstance(table.quant_method, vocab.UnquantizedEmbeddingMethod)
    assert table.params_dtype == torch.bfloat16
    assert table.weight.shape == (128, hidden_size)
    assert table.weight.dtype == torch.bfloat16
    torch.testing.assert_close(table.weight[:65], output_table)
    torch.testing.assert_close(table.weight[65:], torch.zeros_like(table.weight[65:]))
    torch.testing.assert_close(
        model.language_model.model.embed_tokens.weight[:65], input_table
    )
    tokens = torch.tensor([0, 3, 64])
    actual = model._embed_lexical(tokens)
    torch.testing.assert_close(actual, F.embedding(tokens, output_table))
    assert not torch.equal(actual, F.embedding(tokens, input_table))
    assert model.secondary_weights[0].prefix == "pooler.head."
    assert config.quant_config is model.quant_config


@pytest.mark.parametrize("rank", [0, 1])
def test_output_table_preserves_tp_shard_and_vocab_padding(make_model, rank):
    model, _ = make_model(rank=rank, tp_size=2)
    input_table, output_table = _tables(65, 32)
    _load_tables(model, input_table, output_table)
    table = model.language_model.lm_head
    assert table.tp_rank == rank and table.tp_size == 2
    assert not table.disable_tp and table.reduce_results
    assert table.parallel_group is None
    assert table.org_vocab_size == 65 and table.num_embeddings == 65
    assert table.padding_size == 64 and table.num_embeddings_padded == 128
    assert table.weight.shape == (64, 32)
    start = rank * 64
    count = min(64, 65 - start)
    torch.testing.assert_close(
        table.weight[:count], output_table[start : start + count]
    )
    torch.testing.assert_close(
        table.weight[count:], torch.zeros_like(table.weight[count:])
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_tied_table_keeps_input_parameter_identity(make_model, quantized):
    model, _ = make_model(quantized=quantized, tied=True)
    assert (
        model.language_model.lm_head.weight
        is model.language_model.model.embed_tokens.weight
    )
    table, _ = _tables(65, 32)
    model.load_weights([("thinker.model.embed_tokens.weight", table)])
    torch.testing.assert_close(
        model._embed_lexical(torch.tensor([1, 64])), table[[1, 64]]
    )


def test_output_table_does_not_accept_wrong_hidden_width(make_model):
    model, _ = make_model()
    input_table, _ = _tables(65, 32)
    with pytest.raises(RuntimeError, match="size of tensor"):
        _load_tables(model, input_table, torch.zeros(65, 16, dtype=torch.bfloat16))


def test_output_table_does_not_change_expert_quantization(make_model, monkeypatch):
    model, config = make_model()
    cfg = config.quant_config
    expert = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(expert)
    expert.moe_config = SimpleNamespace()
    monkeypatch.setattr(
        modelopt,
        "select_nvfp4_moe_backend",
        lambda **kw: (modelopt.NvFp4MoeBackend.VLLM_CUTLASS, object),
    )
    method = cfg.get_quant_method(expert, "language_model.model.layers.0.mlp.experts")
    assert isinstance(method, modelopt.ModelOptNvFp4FusedMoE)
    assert method.quant_config is cfg
    assert cfg.quant_method == "NVFP4"
    assert isinstance(
        model.language_model.lm_head.quant_method, vocab.UnquantizedEmbeddingMethod
    )


def test_lexical_lookup_keeps_existing_unmapped_lora_wrapper_boundary(make_model):
    from vllm.lora.layers.vocab_parallel_embedding import (
        VocabParallelEmbeddingWithLoRA,
    )

    model, _ = make_model()
    input_table, output_table = _tables(65, 32)
    _load_tables(model, input_table, output_table)
    model.language_model.lm_head = VocabParallelEmbeddingWithLoRA(
        model.language_model.lm_head
    )
    # No adapter mapping or Punica state is installed. The existing lexical
    # path bypasses wrapper.forward and reads the base output table as before.
    tokens = torch.tensor([1, 64])
    torch.testing.assert_close(model._embed_lexical(tokens), output_table[tokens])
