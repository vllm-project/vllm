# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission and shared-weight checks for standalone MTP model families."""

import copy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, PreTrainedConfig

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.weights import materialize_target_shell


class _Model(SimpleNamespace):
    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_vocab_size(self):
        return self.hf_config.vocab_size


MODELS = (
    ("mimo", "MiMoForCausalLM", "MiMoMTPModel", 1),
    ("ernie4_5_moe", "Ernie4_5_MoeForCausalLM", "ErnieMTPModel", 1),
    ("qwen3_next", "Qwen3NextForCausalLM", "Qwen3NextMTP", 3),
    ("glm4_moe", "Glm4MoeForCausalLM", "Glm4MoeMTPModel", 3),
    ("glm4_moe_lite", "Glm4MoeLiteForCausalLM", "Glm4MoeLiteMTPModel", 3),
    ("exaone_moe", "ExaoneMoeForCausalLM", "ExaoneMoeMTP", 3),
    ("nemotron_h", "NemotronHForCausalLM", "NemotronHMTPModel", 3),
    ("hy_v3", "HYV3ForCausalLM", "HYV3MTPModel", 3),
)


def _config(model_type="ernie4_5_moe", architecture="Ernie4_5_MoeForCausalLM", k=1):
    kwargs = dict(
        architectures=[architecture],
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_nextn_predict_layers=1,
    )
    if model_type == "nemotron_h":
        kwargs["mtp_hybrid_override_pattern"] = "*E"
    # MiMo uses the generic remote-code config; the other families have
    # installed Transformers config classes. All use the production override.
    hf_config = (
        PreTrainedConfig(model_type=model_type, **kwargs)
        if model_type == "mimo"
        else AutoConfig.for_model(model_type, **kwargs)
    )
    target = _Model(
        hf_config=hf_config,
        model="test/target",
        revision="a" * 40,
        dtype=torch.bfloat16,
        quantization=None,
        tokenizer="test/target",
        tokenizer_revision="a" * 40,
        tokenizer_mode="auto",
    )
    draft = copy.deepcopy(target)
    draft.hf_config = SpeculativeConfig.hf_config_override(copy.deepcopy(hf_config))
    return SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=draft,
            num_speculative_tokens=k,
            parallel_drafting=False,
            draft_sample_method="greedy",
        ),
    )


@pytest.mark.parametrize("model_type,target_arch,draft_arch,k", MODELS)
def test_native_mtp_config_override_resolves_own_kv_contract(
    model_type, target_arch, draft_arch, k
):
    config = _config(model_type, target_arch, k)
    assert config.speculative_config.draft_model_config.hf_config.architectures == [
        draft_arch
    ]
    contract = create_runtime_contract(config)
    assert contract.capabilities.state_dependency == "own_kv"
    assert contract.capabilities.num_prefill_lookahead == 1
    assert contract.schema.slots[1].trailing_shape == ()
    assert contract.schema.slots[2].kind == "last_hidden_states"
    assert contract.schema.slots[2].trailing_shape == (16,)


@pytest.mark.parametrize("model_type,target_arch,draft_arch,k", MODELS[:2])
def test_non_recurrent_mtp_models_keep_k_one(model_type, target_arch, draft_arch, k):
    config = _config(model_type, target_arch, 2)
    with pytest.raises(ValueError, match="exactly one speculative token"):
        create_runtime_contract(config)


@pytest.mark.parametrize("model_type,target_arch,draft_arch,k", MODELS)
def test_mtp_contract_rejects_missing_or_multiple_checkpoint_modules(
    model_type, target_arch, draft_arch, k
):
    config = _config(model_type, target_arch, k)
    draft = config.speculative_config.draft_model_config.hf_config
    for count in (0, 2):
        draft.num_nextn_predict_layers = count
        with pytest.raises(ValueError, match="exactly one checkpoint MTP module"):
            create_runtime_contract(config)


@pytest.mark.parametrize("kind", ["architecture", "hidden", "vocab"])
def test_mtp_contract_does_not_accept_unrelated_target(kind):
    config = _config()
    target = config.model_config.hf_config
    if kind == "architecture":
        target.architectures = ["Qwen3ForCausalLM"]
    elif kind == "hidden":
        target.hidden_size *= 2
    else:
        target.vocab_size += 1
    with pytest.raises(ValueError, match="matching target"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "architecture",
    ["MiMoMTP", "MiMoV2MTPModel", "Gemma4MTPModel", "Qwen3_5MTP", "DeepSeekV4MTPModel"],
)
def test_unsupported_feature_and_state_families_remain_fail_closed(architecture):
    config = _config()
    config.speculative_config.draft_model_config.hf_config.architectures = [
        architecture
    ]
    with pytest.raises(ValueError, match="unsupported standalone MTP architecture"):
        create_runtime_contract(config)


@pytest.mark.parametrize("field,value", [("hc_mult", 4), ("text_config", {})])
def test_target_feature_contract_rejects_nested_or_multistream_state(field, value):
    config = _config()
    setattr(config.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="scalar positions and text features"):
        create_runtime_contract(config)


def test_mrope_is_not_silently_encoded_as_scalar_positions():
    config = _config()
    config.model_config.hf_config.rope_scaling = {"mrope_section": [2, 2, 4]}
    with pytest.raises(ValueError, match="scalar positions and text features"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "layer_types",
    [[], ["moe"], ["linear_attention", "full_attention", "moe"], ["mlp", "moe"]],
)
def test_nemotron_mtp_does_not_accept_target_mamba_state(layer_types):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3)
    draft = config.speculative_config.draft_model_config.hf_config
    draft.mtp_layers_block_type = layer_types
    with pytest.raises(ValueError, match="attention/MoE"):
        create_runtime_contract(config)


def test_glm_dense_mla_contract_rejects_sparse_indexer_variant():
    config = _config("glm4_moe_lite", "Glm4MoeLiteForCausalLM", 3)
    config.speculative_config.draft_model_config.hf_config.index_topk = 2048
    with pytest.raises(ValueError, match="sparse indexer"):
        create_runtime_contract(config)


@pytest.fixture
def shell_layers(monkeypatch):
    import vllm.model_executor.layers.vocab_parallel_embedding as embeddings

    class Layer(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size, *, params_dtype, quant_config):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(vocab_size, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )

        def weight_loader(self, parameter, loaded):
            parameter.copy_(loaded)

        def tie_weights(self, embedding):
            self.weight = embedding.weight
            return self

    monkeypatch.setattr(embeddings, "VocabParallelEmbedding", Layer)
    monkeypatch.setattr(embeddings, "ParallelLMHead", Layer)


def test_nemotron_shared_embedding_uses_checkpoint_mapper_and_pinned_revision(
    monkeypatch, shell_layers
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.hf_config._commit_hash = "c" * 40
    embedding = torch.arange(512, dtype=torch.bfloat16).reshape(32, 16)
    head = embedding + 16
    calls = []

    def selected(source, revision, names):
        calls.append((source, revision, names))
        return {"backbone.embeddings.weight": embedding, "lm_head.weight": head}

    monkeypatch.setattr(weights, "selected_checkpoint_weights", selected)
    shell = materialize_target_shell(config, torch.device("cpu"))
    assert calls == [
        (
            "test/target",
            "a" * 40,
            (
                "backbone.embeddings.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
            ),
        )
    ]
    torch.testing.assert_close(shell.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(shell.lm_head.weight, head)
    assert list(dict(shell.named_parameters())) == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]


@pytest.mark.parametrize(
    "embedding_name", ["model.embed_tokens.weight", "backbone.embeddings.weight"]
)
def test_tied_nemotron_checkpoint_needs_no_separate_head(
    tmp_path, shell_layers, embedding_name
):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = True
    embedding = torch.randn(32, 16, dtype=torch.bfloat16)
    save_file({embedding_name: embedding}, tmp_path / "model.safetensors")
    shell = materialize_target_shell(config, torch.device("cpu"))
    torch.testing.assert_close(shell.lm_head.weight, embedding)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "tied,include_head", [(True, False), (True, True), (False, True)]
)
def test_shared_weight_storage_preserves_tying(
    tmp_path, monkeypatch, device, tied, include_head
):
    import vllm.model_executor.layers.vocab_parallel_embedding as embeddings
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    monkeypatch.setattr(embeddings, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(embeddings, "get_tensor_model_parallel_world_size", lambda: 1)
    config = _config().model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = tied
    embedding = torch.arange(512, dtype=config.dtype).reshape(32, 16)
    checkpoint = {"model.embed_tokens.weight": embedding}
    if include_head:
        checkpoint["lm_head.weight"] = embedding.clone() if tied else embedding + 16
    save_file(checkpoint, tmp_path / "model.safetensors")
    requested = []
    select_weights = weights.selected_checkpoint_weights
    allocations = []
    create_weights = embeddings.UnquantizedEmbeddingMethod.create_weights

    def select(source, revision, names):
        requested.extend(names)
        return select_weights(source, revision, names)

    def allocate(self, layer, *args, **kwargs):
        create_weights(self, layer, *args, **kwargs)
        allocations.append(layer.weight.device.type)

    monkeypatch.setattr(weights, "selected_checkpoint_weights", select)
    monkeypatch.setattr(
        embeddings.UnquantizedEmbeddingMethod, "create_weights", allocate
    )
    shell = materialize_target_shell(config, torch.device(device))
    embed = shell.model.embed_tokens.weight
    head = shell.lm_head.weight
    assert (embed is head) is tied
    torch.testing.assert_close(embed[:32].cpu(), embedding)
    torch.testing.assert_close(
        head[:32].cpu(), embedding if tied else checkpoint["lm_head.weight"]
    )
    assert ("lm_head.weight" in requested) is not tied
    assert allocations == [device, "meta" if tied else device]
    storages = {
        p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
        for p in shell.parameters()
    }
    matrix_bytes = embeddings.pad_vocab_size(32) * 16 * embedding.element_size()
    assert sum(storages.values()) == matrix_bytes * (1 if tied else 2)
    shell.register_parameter(
        "draft_weight",
        torch.nn.Parameter(torch.empty(3, dtype=config.dtype, device=device)),
    )
    assert weights.model_weight_bytes(shell) == (
        3 * embedding.element_size(),
        matrix_bytes,
        0 if tied else matrix_bytes,
    )


def test_conflicting_nemotron_embedding_aliases_are_rejected(tmp_path, shell_layers):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.model = str(tmp_path)
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros(32, 16),
            "backbone.embeddings.weight": torch.ones(32, 16),
        },
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="exactly one embedding weight"):
        materialize_target_shell(config, torch.device("cpu"))


@pytest.mark.parametrize(
    "weight",
    [
        torch.zeros(32, 15),
        torch.zeros(32, 16, dtype=torch.int8),
        torch.zeros(32, 16, dtype=torch.float8_e4m3fn),
    ],
)
def test_shared_weights_cannot_silently_reinterpret_shape_or_quantization(
    tmp_path, shell_layers, weight
):
    config = _config().model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = True
    save_file({"model.embed_tokens.weight": weight}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="requires an unquantized"):
        materialize_target_shell(config, torch.device("cpu"))


@pytest.mark.parametrize(
    "tied,changed_name",
    [
        (False, "model.embed_tokens.weight"),
        (False, "lm_head.weight"),
        (True, "model.embed_tokens.weight"),
    ],
)
def test_alternate_shared_weights_must_match_the_target_before_loading(
    tmp_path, shell_layers, tied, changed_name
):
    config = _config().model_config
    target, alternate = tmp_path / "target", tmp_path / "shared"
    target.mkdir()
    alternate.mkdir()
    config.model = str(target)
    config.hf_config.tie_word_embeddings = tied
    tensors = {"model.embed_tokens.weight": torch.ones(32, 16)}
    if not tied:
        tensors["lm_head.weight"] = torch.zeros(32, 16)
    save_file(tensors, target / "model.safetensors")
    save_file(tensors, alternate / "model.safetensors")
    shell = materialize_target_shell(
        config, torch.device("cpu"), shared_weights_source=str(alternate)
    )
    assert torch.equal(
        shell.model.embed_tokens.weight, tensors["model.embed_tokens.weight"]
    )
    tensors[changed_name][0, 0] = 2
    save_file(tensors, alternate / "model.safetensors")
    with pytest.raises(ValueError, match="differ from the target"):
        materialize_target_shell(
            config, torch.device("cpu"), shared_weights_source=str(alternate)
        )
    with pytest.raises(ValueError, match="unpinned alternate repositories"):
        materialize_target_shell(
            config, torch.device("cpu"), shared_weights_source="other/repo"
        )
