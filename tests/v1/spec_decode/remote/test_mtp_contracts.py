# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission and shared-weight checks for standalone MTP model families."""

import copy
import json
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, PreTrainedConfig, Qwen3Config

from tests.v1.e2e.spec_decode.remote.test_remote_mtp import _separate_mtp_draft
from tests.v1.e2e.spec_decode.remote.test_remote_mtp import (
    tiny_ernie_mtp_model as tiny_ernie_mtp_model,
)
from tests.v1.e2e.spec_decode.remote.test_remote_mtp import (
    tiny_hyv3_mtp_model as tiny_hyv3_mtp_model,
)
from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from tests.v1.spec_decode.remote.runtime_fixtures import make_config
from vllm.config import ModelConfig, SpeculativeConfig
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.weights import materialize_target_shell

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
NEMOTRON_EMBEDDING_NAMES = (
    "model.embed_tokens.weight",
    "model.embeddings.weight",
    "backbone.embed_tokens.weight",
    "backbone.embeddings.weight",
)

pytestmark = pytest.mark.usefixtures("cpu_parallel_state")


def _mtp_context_config(
    source, destination, *, target_length, draft_length=16, draft_max_length=None, k=1
):
    from vllm.engine.arg_utils import EngineArgs

    draft_path = _separate_mtp_draft(source, destination)
    config_path = Path(draft_path) / "config.json"
    draft_config = json.loads(config_path.read_text())
    draft_config["max_position_embeddings"] = draft_length
    config_path.write_text(json.dumps(draft_config))
    speculative_config = {
        "method": "mtp",
        "model": draft_path,
        "num_speculative_tokens": k,
    }
    if draft_max_length is not None:
        speculative_config["max_model_len"] = draft_max_length
    return EngineArgs(
        model=source,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=target_length,
        enforce_eager=True,
        speculative_config=speculative_config,
    ).create_engine_config()


@pytest.mark.parametrize("family,k", [("ernie", 1), ("hyv3", 3)])
@pytest.mark.parametrize("draft_length,draft_max_length", [(16, None), (128, 16)])
def test_mtp_public_contract_rejects_short_draft_context(
    request, tmp_path, family, k, draft_length, draft_max_length
):
    """Reject a short native draft before prefill, decode or rollout can run."""
    source = request.getfixturevalue(f"tiny_{family}_mtp_model")
    config = _mtp_context_config(
        source,
        tmp_path / "short-draft",
        target_length=64,
        draft_length=draft_length,
        draft_max_length=draft_max_length,
        k=k,
    )
    assert config.speculative_config.draft_model_config.max_model_len == 16
    with pytest.raises(ValueError, match="standalone MTP.*draft.*max_model_len"):
        create_runtime_contract(config)


@pytest.mark.parametrize("prefill_length", [15, 16, 17])
def test_mtp_admitted_context_keeps_planned_positions_inside_native_rope(
    request, tmp_path, prefill_length
):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.spec_decode.remote.protocol import SequenceKey
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow

    config = _mtp_context_config(
        request.getfixturevalue("tiny_ernie_mtp_model"),
        tmp_path / "short-draft",
        target_length=16,
    )
    create_runtime_contract(config)
    draft = config.speculative_config.draft_model_config
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        model = get_model_loader(config.load_config).create_model(config, draft)
    rope = next(iter(model.model.layers.values())).mtp_block.self_attn.rotary_emb
    assert rope.cos_sin_cache.shape[0] == 16
    planner = DraftStepPlanner(
        DraftBlockPool(32, 2),
        max_num_reqs=1,
        max_model_len=config.model_config.max_model_len,
        num_speculative_tokens=1,
    )
    key = SequenceKey("context-boundary", 0, 1)
    assert planner.open(key)
    plan, failures = planner.plan([DraftStepRow(key, prefill_length, 1)])
    if prefill_length > 16:
        assert not plan.rows
        assert "max_model_len=16" in failures[0][1]
    else:
        assert not failures
        positions = torch.from_numpy(plan.positions)
        query = torch.zeros(prefill_length, draft.get_hidden_size())
        actual, _ = rope.forward_native(positions, query, query.clone())
        assert torch.equal(actual, query)
        assert positions.tolist() == list(range(prefill_length))


@pytest.mark.parametrize("mtp_family", ["hyv3", "ernie"])
def test_native_mtp_uses_separate_draft_checkpoint_settings(
    request, tmp_path, mtp_family
):
    """Admitted draft metadata must determine the loaded native head's math."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    source = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    draft_path = _separate_mtp_draft(source, tmp_path / "separate-draft")
    config = EngineArgs(
        model=source,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "model": draft_path,
            "num_speculative_tokens": 1,
        },
    ).create_engine_config()
    target_config = config.model_config.hf_config.to_dict()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    create_runtime_contract(config)
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        loader.load_weights(model, draft)
        layer = next(iter(model.model.layers.values()))
        norm = layer.enorm if mtp_family == "hyv3" else layer.mtp_emb_norm
        inputs = torch.full((1, draft.get_hidden_size()), 0.01, dtype=torch.float32)
        expected = inputs * torch.rsqrt(
            inputs.square().mean(-1, keepdim=True) + draft.hf_config.rms_norm_eps
        )
        expected = (expected.to(norm.weight.dtype) * norm.weight).to(inputs.dtype)
        assert torch.equal(norm.forward_native(inputs), expected)
        assert norm.variance_epsilon == draft.hf_config.rms_norm_eps
        assert (
            layer.mtp_block.self_attn.rotary_emb.base
            == draft.hf_config.rope_parameters["rope_theta"]
        )
    assert config.model_config.hf_config.to_dict() == target_config


@cache
def _template(model_type, architecture, k):
    kwargs = dict(
        architectures=[architecture],
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_nextn_predict_layers=1,
        max_position_embeddings=64,
    )
    if model_type == "nemotron_h":
        kwargs["mtp_hybrid_override_pattern"] = "*E"
    hf_config = (
        PreTrainedConfig(model_type=model_type, **kwargs)
        if model_type == "mimo"
        else AutoConfig.for_model(model_type, **kwargs)
    )
    directory = TemporaryDirectory(prefix="vllm-mtp-contract-")
    root = Path(directory.name)
    Qwen3Config(**kwargs).save_pretrained(root)
    target = ModelConfig(
        model=str(root),
        dtype="bfloat16",
        max_model_len=64,
        skip_tokenizer_init=True,
        hf_overrides=lambda _: copy.deepcopy(hf_config),
    )
    config = make_config()
    config.model_config = target
    draft = ModelConfig(
        model=str(root),
        runner="draft",
        dtype="bfloat16",
        max_model_len=64,
        skip_tokenizer_init=True,
        hf_overrides=lambda _: SpeculativeConfig.hf_config_override(
            copy.deepcopy(hf_config)
        ),
    )
    config.speculative_config.method = "mtp"
    config.speculative_config.num_speculative_tokens = k
    config.speculative_config.draft_model_config = draft
    # Schema tests vary checkpoint identities independently from parsed metadata.
    target.model = target.tokenizer = "test/target"
    target.revision = target.tokenizer_revision = "a" * 40
    draft = config.speculative_config.draft_model_config
    draft.model, draft.revision = "test/target", "a" * 40
    return directory, config


def _config(model_type="ernie4_5_moe", architecture="Ernie4_5_MoeForCausalLM", k=1):
    return copy.deepcopy(_template(model_type, architecture, k)[1])


@pytest.mark.parametrize("draft_limit", [16, 64])
def test_mtp_public_config_requires_draft_context_to_cover_target(draft_limit):
    """A public draft config must cover every context admitted by the target."""
    directory, _ = _template("ernie4_5_moe", "Ernie4_5_MoeForCausalLM", 1)
    config = _config()
    save_file(
        {"identity_only": torch.zeros(1)}, Path(directory.name) / "model.safetensors"
    )
    hf_config = copy.deepcopy(config.model_config.hf_config)
    hf_config.max_position_embeddings = draft_limit
    draft = ModelConfig(
        model=directory.name,
        runner="draft",
        dtype="bfloat16",
        max_model_len=draft_limit,
        skip_tokenizer_init=True,
        hf_overrides=lambda _: SpeculativeConfig.hf_config_override(hf_config),
    )
    assert draft.max_model_len == draft_limit
    config.speculative_config.draft_model_config = draft
    if draft_limit < config.model_config.max_model_len:
        with pytest.raises(ValueError, match="standalone MTP.*draft.*max_model_len"):
            create_runtime_contract(config)
    else:
        assert create_runtime_contract(config).capabilities.state_dependency == "own_kv"


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
    config.model_config.model_arch_config = config.model_config.get_model_arch_config()
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


@pytest.mark.parametrize("embedding_name", NEMOTRON_EMBEDDING_NAMES)
def test_nemotron_shared_embedding_uses_checkpoint_mapper_and_pinned_revision(
    monkeypatch, tmp_path, embedding_name
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.hf_config._commit_hash = "c" * 40
    embedding = torch.arange(512, dtype=torch.bfloat16).reshape(32, 16)
    head = embedding + 16
    calls = []

    save_file(
        {embedding_name: embedding, "lm_head.weight": head},
        tmp_path / "model.safetensors",
    )

    class HubService:
        def hf_hub_download(self, source, filename, *, revision):
            calls.append((source, revision))
            return str(tmp_path / filename)

    monkeypatch.setattr(weights, "hf_api", HubService)
    shell = materialize_target_shell(config, torch.device("cpu"))
    assert calls == [("test/target", "a" * 40)] * 2
    torch.testing.assert_close(shell.model.embed_tokens.weight[:32], embedding)
    torch.testing.assert_close(shell.lm_head.weight[:32], head)
    assert list(dict(shell.named_parameters())) == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]


@pytest.mark.parametrize("embedding_name", NEMOTRON_EMBEDDING_NAMES)
def test_tied_nemotron_checkpoint_needs_no_separate_head(tmp_path, embedding_name):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = True
    embedding = torch.randn(32, 16, dtype=torch.bfloat16)
    save_file({embedding_name: embedding}, tmp_path / "model.safetensors")
    shell = materialize_target_shell(config, torch.device("cpu"))
    assert shell.model.embed_tokens.weight is shell.lm_head.weight
    torch.testing.assert_close(shell.model.embed_tokens.weight[:32], embedding)
    torch.testing.assert_close(shell.lm_head.weight[:32], embedding)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "tied,include_head", [(True, False), (True, True), (False, True)]
)
def test_shared_weight_storage_preserves_tying(tmp_path, device, tied, include_head):
    import vllm.model_executor.layers.vocab_parallel_embedding as embeddings
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    config = _config().model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = tied
    embedding = torch.arange(512, dtype=config.dtype).reshape(32, 16)
    checkpoint = {"model.embed_tokens.weight": embedding}
    if include_head:
        checkpoint["lm_head.weight"] = embedding.clone() if tied else embedding + 16
    save_file(checkpoint, tmp_path / "model.safetensors")
    shell = materialize_target_shell(config, torch.device(device))
    embed = shell.model.embed_tokens.weight
    head = shell.lm_head.weight
    assert (embed is head) is tied
    torch.testing.assert_close(embed[:32].cpu(), embedding)
    torch.testing.assert_close(
        head[:32].cpu(), embedding if tied else checkpoint["lm_head.weight"]
    )
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


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "embedding_names",
    [
        ("model.embed_tokens.weight", "backbone.embeddings.weight"),
        ("model.embed_tokens.weight", "model.embeddings.weight"),
        ("backbone.embeddings.weight", "backbone.embed_tokens.weight"),
    ],
)
def test_conflicting_nemotron_embedding_aliases_are_rejected(
    tmp_path, tied, embedding_names
):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = tied
    # Even identical values cannot disambiguate two checkpoint names that map
    # to the same native embedding parameter.
    checkpoint = {name: torch.zeros(32, 16) for name in embedding_names}
    if not tied:
        checkpoint["lm_head.weight"] = torch.ones(32, 16)
    save_file(checkpoint, tmp_path / "model.safetensors")
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
    tmp_path, weight
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
    tmp_path, tied, changed_name
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
        shell.model.embed_tokens.weight[:32], tensors["model.embed_tokens.weight"]
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


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "omitted,tracking,packed",
    [
        (None, None, False),
        (None, None, True),
        ("eh_proj", None, False),
        ("final_layernorm", None, False),
        ("self_attn.q_proj", None, False),
        ("self_attn.qkv_proj", None, True),
        ("mlp.shared_mlp.gate_proj", None, False),
        ("mlp.experts.0.gate_proj", None, False),
        ("mlp.experts.0.down_proj", None, False),
        ("mlp.experts", None, False),
        ("eh_proj", False, False),
    ],
)
def test_hyv3_native_load_rejects_uninitialized_mtp_parameters(
    request, omitted, tracking, packed, tied
):
    """Whole missing MTP tensors must not bypass native loader completeness."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    source = Path(request.getfixturevalue("tiny_hyv3_mtp_model"))
    config_path = source / "config.json"
    hf_config = json.loads(config_path.read_text())
    hf_config["tie_word_embeddings"] = tied
    config_path.write_text(json.dumps(hf_config))
    path = source / "model.safetensors"
    weights = load_file(path)
    if tied:
        del weights["lm_head.weight"]
    reference = weights.copy()
    prefix = "model.layers.2."
    if packed:
        q, k, v = (
            weights.pop(prefix + f"self_attn.{part}_proj.weight")
            for part in ("q", "k", "v")
        )
        weights[prefix + "self_attn.qkv_proj.weight"] = torch.cat(
            (
                q.reshape(2, 2, 32, 128),
                k.reshape(2, 1, 32, 128),
                v.reshape(2, 1, 32, 128),
            ),
            dim=1,
        ).reshape(-1, 128)
        weights[prefix + "mlp.shared_mlp.gate_and_up_proj.weight"] = torch.cat(
            [
                weights.pop(prefix + f"mlp.shared_mlp.{part}_proj.weight")
                for part in ("up", "gate")
            ]
        )
    if omitted is not None:
        missing_weights = [
            name for name in weights if name.startswith(prefix + omitted + ".")
        ]
        assert missing_weights
        for name in missing_weights:
            del weights[name]
        if omitted == "self_attn.q_proj":
            # A duplicate K write must not count as the absent Q component.
            name = prefix + "self_attn.k_proj.weight"
            save_file({name: weights[name]}, source / "duplicate.safetensors")
    save_file(weights, path, metadata={"format": "pt"})
    config = EngineArgs(
        model=str(source),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        model_loader_extra_config={"enable_weights_track": tracking},
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    assert draft is not None
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        if omitted is not None and tracking is not False:
            missing = {
                "self_attn.q_proj": "qkv_proj",
                "mlp.shared_mlp.gate_proj": "gate_up_proj",
                "mlp.experts.0.gate_proj": "w13_weight",
                "mlp.experts.0.down_proj": "w2_weight",
            }.get(omitted, omitted)
            with pytest.raises(
                ValueError, match=rf"not initialized.*checkpoint.*{missing}"
            ):
                loader.load_weights(model, draft)
        else:
            if omitted is not None:
                with torch.no_grad():
                    model.get_parameter(f"model.layers.2.{omitted}.weight").fill_(-2)
            loader.load_weights(model, draft)
            for name in ("eh_proj", "final_layernorm", "enorm", "hnorm"):
                actual = model.get_parameter(f"model.layers.2.{name}.weight")
                if name == omitted:
                    assert torch.equal(actual, torch.full_like(actual, -2))
                    continue
                assert torch.equal(actual, weights[f"model.layers.2.{name}.weight"])
            assert torch.equal(
                model.get_parameter("model.layers.2.shared_head.head.weight"),
                weights["model.embed_tokens.weight" if tied else "lm_head.weight"],
            )
            for native, checkpoint_parts in (
                (
                    "self_attn.qkv_proj.weight",
                    (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    ),
                ),
                (
                    "mlp.shared_mlp.gate_up_proj.weight",
                    (
                        "mlp.shared_mlp.gate_proj.weight",
                        "mlp.shared_mlp.up_proj.weight",
                    ),
                ),
            ):
                assert torch.equal(
                    model.get_parameter(prefix + "mtp_block." + native),
                    torch.cat([reference[prefix + name] for name in checkpoint_parts]),
                )
            for native, parts in (
                ("w13_weight", ("gate", "up")),
                ("w2_weight", ("down",)),
            ):
                assert torch.equal(
                    model.get_parameter(
                        prefix + "mtp_block.mlp.experts.routed_experts." + native
                    ),
                    torch.stack(
                        [
                            torch.cat(
                                [
                                    reference[
                                        prefix
                                        + f"mlp.experts.{expert}.{part}_proj.weight"
                                    ]
                                    for part in parts
                                ]
                            )
                            for expert in range(4)
                        ]
                    ),
                )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("mode", ["target_shared", "standalone", "missing_fusion"])
def test_hyv3_draft_only_checkpoint_preserves_target_shared_parameters(
    tmp_path, request, tied, mode
):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.spec_decode.remote.runtime.weights import materialize_target_shell

    target = Path(request.getfixturevalue("tiny_hyv3_mtp_model"))
    metadata = json.loads((target / "config.json").read_text())
    metadata["tie_word_embeddings"] = tied
    (target / "config.json").write_text(json.dumps(metadata))
    draft_path = tmp_path / "draft-only"
    draft_path.mkdir()
    (draft_path / "config.json").write_text(json.dumps(metadata))
    weights = load_file(target / "model.safetensors")
    if tied:
        del weights["lm_head.weight"]
        save_file(weights, target / "model.safetensors", metadata={"format": "pt"})
    draft_weights = {
        name: weight
        for name, weight in weights.items()
        if name.startswith("model.layers.2.")
    }
    if mode == "missing_fusion":
        del draft_weights["model.layers.2.eh_proj.weight"]
    save_file(
        draft_weights, draft_path / "model.safetensors", metadata={"format": "pt"}
    )
    config = EngineArgs(
        model=str(target),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "model": str(draft_path),
            "num_speculative_tokens": 1,
        },
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    assert draft is not None
    if mode == "standalone":
        config.speculative_config = None
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        if mode != "target_shared":
            missing = "embed_tokens" if mode == "standalone" else "eh_proj"
            with pytest.raises(
                ValueError, match=rf"not initialized.*checkpoint.*{missing}"
            ):
                loader.load_weights(model, draft)
        else:
            loader.load_weights(model, draft)
            assert torch.equal(
                model.get_parameter("model.layers.2.eh_proj.weight"),
                weights["model.layers.2.eh_proj.weight"],
            )
            target_shell = materialize_target_shell(
                config.model_config, torch.device("cpu")
            )
            assert torch.equal(
                target_shell.model.embed_tokens.weight[:128],
                weights["model.embed_tokens.weight"],
            )
            assert torch.equal(
                target_shell.lm_head.weight[:128],
                weights["model.embed_tokens.weight" if tied else "lm_head.weight"],
            )
