# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the standalone host's wire, batch and partial-weight contracts."""

import json
import shutil

import pytest
import torch
from huggingface_hub import ResolvedRevision
from safetensors.torch import save_file

from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from tests.v1.spec_decode.remote.runtime_fixtures import make_config
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.gpu import make_input_batch
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow
from vllm.v1.spec_decode.remote.runtime.weights import selected_checkpoint_weights
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.worker.gpu.input_batch import InputBuffers

pytestmark = pytest.mark.usefixtures("cpu_parallel_state")


class HubService:
    def __init__(self, download):
        self.hf_hub_download = download


def _config(*, use_aux=False):
    config = make_config()
    target = config.model_config
    draft = config.speculative_config.draft_model_config
    target.model = target.tokenizer = "target"
    target.revision = target.tokenizer_revision = "a" * 40
    target.dtype = draft.dtype = torch.bfloat16
    target.hf_config.vocab_size = draft.hf_config.vocab_size = 32
    target.model_arch_config = target.get_model_arch_config()
    draft.model, draft.revision = "draft", "b" * 40
    draft.hf_config.eagle_config = {"use_aux_hidden_state": use_aux}
    return config


@pytest.mark.parametrize("aux", [False, True])
def test_contract_transports_only_the_checkpoint_hidden_state_kind(aux):
    contract = create_runtime_contract(_config(use_aux=aux))
    hidden = contract.schema.slots[2]
    assert hidden.kind == ("aux_hidden_states" if aux else "last_hidden_states")
    assert hidden.trailing_shape == (48 if aux else 16,)
    assert hidden.dtype == "bfloat16"
    assert len(contract.schema.slots) == 4
    lookahead = contract.schema.slots[-1]
    assert lookahead.optional and lookahead.per_sequence
    assert lookahead.trailing_shape == (1,)


@pytest.mark.parametrize(
    "layers",
    [
        {"target_layer_ids": [0, 1]},
        {"eagle_aux_hidden_state_layer_ids": [1, 2]},
        {"eagle_config": {"use_aux_hidden_state": True, "layer_ids": [1, 2]}},
    ],
)
def test_eagle_contract_matches_native_auxiliary_selection(layers):
    from transformers import Qwen3Config

    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    config = _config(use_aux=True)
    draft = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=16,
        vocab_size=32,
        num_aux_layers=2,
        **layers,
    )
    config.speculative_config.draft_model_config.hf_config = EAGLEConfig(
        model=draft, method="eagle3"
    )
    selected = get_eagle3_aux_layers_from_config(config.speculative_config)
    assert selected == (1, 2)
    feature = create_runtime_contract(config).schema.slots[2]
    assert feature.trailing_shape == (16 * len(selected),)


@pytest.mark.parametrize("count_field", ["num_aux_layers", "num_aux_hidden_states"])
def test_eagle_contract_rejects_projection_and_selected_layer_count_mismatch(
    count_field,
):
    config = _config(use_aux=True)
    draft = config.speculative_config.draft_model_config.hf_config
    draft.target_layer_ids = [0, 1]
    setattr(draft, count_field, 3)
    with pytest.raises(ValueError, match="auxiliary.*count"):
        create_runtime_contract(config)


def test_contract_fingerprint_changes_with_weights_precision_and_feature_policy():
    config = _config()
    baseline = create_runtime_contract(config)
    config.speculative_config.draft_model_config.revision = "c" * 40
    changed_draft = create_runtime_contract(config)
    assert changed_draft.draft_fingerprint != baseline.draft_fingerprint
    assert changed_draft.target_fingerprint == baseline.target_fingerprint
    config.speculative_config.draft_model_config.revision = "b" * 40
    config.speculative_config.num_speculative_tokens = 5
    assert (
        create_runtime_contract(config).draft_fingerprint != baseline.draft_fingerprint
    )
    config.model_config.dtype = torch.float16
    assert (
        create_runtime_contract(config).target_fingerprint
        != baseline.target_fingerprint
    )
    assert create_runtime_contract(_config(use_aux=True)).schema != baseline.schema


@pytest.mark.parametrize("same_commit", [True, False])
def test_tokenizer_handshake_uses_pinned_commit_not_requested_branch(same_commit):
    from vllm.v1.spec_decode.remote.server import FakeDraftAdapter
    from vllm.v1.spec_decode.remote.session import (
        RemoteDraftError,
        RemoteDraftSession,
        VerifierIdentity,
    )

    config = _config()
    config.model_config.tokenizer = "test/tokenizer"
    config.model_config.tokenizer_revision = ResolvedRevision(
        "a" * 40, "main", repo_id="test/tokenizer"
    )
    expected = create_runtime_contract(config)
    config.model_config.tokenizer_revision = (
        "a" * 40
        if same_commit
        else ResolvedRevision("b" * 40, "main", repo_id="test/tokenizer")
    )
    actual = create_runtime_contract(config)
    assert (
        actual.tokenizer_fingerprint == expected.tokenizer_fingerprint
    ) == same_commit
    assert actual.target_fingerprint == expected.target_fingerprint
    adapter = FakeDraftAdapter(
        vocab_size=32,
        feature_schema=expected.schema,
        capabilities=expected.capabilities,
        draft_checkpoint_fingerprint=expected.draft_fingerprint,
    )
    server = RemoteDraftServer(
        adapter,
        target_fingerprint=expected.target_fingerprint,
        tokenizer_fingerprint=expected.tokenizer_fingerprint,
    )
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(
        server.endpoint,
        VerifierIdentity(
            verifier_instance_id="tokenizer-revision",
            target_fingerprint=actual.target_fingerprint,
            tokenizer_fingerprint=actual.tokenizer_fingerprint,
            draft_checkpoint_fingerprint=actual.draft_fingerprint,
            method="eagle3",
            num_speculative_tokens=3,
            parallel_drafting=False,
            provided_features=tuple(slot.kind for slot in actual.schema.slots),
            max_num_seqs=1,
            max_num_batched_tokens=4,
            max_model_len=16,
            vocab_size=32,
        ),
        request_timeout_ms=1000,
        startup_timeout_ms=1000,
    )
    try:
        if same_commit:
            session.connect()
            assert session.connected
        else:
            with pytest.raises(RemoteDraftError, match="tokenizer fingerprint"):
                session.connect()
            assert not session.connected
    finally:
        session.close()
        server.stop()


@pytest.mark.parametrize(
    "revision",
    [
        None,
        "main",
        ResolvedRevision("a" * 40, "main", repo_id="other/tokenizer"),
        ResolvedRevision("a" * 40, "main", repo_id="target", repo_type="dataset"),
    ],
)
def test_contract_rejects_unresolved_or_foreign_tokenizer_revision(revision):
    config = _config()
    config.model_config.tokenizer_revision = revision
    with pytest.raises(ValueError, match="checkpoint revision"):
        create_runtime_contract(config)


def test_local_checkpoint_identity_tracks_bytes_instead_of_deployment_path(tmp_path):
    config = _config()
    source, copied = tmp_path / "source", tmp_path / "copied"
    source.mkdir()
    weight_file = source / "model.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(2, 2)}, weight_file)
    (source / "tokenizer.json").write_text('{"model":{"vocab":{"a":0}}}')
    config.model_config.model = config.model_config.tokenizer = str(source)
    config.model_config.revision = "main"
    original = create_runtime_contract(config)
    shutil.copytree(source, copied)
    config.model_config.model = config.model_config.tokenizer = str(copied)
    duplicate = create_runtime_contract(config)
    assert duplicate == original
    save_file(
        {"model.embed_tokens.weight": torch.zeros(2, 2)}, copied / weight_file.name
    )
    changed = create_runtime_contract(config)
    assert changed.target_fingerprint != original.target_fingerprint
    assert changed.draft_fingerprint == original.draft_fingerprint
    assert changed.tokenizer_fingerprint == original.tokenizer_fingerprint
    (copied / "tokenizer.json").write_text('{"model":{"vocab":{"b":0}}}')
    assert (
        create_runtime_contract(config).tokenizer_fingerprint
        != original.tokenizer_fingerprint
    )


@pytest.mark.parametrize("which", ["target", "draft"])
def test_contract_rejects_a_separate_unverified_weight_source(which):
    config = _config()
    original = create_runtime_contract(config)
    model = (
        config.model_config
        if which == "target"
        else config.speculative_config.draft_model_config
    )
    model.model_weights = model.model
    assert create_runtime_contract(config) == original
    model.model_weights = "other/checkpoint"
    with pytest.raises(ValueError, match="separate model_weights source"):
        create_runtime_contract(config)


def test_contract_rejects_unresolved_revision_or_missing_local_weights(tmp_path):
    config = _config()
    config.model_config.revision = "main"
    with pytest.raises(ValueError, match="immutable checkpoint revision"):
        create_runtime_contract(config)
    config.model_config.hf_config._commit_hash = "d" * 40
    with pytest.raises(ValueError, match="immutable checkpoint revision"):
        create_runtime_contract(config)
    config.model_config.model = str(tmp_path)
    with pytest.raises(ValueError, match="no verifiable"):
        create_runtime_contract(config)


def test_contract_revision_tracks_weight_commit_instead_of_external_config():
    config = _config()
    target = config.model_config
    target.hf_config._commit_hash = "c" * 40
    target.revision = ResolvedRevision("a" * 40, "main", repo_id=target.model)
    first = create_runtime_contract(config)
    target.revision = ResolvedRevision("b" * 40, "main", repo_id=target.model)
    second = create_runtime_contract(config)
    assert second.target_fingerprint != first.target_fingerprint
    target.revision = "b" * 40
    assert create_runtime_contract(config) == second
    target.hf_config._commit_hash = "d" * 40
    assert create_runtime_contract(config) == second


@pytest.mark.parametrize(
    "revision",
    [
        None,
        "main",
        "a" * 39,
        "A" * 40,
        ResolvedRevision("a" * 40, "main", repo_id="other/model"),
        ResolvedRevision("a" * 40, "main", repo_id="target", repo_type="dataset"),
        ResolvedRevision("main", "main", repo_id="target"),
    ],
)
def test_contract_rejects_unresolved_or_foreign_weight_revision(revision):
    config = _config()
    config.model_config.hf_config._commit_hash = "c" * 40
    config.model_config.revision = revision
    with pytest.raises(ValueError, match="checkpoint revision"):
        create_runtime_contract(config)


@pytest.mark.parametrize("checkpoint", ["immutable", "resolved", "local"])
def test_shared_weights_use_weight_revision_instead_of_external_config(
    monkeypatch, tmp_path, checkpoint
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    target = _config().model_config
    target.hf_config._commit_hash = "c" * 40
    expected_revision = "a" * 40
    if checkpoint == "resolved":
        target.revision = ResolvedRevision("a" * 40, "main", repo_id=target.model)
    elif checkpoint == "local":
        target.model = str(tmp_path)
        target.revision = "main"
        expected_revision = None
    embedding = torch.ones(32, 16, dtype=target.dtype)
    save_file(
        {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding + 1},
        tmp_path / "model.safetensors",
    )
    calls = []

    def download(source, filename, *, revision):
        calls.append((source, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(weights, "hf_api", lambda: HubService(download))
    shell = weights.materialize_target_shell(target, torch.device("cpu"))
    torch.testing.assert_close(shell.model.embed_tokens.weight[:32], embedding)
    assert calls == (
        [] if checkpoint == "local" else [(target.model, expected_revision)] * 2
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "draft_model"),
        ("parallel_drafting", True),
        ("draft_sample_method", "probabilistic"),
    ],
)
def test_contract_rejects_unimplemented_drafting_modes(field, value):
    config = _config()
    setattr(config.speculative_config, field, value)
    with pytest.raises(ValueError):
        create_runtime_contract(config)


def test_partial_weights_download_only_shards_containing_shared_tensors(
    tmp_path, monkeypatch
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    embedding = torch.randn(8, 4)
    head = torch.randn(8, 4)
    save_file(
        {"model.embed_tokens.weight": embedding}, tmp_path / "embedding.safetensors"
    )
    save_file({"lm_head.weight": head}, tmp_path / "head.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "embedding.safetensors",
                    "lm_head.weight": "head.safetensors",
                    "model.layers.0.weight": "never-download.safetensors",
                }
            }
        )
    )
    fetched = []

    def download(repo, filename, *, revision):
        assert repo == "test/target" and revision == "immutable-sha"
        fetched.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(weights, "hf_api", lambda: HubService(download))
    actual = selected_checkpoint_weights(
        "test/target", "immutable-sha", ("model.embed_tokens.weight", "lm_head.weight")
    )
    assert set(fetched) == {
        "model.safetensors.index.json",
        "embedding.safetensors",
        "head.safetensors",
    }
    assert torch.equal(actual["model.embed_tokens.weight"], embedding)
    assert torch.equal(actual["lm_head.weight"], head)


@pytest.mark.parametrize("filename", ["model.safetensors", "weights.safetensors"])
def test_indexless_local_weights_accept_native_safetensor_filenames(tmp_path, filename):
    embedding = torch.randn(8, 4)
    save_file(
        {
            "model.embed_tokens.weight": embedding,
            "model.layers.0.weight": torch.ones(3),
        },
        tmp_path / filename,
    )
    actual = selected_checkpoint_weights(
        str(tmp_path), None, ("model.embed_tokens.weight", "lm_head.weight")
    )
    assert set(actual) == {"model.embed_tokens.weight"}
    torch.testing.assert_close(actual["model.embed_tokens.weight"], embedding)


def test_indexless_local_weights_select_across_shards_and_reject_duplicates(
    tmp_path,
):
    embedding, head = torch.randn(8, 4), torch.randn(8, 4)
    save_file(
        {"model.embed_tokens.weight": embedding}, tmp_path / "embedding.safetensors"
    )
    save_file({"lm_head.weight": head}, tmp_path / "head.safetensors")
    names = ("model.embed_tokens.weight", "lm_head.weight")
    actual = selected_checkpoint_weights(str(tmp_path), None, names)
    torch.testing.assert_close(actual["model.embed_tokens.weight"], embedding)
    torch.testing.assert_close(actual["lm_head.weight"], head)

    save_file({"lm_head.weight": head + 1}, tmp_path / "other.safetensors")
    with pytest.raises(ValueError, match="duplicate.*lm_head.weight"):
        selected_checkpoint_weights(str(tmp_path), None, names)


def test_partial_weight_auth_errors_do_not_fall_back_to_full_checkpoint(monkeypatch):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    def denied(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(weights, "hf_api", lambda: HubService(denied))
    with pytest.raises(PermissionError, match="access denied"):
        selected_checkpoint_weights("test/target", "sha", ("lm_head.weight",))


def test_input_batch_preserves_reordered_request_slots_and_rejected_tail():
    planner = DraftStepPlanner(
        DraftBlockPool(32, 4),
        max_num_reqs=3,
        max_model_len=32,
        num_speculative_tokens=3,
    )
    a, b = SequenceKey("v", 1, 0), SequenceKey("v", 2, 0)
    planner.open(a)
    planner.open(b)
    prompt, failures = planner.plan([DraftStepRow(a, 5, 1), DraftStepRow(b, 3, 1)])
    assert not failures
    planner.commit(prompt)
    plan, failures = planner.plan(
        [
            DraftStepRow(b, 4, 2, num_rejected=2, last_sampled=15),
            DraftStepRow(a, 2, 2, last_sampled=19),
        ]
    )
    assert not failures
    batch = make_input_batch(
        plan, torch.tensor([1, 2, 3, 4, 5, 6]), InputBuffers(3, 16, torch.device("cpu"))
    )
    assert batch.idx_mapping.tolist() == [1, 0]
    assert batch.query_start_loc.tolist() == [0, 4, 6]
    assert batch.seq_lens.tolist() == [7, 7]
    assert batch.positions.tolist() == [3, 4, 5, 6, 5, 6]
    assert batch.num_computed_tokens_np.tolist() == [3, 5]
    assert batch.input_ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert not batch.is_padding.any()
    planner.commit(plan)
    assert planner.num_computed(b) == 5
    assert planner.num_computed(a) == 7


def test_final_prompt_chunk_keeps_prefill_metadata_after_an_earlier_chunk():
    planner = DraftStepPlanner(
        DraftBlockPool(32, 4),
        max_num_reqs=1,
        max_model_len=32,
        num_speculative_tokens=3,
    )
    key = SequenceKey("v", 1, 0)
    planner.open(key)
    first, _ = planner.plan(
        [
            DraftStepRow(key, 4, 0, next_prefill_tokens=(5,)),
        ]
    )
    planner.commit(first)
    final, _ = planner.plan([DraftStepRow(key, 3, 1, last_sampled=8)])
    batch = make_input_batch(
        final,
        torch.tensor([5, 6, 7]),
        InputBuffers(1, 16, torch.device("cpu")),
    )
    assert batch.has_prefill
    assert batch.is_prefilling_np.tolist() == [True]
    assert batch.prefill_len_np.tolist() == [7]
    assert batch.positions.tolist() == [4, 5, 6]
