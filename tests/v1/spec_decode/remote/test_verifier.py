# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request/chunk translation against a real socket and stateful CPU KV.

These tests exercise the public propose/lifecycle boundary without CUDA. The
KV digest makes missing, duplicated and rejected context observable in drafts.
"""

import errno
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.remote.test_runtime import (
    VOCAB,
    K,
    KVDigestExecutor,
    digest_row,
    make_planner,
)
from vllm.config.speculative import RemoteDraftConfig
from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    OpenSequence,
    SequenceAck,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.server import AdapterError, RemoteDraftServer
from vllm.v1.spec_decode.remote.transport import ConnectionClosed, TransportTimeout
from vllm.v1.worker.gpu.spec_decode import remote


@pytest.fixture
def cpu_draft_handler(monkeypatch):
    """Run real runner/handler initialization through the host metadata boundary."""
    import vllm.v1.worker.gpu.model_runner as runner_module
    import vllm.v1.worker.gpu.spec_decode.utils as utils

    events = []
    stream = SimpleNamespace(wait_stream=lambda other: events.append("wait"))
    event = SimpleNamespace(
        record=lambda: events.append("record"),
        synchronize=lambda: events.append("synchronize"),
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **kw: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a, **kw: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, stream: events.append("retain")
    )

    def copy(tokens):
        events.append("copy")
        return tokens.numpy().copy()

    monkeypatch.setattr(utils, "async_copy_to_np", copy)
    for name in (
        "JitWarmupRegistry",
        "RequestState",
        "set_default_max_concurrency",
        "get_ec_connector",
        "init_speculator",
        "StepTimingCollector",
    ):
        monkeypatch.setattr(runner_module, name, lambda *a, **kw: None)
    monkeypatch.setattr(runner_module, "requires_aux_hidden_states", lambda spec: False)
    monkeypatch.setattr(
        runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    class HandlerInitialized(Exception):
        pass

    def stop_before_gpu_buffers(*args, **kwargs):
        raise HandlerInitialized

    monkeypatch.setattr(runner_module, "InputBuffers", stop_before_gpu_buffers)

    def create(remote_enabled):
        spec = (
            None
            if remote_enabled is None
            else SimpleNamespace(
                use_dspark=lambda: False,
                use_ngram=lambda: False,
                use_remote_draft=lambda: remote_enabled,
                enable_adaptive_verification=False,
            )
        )
        settings = SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_vocab_size=lambda: 1000,
                max_model_len=64,
                is_encoder_decoder=False,
                supports_multimodal_inputs=False,
                enable_prompt_embeds=False,
                runner_type="generate",
            ),
            cache_config=SimpleNamespace(cache_dtype="auto"),
            compilation_config=SimpleNamespace(),
            lora_config=None,
            load_config=None,
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
                cp_kv_cache_interleave_size=1,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64),
            speculative_config=spec,
            observability_config=None,
            max_concurrent_batches=1,
            num_speculative_tokens=K,
            num_prefill_lookahead_tokens=1,
        )
        runner = object.__new__(runner_module.GPUModelRunner)
        with pytest.raises(HandlerInitialized):
            runner_module.GPUModelRunner.__init__(runner, settings, torch.device("cpu"))
        return runner.draft_tokens_handler

    return create, events


@pytest.mark.parametrize("remote_enabled", [None, False])
def test_native_unstructured_drafts_keep_the_no_copy_path(
    cpu_draft_handler, remote_enabled
):
    create, events = cpu_draft_handler
    handler = create(remote_enabled)
    batch = SimpleNamespace(req_ids=["native"], has_structured_output_reqs=False)
    handler.set_draft_tokens(batch, torch.tensor([[0, 2, 3]]))
    assert handler.get_draft_tokens().draft_token_ids == [[-1, -1, -1]]
    assert events == []


@pytest.mark.parametrize("structured", [False, True])
def test_remote_draft_lengths_reach_scheduler_in_request_order(
    cpu_draft_handler, structured
):
    create, events = cpu_draft_handler
    handler = create(True)
    batch = SimpleNamespace(
        req_ids=["partial", "healthy", "fallback"],
        has_structured_output_reqs=structured,
    )
    handler.set_draft_tokens(
        batch, torch.tensor([[0, -1, -1], [7, 8, 9], [-1, -1, -1]])
    )
    output = handler.get_draft_tokens()
    assert output.req_ids == batch.req_ids
    assert output.draft_token_ids == [[0], [7, 8, 9], []]
    assert events == ["wait", "copy", "retain", "record", "synchronize"]
    handler.set_draft_tokens(batch, torch.full((3, 3), -1, dtype=torch.int64))
    assert handler.get_draft_tokens().draft_token_ids == [[], [], []]
    handler.set_draft_tokens(batch, torch.empty(3, 0, dtype=torch.int64))
    assert handler.get_draft_tokens().draft_token_ids == [[], [], []]


def test_native_structured_drafts_preserve_existing_transfer(cpu_draft_handler):
    create, events = cpu_draft_handler
    handler = create(False)
    batch = SimpleNamespace(req_ids=["structured"], has_structured_output_reqs=True)
    handler.set_draft_tokens(batch, torch.tensor([[0, 2, 3]]))
    assert handler.get_draft_tokens().draft_token_ids == [[0, 2, 3]]
    assert events == ["wait", "copy", "retain", "record", "synchronize"]


@pytest.mark.parametrize("k", [1, 3])
def test_remote_terminal_fallback_schedules_no_speculative_slots(
    cpu_draft_handler, tmp_path, monkeypatch, k
):
    from transformers import Qwen3Config

    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.platforms import current_platform
    from vllm.v1.outputs import ModelRunnerOutput

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    model = tmp_path / "qwen"
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1000,
        max_position_embeddings=64,
    ).save_pretrained(model)
    scheduler = create_scheduler(
        model=str(model),
        skip_tokenizer_init=True,
        num_speculative_tokens=k,
        enable_prefix_caching=False,
        async_scheduling=False,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        num_blocks=64,
    )
    requests = create_requests(num_requests=2, num_tokens=2, max_tokens=16)
    for request in requests:
        scheduler.add_request(request)
    ids = [request.request_id for request in requests]
    first = scheduler.schedule()
    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(ids)},
            sampled_token_ids=[[30], [31]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    create, _ = cpu_draft_handler
    handler = create(True)
    handler.set_draft_tokens(
        SimpleNamespace(req_ids=ids, has_structured_output_reqs=False),
        torch.tensor([[-1] * k, list(range(k))]),
    )
    scheduler.update_draft_token_ids(handler.get_draft_tokens())
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {ids[0]: 1, ids[1]: 1 + k}
    assert output.scheduled_spec_decode_tokens == {ids[1]: list(range(k))}


SCHEMA = TargetFeatureSchema(
    1,
    (
        FeatureSlot("token_ids", "int32", ()),
        FeatureSlot("positions", "int64", ()),
        FeatureSlot("last_hidden_states", "float32", (4,)),
        FeatureSlot(
            "next_prefill_tokens", "int64", (1,), optional=True, per_sequence=True
        ),
    ),
)
CONTRACT = RuntimeContract(
    SCHEMA,
    SpeculatorPlacementCapabilities(
        state_dependency="own_kv",
        required_features=("token_ids", "positions", "last_hidden_states"),
        num_prefill_lookahead=1,
        standalone_weights="complete",
    ),
    "target",
    "tokenizer",
    "draft",
)


class TPGroup:
    def __init__(self):
        self.rank_in_group = 0
        self.payload = None

    def broadcast_object(self, value, src):
        if self.rank_in_group == src:
            self.payload = value
        return self.payload

    def broadcast(self, value, src):
        pytest.fail("remote drafts must not initialize a GPU tensor collective")


def config(endpoint):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False, max_num_seqs=4, max_num_batched_tokens=64
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=64, get_vocab_size=lambda: VOCAB
        ),
        lora_config=None,
        watermark_config=None,
        speculative_config=SimpleNamespace(
            method="eagle3",
            num_speculative_tokens=K,
            enable_adaptive_verification=False,
            remote_draft=RemoteDraftConfig(
                endpoint=endpoint, failure_policy="error", request_timeout_ms=5000
            ),
        ),
    )


@pytest.fixture
def verifier(monkeypatch, request):
    options = getattr(request, "param", "error")
    policy, draft_steps = options if isinstance(options, tuple) else (options, K)
    planner = make_planner()
    planner.num_speculative_tokens = draft_steps
    executor = KVDigestExecutor(planner.block_pool)
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=CONTRACT.capabilities,
        feature_schema=SCHEMA,
        methods={"eagle3", "mtp"},
        draft_checkpoint_fingerprint=CONTRACT.draft_fingerprint,
    )
    server = RemoteDraftServer(
        adapter,
        target_fingerprint=CONTRACT.target_fingerprint,
        tokenizer_fingerprint=CONTRACT.tokenizer_fingerprint,
    )
    server.start("tcp://127.0.0.1:0")
    group = TPGroup()
    monkeypatch.setattr(remote, "get_tp_group", lambda: group)
    monkeypatch.setattr(remote, "create_runtime_contract", lambda _: CONTRACT)
    settings = config(server.endpoint)
    settings.speculative_config.remote_draft.failure_policy = policy
    settings.speculative_config.num_speculative_tokens = draft_steps
    settings.speculative_config.method = "mtp" if draft_steps == 1 else "eagle3"
    instance = remote.RemoteSpeculator(settings, torch.device("cpu"))
    yield instance, executor, settings, group
    instance.close()
    server.stop()


def inputs(ids, tokens, positions, prefill, slots=None):
    count = len(ids)
    slots = slots if slots is not None else list(range(count))
    lens = [len(row) for row in tokens]
    return SimpleNamespace(
        req_ids=ids,
        num_reqs=count,
        num_tokens=sum(lens),
        idx_mapping=torch.tensor(slots),
        query_start_loc_np=np.cumsum([0, *lens]),
        input_ids=torch.tensor(sum(tokens, []), dtype=torch.int32),
        positions=torch.tensor(sum(positions, []), dtype=torch.int64),
        is_prefilling_np=np.array([length > 0 for length in prefill]),
        prefill_len_np=np.array(prefill),
    )


def propose(verifier, batch, sampled, rejected, recovery, lookahead=None, **kwargs):
    last_sampled = torch.zeros(4, 1, dtype=torch.int64)
    last_sampled[batch.idx_mapping, 0] = torch.tensor(recovery)
    next_prefill = torch.zeros(1, 4, dtype=torch.int32)
    if lookahead is not None:
        next_prefill[0, batch.idx_mapping] = torch.tensor(lookahead, dtype=torch.int32)
    return verifier.propose(
        batch,
        {},
        {},
        torch.zeros(batch.num_tokens, 4),
        kwargs.pop("aux_hidden_states", None),
        torch.tensor(sampled),
        torch.tensor(rejected),
        last_sampled,
        next_prefill,
        kwargs.pop("temperature", torch.zeros(4)),
        torch.zeros(4, dtype=torch.int64),
        **kwargs,
    )


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
@pytest.mark.parametrize("registered", [0, 1])
def test_request_registration_allocation_failure_is_terminal(
    verifier, monkeypatch, registered
):
    instance, executor, settings, _ = verifier
    factory = remote._Request
    calls = 0

    def allocate(key):
        nonlocal calls
        calls += 1
        if calls > registered:
            raise MemoryError("request registration allocation failed")
        return factory(key)

    first = inputs(["a", "b"], [[1], [11]], [[0], [0]], [1, 1])
    with monkeypatch.context() as patch:
        patch.setattr(remote, "_Request", allocate)
        if settings.speculative_config.remote_draft.failure_policy == "error":
            with pytest.raises(RuntimeError, match="registration allocation failed"):
                propose(instance, first, [1, 1], [0, 0], [2, 12])
            return
        assert propose(instance, first, [1, 1], [0, 0], [2, 12]).tolist() == [
            [-1] * K,
            [-1] * K,
        ]
    assert executor.plans == []
    assert all(
        record.state in (remote.SequenceState.TARGET_ONLY, remote.SequenceState.INVALID)
        for record in instance._session._records.values()
    )
    followup = inputs(["a", "b"], [[2], [12]], [[1], [1]], [0, 0])
    assert propose(instance, followup, [1, 1], [0, 0], [3, 13]).tolist() == [
        [-1] * K,
        [-1] * K,
    ]
    instance.close_requests(["a", "b"])
    assert not instance._terminal_requests
    assert propose(
        instance, inputs(["a"], [[21]], [[0]], [1]), [1], [0], [22]
    ).tolist() == [digest_row([21], 22)]


@pytest.mark.parametrize("effective_k", [0, 1, K])
def test_empty_proposal_preserves_batch_shape_without_rpc(
    verifier, monkeypatch, effective_k
):
    instance, executor, _, _ = verifier
    batch = inputs([], [], [], [])
    batch.idx_mapping = torch.empty(0, dtype=torch.int64)

    def unexpected_rpc(*args, **kwargs):
        pytest.fail("empty proposals must not create remote work")

    for method in ("open_sequences", "prefill_many", "dispatch"):
        monkeypatch.setattr(instance._session, method, unexpected_rpc)
    result = instance.propose(
        batch,
        {},
        {},
        torch.empty(0, 4),
        None,
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros(4, 1, dtype=torch.int64),
        torch.zeros(1, 4, dtype=torch.int32),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.int64),
        num_speculative_tokens=effective_k,
    )
    assert result.shape == (0, instance.num_speculative_steps)
    assert result.dtype == torch.int64
    assert not instance._requests and not executor.plans


def test_chunks_rejections_and_request_reordering_preserve_verified_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a", "b"], [[11, 12], [21, 22]], [[0, 1], [0, 1]], [4, 2], [3, 1])
    drafts = propose(instance, batch, [0, 1], [0, 0], [0, 30], [13, 0])
    b_drafts = digest_row([21, 22], 30)
    assert drafts.tolist() == [[-1] * K, b_drafts]
    assert executor.plans[0].next_prefill_tokens.tolist() == [[13]]

    # Finish a's prompt while b rejects its last two proposals. The final
    # prompt must be buffered then consumed once by a zero-query round.
    batch = inputs(
        ["b", "a"],
        [[30, *b_drafts], [13, 14]],
        [[2, 3, 4, 5], [2, 3]],
        [0, 4],
        [1, 3],
    )
    drafts = propose(instance, batch, [2, 1], [2, 0], [31, 40])
    b_prefix = [21, 22, 30, b_drafts[0]]
    assert drafts.tolist() == [
        digest_row(b_prefix, 31),
        digest_row([11, 12, 13, 14], 40),
    ]
    assert executor.plans[-1].num_computed_tokens.tolist() == [2, 2]
    assert executor.plans[-1].num_rejected.tolist() == [2, 0]

    batch = inputs(["b"], [[31, *drafts[0].tolist()]], [[4, 5, 6, 7]], [0], [1])
    assert propose(instance, batch, [1], [3], [32]).tolist() == [
        digest_row([*b_prefix, 31], 32)
    ]


@pytest.mark.parametrize(
    "verifier",
    [(policy, k) for policy in ("error", "target_only") for k in (1, 3)],
    indirect=True,
)
@pytest.mark.parametrize("new_first", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_final_prefill_padding_preserves_prompt_and_reordered_decode(
    verifier, new_first, padded
):
    instance, executor, _, _ = verifier
    k = instance.num_speculative_steps
    initial = inputs(["running"], [[21, 22]], [[0, 1]], [2], [1])
    old_drafts = propose(instance, initial, [1], [0], [30])[0].tolist()
    padding = k if padded else 0
    order = ["new", "running"] if new_first else ["running", "new"]
    tokens = {"new": [7, *([999] * padding)], "running": [30, *old_drafts]}
    positions = {"new": list(range(1 + padding)), "running": list(range(2, 3 + k))}
    batch = inputs(
        order,
        [tokens[req] for req in order],
        [positions[req] for req in order],
        [1 if req == "new" else 0 for req in order],
        [3 if req == "new" else 1 for req in order],
    )
    drafts = propose(
        instance,
        batch,
        [1, 1],
        [padding if req == "new" else k for req in order],
        [8 if req == "new" else 31 for req in order],
    ).tolist()
    expected = {
        "new": digest_row([7], 8)[:k],
        "running": digest_row([21, 22, 30], 31)[:k],
    }
    assert drafts == [expected[req] for req in order]
    assert instance._requests["new"].num_computed == 1
    assert instance._requests["running"].num_computed == 3
    assert executor.plans[-1].num_scheduled_tokens.tolist() == [
        1 if req == "new" else 1 + k for req in order
    ]

    # Reorder the rows and accept one new request draft. Padding must neither
    # enter its KV digest nor shift the physical feature offset of its sibling.
    order.reverse()
    tokens = {"new": [8, *expected["new"]], "running": [31, *expected["running"]]}
    positions = {"new": list(range(1, 2 + k)), "running": list(range(3, 4 + k))}
    batch = inputs(
        order,
        [tokens[req] for req in order],
        [positions[req] for req in order],
        [0, 0],
        [3 if req == "new" else 1 for req in order],
    )
    result = propose(
        instance,
        batch,
        [2 if req == "new" else 1 for req in order],
        [k - 1 if req == "new" else k for req in order],
        [9 if req == "new" else 32 for req in order],
    )
    assert result.tolist() == [
        digest_row([7, 8, expected["new"][0]], 9)[:k]
        if req == "new"
        else digest_row([21, 22, 30, 31], 32)[:k]
        for req in order
    ]
    assert instance._requests["new"].num_computed == 3
    assert instance._requests["running"].num_computed == 4


@pytest.mark.parametrize("verifier", ["error", "target_only"], indirect=True)
@pytest.mark.parametrize(
    "query_len,prompt_len,sampled,rejected",
    [
        (4, 1, 0, 3),
        (4, 1, 2, 3),
        (4, 1, 1, 0),
        (4, 1, 1, 2),
        (4, 1, 1, 4),
        (5, 1, 1, 4),
        (2, 5, 0, 1),
        (2, 5, 1, 0),
    ],
)
def test_prefill_padding_does_not_hide_invalid_sampling_counts(
    verifier, query_len, prompt_len, sampled, rejected
):
    instance, executor, _, _ = verifier
    batch = inputs(["new"], [[7] * query_len], [list(range(query_len))], [prompt_len])
    with pytest.raises(
        RuntimeError, match="unexpected sampling counts during remote prefill"
    ):
        propose(instance, batch, [sampled], [rejected], [8])
    assert instance._requests["new"].num_computed == 0
    assert executor.plans == []


def test_close_releases_kv_and_reused_request_id_starts_from_zero(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["request"], [[1, 2]], [[0, 1]], [2], [3])
    propose(instance, batch, [1], [0], [3])
    old_key = executor.plans[-1].rows[0].key
    instance.close_requests({"request", "already-gone"})
    batch = inputs(["request"], [[7]], [[0]], [1], [3])
    assert propose(instance, batch, [1], [0], [8]).tolist() == [digest_row([7], 8)]
    plan = executor.plans[-1]
    assert plan.rows[0].key != old_key
    assert plan.num_computed_tokens.tolist() == [0]
    assert list(plan.block_table_updates.values())[0].start == 0


@pytest.mark.parametrize("flag", ["dummy_run", "is_profile"])
def test_dummy_and_profile_do_not_contact_or_advance_server(
    verifier, monkeypatch, flag
):
    instance, executor, _, _ = verifier

    def reject_network(*args, **kwargs):
        pytest.fail("dummy/profile run contacted remote session")

    monkeypatch.setattr(instance._session, "open_sequences", reject_network)
    monkeypatch.setattr(instance._session, "prefill_many", reject_network)
    monkeypatch.setattr(instance._session, "dispatch", reject_network)
    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2], **{flag: True}).tolist() == [
        [-1] * K
    ]
    assert executor.plans == []


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_profile_budgets_feature_copies_without_mutating_remote_state(
    verifier, monkeypatch, auxiliary, mixed
):
    instance, executor, _, group = verifier
    aux = [torch.zeros(3, 4), torch.ones(3, 4)] if auxiliary else None
    if auxiliary:
        monkeypatch.setattr(
            instance,
            "_contract",
            SimpleNamespace(
                schema=TargetFeatureSchema(
                    1,
                    (
                        *SCHEMA.slots[:2],
                        FeatureSlot("aux_hidden_states", "float32", (8,)),
                        *SCHEMA.slots[3:],
                    ),
                )
            ),
        )
    feature_calls, selected = [], []
    prepare = instance._features
    select = instance._select

    def prepare_features(*args, **kwargs):
        result = prepare(*args, **kwargs)
        feature_calls.append(kwargs)
        return result

    def select_features(features, tokens, rows):
        result = select(features, tokens, rows)
        source, copy = (
            features[instance._contract.schema.slots[2].kind],
            result.slots[2],
        )
        assert source.untyped_storage().data_ptr() != copy.untyped_storage().data_ptr()
        assert torch.equal(source, copy)
        selected.append(tuple(copy.shape))
        return result

    def reject_network(*args, **kwargs):
        pytest.fail("profiling contacted the remote session")

    monkeypatch.setattr(instance, "_features", prepare_features)
    monkeypatch.setattr(instance, "_select", select_features)
    monkeypatch.setattr(instance._session, "open_sequences", reject_network)
    monkeypatch.setattr(instance._session, "prefill_many", reject_network)
    monkeypatch.setattr(instance._session, "dispatch", reject_network)
    batch = inputs(
        ["a", "b"], [[1, 2], [3]], [[0, 1], [2 if mixed else 0]], [2, 0 if mixed else 1]
    )
    before_stats = dict(instance._stats)
    output = propose(
        instance,
        batch,
        [1, 1],
        [0, 0],
        [4, 5],
        aux_hidden_states=aux,
        is_profile=True,
        dummy_run=True,
    )
    assert output.tolist() == [[-1] * K, [-1] * K]
    assert feature_calls == [{"token_rows": [0, 1, 2], "sequence_rows": [0, 1]}]
    assert selected == [(3, 8 if auxiliary else 4)]
    assert instance._stats == before_stats
    assert instance._requests == {}
    assert executor.plans == []
    # Only rank zero owns and profiles the transferred features.
    group.rank_in_group = 1
    propose(instance, batch, [1, 1], [0, 0], [4, 5], is_profile=True)
    assert len(feature_calls) == 1
    group.rank_in_group = 0


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, MemoryError])
def test_profile_allocation_failure_is_not_hidden_by_target_only(
    verifier, monkeypatch, error_type
):
    instance, _, _, _ = verifier

    def fail_features(*args, **kwargs):
        raise error_type("profile feature allocation")

    monkeypatch.setattr(instance, "_features", fail_features)
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(error_type, match="profile feature allocation"):
        propose(instance, batch, [1], [0], [2], is_profile=True)
    assert instance._requests == {}


@pytest.mark.parametrize("effective_k", [0, 1, K])
def test_effective_k_reaches_executor_and_preserves_context(verifier, effective_k):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[11, 12]], [[0, 1]], [2])
    output = propose(
        instance, batch, [1], [0], [13], num_speculative_tokens=effective_k
    ).tolist()[0]
    expected = digest_row([11, 12], 13)[:effective_k]
    assert output == expected + [-1] * (K - effective_k)
    assert executor.effective_ks == [effective_k]
    assert instance._requests["a"].num_computed == 2
    verified = [13, *expected]
    batch = inputs(["a"], [verified], [list(range(2, 2 + len(verified)))], [0])
    assert propose(instance, batch, [len(verified)], [0], [14]).tolist() == [
        digest_row([11, 12, *verified], 14)
    ]
    assert executor.effective_ks == [effective_k, K]


@pytest.mark.parametrize("effective_k", [-1, K + 1])
def test_invalid_effective_k_does_not_open_or_advance_requests(verifier, effective_k):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="draft budget"):
        propose(instance, batch, [1], [0], [2], num_speculative_tokens=effective_k)
    assert instance._requests == {}
    assert executor.plans == []


def test_nonzero_tp_rank_receives_cpu_tokens_without_gpu_collective(verifier):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    assert peer._session is None
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    expected = propose(instance, batch, [1], [0], [2])
    assert group.payload == (None, expected.tolist())
    plan_count = len(executor.plans)
    group.rank_in_group = 1
    actual = propose(peer, batch, [1], [0], [2])
    assert torch.equal(actual, expected)
    assert len(executor.plans) == plan_count
    group.rank_in_group = 0


def test_error_is_shared_instead_of_tokens_with_tp_peer(verifier):
    instance, _, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="greedy target sampling"):
        propose(instance, batch, [1], [0], [2], temperature=torch.ones(4))
    group.rank_in_group = 1
    with pytest.raises(RuntimeError, match="greedy target sampling"):
        propose(peer, batch, [1], [0], [2])
    assert group.payload[1] is None
    group.rank_in_group = 0


def test_request_close_error_is_shared_with_tp_peer_without_session(
    verifier, monkeypatch
):
    instance, _, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    propose(instance, batch, [1], [0], [2])

    def fail(keys):
        raise RuntimeError("close failed")

    monkeypatch.setattr(instance._session, "close_sequences", fail)
    with pytest.raises(RuntimeError, match="request cleanup failed.*close failed"):
        instance.close_requests({"a"})
    group.rank_in_group = 1
    with pytest.raises(RuntimeError, match="request cleanup failed.*close failed"):
        peer.close_requests({"a"})
    group.rank_in_group = 0


def test_shutdown_error_does_not_mask_an_engine_error(verifier, monkeypatch):
    instance, _, _, _ = verifier
    session = instance._session
    close = session.close

    def fail():
        close()
        raise OSError("socket close failed")

    monkeypatch.setattr(session, "close", fail)
    instance.close()
    instance.close()
    assert instance._session is None


@pytest.mark.parametrize("existing,computed", [(True, 0), (False, 4)])
def test_runner_rejects_streaming_and_missing_prefix_before_removing_state(
    existing, computed
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_remote_draft=lambda: True),
        req_states=SimpleNamespace(req_id_to_index={"a": 0} if existing else {}),
    )
    scheduler = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="a", prefill_token_ids=[1, 2], num_computed_tokens=computed
            )
        ]
    )
    with pytest.raises(ValueError, match="streaming input updates"):
        GPUModelRunner.add_requests(runner, scheduler)


def test_invalid_verification_counts_fail_before_advancing_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[1]], [[0]], [1])
    drafts = propose(instance, batch, [1], [0], [2]).tolist()[0]
    plan_count = len(executor.plans)
    batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
    with pytest.raises(RuntimeError, match="counts do not cover"):
        propose(instance, batch, [2], [0], [3])
    assert len(executor.plans) == plan_count


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_failed_remote_row_stays_target_only_without_reusing_stale_kv(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    calls = []

    def fail(plan, features, *, num_speculative_tokens=None):
        calls.append(plan)
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2]).tolist() == [[-1] * K]
    batch = inputs(["a"], [[2]], [[1]], [0])
    assert propose(instance, batch, [1], [0], [3]).tolist() == [[-1] * K]
    assert len(calls) == 1


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
def test_partial_prefill_ack_disconnect_preserves_fallback_and_shares_policy(
    verifier, monkeypatch
):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    session = instance._session
    prefill_many = session.prefill_many
    acknowledged: list[SequenceAck] = []

    def disconnect_after_first_ack(chunks, **kwargs):
        await_ack = session._await

        def receive(*args, **kwargs):
            if acknowledged:
                raise ConnectionClosed("connection lost after first prefill ACK")
            envelope, ack = await_ack(*args, **kwargs)
            assert isinstance(ack, SequenceAck)
            assert ack.status is SpeculatorStatusCode.OK
            acknowledged.append(ack)
            return envelope, ack

        with monkeypatch.context() as patch:
            patch.setattr(session, "_await", receive)
            return prefill_many(chunks, **kwargs)

    monkeypatch.setattr(session, "prefill_many", disconnect_after_first_ack)
    batch = inputs(["a", "b"], [[1, 2], [11, 12]], [[0, 1], [0, 1]], [2, 2])
    policy = settings.speculative_config.remote_draft.failure_policy
    error = "RemoteDraftError: prefill failed for"
    if policy == "error":
        with pytest.raises(RuntimeError, match=error):
            propose(instance, batch, [1, 1], [0, 0], [3, 13])
    else:
        assert propose(instance, batch, [1, 1], [0, 0], [3, 13]).tolist() == [
            [-1] * K,
            [-1] * K,
        ]
    assert len(acknowledged) == 1
    assert not session.connected
    assert all(
        session.sequence_state(instance._requests[req_id].key)
        is remote.SequenceState.INVALID
        for req_id in batch.req_ids
    )
    assert executor.plans == []

    group.rank_in_group = 1
    try:
        if policy == "error":
            with pytest.raises(RuntimeError, match=error):
                propose(peer, batch, [1, 1], [0, 0], [3, 13])
        else:
            assert propose(peer, batch, [1, 1], [0, 0], [3, 13]).tolist() == [
                [-1] * K,
                [-1] * K,
            ]
    finally:
        group.rank_in_group = 0

    def reject_features(*args, **kwargs):
        pytest.fail("disconnected own-KV requests resumed remote decoding")

    monkeypatch.setattr(instance, "_features", reject_features)
    batch = inputs(["a", "b"], [[3], [13]], [[2], [2]], [0, 0])
    assert propose(instance, batch, [1, 1], [0, 0], [4, 14]).tolist() == [
        [-1] * K,
        [-1] * K,
    ]


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
@pytest.mark.parametrize("healthy_sibling", [True, False])
def test_new_requests_cannot_refresh_the_round_budget_for_existing_decode(
    verifier, monkeypatch, healthy_sibling
):
    instance, _, _, _ = verifier
    if healthy_sibling:
        propose(instance, inputs(["good"], [[1]], [[0]], [1]), [1], [0], [2])
    session = instance._session
    now = [10.0]
    session._clock = lambda: now[0]
    session._request_timeout_s = 0.1
    send, wait = session._send, session._await
    opened = set()
    events = []
    deadlines = []

    def record_open(message, **kwargs):
        request_id = send(message, **kwargs)
        if isinstance(message, OpenSequence):
            opened.add(request_id)
            events.append("send")
        return request_id

    def timeout_opens(matches, deadline, *, request_ids):
        if opened.intersection(request_ids):
            events.append("wait")
            deadlines.append(deadline)
            now[0] = deadline
            raise TransportTimeout("open acknowledgement deadline elapsed")
        return wait(matches, deadline, request_ids=request_ids)

    monkeypatch.setattr(session, "_send", record_open)
    monkeypatch.setattr(session, "_await", timeout_opens)
    if healthy_sibling:
        batch = inputs(
            ["good", "a", "b", "c"],
            [[2], [5], [6], [7]],
            [[1], [0], [0], [0]],
            [0, 1, 1, 1],
        )
        recovery = [3, 8, 9, 10]
    else:
        batch = inputs(["a", "b", "c"], [[5], [6], [7]], [[0], [0], [0]], [1] * 3)
        recovery = [8, 9, 10]
    drafts = propose(
        instance, batch, [1] * batch.num_reqs, [0] * batch.num_reqs, recovery
    )
    assert events == ["send", "send", "send", "wait"]
    assert deadlines == [pytest.approx(10.1)]
    expected = [[-1] * K] * batch.num_reqs
    assert drafts.tolist() == expected
    assert not session.connected
    assert all(
        session.sequence_state(instance._requests[req_id].key)
        in (remote.SequenceState.TARGET_ONLY, remote.SequenceState.INVALID)
        for req_id in ["a", "b", "c"]
    )


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_target_only_round_skips_feature_preparation_and_still_closes(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier

    def fail(plan, features, *, num_speculative_tokens=None):
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    batch = inputs(["a"], [[1]], [[0]], [1])
    propose(instance, batch, [1], [0], [2])

    def reject_features(*args, **kwargs):
        pytest.fail("target-only round prepared remote features")

    monkeypatch.setattr(instance, "_features", reject_features)
    batch = inputs(["a"], [[2]], [[1]], [0])
    assert propose(instance, batch, [1], [0], [3]).tolist() == [[-1] * K]
    key = instance._requests["a"].key
    instance.close_requests({"a"})
    assert instance._requests == {}
    assert instance._session.sequence_state(key) is None


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
@pytest.mark.parametrize("reconnect", [False, True])
def test_disconnected_new_request_attempts_reconnect_before_feature_preparation(
    verifier, monkeypatch, reconnect
):
    instance, _, _, _ = verifier
    session = instance._session
    session._invalidate("test connection loss")
    attempts = []
    maybe_reconnect = session.maybe_reconnect

    def unavailable(**kwargs):
        attempts.append(True)
        return maybe_reconnect(**kwargs) if reconnect else False

    def reject_features(*args, **kwargs):
        pytest.fail("disconnected request prepared remote features")

    monkeypatch.setattr(session, "maybe_reconnect", unavailable)
    if not reconnect:
        monkeypatch.setattr(instance, "_features", reject_features)
    batch = inputs(["new"], [[1, 2]], [[0, 1]], [2])
    expected = digest_row([1, 2], 3) if reconnect else [-1] * K
    assert propose(instance, batch, [1], [0], [3]).tolist() == [expected]
    assert attempts
    key = instance._requests["new"].key
    instance.close_requests({"new"})
    assert session.sequence_state(key) is None


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_mixed_target_only_batch_transfers_only_active_ragged_features(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    execute = executor.execute

    def fail(plan, features, *, num_speculative_tokens=None):
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    propose(instance, inputs(["dead"], [[1]], [[0]], [1]), [1], [0], [2])
    monkeypatch.setattr(executor, "execute", execute)
    transferred = []
    tensor_to = torch.Tensor.to

    def track_hidden(tensor, *args, **kwargs):
        if tensor.ndim == 2 and tensor.dtype == torch.float32:
            transferred.append(tensor[:, 0].tolist())
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", track_hidden)
    batch = inputs(
        ["b", "dead", "c"],
        [[11, 12], [2], [21, 22, 23]],
        [[0, 1], [1], [0, 1, 2]],
        [4, 0, 3],
        [3, 0, 1],
    )
    result = instance._propose(
        batch,
        torch.arange(6, dtype=torch.float32)[:, None].expand(-1, 4),
        None,
        torch.tensor([0, 1, 1]),
        torch.tensor([0, 0, 0]),
        torch.tensor([[3], [30], [0], [0]]),
        torch.tensor([[0, 0, 0, 13]]),
    )
    assert result.tolist() == [[-1] * K, [-1] * K, digest_row([21, 22, 23], 30)]
    assert transferred == [[0, 1, 3, 4, 5]]
    assert executor.plans[-2].next_prefill_tokens.tolist() == [[13]]


def test_auxiliary_features_are_selected_before_concatenation(monkeypatch):
    instance = object.__new__(remote.RemoteSpeculator)
    instance.device = torch.device("cpu")
    instance._session = SimpleNamespace(selected_transport="inline")
    instance._contract = SimpleNamespace(
        schema=TargetFeatureSchema(
            1, (FeatureSlot("aux_hidden_states", "float32", (8,)),)
        )
    )
    batch = inputs(["a", "b"], [[1, 2], [3]], [[0, 1], [0]], [2, 1])
    concatenated = []
    tensor_cat = torch.cat

    def track_aux(tensors, **kwargs):
        concatenated.append([tensor[:, 0].tolist() for tensor in tensors])
        return tensor_cat(tensors, **kwargs)

    monkeypatch.setattr(torch, "cat", track_aux)
    state = torch.arange(3, dtype=torch.float32)[:, None].expand(-1, 4)
    features = instance._features(
        batch,
        state,
        [state, state + 10],
        torch.zeros(1, 2, dtype=torch.int64),
        token_rows=[2],
        sequence_rows=[1],
    )
    assert concatenated == [[[2], [12]]]
    assert features["aux_hidden_states"].shape == (1, 8)


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
@pytest.mark.parametrize("stage", ["aux_cat", "prefill_select", "decode_select"])
@pytest.mark.parametrize(
    "allocation_error",
    [
        torch.OutOfMemoryError("remote feature allocation failed"),
        MemoryError("remote feature allocation failed"),
        OSError(errno.ENOMEM, "remote feature allocation failed"),
        RuntimeError(
            "[enforce fail at alloc_cpu.cpp:127] err == 0. "
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
            "67108864 bytes. Error code 12 (Cannot allocate memory)"
        ),
    ],
)
def test_remote_feature_oom_latches_affected_rows_and_shares_policy(
    verifier, monkeypatch, stage, allocation_error
):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    policy = settings.speculative_config.remote_draft.failure_policy
    aux = None
    if stage == "decode_select":
        first = inputs(["a"], [[1]], [[0]], [1])
        drafts = propose(instance, first, [1], [0], [2])[0].tolist()
        batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
        sampled, rejected, recovery = [1], [3], [3]
        expected_computed = [2]
    elif stage == "prefill_select":
        batch = inputs(["a", "b"], [[1], [3]], [[0], [0]], [4, 4])
        sampled, rejected, recovery = [0, 0], [0, 0], [0, 0]
        expected_computed = [1, 0]
    else:
        batch = inputs(["a"], [[1]], [[0]], [1])
        sampled, rejected, recovery = [1], [0], [2]
        expected_computed = [0]
        aux = [torch.zeros(1, 4), torch.zeros(1, 4)]
    plan_count = len(executor.plans)
    original_select = instance._select
    select_calls = 0

    def fail_cat(*args, **kwargs):
        raise allocation_error

    def select_with_allocation_failure(features, token_rows, sequence_rows):
        nonlocal select_calls
        select_calls += 1
        fail_on_call = 2 if stage == "prefill_select" else 1
        if select_calls != fail_on_call:
            return original_select(features, token_rows, sequence_rows)
        getitem = torch.Tensor.__getitem__

        def fail_feature_index(tensor, index):
            if any(tensor is feature for feature in features.values()):
                raise allocation_error
            return getitem(tensor, index)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "__getitem__", fail_feature_index)
            return original_select(features, token_rows, sequence_rows)

    with monkeypatch.context() as patch:
        if stage == "aux_cat":
            # Failure happens in the real _features before any tensor is sent;
            # the real session/server keep their negotiated contract unchanged.
            patch.setattr(
                instance,
                "_contract",
                SimpleNamespace(
                    schema=TargetFeatureSchema(
                        1,
                        (
                            *SCHEMA.slots,
                            FeatureSlot("aux_hidden_states", "float32", (8,)),
                        ),
                    )
                ),
            )
            patch.setattr(torch, "cat", fail_cat)
        else:
            patch.setattr(instance, "_select", select_with_allocation_failure)
        if policy == "error":
            with pytest.raises(RuntimeError, match="remote feature preparation failed"):
                propose(
                    instance, batch, sampled, rejected, recovery, aux_hidden_states=aux
                )
        else:
            assert propose(
                instance, batch, sampled, rejected, recovery, aux_hidden_states=aux
            ).tolist() == [[-1] * K for _ in batch.req_ids]
    assert len(executor.plans) == plan_count
    assert [
        instance._requests[req_id].num_computed for req_id in batch.req_ids
    ] == expected_computed
    assert all(
        instance._session.sequence_state(instance._requests[req_id].key)
        is remote.SequenceState.TARGET_ONLY
        for req_id in batch.req_ids
    )
    group.rank_in_group = 1
    try:
        if policy == "error":
            with pytest.raises(RuntimeError, match="remote feature preparation failed"):
                propose(peer, batch, sampled, rejected, recovery)
        else:
            assert propose(peer, batch, sampled, rejected, recovery).tolist() == [
                [-1] * K for _ in batch.req_ids
            ]
    finally:
        group.rank_in_group = 0

    # Do not resume an own-KV sequence after skipping its verified context.
    # A separate request can still use the healthy connection and clean up.
    with monkeypatch.context() as patch:

        def reject_features(*args, **kwargs):
            pytest.fail("an allocation-failed request resumed remote decoding")

        patch.setattr(instance, "_features", reject_features)
        next_batch = inputs(
            batch.req_ids,
            [[7]] * batch.num_reqs,
            [[8]] * batch.num_reqs,
            [0] * batch.num_reqs,
        )
        assert propose(
            instance,
            next_batch,
            [1] * batch.num_reqs,
            [0] * batch.num_reqs,
            [8] * batch.num_reqs,
        ).tolist() == [[-1] * K for _ in batch.req_ids]
    fresh = inputs(["fresh"], [[11]], [[0]], [1])
    assert propose(instance, fresh, [1], [0], [12]).tolist() == [digest_row([11], 12)]
    instance.close_requests({*batch.req_ids, "fresh"})
    assert instance._requests == {}


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_feature_failure_retires_remote_kv_without_waiting_for_an_ack(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    session = instance._session
    free_blocks = executor.pool.num_free_blocks
    propose(instance, inputs(["a"], [[1, 2]], [[0, 1]], [2]), [1], [0], [3])
    key = instance._requests["a"].key
    assert executor.pool.num_free_blocks < free_blocks
    released = threading.Event()
    release = executor.pool.release

    def release_blocks(req_index):
        release(req_index)
        released.set()

    def fail_features(*args, **kwargs):
        raise torch.OutOfMemoryError("remote feature allocation failed")

    with monkeypatch.context() as patch:
        patch.setattr(executor.pool, "release", release_blocks)
        patch.setattr(instance, "_features", fail_features)
        patch.setattr(
            session,
            "_await",
            lambda *args, **kwargs: pytest.fail("cleanup waited for ACK"),
        )
        batch = inputs(["a"], [[3]], [[2]], [0])
        assert propose(instance, batch, [1], [0], [4]).tolist() == [[-1] * K]
        assert released.wait(2), "terminal fallback retained remote draft KV"
    assert executor.pool.num_free_blocks == free_blocks
    assert session.sequence_state(key) is remote.SequenceState.TARGET_ONLY
    assert session.connected
    assert instance._requests["a"].key == key

    # The late close ACK must not interfere with a healthy new request.
    assert propose(
        instance, inputs(["b"], [[11]], [[0]], [1]), [1], [0], [12]
    ).tolist() == [digest_row([11], 12)]


@pytest.mark.parametrize("num_requests", [1, 2])
@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_all_remote_timeouts_release_kv_before_target_only_steps(
    verifier, monkeypatch, num_requests
):
    instance, executor, _, _ = verifier
    session = instance._session
    pool = executor.pool
    initial_free = pool.num_free_blocks
    ids = [f"r{i}" for i in range(num_requests)]
    first = inputs(
        ids, [[1, 2]] * num_requests, [[0, 1]] * num_requests, [2] * num_requests
    )
    propose(instance, first, [1] * num_requests, [0] * num_requests, [3] * num_requests)
    keys = [instance._requests[r].key for r in ids]
    assert pool.num_free_blocks < initial_free
    released = threading.Event()
    release = pool.release

    def release_blocks(index):
        release(index)
        if pool.num_free_blocks == initial_free:
            released.set()

    now = session._clock()
    monkeypatch.setattr(pool, "release", release_blocks)
    monkeypatch.setattr(session, "_clock", lambda: now)

    def timeout(*args, **kwargs):
        nonlocal now
        now += 10.0
        raise TransportTimeout("proposal receive deadline expired")

    monkeypatch.setattr(session, "_await", timeout)
    batch = inputs(ids, [[3]] * num_requests, [[2]] * num_requests, [0] * num_requests)
    expected = [[-1] * K] * num_requests
    assert (
        propose(
            instance, batch, [1] * num_requests, [0] * num_requests, [4] * num_requests
        ).tolist()
        == expected
    )
    assert not session.connected and not session._retiring
    assert released.wait(2), "timeout fallback retained draft KV without future RPCs"
    assert all(
        session.sequence_state(k) is remote.SequenceState.TARGET_ONLY for k in keys
    )
    assert pool.num_free_blocks == initial_free

    def reject_remote_work(*args, **kwargs):
        pytest.fail("terminal target-only steps prepared features or sent an RPC")

    monkeypatch.setattr(instance, "_features", reject_remote_work)
    monkeypatch.setattr(session, "_send", reject_remote_work)
    for _ in range(200):
        assert (
            propose(
                instance,
                batch,
                [1] * num_requests,
                [0] * num_requests,
                [4] * num_requests,
            ).tolist()
            == expected
        )
    assert [instance._requests[r].key for r in ids] == keys


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
@pytest.mark.parametrize("is_final", [False, True])
def test_prefill_timeout_releases_kv_without_a_fresh_cleanup_budget(
    verifier, monkeypatch, is_final
):
    instance, executor, settings, _ = verifier
    session = instance._session
    now = [10.0]
    session._clock = lambda: now[0]
    session._request_timeout_s = 0.1
    initial_free = executor.pool.num_free_blocks
    propose(instance, inputs(["a"], [[1, 2]], [[0, 1]], [6]), [0], [0], [3], [3])
    assert executor.pool.num_free_blocks < initial_free
    completed, released = threading.Event(), threading.Event()
    prefill = TargetConditionedDraftAdapter.prefill
    release = executor.pool.release

    def prefill_and_notify(adapter, *args, **kwargs):
        result = prefill(adapter, *args, **kwargs)
        completed.set()
        return result

    def release_blocks(index):
        release(index)
        if executor.pool.num_free_blocks == initial_free:
            released.set()

    monkeypatch.setattr(TargetConditionedDraftAdapter, "prefill", prefill_and_notify)
    monkeypatch.setattr(executor.pool, "release", release_blocks)
    prefill_many = session.prefill_many

    def timeout_prefill(chunks, **kwargs):
        def timeout(matches, deadline, **kwargs):
            assert completed.wait(2), "CPU adapter did not finish real prefill"
            now[0] = deadline
            raise TransportTimeout("prefill ACK deadline expired")

        with monkeypatch.context() as patch:
            patch.setattr(session, "_await", timeout)
            return prefill_many(chunks, **kwargs)

    retire = session._send_retirements

    def reject_late_retirement(*args, **kwargs):
        if session._retiring:
            pytest.fail("prefill timeout granted a new retirement write budget")
        return retire(*args, **kwargs)

    monkeypatch.setattr(session, "prefill_many", timeout_prefill)
    monkeypatch.setattr(session, "_send_retirements", reject_late_retirement)
    batch = inputs(["a"], [[3, 4]], [[2, 3]], [4 if is_final else 6])
    if settings.speculative_config.remote_draft.failure_policy == "error":
        with pytest.raises(RuntimeError, match="RemoteDraftError: prefill failed"):
            propose(instance, batch, [int(is_final)], [0], [3], [3])
    else:
        assert propose(instance, batch, [int(is_final)], [0], [3], [3]).tolist() == [
            [-1] * K
        ]
    key = instance._requests["a"].key
    assert session.sequence_state(key) is remote.SequenceState.TARGET_ONLY
    assert not session.connected and not session._retiring
    assert released.wait(2), "prefill timeout retained remote KV without future RPCs"
    assert executor.pool.num_free_blocks == initial_free

    def reject_remote_work(*args, **kwargs):
        pytest.fail("terminal fallback resumed feature preparation or remote work")

    monkeypatch.setattr(instance, "_features", reject_remote_work)
    monkeypatch.setattr(session, "_send", reject_remote_work)
    fallback = inputs(["a"], [[5]], [[4]], [0 if is_final else 6])
    for _ in range(3):
        assert propose(instance, fallback, [1], [0], [4]).tolist() == [[-1] * K]


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
@pytest.mark.parametrize(
    "kind",
    [
        "cuda",
        "runtime",
        "errno",
        "cpu_einval",
        "memory_fatal",
        "torch_fatal",
        "wrapped_memory",
        "context_fatal",
    ],
)
def test_remote_feature_unrecoverable_error_is_shared_and_not_fallback(
    verifier, monkeypatch, kind
):
    instance, _, _, group = verifier

    def fail_features(*args, **kwargs):
        error = {
            "cuda": RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            ),
            "runtime": RuntimeError("unrelated allocation failed"),
            "errno": OSError(errno.EINVAL, "invalid allocation argument"),
            "cpu_einval": RuntimeError(
                "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
                "64 bytes. Error code 22 (Invalid argument)"
            ),
            "context_fatal": MemoryError("allocation after CUDA failure"),
            "memory_fatal": MemoryError("allocation after CUDA failure"),
            "torch_fatal": torch.OutOfMemoryError("allocation after CUDA failure"),
            "wrapped_memory": ValueError("invalid shape"),
        }[kind]
        if kind.endswith("fatal"):
            error.__cause__ = RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            )
        if kind == "context_fatal":
            error.__context__ = error.__cause__
            error.__cause__ = ValueError("nonfatal explicit cause")
        elif kind == "wrapped_memory":
            error.__cause__ = MemoryError("prior allocation failure")
        raise error

    monkeypatch.setattr(instance, "_features", fail_features)
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="remote drafter proposal failed"):
        propose(instance, batch, [1], [0], [2])
    assert group.payload[1] is None


@pytest.mark.parametrize("option", ["prefix", "async", "tp4", "pp2", "embeddings"])
def test_unsupported_placement_fails_before_connecting(option, monkeypatch):
    settings = config("tcp://127.0.0.1:1")
    if option == "prefix":
        settings.cache_config.enable_prefix_caching = True
    elif option == "async":
        settings.scheduler_config.async_scheduling = True
    elif option == "tp4":
        settings.parallel_config.tensor_parallel_size = 4
    elif option == "pp2":
        settings.parallel_config.pipeline_parallel_size = 2
    else:
        settings.model_config.enable_prompt_embeds = True
    with pytest.raises(ValueError, match="remote drafter"):
        remote.RemoteSpeculator(settings, torch.device("cpu"))
