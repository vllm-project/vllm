# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request/chunk translation through real configuration, Gloo, and sockets."""

import threading

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.remote.runtime_fixtures import (
    K,
    KVDigestExecutor,
    digest_row,
    make_config,
    make_planner,
)
from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from vllm.config.speculative import RemoteDraftConfig
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode import remote

pytestmark = pytest.mark.usefixtures("cpu_parallel_state")


class ControlledExecutor(KVDigestExecutor):
    def __init__(self, pool):
        super().__init__(pool)
        self.error = None
        self.attempts = 0
        self.failure_after = 0
        self.gate = None
        self.entered = threading.Event()

    def execute(self, plan, features, *, num_speculative_tokens=None):
        self.attempts += 1
        if self.gate is not None:
            self.entered.set()
            assert self.gate.wait(5), "external executor was not released"
        if self.error is not None and self.attempts >= self.failure_after:
            raise self.error
        return super().execute(
            plan, features, num_speculative_tokens=num_speculative_tokens
        )


class ControlledAdapter(TargetConditionedDraftAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = threading.Event()
        self.close_error = None

    def close_sequences(self, keys):
        if self.close_error is not None:
            raise self.close_error
        super().close_sequences(keys)
        self.closed.set()


@pytest.fixture
def verifier(request):
    options = getattr(request, "param", "error")
    if isinstance(options, tuple):
        policy, draft_steps, *timeouts = options
    else:
        policy, draft_steps, timeouts = options, K, []
    settings = make_config(num_speculative_tokens=draft_steps)
    contract = create_runtime_contract(settings)
    planner = make_planner()
    planner.num_speculative_tokens = draft_steps
    executor = ControlledExecutor(planner.block_pool)
    adapter = ControlledAdapter(
        executor,
        planner,
        capabilities=contract.capabilities,
        feature_schema=contract.schema,
        methods={"eagle3"},
        draft_checkpoint_fingerprint=contract.draft_fingerprint,
    )
    server = RemoteDraftServer(
        adapter,
        target_fingerprint=contract.target_fingerprint,
        tokenizer_fingerprint=contract.tokenizer_fingerprint,
    )
    server.start("tcp://127.0.0.1:0")
    settings.speculative_config.remote_draft = RemoteDraftConfig(
        endpoint=server.endpoint,
        failure_policy=policy,
        request_timeout_ms=timeouts[0] if timeouts else 5000,
    )
    instance = remote.RemoteSpeculator(settings, torch.device("cpu"))
    yield instance, executor, settings, server
    instance.close()
    server.stop()


def inputs(ids, tokens, positions, prefill, slots=None):
    count = len(ids)
    slots = slots if slots is not None else list(range(count))
    lens = np.array([len(row) for row in tokens], dtype=np.int32)
    indices = torch.tensor(slots, dtype=torch.int64)
    starts = np.cumsum([0, *lens], dtype=np.int32)
    computed = np.array([row[0] if row else 0 for row in positions], dtype=np.int32)
    seq_lens = torch.from_numpy(computed + lens)
    is_prefilling = np.array([length > 0 for length in prefill])
    return InputBatch(
        req_ids=ids,
        num_reqs=count,
        num_reqs_after_padding=count,
        idx_mapping=indices,
        idx_mapping_np=np.array(slots),
        expanded_idx_mapping=indices,
        expanded_local_pos=torch.zeros(count, dtype=torch.int32),
        num_scheduled_tokens=lens,
        num_tokens=int(sum(lens)),
        num_tokens_after_padding=int(sum(lens)),
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=torch.from_numpy(starts),
        query_start_loc_np=starts,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        dcp_local_seq_lens=None,
        num_computed_tokens_np=computed,
        prefill_len_np=np.array(prefill),
        num_computed_prefill_tokens_np=computed,
        is_prefilling_np=is_prefilling,
        has_prefill=bool(is_prefilling.any()),
        decode_graph_eligible=not is_prefilling.any(),
        input_ids=torch.tensor(sum(tokens, []), dtype=torch.int32),
        positions=torch.tensor(sum(positions, []), dtype=torch.int64),
        is_padding=torch.zeros(int(sum(lens)), dtype=torch.bool),
        logits_indices=torch.arange(count),
        cu_num_logits=torch.arange(count + 1),
        cu_num_logits_np=np.arange(count + 1),
        has_structured_output_reqs=False,
        prompt_lens=None,
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
        torch.zeros(batch.num_tokens, 16),
        kwargs.pop("aux_hidden_states", None),
        torch.tensor(sampled),
        torch.tensor(rejected),
        last_sampled,
        next_prefill,
        kwargs.pop("temperature", torch.zeros(4)),
        torch.zeros(4, dtype=torch.int64),
        **kwargs,
    )


@pytest.mark.parametrize("effective_k", [0, 1, K])
def test_empty_proposal_preserves_batch_shape_without_rpc(verifier, effective_k):
    instance, executor, _, _ = verifier
    batch = inputs([], [], [], [])

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
    [("error", k) for k in (1, 3)],
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
def test_dummy_and_profile_do_not_contact_or_advance_server(verifier, flag):
    instance, executor, _, _ = verifier

    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2], **{flag: True}).tolist() == [
        [-1] * K
    ]
    assert executor.plans == []
    assert len(verifier[3].registry) == 0


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


def test_invalid_verification_counts_fail_before_advancing_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[1]], [[0]], [1])
    drafts = propose(instance, batch, [1], [0], [2]).tolist()[0]
    plan_count = len(executor.plans)
    batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
    with pytest.raises(RuntimeError, match="counts do not cover"):
        propose(instance, batch, [2], [0], [3])
    assert len(executor.plans) == plan_count


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
@pytest.mark.parametrize("stage", ["prefill_chunk", "final_prefill", "decode"])
def test_executor_oom_latches_rows_reclaims_kv_and_allows_healthy_siblings(
    verifier, stage
):
    instance, executor, settings, server = verifier
    pool = executor.pool
    free = pool.num_free_blocks
    if stage == "decode":
        drafts = propose(instance, inputs(["a"], [[1]], [[0]], [1]), [1], [0], [2])[
            0
        ].tolist()
        batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
        sampled, rejected, recovery = [1], [3], [3]
    else:
        batch = inputs(["a"], [[1]], [[0]], [1 if stage == "final_prefill" else 4])
        sampled, rejected, recovery = [int(stage == "final_prefill")], [0], [2]
    before = executor.attempts
    executor.error = torch.OutOfMemoryError("draft KV exhausted")
    if settings.speculative_config.remote_draft.failure_policy == "error":
        with pytest.raises(RuntimeError, match="failed"):
            propose(instance, batch, sampled, rejected, recovery, [2])
    else:
        assert propose(instance, batch, sampled, rejected, recovery, [2]).tolist() == [
            [-1] * K
        ]
    assert executor.attempts == before + 1
    assert pool.num_free_blocks == free
    executor.error = None
    # A terminal row cannot resume after omitting verified context.
    followup = inputs(["a", "fresh"], [[7], [11]], [[8], [0]], [0, 1])
    assert propose(instance, followup, [1, 1], [0, 0], [8, 12]).tolist() == [
        [-1] * K,
        digest_row([11], 12),
    ]
    assert executor.attempts == before + 2
    assert server.healthy
    instance.close_requests({"a", "fresh"})
    assert pool.num_free_blocks == free
    assert len(server.registry) == 0


@pytest.mark.parametrize("option", ["prefix", "async", "tp4", "pp2", "embeddings"])
def test_unsupported_placement_fails_before_connecting(option):
    settings = make_config()
    settings.speculative_config.remote_draft = RemoteDraftConfig(
        endpoint="tcp://127.0.0.1:1"
    )
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


def _tp_proposal_worker(rank, settings, rendezvous, output_dir, failure):
    import json
    from pathlib import Path

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=rendezvous,
        backend="gloo",
    )
    with set_current_vllm_config(settings):
        initialize_model_parallel(tensor_model_parallel_size=2, backend="gloo")
        instance = remote.RemoteSpeculator(settings, torch.device("cpu"))
        try:
            batch = inputs(["tp"], [[1]], [[0]], [1])
            tokens = propose(instance, batch, [1], [0], [2]).tolist()
            with pytest.raises(RuntimeError, match="greedy target sampling") as invalid:
                propose(instance, batch, [1], [0], [2], temperature=torch.ones(4))
            error = None
            if failure == "fatal":
                batch = inputs(["tp"], [[2, *tokens[0]]], [[1, 2, 3, 4]], [0])
                with pytest.raises(RuntimeError, match="proposal failed") as caught:
                    propose(instance, batch, [1], [3], [3])
                error = str(caught.value)
            else:
                instance.close_requests({"tp"})
            Path(output_dir, f"rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "tokens": tokens,
                        "error": error,
                        "validation_error": str(invalid.value),
                    }
                )
            )
        finally:
            instance.close()
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.parametrize("failure", ["fatal", "close"])
def test_tp_processes_share_rank_zero_tokens_and_terminal_outcomes(
    verifier, tmp_path, monkeypatch, failure
):
    import json

    from vllm.v1.spec_decode.remote.errors import CudaRuntimeError

    instance, executor, settings, server = verifier
    instance.close()
    settings.parallel_config.tensor_parallel_size = 2
    settings.speculative_config.remote_draft.failure_policy = "target_only"
    if failure == "fatal":
        settings.speculative_config.remote_draft.failure_policy = "error"
        executor.error = CudaRuntimeError("executor failed", cuda_error_code=700)
        executor.failure_after = 2
    else:
        server.adapter.close_error = MemoryError("external executor retirement failed")
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.multiprocessing.spawn(
        _tp_proposal_worker,
        args=(settings, f"file://{tmp_path / 'tp-store'}", str(tmp_path), failure),
        nprocs=2,
        join=True,
    )
    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    assert results[0] == results[1]
    assert results[0]["tokens"] == [digest_row([1], 2)]
    assert len(executor.plans) == 1
    assert not server.healthy
    assert executor.pool.num_free_blocks < executor.pool.num_blocks - 1
    assert len(server.registry) == 1


@pytest.mark.parametrize(
    "verifier", [("target_only", K, 1000), ("error", K, 1000)], indirect=True
)
@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_timeout_retires_owned_kv_before_terminal_target_only_steps(verifier, stage):
    instance, executor, settings, server = verifier
    free = executor.pool.num_free_blocks
    if stage == "decode":
        drafts = propose(instance, inputs(["a"], [[1]], [[0]], [1]), [1], [0], [2])[
            0
        ].tolist()
        batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
        sampled, rejected, recovery = [1], [3], [3]
    else:
        batch = inputs(["a"], [[1, 2]], [[0, 1]], [4])
        sampled, rejected, recovery = [0], [0], [3]
    executor.gate = threading.Event()
    try:
        if settings.speculative_config.remote_draft.failure_policy == "error":
            with pytest.raises(RuntimeError, match="failed"):
                propose(instance, batch, sampled, rejected, recovery, [3])
        else:
            assert propose(
                instance, batch, sampled, rejected, recovery, [3]
            ).tolist() == [[-1] * K]
        assert executor.entered.is_set()
        assert executor.pool.num_free_blocks < free
    finally:
        executor.gate.set()
    assert server.adapter.closed.wait(5), (
        "timeout retained draft KV without future RPCs"
    )
    assert executor.pool.num_free_blocks == free
    before = executor.attempts
    for _ in range(3):
        assert propose(
            instance, inputs(["a"], [[7]], [[8]], [0]), [1], [0], [8]
        ).tolist() == [[-1] * K]
    assert executor.attempts == before
    instance.close_requests({"a"})
    executor.gate = None
    assert propose(
        instance, inputs(["fresh"], [[11]], [[0]], [1]), [1], [0], [12]
    ).tolist() == [digest_row([11], 12)]


@pytest.mark.parametrize(
    "remote_enabled,resumable", [(True, True), (False, True), (True, False)]
)
def test_resumable_input_admission_uses_real_input_processor(remote_enabled, resumable):
    from vllm import SamplingParams
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    config = make_config()
    if remote_enabled:
        config.speculative_config.remote_draft = RemoteDraftConfig(
            endpoint="tcp://127.0.0.1:1"
        )
    processor = InputProcessor(config)
    if remote_enabled and resumable:
        with pytest.raises(
            VLLMValidationError, match="does not support input streaming"
        ):
            processor.process_inputs(
                "stream",
                {"prompt_token_ids": [1, 2]},
                SamplingParams(temperature=0, max_tokens=2),
                ("generate",),
                resumable=resumable,
            )
    else:
        request = processor.process_inputs(
            "stream",
            {"prompt_token_ids": [1, 2]},
            SamplingParams(temperature=0, max_tokens=2),
            ("generate",),
            resumable=resumable,
        )
        assert request.resumable is resumable
        assert request.prompt_token_ids == [1, 2]
