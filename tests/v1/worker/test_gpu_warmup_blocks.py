# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 warmup must reserve the KV blocks the scheduler would reserve.

`KVCacheManager.allocate_slots` sizes every allocation for
`num_computed + num_scheduled + num_lookahead_tokens`, because the speculator
writes KV past the token range the target model was scheduled for. The warmup
hand-builds its `SchedulerOutput`s, so it has to reserve the same blocks.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager
from tests.v1.core.utils import create_requests, new_kv_cache_group
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    MambaSpec,
)
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import (
    _reserved_block_count,
    run_context_prefill_warmup,
    run_mixed_prefill_decode_warmup,
    warmup_kernels,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
NUM_SPEC_STEPS = 3

# `warmup_kernels` ends on `torch.accelerator.synchronize()`.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="warmup synchronizes on the accelerator",
)


def _attention_group() -> KVCacheGroup:
    return new_kv_cache_group(
        ["layer"],
        AttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )


def _circular_group() -> KVCacheGroup:
    return new_kv_cache_group(
        ["circular"],
        CircularBufferManager.layer_config(
            AttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            )
        ),
    )


def _mamba_group(mamba_cache_mode: str) -> KVCacheGroup:
    # Name carries the mode so several groups can coexist in one config.
    return new_kv_cache_group(
        [f"mamba_{mamba_cache_mode}"],
        MambaManager.layer_config(
            MambaSpec(block_size=BLOCK_SIZE, shapes=((1,),), dtypes=(torch.float32,)),
            mamba_cache_mode=mamba_cache_mode,
            num_speculative_blocks=NUM_SPEC_STEPS,
        ),
    )


def _make_runner(
    kv_cache_groups: list[KVCacheGroup],
    num_lookahead_tokens: int,
    num_spec_steps: int = NUM_SPEC_STEPS,
) -> SimpleNamespace:
    """Stub model runner exposing only what the warmup entry points read."""
    return SimpleNamespace(
        num_speculative_steps=num_spec_steps,
        speculative_config=(
            SimpleNamespace(max_num_new_slots_for_drafting=0)
            if num_spec_steps
            else None
        ),
        adaptive_verification=None,
        speculator=None,
        rejection_sampler=None,
        decode_query_len=num_spec_steps + 1,
        is_pooling_model=False,
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        max_num_reqs=4,
        max_num_tokens=2048,
        max_model_len=MAX_MODEL_LEN,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        model_state=SimpleNamespace(max_encoder_len=0),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=kv_cache_groups, num_blocks=1024
        ),
        vllm_config=SimpleNamespace(
            num_lookahead_tokens=num_lookahead_tokens,
            num_scheduled_spec_tokens=num_spec_steps,
            is_mm_encoder_only=False,
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        extensible_kv_cache=None,
    )


class _StepRecorder:
    """Rebuilds each request's per-group block holdings from the warmup steps."""

    def __init__(self) -> None:
        # (blocks held per group, num_computed_tokens, num_scheduled_tokens)
        self.steps: list[tuple[list[int], int, int]] = []
        self.num_spec_tokens_to_schedule: list[int] = []
        self._held: dict[str, list[int]] = {}

    def execute_model(self, scheduler_output) -> None:
        if scheduler_output.total_num_scheduled_tokens:
            self.num_spec_tokens_to_schedule.append(
                scheduler_output.num_spec_tokens_to_schedule
            )
        for new_req in scheduler_output.scheduled_new_reqs:
            self._held[new_req.req_id] = [len(ids) for ids in new_req.block_ids]
            self._record(new_req.req_id, new_req.num_computed_tokens, scheduler_output)
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                self._held[req_id] = [
                    held + len(ids)
                    for held, ids in zip(self._held[req_id], new_block_ids)
                ]
            self._record(req_id, cached.num_computed_tokens[i], scheduler_output)

    def _record(self, req_id: str, num_computed: int, scheduler_output) -> None:
        self.steps.append(
            (
                list(self._held[req_id]),
                num_computed,
                scheduler_output.num_scheduled_tokens[req_id],
            )
        )

    def sample_tokens(self, grammar_output=None) -> None:
        return None


def _assert_covers_lookahead(
    steps: list[tuple[list[int], int, int]], num_lookahead_tokens: int
) -> None:
    assert steps, "warmup ran no steps"
    for num_blocks, num_computed, num_scheduled in steps:
        num_tokens = min(
            num_computed + num_scheduled + num_lookahead_tokens, MAX_MODEL_LEN
        )
        assert num_blocks[0] >= cdiv(num_tokens, BLOCK_SIZE), (
            f"{num_blocks[0]} blocks for {num_computed}+{num_scheduled} tokens "
            f"and {num_lookahead_tokens} lookahead tokens"
        )


# 0 covers eagle / MTP / draft models, 1 covers DFlash's extra in-fill query.
@pytest.mark.parametrize("extra_lookahead", [0, 1])
@pytest.mark.parametrize("num_spec_steps", [1, 2, 3, 5, 7])
def test_warmup_kernels_reserves_lookahead_blocks(num_spec_steps, extra_lookahead):
    num_lookahead_tokens = num_spec_steps + extra_lookahead
    recorder = _StepRecorder()

    warmup_kernels(
        _make_runner([_attention_group()], num_lookahead_tokens, num_spec_steps),
        recorder.execute_model,
        recorder.sample_tokens,
    )

    _assert_covers_lookahead(recorder.steps, num_lookahead_tokens)
    assert set(recorder.num_spec_tokens_to_schedule) == {num_spec_steps}


def test_mixed_warmup_reserves_lookahead_blocks():
    num_lookahead_tokens = NUM_SPEC_STEPS + 1
    recorder = _StepRecorder()

    assert run_mixed_prefill_decode_warmup(
        _make_runner([_attention_group()], num_lookahead_tokens),
        worker_execute_model=recorder.execute_model,
        worker_sample_tokens=recorder.sample_tokens,
        num_tokens=128,
    )

    _assert_covers_lookahead(recorder.steps, num_lookahead_tokens)
    assert set(recorder.num_spec_tokens_to_schedule) == {NUM_SPEC_STEPS}


def test_warmup_kernels_reserves_drafting_slots(monkeypatch):
    runner = _make_runner([_attention_group()], NUM_SPEC_STEPS)
    runner.speculative_config.max_num_new_slots_for_drafting = 1
    runner.scheduler_config.max_num_seqs = 1000
    runner.kv_cache_config.num_blocks = 100_000
    outputs: list[SchedulerOutput] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    warmup._warmup_kernels(runner, outputs.append, lambda grammar_output: None)

    prefill_output = outputs[0]
    num_reqs = len(prefill_output.scheduled_new_reqs)
    assert num_reqs == 2048 // (runner.decode_query_len + 1 + 1)
    assert prefill_output.total_num_scheduled_tokens + num_reqs <= runner.max_num_tokens


class _StubExtensibleCache:
    def __init__(self, warmup_committable_blocks: int) -> None:
        self.warmup_committable_blocks = warmup_committable_blocks
        self.commits: list[int] = []

    def commit(self, num_blocks: int) -> None:
        self.commits.append(num_blocks)


def test_warmup_sizes_batches_by_the_rank_agreed_committable_count():
    """Each rank measures its own free memory; the warmup batch and what it
    commits must follow the smallest count across ranks, or ranks run
    different shapes and can skip steps the others wait on."""
    runner = _make_runner(
        [_attention_group()], num_lookahead_tokens=0, num_spec_steps=0
    )
    # The engine agreed on 3 blocks, whatever this rank alone could commit.
    cache = _StubExtensibleCache(3)
    runner.extensible_kv_cache = cache
    recorder = _StepRecorder()

    warmup_kernels(runner, recorder.execute_model, recorder.sample_tokens)

    # One block per request at this length: 3 blocks less the null block fit
    # two requests, not the four `max_num_seqs` allows.
    assert len(recorder._held) == 2
    assert cache.commits[0] == 3


def test_mixed_warmup_skips_when_ranks_agree_too_few_blocks():
    runner = _make_runner([_attention_group()], num_lookahead_tokens=0)
    cache = _StubExtensibleCache(9)
    runner.extensible_kv_cache = cache
    recorder = _StepRecorder()

    # 128 tokens need 9 blocks (8 prefill + 1 decode) beyond the null block.
    assert not run_mixed_prefill_decode_warmup(
        runner, recorder.execute_model, recorder.sample_tokens, num_tokens=128
    )
    assert cache.commits == []
    assert recorder.steps == []


def test_context_prefill_warmup_attends_to_the_requested_context():
    """Full-length requests resume from a prefix hit until their cached
    context covers the requested tokens, holding what the scheduler would."""
    num_lookahead_tokens = NUM_SPEC_STEPS + 1
    runner = _make_runner([_attention_group()], num_lookahead_tokens)
    runner.extensible_kv_cache = cache = _StubExtensibleCache(1024)
    recorder = _StepRecorder()

    assert run_context_prefill_warmup(
        runner,
        recorder.execute_model,
        recorder.sample_tokens,
        num_context_tokens=2 * MAX_MODEL_LEN,
    )

    _assert_covers_lookahead(recorder.steps, num_lookahead_tokens)
    # Each request fills the model length, so three cover twice the context.
    assert len(recorder.steps) == 3
    assert sum(num_computed for _, num_computed, _ in recorder.steps) >= (
        2 * MAX_MODEL_LEN
    )
    for _, num_computed, num_scheduled in recorder.steps:
        assert num_computed + num_scheduled < MAX_MODEL_LEN
        assert num_scheduled == runner.decode_query_len + 1
    assert cache.commits == [1 + 3 * cdiv(MAX_MODEL_LEN, BLOCK_SIZE)]


def test_context_prefill_warmup_stays_within_the_context_and_vocabulary():
    """A long model length must not inflate the context past what the step
    needs, nor its prompt ids past the vocabulary they embed from."""
    runner = _make_runner([_attention_group()], num_lookahead_tokens=0)
    runner.extensible_kv_cache = _StubExtensibleCache(1024)
    outputs: list[SchedulerOutput] = []

    assert run_context_prefill_warmup(
        runner, outputs.append, lambda grammar_output: None, num_context_tokens=100
    )

    (request,) = outputs[0].scheduled_new_reqs
    assert request.num_computed_tokens == 100
    vocab_size = runner.model_config.get_vocab_size()
    assert max(request.prompt_token_ids) < vocab_size < len(request.prompt_token_ids)


def test_context_prefill_warmup_reserves_drafting_slots():
    runner = _make_runner([_attention_group()], NUM_SPEC_STEPS)
    runner.speculative_config.max_num_new_slots_for_drafting = 1
    runner.max_num_reqs = 1000
    runner.max_model_len = 16
    runner.kv_cache_config.num_blocks = 1024
    outputs: list[SchedulerOutput] = []

    assert run_context_prefill_warmup(
        runner, outputs.append, lambda grammar_output: None, num_context_tokens=10_000
    )

    prefill_output = outputs[0]
    num_reqs = len(prefill_output.scheduled_new_reqs)
    assert num_reqs == runner.max_num_tokens // (runner.decode_query_len + 2)
    assert prefill_output.total_num_scheduled_tokens + num_reqs <= runner.max_num_tokens


def test_context_prefill_warmup_is_capped_by_the_committable_blocks():
    runner = _make_runner([_attention_group()], num_lookahead_tokens=0)
    blocks_per_req = cdiv(MAX_MODEL_LEN, BLOCK_SIZE)
    runner.extensible_kv_cache = _StubExtensibleCache(blocks_per_req)
    recorder = _StepRecorder()

    # The null block leaves one block short of a single full-length request.
    assert not run_context_prefill_warmup(
        runner, recorder.execute_model, recorder.sample_tokens, MAX_MODEL_LEN
    )
    assert recorder.steps == []


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_warmup_reserves_mamba_speculative_blocks(mamba_cache_mode):
    """Mamba groups hold the running-state block plus the speculative tail.

    `MambaManager` reserves `num_speculative_blocks` blocks past the token
    range in every cache mode, and the mamba kernels read all
    `1 + num_speculative_blocks` of those block-table columns (see
    `mamba_get_block_table_tensor`).
    """
    recorder = _StepRecorder()

    warmup_kernels(
        _make_runner(
            [_attention_group(), _mamba_group(mamba_cache_mode)], NUM_SPEC_STEPS
        ),
        recorder.execute_model,
        recorder.sample_tokens,
    )

    assert recorder.steps, "warmup ran no steps"
    for num_blocks, num_computed, num_scheduled in recorder.steps:
        # `MambaManager` drops the lookahead tokens in align mode to keep the
        # allocation block-aligned, and keeps them otherwise. Pin the token
        # range and the speculative tail separately, so an implementation that
        # returned a flat `1 + num_speculative_blocks` would fail.
        if mamba_cache_mode == "align":
            # Align mode sizes from the uncapped main-model range.
            lookahead, num_tokens = 0, num_computed + num_scheduled
        else:
            lookahead = NUM_SPEC_STEPS
            num_tokens = min(num_computed + num_scheduled + lookahead, MAX_MODEL_LEN)
        assert num_blocks[1] == cdiv(num_tokens, BLOCK_SIZE) + NUM_SPEC_STEPS, (
            f"{mamba_cache_mode}: {num_blocks[1]} mamba blocks for "
            f"{num_computed}+{num_scheduled} tokens and {lookahead} lookahead"
        )


def _hybrid_kv_cache_config(num_blocks: int) -> KVCacheConfig:
    """Attention, circular, and Mamba groups exercise every reservation branch."""
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            _attention_group(),
            _circular_group(),
            _mamba_group("none"),
            _mamba_group("align"),
        ],
    )


def _row_length_config(dcp_world_size: int = 1) -> SimpleNamespace:
    """What `max_num_blocks_per_req` reads; align-mode Mamba specs return
    before the row clamp."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_world_size),
    )


def test_reserved_block_count_matches_real_kv_cache_manager():
    """`_reserved_block_count` must predict exactly what the real
    `KVCacheManager.allocate_slots` consumes, for every group, at every step
    of a warmup-realistic trajectory: a prefill followed by decode steps that
    mix spec and non-spec shapes, all at the fixed `num_lookahead_tokens`
    `_warmup_block_counter` binds once per model runner.

    The existing tests above pin this arithmetic against a hand-built
    `_StepRecorder` fake, which only proves the fake and the production
    function agree with each other. Driving `allocate_slots` itself is the
    only way to catch a real divergence from the allocator warmup has to
    match.

    Since `num_computed_tokens` only grows and `num_lookahead_tokens` is held
    fixed across the trajectory, the real manager's held-block count is
    non-decreasing in lockstep with the prediction, so equality (not just
    the reservation lower bound) holds at every step -- exactly the shape of
    trajectory `warmup_kernels` itself drives the allocator through.
    """
    num_lookahead_tokens = NUM_SPEC_STEPS + 1
    decode_query_len = NUM_SPEC_STEPS + 1

    kv_cache_config = _hybrid_kv_cache_config(num_blocks=256)
    groups = kv_cache_config.kv_cache_groups
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=MAX_MODEL_LEN,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=decode_query_len + 1, block_size=BLOCK_SIZE
    )

    def _step(num_new_tokens: int) -> None:
        computed_blocks, num_new_computed, _ = manager.get_computed_blocks(request)
        blocks = manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed,
            computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        assert blocks is not None, "block pool exhausted"
        held = manager.get_block_ids(request.request_id)
        num_tokens = request.num_computed_tokens + num_new_tokens
        for group, group_held in zip(groups, held):
            expected = _reserved_block_count(
                num_tokens,
                group,
                vllm_config=_row_length_config(),
                num_lookahead_tokens=num_lookahead_tokens,
                max_model_len=MAX_MODEL_LEN,
                max_encoder_len=0,
            )
            mode = getattr(group.manager_config, "mamba_cache_mode", None)
            assert len(group_held) == expected, (
                f"{group.manager_cls.__name__} mode={mode}: real manager "
                f"holds {len(group_held)} blocks for {num_tokens} tokens, "
                f"_reserved_block_count predicts {expected}"
            )
        request.num_computed_tokens = num_tokens

    # Prefill, a spec decode step, a no-draft decode step, and another spec
    # decode step: the same shapes `warmup_kernels` drives its decode steps
    # through (see `decode_steps` in `warmup.py`).
    _step(decode_query_len + 1)
    _step(decode_query_len)
    _step(1)
    _step(decode_query_len)

    manager.free(request)


def test_reserved_block_count_scales_dcp_sharded_blocks():
    """Under DCP the scheduler's block of a sharded group spans every rank, so
    a long prompt holds `dcp_world_size` times fewer blocks than locally."""
    dcp_world_size = 4
    kv_cache_config = KVCacheConfig(
        num_blocks=256, kv_cache_tensors=[], kv_cache_groups=[_attention_group()]
    )
    (group,) = kv_cache_config.kv_cache_groups
    manager = make_kv_cache_manager(
        kv_cache_config,
        max_model_len=MAX_MODEL_LEN,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
        dcp_world_size=dcp_world_size,
        scheduler_block_size=BLOCK_SIZE * dcp_world_size,
    )
    num_tokens = MAX_MODEL_LEN - 1
    (request,) = create_requests(
        num_requests=1, num_tokens=num_tokens, block_size=BLOCK_SIZE
    )
    computed_blocks, num_new_computed, _ = manager.get_computed_blocks(request)
    assert manager.allocate_slots(
        request, num_tokens, num_new_computed, computed_blocks
    )
    (held,) = manager.get_block_ids(request.request_id)

    assert len(held) == _reserved_block_count(
        num_tokens,
        group,
        vllm_config=_row_length_config(dcp_world_size),
        num_lookahead_tokens=0,
        max_model_len=MAX_MODEL_LEN,
        max_encoder_len=0,
        dcp_world_size=dcp_world_size,
    )
    assert len(held) == cdiv(num_tokens, BLOCK_SIZE * dcp_world_size)
    manager.free(request)


def _config_without_post_init(
    speculative_config: SpeculativeConfig | None,
) -> VllmConfig:
    """A real `VllmConfig` built without `__post_init__`, which would otherwise
    require a resolvable draft model. The lookahead properties only read
    `speculative_config` and `diffusion_config`."""
    config = object.__new__(VllmConfig)
    object.__setattr__(config, "speculative_config", speculative_config)
    object.__setattr__(config, "diffusion_config", None)
    return config


@pytest.mark.parametrize(
    ("method", "draft_hf_config", "expected"),
    [
        ("eagle", None, NUM_SPEC_STEPS),
        ("eagle3", None, NUM_SPEC_STEPS),
        ("mtp", None, NUM_SPEC_STEPS),
        ("dspark", None, NUM_SPEC_STEPS),
        ("dspark", {"sample_from_anchor": True}, NUM_SPEC_STEPS),
        ("dspark", {"sample_from_anchor": False}, NUM_SPEC_STEPS + 1),
        ("draft_model", None, NUM_SPEC_STEPS),
        # DFlash's in-fill decoding adds a query for the last sampled token.
        ("dflash", None, NUM_SPEC_STEPS + 1),
        ("ngram", None, 0),
        ("ngram_gpu", None, 0),
        ("suffix", None, 0),
        ("extract_hidden_states", None, 0),
    ],
)
def test_num_lookahead_tokens_per_method(
    method: str, draft_hf_config: dict | None, expected: int
):
    """`VllmConfig.num_lookahead_tokens` is the single source of the reservation.

    Both the scheduler and the warmup read it, so a wrong answer here silently
    changes how many blocks every request holds. Exercises the real property
    and the real `SpeculativeConfig` predicates; the config is built without
    `__post_init__` because the speculative methods otherwise require a draft
    model to be resolvable.
    """
    speculative_config = object.__new__(SpeculativeConfig)
    object.__setattr__(speculative_config, "method", method)
    object.__setattr__(speculative_config, "num_speculative_tokens", NUM_SPEC_STEPS)
    hf_config = SimpleNamespace(**(draft_hf_config or {}))
    object.__setattr__(
        speculative_config,
        "draft_model_config",
        SimpleNamespace(hf_config=hf_config, architectures=[]),
    )

    config = _config_without_post_init(speculative_config)

    assert config.num_lookahead_tokens == expected


def test_num_lookahead_tokens_without_speculation():
    config = _config_without_post_init(None)

    assert config.num_lookahead_tokens == 0


@pytest.mark.parametrize("max_num_reqs", [4, 2, 1])
@pytest.mark.parametrize("max_logprobs", [0, 5, -1])
def test_sizing_warmup_requests_prompt_logprobs_on_its_prefill(
    max_logprobs, max_num_reqs
):
    """Prompt logprobs materialize full-vocab logits a chunk at a time, which
    no other step reaches. The sizing step requests them on its real prefill,
    so the runtime path computes them from the model's own hidden states.

    They are computed for every token of a step that requests them, so with a
    budget beyond max_model_len they are measured on a single-prompt step and
    the full-budget step runs without them. A runner without room for a second
    prefill request runs one step."""
    runner = _make_runner([_attention_group()], num_lookahead_tokens=0)
    runner.model_config = SimpleNamespace(
        get_vocab_size=lambda: 64, max_logprobs=max_logprobs, use_mla=False
    )
    runner.max_num_reqs = max_num_reqs
    recorder = _StepRecorder()
    # (scheduled tokens, prompt_logprobs of each prefill request) per step
    steps: list[tuple[int, list[int | None]]] = []

    def execute_model(scheduler_output) -> None:
        prefill = [
            r.sampling_params.prompt_logprobs
            for r in scheduler_output.scheduled_new_reqs
            if "_prefill_" in r.req_id
        ]
        if prefill:
            steps.append((scheduler_output.total_num_scheduled_tokens, prefill))
        recorder.execute_model(scheduler_output)

    warmup.run_kv_cache_sizing_warmup(runner, execute_model, recorder.sample_tokens)

    expected = None if max_logprobs == 0 else max_logprobs
    if max_num_reqs == 1:
        assert steps == [(MAX_MODEL_LEN - 1, [expected])]
        return
    logprob_steps = [step for step in steps if step[1] != [None] * len(step[1])]
    if expected is None:
        assert logprob_steps == []
    else:
        assert logprob_steps == [(MAX_MODEL_LEN, [expected])]
    budget = runner.scheduler_config.max_num_batched_tokens
    assert max(n for n, _ in steps) == min(budget, max_num_reqs * (MAX_MODEL_LEN - 1))


def test_sizing_warmup_waits_for_each_step(monkeypatch):
    """UvaBufferPool reuses a staging slot two writes later. Warmup discards
    step outputs, so it must wait for each step before running the next."""
    runner = _make_runner([_attention_group()], num_lookahead_tokens=0)
    runner.model_config = SimpleNamespace(
        get_vocab_size=lambda: 64, max_logprobs=0, use_mla=False
    )
    events: list[str] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))

    warmup.run_kv_cache_sizing_warmup(
        runner, lambda output: events.append("step"), lambda grammar_output: None
    )

    assert events
    assert events == ["step", "sync"] * (len(events) // 2)


@pytest.mark.parametrize(
    "max_num_reqs, committable_blocks, expected_prefill_lens",
    [
        # Two full-length prompts plus the remainder fill the 2048-token budget.
        (4, None, [MAX_MODEL_LEN - 1, MAX_MODEL_LEN - 1, 1]),
        # Two request slots go to prefills rather than to a decode request.
        (2, None, [MAX_MODEL_LEN - 1, MAX_MODEL_LEN - 1]),
        # 1 decode + 64 prefill blocks fit; a second 64-block prompt does not.
        (4, 100, [MAX_MODEL_LEN - 1]),
        # A single-request runner measures the prefill alone.
        (1, None, [MAX_MODEL_LEN - 1]),
    ],
)
def test_sizing_warmup_fills_token_budget_beyond_max_model_len(
    max_num_reqs, committable_blocks, expected_prefill_lens
):
    """max_num_batched_tokens can exceed max_model_len, so the scheduler can
    pack several prompts into one step. The sizing step must reach that
    budget, splitting it across requests the runner and KV cache can hold."""
    runner = _make_runner([_attention_group()], num_lookahead_tokens=0)
    runner.model_config = SimpleNamespace(
        get_vocab_size=lambda: 64, max_logprobs=0, use_mla=False
    )
    runner.max_num_reqs = max_num_reqs
    if committable_blocks is not None:
        runner.extensible_kv_cache = _StubExtensibleCache(committable_blocks)
    outputs: list[SchedulerOutput] = []

    warmup.run_kv_cache_sizing_warmup(
        runner, outputs.append, lambda grammar_output: None
    )

    step = max(outputs, key=lambda o: o.total_num_scheduled_tokens)
    prefill_lens = [len(r.prompt_token_ids) for r in step.scheduled_new_reqs]
    assert prefill_lens == expected_prefill_lens
    num_decode_reqs = len(step.scheduled_cached_reqs.req_ids)
    assert num_decode_reqs == (1 if max_num_reqs > 2 else 0)
    assert step.total_num_scheduled_tokens == num_decode_reqs + sum(prefill_lens)
    assert step.total_num_scheduled_tokens == sum(step.num_scheduled_tokens.values())
