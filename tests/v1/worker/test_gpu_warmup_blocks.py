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
from tests.v1.core.utils import create_requests
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.warmup import (
    _reserved_block_count,
    run_mixed_prefill_decode_warmup,
    warmup_kernels,
)

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
NUM_SPEC_STEPS = 3

# `warmup_kernels` ends on `torch.accelerator.synchronize()`.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="warmup synchronizes on the accelerator"
)


def _attention_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )


def _mamba_group(mamba_cache_mode: str) -> KVCacheGroupSpec:
    # Name carries the mode so several groups can coexist in one config.
    return KVCacheGroupSpec(
        [f"mamba_{mamba_cache_mode}"],
        MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode=mamba_cache_mode,
            num_speculative_blocks=NUM_SPEC_STEPS,
        ),
    )


def _make_runner(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_lookahead_tokens: int,
    num_spec_steps: int = NUM_SPEC_STEPS,
) -> SimpleNamespace:
    """Stub model runner exposing only what the warmup entry points read."""
    return SimpleNamespace(
        num_speculative_steps=num_spec_steps,
        decode_query_len=num_spec_steps + 1,
        is_pooling_model=False,
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        model_state=SimpleNamespace(max_encoder_len=0),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=kv_cache_groups, num_blocks=1024
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=num_lookahead_tokens),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )


class _StepRecorder:
    """Rebuilds each request's per-group block holdings from the warmup steps."""

    def __init__(self) -> None:
        # (blocks held per group, num_computed_tokens, num_scheduled_tokens)
        self.steps: list[tuple[list[int], int, int]] = []
        self._held: dict[str, list[int]] = {}

    def execute_model(self, scheduler_output) -> None:
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
@pytest.mark.parametrize("num_spec_steps", [2, 3, 5, 7])
def test_warmup_kernels_reserves_lookahead_blocks(num_spec_steps, extra_lookahead):
    num_lookahead_tokens = num_spec_steps + extra_lookahead
    recorder = _StepRecorder()

    warmup_kernels(
        _make_runner([_attention_group()], num_lookahead_tokens, num_spec_steps),
        recorder.execute_model,
        recorder.sample_tokens,
    )

    _assert_covers_lookahead(recorder.steps, num_lookahead_tokens)


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


@pytest.mark.parametrize("mamba_cache_mode", ["none", "all", "align"])
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
    """Full attention plus one Mamba group per cache mode, so a single manager
    exercises every branch `_reserved_block_count` has.

    "none" and "all" reach the same branch of both `_reserved_block_count` and
    `MambaManager`, which tests only for "align". They are still both listed:
    the mode is a spec-level input, and having the real manager confirm the
    prediction for each is what keeps a future divergence between them from
    landing unnoticed.
    """
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            _attention_group(),
            _mamba_group("none"),
            _mamba_group("all"),
            _mamba_group("align"),
        ],
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
    specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups]
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
        for spec, group_held in zip(specs, held):
            expected = _reserved_block_count(
                num_tokens,
                spec,
                num_lookahead_tokens=num_lookahead_tokens,
                max_model_len=MAX_MODEL_LEN,
                max_encoder_len=0,
            )
            assert len(group_held) == expected, (
                f"{type(spec).__name__} "
                f"mode={getattr(spec, 'mamba_cache_mode', None)}: real manager "
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


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("eagle", NUM_SPEC_STEPS),
        ("eagle3", NUM_SPEC_STEPS),
        ("mtp", NUM_SPEC_STEPS),
        ("dspark", NUM_SPEC_STEPS),
        ("draft_model", NUM_SPEC_STEPS),
        # DFlash's in-fill decoding adds a query for the last sampled token.
        ("dflash", NUM_SPEC_STEPS + 1),
        ("ngram", 0),
        ("ngram_gpu", 0),
        ("medusa", 0),
        ("mlp_speculator", 0),
        ("suffix", 0),
        ("extract_hidden_states", 0),
    ],
)
def test_num_lookahead_tokens_per_method(method: str, expected: int):
    """`VllmConfig.num_lookahead_tokens` is the single source of the reservation.

    Both the scheduler and the warmup read it, so a wrong answer here silently
    changes how many blocks every request holds. Exercises the real property
    and the real `SpeculativeConfig` predicates; the config is built without
    `__post_init__` because the speculative methods otherwise require a draft
    model to be resolvable.
    """

    class _Config:
        num_speculative_tokens = VllmConfig.num_speculative_tokens
        num_lookahead_tokens = VllmConfig.num_lookahead_tokens

    speculative_config = object.__new__(SpeculativeConfig)
    object.__setattr__(speculative_config, "method", method)
    object.__setattr__(speculative_config, "num_speculative_tokens", NUM_SPEC_STEPS)

    config = _Config()
    config.speculative_config = speculative_config
    config.diffusion_config = None

    assert config.num_lookahead_tokens == expected


def test_num_lookahead_tokens_without_speculation():
    class _Config:
        num_speculative_tokens = VllmConfig.num_speculative_tokens
        num_lookahead_tokens = VllmConfig.num_lookahead_tokens

    config = _Config()
    config.speculative_config = None
    config.diffusion_config = None

    assert config.num_lookahead_tokens == 0
