# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `num_speculative_tokens_under_prefill_load`: shorter drafts on steps
where the prefills that can run fill the token budget even next to shortened
drafts."""

import pytest

from tests.v1.core.test_scheduler import _model_output
from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.config import SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _scheduler(
    max_num_batched_tokens: int = 256,
    num_speculative_tokens: int = 16,
    num_speculative_tokens_under_prefill_load: int | None = 4,
    max_num_seqs: int = 8,
    **kwargs,
):
    return create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_tokens_under_prefill_load=num_speculative_tokens_under_prefill_load,
        speculative_method="ngram",
        **kwargs,
    )


def _add(sched, req_id: str, num_tokens: int):
    (request,) = create_requests(
        num_requests=1, num_tokens=num_tokens, req_ids=[req_id], max_tokens=256
    )
    sched.add_request(request)
    return request


def _step(sched, num_drafts: int | None = None):
    """Run a step as the worker would: sample a token for each request past its
    prompt and give it `num_drafts` drafts (default: as many as requested)."""
    out = sched.schedule()
    decodes = [
        rid
        for rid in out.num_scheduled_tokens
        if not sched.requests[rid].is_prefill_chunk
    ]
    _model_output(
        sched,
        out,
        [[100] if rid in decodes else [] for rid in out.num_scheduled_tokens],
    )
    if num_drafts is None:
        num_drafts = out.num_spec_tokens_to_schedule
    sched.update_draft_token_ids(
        DraftTokenIds(decodes, [[1] * num_drafts] * len(decodes))
    )
    return out


def test_no_reduction_when_unset():
    sched = _scheduler(num_speculative_tokens_under_prefill_load=None)
    _add(sched, "dec", 4)
    _add(sched, "pre", 1000)
    _step(sched)

    out = sched.schedule()
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 16
    assert out.num_spec_tokens_to_schedule == 16


def test_reduction_when_running_prefill_exceeds_budget():
    sched = _scheduler()
    _add(sched, "dec", 4)
    _add(sched, "pre", 1000)
    _step(sched, num_drafts=16)

    out = sched.schedule()
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 4
    assert out.num_spec_tokens_to_schedule == 4


def test_reduction_when_waiting_prefill_exceeds_budget():
    sched = _scheduler()
    _add(sched, "d0", 4)
    _add(sched, "d1", 4)
    _step(sched)
    _add(sched, "pre", 1000)

    out = sched.schedule()
    assert "pre" in out.num_scheduled_tokens
    assert len(out.scheduled_spec_decode_tokens["d0"]) == 4
    assert len(out.scheduled_spec_decode_tokens["d1"]) == 4


def test_no_reduction_when_prefill_fits_next_to_full_drafts():
    sched = _scheduler(max_num_batched_tokens=512)
    _add(sched, "dec", 4)
    _step(sched)
    _add(sched, "small", 100)

    out = sched.schedule()
    assert out.num_scheduled_tokens["small"] == 100
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 16
    assert out.num_spec_tokens_to_schedule == 16


@pytest.mark.parametrize(("prompt_len", "num_drafts"), [(250, 16), (251, 4)])
def test_reduction_threshold(prompt_len, num_drafts):
    """A decode with shortened drafts takes 1 + 4 tokens, leaving 251 of the
    budget. A shorter prefill keeps full drafts and is chunked next to them
    instead; one that fills it gets shortened drafts. Either way the step fills
    the budget."""
    sched = _scheduler()
    _add(sched, "dec", 4)
    _step(sched)
    _add(sched, "pre", prompt_len)

    out = sched.schedule()
    assert len(out.scheduled_spec_decode_tokens["dec"]) == num_drafts
    assert out.total_num_scheduled_tokens == 256


def test_full_drafts_resume_once_prefill_fits_next_to_short_drafts():
    """The drafts proposed for the next step are full length once the prefill
    left fits next to shortened drafts."""
    sched = _scheduler()
    _add(sched, "dec", 4)
    pre = _add(sched, "pre", 1000)
    proposed = []
    for _ in range(10):
        proposed.append(_step(sched).num_spec_tokens_to_schedule)
        if not pre.is_prefill_chunk:
            break
    # Prefill left after each step: 748, 497, 246 (fits in 251), 7, 0.
    assert proposed == [4, 4, 16, 16, 16]

    out = sched.schedule()
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 16


def test_no_reduction_on_steps_that_hold_prefill():
    """With prefill_schedule_interval, only steps carrying prefill shorten."""
    sched = _scheduler(
        max_num_batched_tokens=512,
        num_speculative_tokens=8,
        num_speculative_tokens_under_prefill_load=2,
        prefill_schedule_interval=4,
    )
    _add(sched, "dec", 4)
    _step(sched)
    _add(sched, "pre", 2000)

    carried_prefill = []
    for _ in range(8):
        out = _step(sched)
        has_prefill = "pre" in out.num_scheduled_tokens
        carried_prefill.append(has_prefill)
        expected = 2 if has_prefill else 8
        assert len(out.scheduled_spec_decode_tokens["dec"]) == expected
    assert any(carried_prefill) and not all(carried_prefill)


def test_no_reduction_when_waiting_prefill_cannot_be_admitted():
    sched = _scheduler(max_num_batched_tokens=512, max_num_seqs=1)
    _add(sched, "dec", 4)
    _step(sched)
    _add(sched, "pre", 1000)

    out = sched.schedule()
    assert "pre" not in out.num_scheduled_tokens
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 16
    assert out.num_spec_tokens_to_schedule == 16


def test_no_reduction_for_queue_blocked_prefill():
    sched = _scheduler(max_num_batched_tokens=512)
    _add(sched, "dec", 4)
    _step(sched)
    (pre,) = create_requests(num_requests=1, num_tokens=1000, req_ids=["pre"])
    pre.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    sched.add_request(pre)

    out = sched.schedule()
    assert "pre" not in out.num_scheduled_tokens
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 16


def test_no_reduction_for_remote_prefill_arrival(monkeypatch):
    """A P/D decode instance loads the prompt's KV instead of computing it."""
    sched = _scheduler(
        max_num_batched_tokens=512,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    # The remote prefill instance holds all but the last block of the prompt.
    monkeypatch.setattr(
        sched.connector,
        "get_num_new_matched_tokens",
        lambda request, _: (992, True)
        if request.request_id == "remote"
        else (0, False),
    )
    _add(sched, "dec", 4)
    _step(sched)
    (remote,) = create_requests(num_requests=1, num_tokens=1000, req_ids=["remote"])
    remote.kv_transfer_params = {"do_remote_prefill": True}
    sched.add_request(remote)

    out = sched.schedule()
    assert remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 16
    assert out.num_spec_tokens_to_schedule == 16


def test_padding_matches_reduced_k():
    sched = _scheduler(
        max_num_batched_tokens=512,
        num_speculative_tokens=8,
        num_speculative_tokens_under_prefill_load=2,
    )
    _add(sched, "d0", 4)
    _step(sched)
    # "d1" is admitted as a padded decode while "pre" keeps the reduction on.
    _add(sched, "d1", 1)
    _add(sched, "pre", 4000)

    out = sched.schedule()
    assert out.num_spec_tokens_to_schedule == 2
    decode_query_lens = {
        out.num_scheduled_tokens[rid] for rid in out.scheduled_spec_decode_tokens
    }
    assert decode_query_lens == {3}


def test_spec_token_ids_not_truncated_in_place():
    """AsyncScheduler shares one placeholder list across requests."""
    sched = _scheduler(num_speculative_tokens_under_prefill_load=2)
    dec = _add(sched, "dec", 4)
    _add(sched, "pre", 1000)
    _step(sched)
    shared = [1] * 16
    dec.spec_token_ids = shared

    out = sched.schedule()
    assert len(out.scheduled_spec_decode_tokens["dec"]) == 2
    assert shared == [1] * 16


def test_zero_spec_tokens_under_prefill_load():
    sched = _scheduler(num_speculative_tokens_under_prefill_load=0)
    _add(sched, "dec", 4)
    _add(sched, "pre", 1000)
    _step(sched, num_drafts=16)

    out = sched.schedule()
    assert out.num_spec_tokens_to_schedule == 0
    assert not out.scheduled_spec_decode_tokens.get("dec")


def test_config_validation():
    for value in (0, 4, 16):
        SpeculativeConfig(
            model="ngram",
            num_speculative_tokens=16,
            num_speculative_tokens_under_prefill_load=value,
        )
    for value in (-1, 17):
        with pytest.raises(ValueError):
            SpeculativeConfig(
                model="ngram",
                num_speculative_tokens=16,
                num_speculative_tokens_under_prefill_load=value,
            )


def _vllm_config(
    method: str = "ngram",
    data_parallel_size: int = 1,
    num_speculative_tokens_under_prefill_load=2,
    enable_chunked_prefill: bool = True,
) -> VllmConfig:
    model = ModelConfig(model="facebook/opt-125m", max_model_len=2048)
    spec = SpeculativeConfig(
        method=method,
        num_speculative_tokens=8,
        num_speculative_tokens_under_prefill_load=num_speculative_tokens_under_prefill_load,
        prompt_lookup_max=8,
        prompt_lookup_min=1,
        target_model_config=model,
        target_parallel_config=ParallelConfig(),
    )
    return VllmConfig(
        model_config=model,
        speculative_config=spec,
        parallel_config=ParallelConfig(data_parallel_size=data_parallel_size),
        scheduler_config=SchedulerConfig.default_factory(
            max_model_len=2048, enable_chunked_prefill=enable_chunked_prefill
        ),
    )


def test_disabled_under_data_parallelism():
    assert (
        _vllm_config().speculative_config.num_speculative_tokens_under_prefill_load == 2
    )
    config = _vllm_config(data_parallel_size=2)
    assert config.speculative_config.num_speculative_tokens_under_prefill_load is None


def test_requires_chunked_prefill():
    """Without chunking, a prefill that does not fit next to full drafts would
    not run at all instead of being chunked."""
    with pytest.raises(ValueError, match="requires chunked prefill"):
        _vllm_config(enable_chunked_prefill=False)


@pytest.mark.parametrize(
    ("method", "use_v2_model_runner", "expected"),
    [("ngram_gpu", "0", None), ("ngram_gpu", "1", 2), ("ngram", "0", 2)],
)
def test_disabled_on_v1_for_fixed_length_proposers(
    monkeypatch, method, use_v2_model_runner, expected
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", use_v2_model_runner)
    config = _vllm_config(method=method)
    assert (
        config.speculative_config.num_speculative_tokens_under_prefill_load == expected
    )


@pytest.mark.parametrize(("num_reduced", "query_lens"), [(None, {9}), (2, {9, 3})])
def test_cudagraph_query_lens_cover_reduced_drafts(num_reduced, query_lens):
    config = _vllm_config(num_speculative_tokens_under_prefill_load=num_reduced)
    max_num_seqs = config.scheduler_config.max_num_seqs
    assert config.get_decode_cudagraph_query_lens(9) == dict.fromkeys(
        query_lens, max_num_seqs
    )
