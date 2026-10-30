# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpeculatorConfig.admit_limit: the speculator's config lowers admission."""

from collections.abc import Callable

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu.spec_decode.ngram.config import NgramSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.round import BudgetView

from .utils import create_requests, create_scheduler, new_mamba_spec, runner_output

pytestmark = pytest.mark.cpu_test


def _scheduler(
    monkeypatch: pytest.MonkeyPatch,
    admit_limit: Callable[[BudgetView], int],
    **kwargs,
) -> Scheduler:
    """A spec-decode scheduler whose speculator config admits up to
    `admit_limit(view)` running requests."""
    monkeypatch.setattr(
        NgramSpeculatorConfig,
        "admit_limit",
        lambda self, view: admit_limit(view),
        raising=False,
    )
    scheduler = create_scheduler(num_speculative_tokens=1, **kwargs)
    assert isinstance(scheduler.speculator_config, NgramSpeculatorConfig)
    return scheduler


def _recording(views: list[BudgetView], value: int) -> Callable[[BudgetView], int]:
    """A budget function that appends each view it is asked with to `views`
    and returns `value`."""

    def budget(view: BudgetView) -> int:
        views.append(view)
        return value

    return budget


def _output(scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
    return runner_output(list(scheduler_output.num_scheduled_tokens))


def _step(scheduler: Scheduler) -> list[str]:
    """Run one synchronous step; return the ids of the requests it admitted."""
    output = scheduler.schedule()
    admitted = [req.req_id for req in output.scheduled_new_reqs]
    cached = output.scheduled_cached_reqs
    admitted += [r for r in cached.req_ids if r in cached.resumed_req_ids]
    scheduler.update_from_output(output, _output(output))
    return admitted


def _add(scheduler: Scheduler, requests: list[Request]) -> None:
    for request in requests:
        scheduler.add_request(request)


def test_admit_limit_lowers_admission(monkeypatch):
    views: list[BudgetView] = []
    scheduler = _scheduler(monkeypatch, _recording(views, 3))
    _add(scheduler, create_requests(num_requests=10, max_tokens=100))

    assert len(_step(scheduler)) == 3
    assert (len(scheduler.running), len(scheduler.waiting)) == (3, 7)
    # Before admission: nothing running yet, all ten waiting.
    assert views[0] == BudgetView(
        num_scheduled_reqs=0,
        num_running=0,
        num_waiting=10,
        num_free_blocks=views[0].num_free_blocks,
        max_num_seqs=16,
        block_size=scheduler.block_size,
    )
    assert _step(scheduler) == []
    # The three running requests were scheduled before the limit was asked.
    assert (views[1].num_scheduled_reqs, views[1].num_running) == (3, 3)
    assert len(views) == 2


def test_admit_limit_is_not_asked_without_waiting_requests(monkeypatch):
    views: list[BudgetView] = []
    scheduler = _scheduler(monkeypatch, _recording(views, 16))
    _add(scheduler, create_requests(num_requests=2, max_tokens=100))
    assert len(_step(scheduler)) == 2
    assert _step(scheduler) == []
    assert len(views) == 1


def test_admit_limit_and_budget_see_the_block_size(monkeypatch):
    """On a recurrent model both of a step's views carry the block size and the
    state blocks held for draft rows."""
    admit_views: list[BudgetView] = []
    budget_views: list[BudgetView] = []
    admit_limit = _recording(admit_views, 2)
    candidate_budget = _recording(budget_views, 4)
    monkeypatch.setattr(
        NgramSpeculatorConfig,
        "admit_limit",
        lambda self, view: admit_limit(view),
        raising=False,
    )
    monkeypatch.setattr(
        NgramSpeculatorConfig,
        "candidate_budget",
        lambda self, view: candidate_budget(view),
    )
    scheduler = create_scheduler(
        num_speculative_tokens=3,
        max_num_batched_tokens=64,
        max_model_len=256,
        block_size=256,
        kv_cache_spec=new_mamba_spec(
            block_size=256, shapes=((1, 1),), num_speculative_blocks=3
        ),
        use_v2_model_runner=True,
    )
    assert scheduler.dynamic_spec_blocks
    _add(scheduler, create_requests(num_requests=3, num_tokens=8, max_tokens=64))
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 2
    scheduler.update_from_output(output, _output(output))
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, [[7] * 3 for _ in req_ids]))

    # A decode step with three drafts each; the third request still waits.
    scheduler.schedule()
    for view in (admit_views[-1], budget_views[-1]):
        assert view.block_size == scheduler.block_size == 256


def test_default_admits_up_to_the_static_limit():
    scheduler = create_scheduler(num_speculative_tokens=1, max_num_active_seqs=4)
    _add(scheduler, create_requests(num_requests=10))
    assert len(_step(scheduler)) == 4


@pytest.mark.parametrize("limit", [5, 16, 1000])
def test_admit_limit_never_raises_the_static_limit(monkeypatch, limit):
    scheduler = _scheduler(monkeypatch, lambda view: limit, max_num_active_seqs=4)
    _add(scheduler, create_requests(num_requests=10, max_tokens=100))
    assert len(_step(scheduler)) == 4
    assert len(_step(scheduler)) == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_admit_limit_below_one_is_rejected(monkeypatch, limit):
    scheduler = _scheduler(monkeypatch, lambda view: limit)
    _add(scheduler, create_requests(num_requests=2))
    with pytest.raises(ValueError, match="admit_limit returned"):
        scheduler.schedule()


def test_admit_limit_never_preempts(monkeypatch):
    """Lowering the limit under a load admitted earlier stops admission; the
    running requests keep running until they finish."""
    limit = [3]
    scheduler = _scheduler(monkeypatch, lambda view: limit[0])
    requests = create_requests(num_requests=5, max_tokens=4)
    _add(scheduler, requests)
    assert _step(scheduler) == ["0", "1", "2"]

    limit[0] = 1
    for _ in range(2):
        output = scheduler.schedule()
        assert sorted(output.num_scheduled_tokens) == ["0", "1", "2"]
        assert not output.scheduled_new_reqs
        assert output.preempted_req_ids is None or not output.preempted_req_ids
        scheduler.update_from_output(output, _output(output))
    assert [r.status for r in requests[:3]] == [RequestStatus.RUNNING] * 3

    # Admission resumes, one at a time, once the running requests finish.
    admitted: list[str] = []
    while len(admitted) < 2:
        admitted += _step(scheduler)
        assert not admitted or len(scheduler.running) <= 1
    assert admitted == ["3", "4"]


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_admit_limit_never_reorders(monkeypatch, policy):
    """A changing limit admits in the same order as no limit at all."""

    def requests() -> list[Request]:
        reqs = create_requests(num_requests=8, max_tokens=100)
        if policy == "priority":
            for req, priority in zip(reqs, [3, 1, 2, 0, 3, 1, 0, 2]):
                req.priority = priority
        return reqs

    unlimited = create_scheduler(num_speculative_tokens=1, scheduling_policy=policy)
    _add(unlimited, requests())
    expected = _step(unlimited)
    assert len(expected) == 8
    if policy == "priority":
        assert expected == ["3", "6", "1", "5", "2", "7", "0", "4"]

    step_limits = iter([1, 3, 3, 4, 8])
    scheduler = _scheduler(
        monkeypatch, lambda view: next(step_limits), scheduling_policy=policy
    )
    _add(scheduler, requests())
    admitted: list[str] = []
    for num_admitted in [1, 2, 0, 1, 4]:
        new = _step(scheduler)
        assert len(new) == num_admitted
        admitted += new
    assert admitted == expected


def test_preempted_requests_resume_under_the_limit(monkeypatch):
    """A preempted request goes back to the head of the queue: a lowered
    limit delays it, it is never skipped, and it resumes normally."""
    limit = [2]
    # 10 usable blocks of 16 tokens: two 80-token prompts fill them.
    scheduler = _scheduler(
        monkeypatch,
        lambda view: limit[0],
        max_num_batched_tokens=200,
        num_blocks=11,
        block_size=16,
    )
    req0, req1 = create_requests(num_requests=2, num_tokens=80, max_tokens=3)
    (req2,) = create_requests(num_requests=1, num_tokens=8, req_ids=["2"])
    _add(scheduler, [req0, req1, req2])

    assert _step(scheduler) == ["0", "1"]
    # The first decode needs a sixth block for request 0: request 1 is
    # preempted to make room.
    _step(scheduler)
    assert req1.status == RequestStatus.PREEMPTED
    assert scheduler.waiting.peek_request() is req1

    limit[0] = 1
    admitted: list[str] = []
    while req0.status == RequestStatus.RUNNING:
        admitted += _step(scheduler)
    assert admitted == []
    # Request 0 is done: request 1 resumes ahead of request 2, which waits.
    assert _step(scheduler) == ["1"]
    assert req1.status == RequestStatus.RUNNING
    assert req2.status == RequestStatus.WAITING
    while req1.status == RequestStatus.RUNNING:
        assert _step(scheduler) == []
    assert req1.num_output_tokens == 3
    assert _step(scheduler) == ["2"]


@pytest.mark.parametrize("by_policy", [False, True])
def test_async_scheduling_counts_requests_still_executing(monkeypatch, by_policy):
    """Under async scheduling a request finishing in the step still executing
    holds its place; the next request is admitted by the first schedule()
    after the output that finishes it. Same timing as max_num_active_seqs."""
    # Async scheduling with ngram needs the V2 model runner's GPU ngram.
    async_kwargs = dict(
        async_scheduling=True, speculative_method="ngram_gpu", use_v2_model_runner=True
    )
    views: list[BudgetView] = []
    if by_policy:
        scheduler = _scheduler(monkeypatch, _recording(views, 1), **async_kwargs)
    else:
        scheduler = create_scheduler(
            num_speculative_tokens=1, max_num_active_seqs=1, **async_kwargs
        )
    req0, req1 = create_requests(num_requests=2, max_tokens=2)
    _add(scheduler, [req0, req1])

    out0 = scheduler.schedule()
    assert [r.req_id for r in out0.scheduled_new_reqs] == ["0"]
    out1 = scheduler.schedule()  # out0 executing
    assert not out1.scheduled_new_reqs
    scheduler.update_from_output(out0, _output(out0))
    out2 = scheduler.schedule()  # out1 executing; it finishes request 0
    assert not out2.scheduled_new_reqs
    scheduler.update_from_output(out1, _output(out1))
    assert req0.is_finished()
    out3 = scheduler.schedule()
    assert [r.req_id for r in out3.scheduled_new_reqs] == ["1"]
    if by_policy:
        assert [v.num_running for v in views] == [0, 1, 1, 0]
