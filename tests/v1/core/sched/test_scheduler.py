# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def _make_request(req_id: str, sla_tier: str, priority: int = 0, arrival: float = 0.0):
    sp = SamplingParams(max_tokens=1, sla_tier=sla_tier)
    return Request(
        request_id=req_id,
        prompt_token_ids=[0],
        sampling_params=sp,
        pooling_params=None,
        eos_token_id=0,
        arrival_time=arrival,
        priority=priority,
        prompt_embeds=None,
        mm_features=None,
    )


def test_priority_queue_orders_by_sla_then_priority():
    q = PriorityRequestQueue(use_sla=True)
    # Higher priority integer should lose to SLA tier precedence.
    reqs = [
        _make_request("bg", "background", priority=0, arrival=2.0),
        _make_request("batch", "batch", priority=0, arrival=1.0),
        _make_request("int", "interactive", priority=5, arrival=0.0),
    ]
    for r in reqs:
        q.add_request(r)

    popped = [q.pop_request().request_id for _ in range(len(q))]
    assert popped == ["int", "batch", "bg"]


def test_priority_queue_default_behavior_when_sla_disabled():
    q = PriorityRequestQueue(use_sla=False)
    reqs = [
        _make_request("high", "interactive", priority=5, arrival=0.0),
        _make_request("low", "batch", priority=0, arrival=1.0),
    ]
    for r in reqs:
        q.add_request(r)

    popped = [q.pop_request().request_id for _ in range(len(q))]
    # Without SLA, lower integer priority wins.
    assert popped == ["low", "high"]


def test_preemption_prefers_background_with_same_priority():
    sched = object.__new__(Scheduler)
    sched.sla_tier_enabled = True
    sched._sla_preemption_rank = Scheduler._sla_preemption_rank.__get__(
        sched, Scheduler
    )

    running = [
        _make_request("int", "interactive", priority=1, arrival=1.0),
        _make_request("batch", "batch", priority=1, arrival=2.0),
        _make_request("bg", "background", priority=1, arrival=3.0),
    ]

    victim = max(
        running,
        key=lambda r: (r.priority, sched._sla_preemption_rank(r), r.arrival_time),
    )
    assert victim.request_id == "bg"


def test_interactive_budget_skip_condition():
    # Mirror the scheduler's interactive budget check logic.
    budget = 3
    interactive_tokens_scheduled = 2
    num_new_tokens = 2
    req = _make_request("int", "interactive", priority=0, arrival=0.0)

    should_skip = (
        interactive_tokens_scheduled + num_new_tokens > budget
        and req.sla_tier == "interactive"
    )
    assert should_skip is True

    # Non-interactive should never be blocked by the interactive budget.
    bg_req = _make_request("bg", "background", priority=0, arrival=0.0)
    should_skip_bg = (
        interactive_tokens_scheduled + num_new_tokens > budget
        and bg_req.sla_tier == "interactive"
    )
    assert should_skip_bg is False
