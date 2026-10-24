# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLMScheduler on a full attention + Mamba ("align" cache mode) model."""

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.prefix_lm_scheduler import (
    PREFIX_LM_LEN,
    PrefixLMAsyncScheduler,
    PrefixLMScheduler,
    prefix_lm_salt,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

#: A Mamba backend that saves prefill checkpoints, as FlashInfer GDN does.
MAMBA = MambaSpec(
    block_size=16,
    shapes=((1, 1),),
    dtypes=(torch.float32,),
    mamba_cache_mode="align",
    num_prefill_checkpoint_blocks=1,
    prefill_checkpoint_alignment=16,
)


def make_scheduler(budget: int = 256, cls: type[Scheduler] = PrefixLMScheduler):
    return create_scheduler(
        max_num_batched_tokens=budget,
        max_model_len=2048,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        mamba_spec=MAMBA,
        scheduler_cls=cls,
    )


def make_request(req_id, prompt, prefix_len, salt="own", resumable=False):
    """A PrefixLM request with the strategy's salt, or causal if prefix_len is None."""
    if salt == "own":
        salt = None if prefix_len is None else prefix_lm_salt(prompt[:prefix_len], None)
    extra_args = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
    params = SamplingParams(max_tokens=1, extra_args=extra_args)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    init_none_hash(sha256)
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        cache_salt=salt,
        resumable=resumable,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def step(scheduler: Scheduler, *requests: Request) -> dict[str, tuple[int, int]]:
    """Add the requests and run one step. Returns the cached and scheduled
    tokens of each new request."""
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(ids, {i: n for n, i in enumerate(ids)}, [[0]] * len(ids)),
    )
    return {
        r.req_id: (r.num_computed_tokens, output.num_scheduled_tokens[r.req_id])
        for r in output.scheduled_new_reqs
    }


@pytest.mark.parametrize("cls", [PrefixLMScheduler, PrefixLMAsyncScheduler])
def test_hits_stop_at_the_prefix_and_prompts_prefill_whole(cls):
    """Hits stop at a prefix of the same tokens and length, and the rest of the
    prompt is prefilled in one step, also where Mamba alignment would split it."""
    scheduler = make_scheduler(cls=cls)
    prefix = list(range(100, 140))
    # The questions share their first 24 tokens: blocks [0, 64) match.
    q1, q2, q3 = (prefix + list(range(200, 224)) + [i] * 6 for i in (1, 2, 3))

    assert step(scheduler, make_request("q1", q1, 40)) == {"q1": (0, 70)}
    # q1 left Mamba state only at 64, past the prefix. q2's attention hit at 32
    # is a junction where alignment would stop, and the step would wait forever.
    assert step(scheduler, make_request("q2", q2, 40)) == {"q2": (0, 70)}
    # The prefix alone leaves its state at 32.
    assert step(scheduler, make_request("prefix", prefix, 40)) == {"prefix": (0, 40)}
    assert step(
        scheduler, make_request("q3", q3, 40), make_request("q1 again", q1, 40)
    ) == {"q3": (32, 38), "q1 again": (32, 38)}
    # Another prefix length, or none, shares no block with these prompts.
    assert step(
        scheduler, make_request("p48", q1, 48), make_request("causal", q1, None)
    ) == {"p48": (0, 70), "causal": (0, 70)}


def test_prompt_one_token_over_budget_waits():
    # a leaves 32 tokens of budget for b's 33, and 32 is block aligned.
    scheduler = make_scheduler(102)
    a, b = list(range(70)), list(range(100, 133))
    assert step(scheduler, make_request("a", a, 40), make_request("b", b, 20)) == {
        "a": (0, 70)
    }
    assert step(scheduler) == {"b": (0, 33)}


def test_waiting_prompt_holds_the_queue():
    """A PrefixLM prompt that does not fit is not skipped by the causal c behind it."""
    scheduler = make_scheduler(100)
    a, b, c = list(range(64)), list(range(100, 164)), list(range(200, 216))
    assert step(
        scheduler,
        make_request("a", a, None),
        make_request("b", b, 32),
        make_request("c", c, None),
    ) == {"a": (0, 64)}
    assert step(scheduler) == {"b": (0, 64), "c": (0, 16)}


@pytest.mark.parametrize(
    "prefix_len,salt,hit,expected",
    [
        (40, None, 32, "{own}"),
        (40, "{own}", 32, "{own}"),
        (40, "{own}:user", 0, "{own}:user"),
        (40, "user", 0, "{own}:user"),
        (40, "{p1}", 0, "{own}:{p1}"),
        (None, "{own}", 0, "causal:{own}"),
    ],
)
def test_requests_get_the_salt_of_their_prefix(prefix_len, salt, hit, expected):
    """A request lacking its prefix's salt is moved into it, a forged one misses,
    and a causal request leaves the PrefixLM namespace."""
    scheduler = make_scheduler()
    p1 = list(range(100, 140))
    p2 = p1[:35] + [7] * 5
    step(scheduler, make_request("p1", p1, 40))
    # Same length and first 32 tokens, but another prefix: no hit.
    assert step(scheduler, make_request("p2", p2, 40)) == {"p2": (0, 40)}
    salts = {"own": prefix_lm_salt(p2, None), "p1": prefix_lm_salt(p1, None)}
    salt = salt and salt.format(**salts)
    request = make_request("q", p2 + list(range(200, 230)), prefix_len, salt=salt)
    assert step(scheduler, request) == {"q": (hit, 70 - hit)}
    assert request.cache_salt == expected.format(**salts)


def test_streamed_requests_are_read_causally():
    request = make_request("streamed", list(range(150)), 40, resumable=True)
    assert step(make_scheduler(100), request) == {"streamed": (0, 96)}


def test_needs_mamba_align_mode():
    with pytest.raises(ValueError, match="align"):
        create_scheduler(skip_tokenizer_init=True, scheduler_cls=PrefixLMScheduler)


@pytest.mark.parametrize("kind", ["", "Async"])
def test_async_scheduling_takes_the_async_twin(monkeypatch, kind):
    # Keep the platform from deciding async scheduling.
    monkeypatch.setattr(
        type(current_platform), "check_and_update_config", classmethod(lambda *_: None)
    )
    cls = "vllm.v1.core.sched.prefix_lm_scheduler.PrefixLM{}Scheduler"
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", skip_tokenizer_init=True),
        scheduler_config=SchedulerConfig.default_factory(
            scheduler_cls=cls.format(""), async_scheduling=kind == "Async"
        ),
    )
    assert config.scheduler_config.scheduler_cls == cls.format(kind)
