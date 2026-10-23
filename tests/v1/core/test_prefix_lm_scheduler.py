# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLMScheduler on a full attention + Mamba ("align" cache mode) model."""

import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
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
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
#: The salt the strategy sends.
OWN = object()


def make_scheduler(
    max_num_batched_tokens: int = 256,
    scheduler_cls: type[Scheduler] = PrefixLMScheduler,
    mamba: bool = True,
) -> Scheduler:
    """A Mamba backend that saves prefill checkpoints, as FlashInfer GDN does."""
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = 100
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=1,
                    prefill_checkpoint_alignment=BLOCK_SIZE,
                ),
            ),
        ][: 2 if mamba else 1],
    )
    register_all_kvcache_specs(vllm_config)
    return scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        log_stats=True,
    )


def make_request(
    req_id: str,
    prompt: list[int],
    prefix_len: int | None,
    salt: object = OWN,
    resumable: bool = False,
) -> Request:
    """A request with a prefix length, or a causal one if it is None. By default
    a PrefixLM request has the salt the strategy sends and a causal one none."""
    if salt is OWN:
        salt = None if prefix_len is None else prefix_lm_salt(prompt[:prefix_len], None)
    assert salt is None or isinstance(salt, str)
    params = SamplingParams(
        max_tokens=1,
        extra_args=None if prefix_len is None else {PREFIX_LM_LEN: prefix_len},
    )
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    init_none_hash(sha256)
    return Request(
        request_id=req_id,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        cache_salt=salt,
        resumable=resumable,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def step(scheduler: Scheduler, *requests: Request) -> dict[str, tuple[int, int]]:
    """Add the requests and run one step. Returns the cached and scheduled
    tokens of each new request."""
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[[0]] * len(req_ids),
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return {
        r.req_id: (r.num_computed_tokens, output.num_scheduled_tokens[r.req_id])
        for r in output.scheduled_new_reqs
    }


def test_overridden_methods_exist():
    # The scheduler hooks into these private methods; a rename would silently
    # bypass the PrefixLM rules.
    for name in ("_get_local_prefix_cache_hit", "_mamba_block_aligned_split"):
        assert name in vars(Scheduler)
        assert name in vars(PrefixLMScheduler)
    assert issubclass(PrefixLMAsyncScheduler, PrefixLMScheduler)


@pytest.mark.parametrize("scheduler_cls", [PrefixLMScheduler, PrefixLMAsyncScheduler])
def test_hits_stop_at_the_prefix_and_prompts_prefill_whole(scheduler_cls):
    """Hits stop at a prefix of the same tokens and length, and the rest of the
    prompt is prefilled in one step, also where Mamba alignment would split it."""
    scheduler = make_scheduler(scheduler_cls=scheduler_cls)
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


def test_scheduler_alone_hits_past_the_prefix():
    """The cap is PrefixLMScheduler's: Scheduler reuses q1's blocks up to 64."""
    prefix = list(range(100, 140))
    q1 = prefix + list(range(200, 230))
    for scheduler_cls, hit in ((Scheduler, 64), (PrefixLMScheduler, 32)):
        scheduler = make_scheduler(scheduler_cls=scheduler_cls)
        step(scheduler, make_request("q1", q1, 40))
        step(scheduler, make_request("prefix", prefix, 40))
        assert step(scheduler, make_request("again", q1, 40)) == {
            "again": (hit, 70 - hit)
        }


def test_prompt_waits_rather_than_split():
    scheduler = make_scheduler(max_num_batched_tokens=100)
    a, b = list(range(70)), list(range(100, 170))
    assert step(scheduler, make_request("a", a, 40), make_request("b", b, 40)) == {
        "a": (0, 70)
    }
    assert step(scheduler) == {"b": (0, 70)}


def test_prompt_one_token_over_budget_waits():
    # a leaves 32 tokens of budget for b's 33, and 32 is block aligned.
    scheduler = make_scheduler(max_num_batched_tokens=102)
    a, b = list(range(70)), list(range(100, 133))
    assert step(scheduler, make_request("a", a, 40), make_request("b", b, 20)) == {
        "a": (0, 70)
    }
    assert step(scheduler) == {"b": (0, 33)}


def test_waiting_prompt_holds_the_queue():
    """A PrefixLM prompt that does not fit waits at the head of the queue: later
    requests do not overtake it."""
    scheduler = make_scheduler(max_num_batched_tokens=100)
    a, b, c = list(range(64)), list(range(100, 164)), list(range(200, 216))
    assert step(
        scheduler,
        make_request("a", a, None),
        make_request("b", b, 32),
        make_request("c", c, None),
    ) == {"a": (0, 64)}
    assert step(scheduler) == {"b": (0, 64), "c": (0, 16)}


def test_causal_prompts_still_chunk():
    scheduler = make_scheduler(max_num_batched_tokens=100)
    assert step(scheduler, make_request("causal", list(range(150)), None)) == {
        "causal": (0, 96)
    }


def test_no_hit_across_prefixes_of_same_length():
    """The first block's key covers the prefix tokens, not only its length."""
    scheduler = make_scheduler()
    p1 = list(range(100, 140))
    p2 = p1[:35] + [7, 7, 7, 7, 7]
    assert step(scheduler, make_request("p1", p1, 40)) == {"p1": (0, 40)}
    q = p2 + list(range(200, 230))
    assert step(scheduler, make_request("q", q, 40)) == {"q": (0, 70)}


@pytest.mark.parametrize("salt,hit", [(None, 32), ("user", 0), ("forged", 0)])
def test_prefix_lm_requests_get_the_salt_of_their_prefix(salt, hit):
    """A PrefixLM request without its own prefix's salt is moved into it, after
    any salt it sent. A forged salt, of another prefix of the same length whose
    first blocks match, gets no hit from that prefix."""
    scheduler = make_scheduler()
    p1 = list(range(100, 140))
    p2 = p1[:35] + [7, 7, 7, 7, 7]
    own = prefix_lm_salt(p2, None)
    step(scheduler, make_request("p1", p1, 40), make_request("p2", p2, 40))

    if salt == "forged":
        salt = prefix_lm_salt(p1, None)
    request = make_request("q", p2 + list(range(200, 230)), 40, salt=salt)
    assert step(scheduler, request) == {"q": (hit, 70 - hit)}
    assert request.cache_salt == (f"{own}:{salt}" if salt else own)


def test_strategy_salt_is_kept():
    scheduler = make_scheduler()
    prompt = list(range(100, 170))
    request = make_request("q", prompt, 40, salt=prefix_lm_salt(prompt[:40], "user"))
    block_hashes = list(request.block_hashes)
    scheduler.add_request(request)
    assert request.cache_salt == prefix_lm_salt(prompt[:40], "user")
    assert request.block_hashes == block_hashes


def test_causal_requests_leave_the_prefix_lm_namespace():
    scheduler = make_scheduler()
    prefix = list(range(100, 140))
    assert step(scheduler, make_request("prefix", prefix, 40)) == {"prefix": (0, 40)}
    salt = prefix_lm_salt(prefix, None)
    causal = make_request("causal", prefix + list(range(200, 230)), None, salt=salt)
    assert step(scheduler, causal) == {"causal": (0, 70)}
    assert causal.cache_salt == f"causal:{salt}"


def test_streamed_requests_are_read_causally():
    scheduler = make_scheduler(max_num_batched_tokens=100)
    request = make_request("streamed", list(range(150)), 40, resumable=True)
    assert step(scheduler, request) == {"streamed": (0, 96)}
    assert request.sampling_params is not None
    assert request.sampling_params.extra_args == {}


def test_needs_mamba_align_mode():
    with pytest.raises(ValueError, match="align"):
        make_scheduler(mamba=False)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_async_scheduling_takes_the_async_twin(monkeypatch, async_scheduling):
    # Keep the platform from deciding async scheduling.
    monkeypatch.setattr(
        type(current_platform), "check_and_update_config", classmethod(lambda *_: None)
    )
    prefix_lm = "vllm.v1.core.sched.prefix_lm_scheduler.PrefixLM"
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", skip_tokenizer_init=True),
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            scheduler_cls=f"{prefix_lm}Scheduler",
            async_scheduling=async_scheduling,
        ),
    )
    kind = "Async" if async_scheduling else ""
    assert config.scheduler_config.scheduler_cls == f"{prefix_lm}{kind}Scheduler"
