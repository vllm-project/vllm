# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV events across cache layouts and speculative decoding, on a real engine.

Every configuration starts one ``vllm serve`` and runs the same checks: the
reference consumer predicts each request's prefix-cache hit from the event
stream alone and the ``cached_tokens`` the engine reports must agree. Hybrid
layouts publish one event group per KV cache type (full attention, sliding
window, Mamba) and speculative decoding changes which tokens get cached, so
these are the cases where a consumer's model of the engine most easily drifts.

Prompts are sized in the block size the engine really uses, learned from the
events: hybrid models grow the attention block until its page matches the
Mamba state page (576 tokens for Qwen3.5-2B, 528 for granite-4.0-h-micro),
and decode context parallelism publishes blocks that span all its ranks
(128 tokens for ``--block-size 64`` with two), so ``--block-size`` is not what
a consumer sees.

The small configurations fit next to other work on one GPU; the ones marked
``optional`` need a whole GPU each, four for DCP (``pytest --optional``).
"""

import threading
from collections.abc import Iterator

import pytest

from vllm.utils.network_utils import get_open_ports_list

from .engine_harness import (
    REQUEST_TIMEOUT_S,
    ServedEngine,
    ServerConfig,
    repetitive_prompt_ids,
    serve,
)
from .reference_consumer import EventStream

SPEC_ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens_total"


def _ngram(num_speculative_tokens: int = 3) -> dict:
    return {
        "method": "ngram",
        "num_speculative_tokens": num_speculative_tokens,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 4,
    }


# Multimodal checkpoints: text only, no vision profiling.
TEXT_ONLY = ("--limit-mm-per-prompt", '{"image": 0, "video": 0}')
# Hybrid models: blocks of several hundred tokens, so longer contexts and
# three-block prompts keep the checks meaningful.
HYBRID = dict(
    max_model_len=4096,
    num_gpu_blocks=48,
    prompt_blocks=3,
    concurrent_blocks=6,
    gpu_gib=10,
)
# Ten cache groups: a request holds ten blocks per 528 tokens.
MAMBA2 = {**HYBRID, "num_gpu_blocks": 96, "concurrent_blocks": 4, "gpu_gib": 11}
# 64-token blocks: eight five-block prompts fit the cache only until they
# decode, so the concurrent check preempts.
MLA = dict(block_size=64, num_gpu_blocks=48, concurrent_blocks=5, start_timeout_s=1800)

MATRIX = [
    ServerConfig(
        name="dense-ngram",
        model="Qwen/Qwen3-0.6B",
        speculative=_ngram(),
        description="dense GQA, n-gram drafts",
    ),
    ServerConfig(
        name="dense-eagle3",
        model="Qwen/Qwen3-1.7B",
        gpu_gib=10,
        speculative={
            "method": "eagle3",
            "model": "AngelSlim/Qwen3-1.7B_eagle3",
            "num_speculative_tokens": 3,
        },
        description="dense GQA with an EAGLE3 drafter (same cache group)",
    ),
    ServerConfig(
        name="dense-eagle3-drop",
        model="Qwen/Qwen3-1.7B",
        gpu_gib=10,
        speculative={
            "method": "eagle3",
            "model": "AngelSlim/Qwen3-1.7B_eagle3",
            "num_speculative_tokens": 3,
            "disable_lookahead_block_hashes": True,
        },
        description="EAGLE3 without lookahead hashes: the last matched block "
        "is dropped instead",
    ),
    ServerConfig(
        name="gdn-hybrid",
        model="Qwen/Qwen3.5-2B",
        extra_args=TEXT_ONLY,
        description="full attention + Gated DeltaNet (Mamba-type) groups",
        **HYBRID,
    ),
    ServerConfig(
        name="gdn-hybrid-mtp",
        model="Qwen/Qwen3.5-2B",
        extra_args=TEXT_ONLY,
        speculative={"method": "mtp", "num_speculative_tokens": 2},
        description="Gated DeltaNet hybrid with its MTP head",
        # Draft tokens reserve speculative Mamba blocks per request; how many
        # requests the budget admits depends on the drafter, so preemption is
        # not required here (the predictions are checked either way).
        expect_preemption=False,
        **{**HYBRID, "num_gpu_blocks": 64},
    ),
    ServerConfig(
        name="mamba2-hybrid",
        model="ibm-granite/granite-4.0-h-micro",
        description="full attention + Mamba2 groups",
        **MAMBA2,
    ),
    ServerConfig(
        name="mamba2-hybrid-ngram",
        model="ibm-granite/granite-4.0-h-micro",
        speculative=_ngram(),
        description="Mamba2 hybrid with speculative blocks",
        **MAMBA2,
    ),
    pytest.param(
        ServerConfig(
            name="swa-hybrid",
            model="openai/gpt-oss-20b",
            num_gpu_blocks=192,
            gpu_gib=30,
            description="full attention + sliding window (128) groups",
        ),
        marks=pytest.mark.optional,
    ),
    pytest.param(
        ServerConfig(
            name="swa-hybrid-ngram",
            model="openai/gpt-oss-20b",
            num_gpu_blocks=192,
            gpu_gib=30,
            speculative=_ngram(),
            description="sliding-window hybrid with n-gram drafts",
        ),
        marks=pytest.mark.optional,
    ),
    pytest.param(
        ServerConfig(
            name="mla",
            model="deepseek-ai/DeepSeek-V2-Lite",
            gpu_gib=48,
            description="MLA, 64-token blocks",
            **MLA,
        ),
        marks=pytest.mark.optional,
    ),
    pytest.param(
        ServerConfig(
            name="mla-ngram",
            model="deepseek-ai/DeepSeek-V2-Lite",
            gpu_gib=48,
            speculative=_ngram(),
            description="MLA with n-gram drafts",
            **MLA,
        ),
        marks=pytest.mark.optional,
    ),
    pytest.param(
        ServerConfig(
            name="dense-mtp",
            model="XiaomiMiMo/MiMo-7B-RL",
            gpu_gib=30,
            start_timeout_s=900,
            speculative={"method": "mtp", "num_speculative_tokens": 1},
            extra_args=("--trust-remote-code",),
            description="dense model with its MTP head in the target's group",
        ),
        marks=pytest.mark.optional,
    ),
    pytest.param(
        ServerConfig(
            name="mla-dcp2",
            model="deepseek-ai/DeepSeek-V2-Lite",
            gpu_gib=11,
            gpus=4,
            # The published block spans both DCP ranks: 128 tokens.
            prompt_blocks=5,
            **MLA,
            extra_args=(
                "--tensor-parallel-size",
                "4",
                "--decode-context-parallel-size",
                "2",
                "--max-num-batched-tokens",
                "1024",
            ),
            description="MLA with decode context parallel: KV sharded over two ranks",
        ),
        marks=pytest.mark.optional,
    ),
]


@pytest.fixture(scope="module", params=MATRIX, ids=lambda config: config.name)
def engine(request: pytest.FixtureRequest) -> Iterator[ServedEngine]:
    pub_port, replay_port = get_open_ports_list(2)
    with serve(request.param, pub_port, replay_port) as engine:
        print(f"[{engine.config.name}] block_size={engine.block_size}; {engine.groups}")
        yield engine


@pytest.fixture
def stream(engine: ServedEngine) -> Iterator[EventStream]:
    with engine.subscribe() as stream:
        engine.fresh_state(stream)
        yield stream
        print(
            f"[{engine.config.name}] {dict(stream.index.stats)}; "
            f"notes={stream.index.notes}"
        )
        assert not stream.index.anomalies, stream.index.anomalies


def test_store_then_hit(engine: ServedEngine, stream: EventStream) -> None:
    block_size = engine.block_size
    blocks = engine.config.prompt_blocks
    prompt = engine.prompt(1, blocks)
    assert engine.check(stream, prompt, "cold prompt") == 0
    warm = engine.check(stream, prompt, "same prompt again")
    assert warm > 0, "nothing of the warm prompt was predicted or found cached"
    shared = max(1, blocks // 2)
    divergent = prompt[: shared * block_size] + engine.prompt(2, blocks - shared)
    engine.check(stream, divergent, "shared prefix")
    aligned = engine.prompt(3, blocks, extra_tokens=0)
    engine.check(stream, aligned, "cold aligned prompt")
    engine.check(stream, aligned, "aligned prompt again")


def test_generated_tokens_are_published(
    engine: ServedEngine, stream: EventStream
) -> None:
    """Blocks filled while decoding are cached too; with speculative decoding
    only accepted tokens may end up in them."""
    config = engine.config
    block_size = engine.block_size
    prompt = repetitive_prompt_ids(4, config.prompt_blocks, block_size)
    engine.check(stream, prompt, "cold prompt")
    accepted_before = engine.metric(SPEC_ACCEPTED_METRIC)
    max_tokens = 2 * block_size + 8
    completion = engine.complete(
        prompt, max_tokens=max_tokens, ignore_eos=True, return_token_ids=True
    )
    output = completion["choices"][0]["token_ids"]
    assert len(output) == max_tokens
    if config.speculative is not None:
        assert engine.metric(SPEC_ACCEPTED_METRIC) > accepted_before, (
            "no draft token was accepted: speculative decoding was not exercised"
        )
    stream.drain()
    warm = engine.check(stream, prompt, "warm prompt")
    extended = engine.check(stream, prompt + output, "prompt plus its continuation")
    # Whether blocks filled while decoding are cached depends on the layout
    # (Mamba keeps states at block boundaries only while prefilling); the
    # prediction must agree either way.
    cached_decode_blocks = (extended - warm) // block_size
    print(f"[{config.name}] decode-filled blocks cached: {cached_decode_blocks}")
    assert extended >= warm


def test_eviction_matches_engine(engine: ServedEngine, stream: EventStream) -> None:
    blocks = engine.config.prompt_blocks
    prompts = [engine.prompt(100 + i, blocks) for i in range(12)]
    for i, prompt in enumerate(prompts):
        assert engine.check(stream, prompt, f"cold prompt {i}") == 0
    assert stream.index.stats["removed"] > 0, "expected evictions"
    # Most recent first, so the probes do not evict what the next probe
    # expects to find.
    hits = [
        engine.check(stream, prompt, f"prompt {i} after eviction")
        for i, prompt in reversed(list(enumerate(prompts)))
    ]
    assert hits[0] > 0, hits
    assert hits[-1] == 0, hits


def test_concurrent_requests_and_preemption(
    engine: ServedEngine, stream: EventStream
) -> None:
    """Block-aligned prompts whose decode crosses into a new block, more of
    them than the cache holds: some get preempted and recomputed."""
    block_size = engine.block_size
    prompts = [
        engine.prompt(200 + i, engine.config.concurrent_blocks, extra_tokens=0)
        for i in range(8)
    ]
    max_tokens = max(128, block_size + 8)
    preemptions_before = engine.num_preemptions()
    errors: list[BaseException] = []

    def run(prompt: list[int]) -> None:
        try:
            engine.complete(prompt, max_tokens=max_tokens, ignore_eos=True)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(prompt,)) for prompt in prompts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=REQUEST_TIMEOUT_S)
    assert not errors, errors
    preemptions = engine.num_preemptions() - preemptions_before
    print(f"[{engine.config.name}] preemptions: {preemptions:.0f}")
    if engine.config.expect_preemption:
        assert preemptions > 0, "no preemption happened"
    stream.drain()
    for i, prompt in enumerate(prompts):
        engine.check(stream, prompt, f"prompt {i} after concurrent run")
