# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end restore through the MORI-backed UMBP runtime.

Batch-invariant kernels make a restore bitwise comparable with recomputing the
prompt, so restored KV must reproduce the cold run's tokens and logprobs.
"""

import time

import pytest

import vllm.envs as envs
from tests.v1.kv_connector.rocm_pd_accuracy_utils import _REVISIONS, QWEN_MODEL
from tests.v1.kv_connector.umbp_test_utils import mori_has_umbp
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_has_umbp()),
    reason="requires ROCm and MORI built with UMBP",
)

BLOCK_SIZE = 16
PARAMS = SamplingParams(temperature=0, max_tokens=16, logprobs=5, ignore_eos=True)


@pytest.fixture
def umbp_llm(request, vllm_runner, socket_dir, monkeypatch):
    pool = getattr(request, "param", {"capacity_bytes": 1 << 30})
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with vllm_runner(
        QWEN_MODEL,
        revision=_REVISIONS[QWEN_MODEL],
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                "lookup_dir": socket_dir,
                **pool,
            },
        ),
    ) as runner:
        yield runner.llm


def _prompt(seed: int, num_tokens: int) -> TokensPrompt:
    return TokensPrompt(
        prompt_token_ids=[
            100 + (seed * 1009 + i * 37) % 5000 for i in range(num_tokens)
        ]
    )


def _external_tokens(llm) -> int:
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if metric.name == "vllm:prompt_tokens_by_source"
        and metric.labels.get("source") == "external_kv_transfer"
    )


def _reset_gpu_prefix_cache(llm) -> None:
    """Drop the GPU prefix cache once UMBP has released the stored blocks."""
    deadline = time.monotonic() + 60
    while not llm.reset_prefix_cache():
        assert time.monotonic() < deadline, "UMBP stores never released GPU blocks"
        llm.generate(
            TokensPrompt(prompt_token_ids=[0]),
            SamplingParams(max_tokens=1),
            use_tqdm=False,
        )


def _assert_same_outputs(expected, actual) -> None:
    for want, got in zip(expected, actual, strict=True):
        want_out, got_out = want.outputs[0], got.outputs[0]
        assert got_out.token_ids == want_out.token_ids
        assert [
            {token: entry.logprob for token, entry in step.items()}
            for step in got_out.logprobs
        ] == [
            {token: entry.logprob for token, entry in step.items()}
            for step in want_out.logprobs
        ]


def test_restore_matches_recompute(umbp_llm):
    """A batch of prompts ending mid-block restores every full block exactly."""
    prompts = [
        _prompt(seed, 4 * BLOCK_SIZE + tail)
        for seed, tail in enumerate((1, 7, BLOCK_SIZE - 1))
    ]
    computed = umbp_llm.generate(prompts, PARAMS, use_tqdm=False)
    _reset_gpu_prefix_cache(umbp_llm)

    before = _external_tokens(umbp_llm)
    restored = umbp_llm.generate(prompts, PARAMS, use_tqdm=False)

    assert _external_tokens(umbp_llm) - before == 3 * 4 * BLOCK_SIZE
    _assert_same_outputs(computed, restored)


def test_restore_extends_a_local_prefix_hit(umbp_llm):
    """UMBP fills in only the blocks after a shorter GPU prefix-cache hit."""
    prompt = _prompt(0, 8 * BLOCK_SIZE + 7)
    computed = umbp_llm.generate(prompt, PARAMS, use_tqdm=False)
    _reset_gpu_prefix_cache(umbp_llm)
    short = PARAMS.clone()
    short.skip_reading_prefix_cache = True
    local_prefix = TokensPrompt(
        prompt_token_ids=prompt["prompt_token_ids"][: 2 * BLOCK_SIZE + 1]
    )
    umbp_llm.generate(local_prefix, short, use_tqdm=False)

    before = _external_tokens(umbp_llm)
    mixed = umbp_llm.generate(prompt, PARAMS, use_tqdm=False)

    assert mixed[0].num_cached_tokens == 8 * BLOCK_SIZE
    assert _external_tokens(umbp_llm) - before == 6 * BLOCK_SIZE
    _assert_same_outputs(computed, mixed)


# Eight full blocks of Qwen3-0.6B KV take 14 MiB, so this pool starts evicting
# after about three prompts.
@pytest.mark.parametrize(
    "umbp_llm",
    [
        {
            "capacity_bytes": 64 << 20,
            "dram_high_watermark": 0.75,
            "dram_low_watermark": 0.5,
        }
    ],
    ids=["64MiB"],
    indirect=True,
)
def test_restore_under_pool_eviction(umbp_llm):
    """The newest prompt survives eviction exactly and the oldest is gone."""
    prompts = [_prompt(seed, 8 * BLOCK_SIZE + 7) for seed in range(8)]
    computed = [umbp_llm.generate(p, PARAMS, use_tqdm=False)[0] for p in prompts]
    _reset_gpu_prefix_cache(umbp_llm)

    before = _external_tokens(umbp_llm)
    newest = umbp_llm.generate(prompts[-1], PARAMS, use_tqdm=False)
    assert _external_tokens(umbp_llm) - before == 8 * BLOCK_SIZE
    _assert_same_outputs([computed[-1]], newest)

    before = _external_tokens(umbp_llm)
    umbp_llm.generate(prompts[0], PARAMS, use_tqdm=False)
    assert _external_tokens(umbp_llm) == before
