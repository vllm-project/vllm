# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end restore through the MORI-backed UMBP runtime.

Batch-invariant kernels make a restore bitwise comparable with recomputing the
prompt, so restored KV must reproduce the cold run's tokens and logprobs. Mamba
hybrids have no batch-invariant kernels, so they compare with a GPU prefix hit.
"""

import contextlib
import time

import pytest

import vllm.envs as envs
from tests.utils import multi_gpu_marks
from tests.v1.kv_connector.rocm_pd_accuracy_utils import _REVISIONS, QWEN_MODEL
from tests.v1.kv_connector.umbp_test_utils import mori_has_umbp, standalone_server
from vllm import SamplingParams, TokensPrompt
from vllm.config import CUDAGraphMode, KVTransferConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_has_umbp()),
    reason="requires ROCm and MORI built with UMBP",
)

BLOCK_SIZE = 16
PARAMS = SamplingParams(temperature=0, max_tokens=16, logprobs=5, ignore_eos=True)


@pytest.fixture
def umbp_llm(vllm_runner, socket_dir, tmp_path, monkeypatch, request):
    """Qwen3 on batch-invariant kernels, storing its prefixes through UMBP.

    `request.param` holds connector options, plus `runner` for `vllm_runner`
    overrides and `standalone` to store through a `umbp_standalone_server`.
    """
    options = {"capacity_bytes": 1 << 30, **getattr(request, "param", {})}
    runner_kwargs = {"enforce_eager": True, **options.pop("runner", {})}
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    with contextlib.ExitStack() as stack:
        if options.pop("standalone", False):
            server = stack.enter_context(
                standalone_server(tmp_path, socket_dir, options.pop("capacity_bytes"))
            )
            pool_config = {
                "mode": "standalone",
                "endpoint": server.address,
                "startup_timeout_ms": 60000,
            }
        else:
            pool_config = {"mode": "embedded", "lookup_dir": socket_dir}
        runner = stack.enter_context(
            vllm_runner(
                QWEN_MODEL,
                revision=_REVISIONS[QWEN_MODEL],
                block_size=BLOCK_SIZE,
                enable_prefix_caching=True,
                disable_log_stats=False,
                kv_transfer_config=KVTransferConfig(
                    kv_connector="UMBPStoreConnector",
                    kv_role="kv_both",
                    kv_connector_extra_config={**pool_config, **options},
                ),
                **runner_kwargs,
            )
        )
        cudagraph_mode = runner_kwargs.get("compilation_config", {}).get(
            "cudagraph_mode"
        )
        if cudagraph_mode is not None:
            compilation_config = runner.llm.llm_engine.vllm_config.compilation_config
            assert compilation_config.cudagraph_mode == CUDAGraphMode[cudagraph_mode]
        yield runner.llm


# The default lazy scan covers 2 * 1024 / 16 = 128 free-queue blocks, so a
# finished prompt is stored only once the GPU is about to reuse its blocks. The
# pool holds every block that reuse can store, so only GPU memory evicts.
_LAZY_OPTIONS = {"lazy_offload": True, "capacity_bytes": 2 << 30}
_LAZY_RUNNER = {"max_num_batched_tokens": 1024, "num_gpu_blocks_override": 512}
LAZY_UMBP_LLMS = [
    pytest.param({**_LAZY_OPTIONS, "runner": _LAZY_RUNNER}, id="eager-embedded"),
    pytest.param(
        {
            **_LAZY_OPTIONS,
            "standalone": True,
            "runner": {
                **_LAZY_RUNNER,
                "enforce_eager": False,
                "compilation_config": {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 2, 4],
                },
            },
        },
        id="full_graph-standalone",
    ),
]


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


# Filler prompts cover more than one full cycle of the GPU pool, so every block
# of an earlier prompt is reused. Their seeds stay clear of the test prompts.
_FILLER_POOL_CYCLES = 1.5
_FILLER_BLOCKS = 32
_FILLER_SEED = 1000


def _reuse_gpu_blocks(llm) -> None:
    """Cycle every GPU block, so lazy offload stores the prefixes it reuses."""
    num_gpu_blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    num_filler_blocks = int(num_gpu_blocks * _FILLER_POOL_CYCLES)
    prompts = [
        _prompt(_FILLER_SEED + seed, _FILLER_BLOCKS * BLOCK_SIZE)
        for seed in range(cdiv(num_filler_blocks, _FILLER_BLOCKS))
    ]
    llm.generate(prompts, SamplingParams(max_tokens=1), use_tqdm=False)


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


@pytest.mark.parametrize(
    "umbp_llm",
    [
        pytest.param({"runner": {"tensor_parallel_size": 1}}, id="tp1"),
        pytest.param(
            {"runner": {"tensor_parallel_size": 2}},
            id="tp2",
            marks=multi_gpu_marks(num_gpus=2),
        ),
    ],
    indirect=True,
)
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


HYBRID_MODEL = "ibm-granite/granite-4.0-h-350m"
HYBRID_REVISION = "3b17b717b8f2f5d305b0a92c1491e239aeda19c8"


@pytest.fixture
def umbp_hybrid_llm(vllm_runner, socket_dir):
    with vllm_runner(
        HYBRID_MODEL,
        revision=HYBRID_REVISION,
        enforce_eager=True,
        max_model_len=2048,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                "capacity_bytes": 1 << 30,
                "lookup_dir": socket_dir,
            },
        ),
    ) as runner:
        yield runner.llm


def test_hybrid_restore_matches_local_prefix_hit(umbp_hybrid_llm):
    """Mamba boundary states restored through UMBP resume like a GPU hit.

    A prompt ending exactly on a block boundary still recomputes its last block.
    """
    llm = umbp_hybrid_llm
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    for seed, tail in enumerate((7, block_size)):
        prompt = _prompt(seed, 2 * block_size + tail)
        llm.generate(prompt, PARAMS, use_tqdm=False)
        local_hit = llm.generate(prompt, PARAMS, use_tqdm=False)
        assert local_hit[0].num_cached_tokens == 2 * block_size
        _reset_gpu_prefix_cache(llm)

        before = _external_tokens(llm)
        restored = llm.generate(prompt, PARAMS, use_tqdm=False)

        assert restored[0].num_cached_tokens == 2 * block_size
        assert _external_tokens(llm) - before == 2 * block_size
        _assert_same_outputs(local_hit, restored)


@pytest.mark.parametrize("umbp_llm", LAZY_UMBP_LLMS, indirect=True)
def test_lazy_offload_defers_stores_until_gpu_reuse(umbp_llm):
    """Finished prefixes stay GPU-only while their blocks are not reused."""
    prompts = [_prompt(seed, 4 * BLOCK_SIZE + 1) for seed in range(3)]
    umbp_llm.generate(prompts, PARAMS, use_tqdm=False)
    _reset_gpu_prefix_cache(umbp_llm)

    before = _external_tokens(umbp_llm)
    again = umbp_llm.generate(prompts, PARAMS, use_tqdm=False)

    assert _external_tokens(umbp_llm) - before == 0
    assert all(output.num_cached_tokens == 0 for output in again)


@pytest.mark.parametrize("umbp_llm", LAZY_UMBP_LLMS, indirect=True)
def test_lazy_offload_restores_reused_blocks(umbp_llm):
    """Full blocks stored when the GPU reuses them restore exactly."""
    prompts = [
        _prompt(seed, 4 * BLOCK_SIZE + tail)
        for seed, tail in enumerate((1, 7, BLOCK_SIZE - 1))
    ]
    computed = umbp_llm.generate(prompts, PARAMS, use_tqdm=False)
    _reuse_gpu_blocks(umbp_llm)
    _reset_gpu_prefix_cache(umbp_llm)

    before = _external_tokens(umbp_llm)
    restored = umbp_llm.generate(prompts, PARAMS, use_tqdm=False)

    assert _external_tokens(umbp_llm) - before == 3 * 4 * BLOCK_SIZE
    _assert_same_outputs(computed, restored)
