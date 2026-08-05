# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import deepseek_v4_mhc_warmup as mhc_warmup
from vllm.model_executor.warmup import deepseek_v4_sm12x_warmup as sm12x_warmup
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="compile keys read the local GPU's SM count",
)


def _mtp_runner(query_len: int = 3):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        num_spec_tokens=query_len - 1,
        uniform_decode_query_len=query_len,
    )


def test_deepseek_v4_mtp_uniform_decode_warmup_covers_c256():
    requests = sm12x_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=256,
    )

    assert requests == (1, 2, 4, 8, 16, 24, 32, 256)


def test_deepseek_v4_mtp_uniform_decode_warmup_still_respects_limits():
    assert sm12x_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=24,
    ) == (1, 2, 4, 8, 16, 24)
    assert sm12x_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=96,
        max_reqs=256,
    ) == (1, 2, 4, 8, 16, 24, 32)


def _dsv4_vllm_config(model_type: str = "deepseek_v4", **overrides):
    hf_config = SimpleNamespace(
        model_type=model_type,
        hidden_size=7168,
        hc_mult=4,
        hc_sinkhorn_iters=3,
        hc_eps=1e-4,
    )
    for name, value in overrides.items():
        if value is None:
            delattr(hf_config, name)
        else:
            setattr(hf_config, name, value)
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))


class _Runner:
    """Records dummy runs. Deliberately has no ``get_model``: the warmup's
    run/skip decision precedes TP collectives, so it must be a pure function
    of vllm_config — per-rank module state consulted here is how one rank
    skips the warmup another rank enters, deadlocking both."""

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.ran: list[int] = []

    def _dummy_run(self, num_tokens):
        self.ran.append(num_tokens)


def test_mhc_warmup_gates_on_config_alone():
    for config in (
        _dsv4_vllm_config(model_type="deepseek_v3"),
        _dsv4_vllm_config(hc_mult=None),
    ):
        runner = _Runner(config)
        mhc_warmup.deepseek_v4_mhc_warmup(runner, max_tokens=2048)
        assert runner.ran == []


@requires_gpu
def test_mhc_warmup_token_sizes_cover_the_compile_key_thresholds():
    """tile_n switches below 8, n_splits at 16; a power-of-two ladder misses
    steps."""
    sizes = mhc_warmup._token_sizes_to_warm(
        max_tokens=4096, hidden_size=7168, hc_mult=4, capture_sizes=[]
    )
    assert any(size < 8 for size in sizes)
    assert any(8 <= size <= 16 for size in sizes)
    assert any(size > 16 for size in sizes)
    assert sizes == sorted(set(sizes))
    # Steps the power-of-two ladder cannot reach are exactly what used to JIT
    # inside a forward pass.
    assert any(size & (size - 1) for size in sizes)


@requires_gpu
def test_mhc_warmup_filters_capture_coverage_by_key_not_size():
    """num_tokens is T.dynamic: capture size 32 compiles the one cubin that
    serves all of [17, 64], so that whole bucket needs no dummy run."""
    uncovered = mhc_warmup._token_sizes_to_warm(
        max_tokens=4096, hidden_size=7168, hc_mult=4, capture_sizes=[]
    )
    filtered = mhc_warmup._token_sizes_to_warm(
        max_tokens=4096, hidden_size=7168, hc_mult=4, capture_sizes=[1, 16, 32]
    )
    key = lambda t: mhc_warmup._compile_key(t, 7168, 4)  # noqa: E731
    covered_keys = {key(1), key(16), key(32)}
    assert set(filtered) == {t for t in uncovered if key(t) not in covered_keys}
    assert all(t > 64 for t in filtered)


@requires_gpu
def test_mhc_warmup_drives_dummy_runs_for_uncovered_keys():
    runner = _Runner(_dsv4_vllm_config())
    expected = mhc_warmup._token_sizes_to_warm(
        max_tokens=2048, hidden_size=7168, hc_mult=4, capture_sizes=[1, 16, 32]
    )
    assert expected, "capture list unexpectedly covers every compile key"

    mhc_warmup.deepseek_v4_mhc_warmup(
        runner, max_tokens=2048, cudagraph_capture_sizes=[1, 16, 32]
    )

    assert runner.ran == expected
