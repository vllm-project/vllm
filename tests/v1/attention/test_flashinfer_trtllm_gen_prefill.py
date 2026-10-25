# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel choice for chunked prefill on the trtllm-gen generation kernel."""

import pytest

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    trtllm_gen_prefill_sm_count,
)

HQ, HKV, D, SMS = 16, 2, 256, 212
WS = 394 * 1024 * 1024


def choose(q, kv, ws=WS):
    return choose_batch([q], [kv], ws)


def choose_batch(qs, kvs, ws=WS):
    return trtllm_gen_prefill_sm_count(qs, kvs, HQ, HKV, D, SMS, ws)


@pytest.mark.parametrize("q,kv", [(4096, 4096), (2048, 10240), (4096, 30000)])
def test_short_prefix_keeps_context_kernel(q, kv):
    # Causal tiles of a chunk without a long prefix are unequal; the wave model
    # does not hold there.
    assert choose(q, kv) is None


@pytest.mark.parametrize(
    "q,kv",
    [(128, 131200), (512, 98816), (2096, 123664), (2144, 40736), (3336, 131976)],
)
def test_split_count_is_a_multiple_of_ctas_and_fits(q, kv):
    sm = choose(q, kv)
    assert sm is not None
    num_ctas = cdiv(q, 16) * HKV  # 128 rows = 16 tokens x 8 q heads per CTA
    assert sm % num_ctas == 0
    splits = sm // num_ctas
    assert 1 <= splits <= 8
    assert splits == 1 or kv // splits >= 8192
    # bf16 partial O + float2 stats for sm_count x 128 rows within half the workspace
    assert sm * 128 * (2 * D + 8) <= WS // 2


def test_workspace_caps_splits():
    q, kv = 2144, 190000
    full = choose(q, kv)
    num_ctas = cdiv(q, 16) * HKV
    tight = choose(q, kv, ws=2 * num_ctas * 128 * (2 * D + 8))
    assert full is not None and full > num_ctas
    assert tight == num_ctas or tight is None


def test_full_waves_keep_context_kernel():
    # 128-token context tiles fill whole waves (17 * 16 = 272 CTAs = 2 waves are
    # not full, but 13 * 16 = 208 CTAs are one full wave of 212 SMs).
    assert choose(13 * 128, 13 * 128 * 40) is None


@pytest.mark.parametrize(
    "qs,kvs",
    [
        ([128, 37], [131200, 90000]),
        ([512, 64, 300], [98816, 140000, 60000]),
        ([640, 128, 128, 16], [120000, 138000, 50000, 70000]),
    ],
)
def test_multi_request_split_is_a_multiple_of_the_padded_grid(qs, kvs):
    # FlashInfer divides the SM count by the grid padded to the longest chunk.
    sm = choose_batch(qs, kvs)
    assert sm is not None
    grid_ctas = cdiv(max(qs), 16) * HKV * len(qs)
    assert sm % grid_ctas == 0
    splits = sm // grid_ctas
    assert 1 <= splits <= 8
    assert splits == 1 or max(kvs) // splits >= 8192
    assert sm * 128 * (2 * D + 8) <= WS // 2


def test_long_sequence_alone_bounds_the_split():
    # One long sequence sets the critical path; a short one in the same launch
    # neither forbids the split nor caps it at its own length.
    sm = choose_batch([128, 128], [131200, 4000])
    grid_ctas = cdiv(128, 16) * HKV * 2
    assert sm is not None and sm % grid_ctas == 0
    assert 1 < sm // grid_ctas <= 131200 // 8192


def test_multi_request_workspace_cap_uses_padded_grid():
    qs, kvs = [2048, 16], [190000, 190000]
    grid_ctas = cdiv(max(qs), 16) * HKV * len(qs)
    # Room for one split of the padded grid only.
    tight = choose_batch(qs, kvs, ws=2 * grid_ctas * 128 * (2 * D + 8))
    assert tight == grid_ctas or tight is None
