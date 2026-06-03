# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.warmup import kernel_warmup


def _mtp_runner(query_len: int = 3):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        num_spec_tokens=query_len - 1,
        uniform_decode_query_len=query_len,
    )


def test_deepseek_v4_mtp_uniform_decode_warmup_covers_c256():
    requests = kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=256,
    )

    assert requests == (1, 2, 4, 8, 16, 24, 32, 256)


def test_deepseek_v4_mtp_uniform_decode_warmup_still_respects_limits():
    assert kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=4096,
        max_reqs=24,
    ) == (1, 2, 4, 8, 16, 24)
    assert kernel_warmup._deepseek_v4_mtp_uniform_decode_warmup_requests(
        _mtp_runner(),
        max_tokens=96,
        max_reqs=256,
    ) == (1, 2, 4, 8, 16, 24, 32)
