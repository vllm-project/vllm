# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.warmup.kernel_warmup import (
    _deepseek_v4_mtp_uniform_decode_warmup_requests,
)


def test_deepseek_v4_mtp_uniform_decode_warmup_caps_large_max_num_seqs():
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
        num_spec_tokens=2,
        uniform_decode_query_len=3,
    )

    assert _deepseek_v4_mtp_uniform_decode_warmup_requests(
        runner,
        max_tokens=4096,
        max_reqs=1024,
    ) == (1, 2, 4, 8, 16, 24, 32)
