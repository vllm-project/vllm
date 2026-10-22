# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.compilation import CompilationConfig


@pytest.mark.parametrize(
    "query_len,tp_size,enable_sp,expected",
    [
        (5, 2, True, [10, 20]),
        (5, 4, True, [20]),
        (4, 2, True, [4, 8, 12, 20]),
        (5, 2, False, [5, 10, 20]),
    ],
)
def test_spec_decode_graph_sizes_respect_token_and_tp_alignment(
    query_len, tp_size, enable_sp, expected
):
    config = CompilationConfig(
        cudagraph_capture_sizes=[1, 5, 10, 20],
        max_cudagraph_capture_size=20,
        pass_config={"enable_sp": enable_sp},
    )

    config.adjust_cudagraph_sizes_for_spec_decode(query_len, tp_size)

    assert config.cudagraph_capture_sizes == expected
    assert config.max_cudagraph_capture_size == expected[-1]
    assert config.pass_config.enable_sp == enable_sp
