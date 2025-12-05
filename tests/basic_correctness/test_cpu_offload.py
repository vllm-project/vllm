# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ..utils import compare_two_settings


def test_cpu_offload():
    compare_two_settings(
        "hmellor/tiny-random-LlamaForCausalLM", [], ["--cpu-offload-gb", "1"]
    )
