# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import prewarm_hf_cache


@pytest.fixture(scope="session", autouse=True)
def _prewarm_hf_cache():
    # tokenization_qwen.py downloads SimSun.ttf from
    # qianwen-res.oss-cn-beijing.aliyuncs.com; both Qwen/Qwen-VL and
    # Qwen/Qwen-VL-Chat look it up from the Chat repo.
    prewarm_hf_cache([("Qwen/Qwen-VL-Chat", "SimSun.ttf")])
