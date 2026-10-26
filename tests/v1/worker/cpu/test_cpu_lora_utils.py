# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.v1.worker.cpu.lora_utils import LoraState

pytestmark = pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU only")


def test_cpu_lora_state_maps_each_logits_row_to_its_request() -> None:
    state = LoraState(max_num_reqs=4)
    first = LoRARequest("first", 3, "/tmp/first")
    second = LoRARequest("second", 7, "/tmp/second")
    state.add_request("first-request", 1, first)
    state.add_request("base-request", 2, None)
    state.add_request("second-request", 3, second)

    prompt_mapping, token_mapping, active = state.make_lora_inputs(
        req_ids=["second-request", "base-request"],
        idx_mapping=np.array([3, 2], dtype=np.intp),
        num_scheduled_tokens=np.array([2, 3], dtype=np.int32),
        num_logits=np.array([1, 1], dtype=np.int32),
    )

    assert prompt_mapping == (7, 0)
    assert token_mapping == (7, 7, 0, 0, 0)
    assert active == {second}
