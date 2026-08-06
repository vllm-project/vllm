# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams

pytestmark = pytest.mark.skip_global_cleanup


def test_duplicate_stop_token_ids_are_deduplicated_in_order():
    params = SamplingParams(stop_token_ids=[42, 7, 42, 9, 7])

    assert params.stop_token_ids == [42, 7, 9]
    assert params.all_stop_token_ids == {7, 9, 42}
