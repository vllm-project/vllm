# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A sliding-window DSpark drafter only inserts context KV for the last
``window`` scheduled tokens of each request."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA buffers")


def _rows(window, query_lens):
    speculator = SimpleNamespace(
        context_window=window, _context_row_pool=UvaBufferPool(512, torch.int64)
    )
    batch = SimpleNamespace(
        num_reqs=len(query_lens),
        query_start_loc_np=np.concatenate([[0], np.cumsum(query_lens)]).astype(
            np.int32
        ),
    )
    return DSparkSpeculator._context_rows(speculator, batch)


def test_context_rows_keep_each_request_tail():
    rows = _rows(128, [1, 300, 100])
    assert rows.tolist() == [0, *range(301 - 128, 301), *range(301, 401)]


@pytest.mark.parametrize(
    "window, query_lens",
    [(128, [6, 6, 100]), (None, [1, 300])],
    ids=["all fit the window", "no window"],
)
def test_context_rows_none_when_nothing_to_skip(window, query_lens):
    assert _rows(window, query_lens) is None
