# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A dummy batch must stay marked wherever the metadata is rebuilt.

Under DCP a PAD slot means both "dummy token" and "another rank owns this
position". A cache that re-derives its own slot mapping cannot tell the two
apart, so it cannot recover the flag once a rebuild drops it.
"""

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata

BATCH_SPEC = BatchSpec(seq_lens=[64, 64], query_lens=[8, 8])


@pytest.mark.parametrize("is_dummy", [True, False])
def test_unpadded_keeps_is_dummy_batch(is_dummy):
    metadata = create_common_attn_metadata(
        BATCH_SPEC, block_size=16, device=torch.device("cpu")
    )
    metadata.is_dummy_batch = is_dummy
    unpadded = metadata.unpadded(num_actual_tokens=8, num_actual_reqs=1)
    assert unpadded.is_dummy_batch is is_dummy
