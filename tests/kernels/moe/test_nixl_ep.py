# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.nixl_ep


@pytest.mark.parametrize(
    "tokens,sp,dispatchers,expected", [(128, 1, 2, 128), (64, 1, 8, 64), (5, 2, 4, 3)]
)
def test_runtime_capacity_uses_dp_and_sp_execution(tokens, sp, dispatchers, expected):
    from unittest.mock import Mock

    import nixl_ep

    from vllm.forward_context import (
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        NixlEPPrepareAndFinalize,
    )

    prepare = NixlEPPrepareAndFinalize(Mock(spec=nixl_ep.Buffer), 128, dispatchers, 8)
    metadata = DPMetadata(torch.tensor([tokens, tokens], dtype=torch.int32))
    ctx = ForwardContext({}, {}, {}, dp_metadata=metadata)
    assert prepare._get_runtime_dispatch_capacity() == 128
    with override_forward_context(ctx), metadata.sp_local_sizes(sp, use_ep=True):
        assert prepare._get_runtime_dispatch_capacity() == expected
    assert prepare.max_num_tokens_per_rank() == 128
