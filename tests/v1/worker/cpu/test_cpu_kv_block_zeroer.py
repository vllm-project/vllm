# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout
from vllm.v1.worker.cpu.utils import CPUKVBlockZeroer
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="Requires a non-x86 CPU without Triton",
)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_cpu_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    device = torch.device("cpu")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.ones(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    )
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    zeroer = CPUKVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context={
            f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([2])

    for view in views:
        assert (view[2] == 0).all(), layout
        assert (view[[0, 1, 3]].view(torch.int8) == 1).all(), layout
    assert int((raw == 0).sum()) == num_layers * spec.page_size_bytes, layout


def test_cpu_zeroes_every_kernel_block_in_a_logical_block():
    device = torch.device("cpu")
    num_blocks = 4
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
    )
    raw = torch.ones(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    cache = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers=1,
        layout=KVCacheLayout.LBNHC,
        kernel_block_size=2,
    )[0]
    zeroer = CPUKVBlockZeroer(
        device,
        attn_groups_iter=iter(
            [AttentionGroup(None, ["layer"], spec, kv_cache_group_id=0)]
        ),
        kernel_block_sizes=[2],
        static_forward_context={"layer": SimpleNamespace(kv_cache=cache)},
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([1, 3])

    assert (cache[[2, 3, 6, 7]] == 0).all()
    assert (cache[[0, 1, 4, 5]].view(torch.int8) == 1).all()
