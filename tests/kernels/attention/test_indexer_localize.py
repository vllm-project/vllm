# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local sparse-attention indexer decode (`VLLM_INDEXER_LOCALIZE`).

The placement test needs a GPU with two memory locality domains and
`torch.cuda.execute_on_streams`.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)

from vllm.v1.attention.ops.indexer_localize import placement  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda", torch.accelerator.current_device_index())


@pytest.fixture(scope="module")
def domains() -> placement.DomainResources | None:
    torch.empty(1, device=_device())  # the green contexts need a CUDA context
    res = placement.get_domain_resources(_device())
    if res is None:
        pytest.skip("needs a GPU with two memory locality domains")
    return res


def _cache_config(num_blocks: int, page_bytes: int, layers: int) -> KVCacheConfig:
    """A block-outermost KV cache: block b holds every layer's page b."""
    block_bytes = math.ceil(layers * page_bytes / 512) * 512
    tensor = KVCacheTensor(
        size=num_blocks * block_bytes,
        layers=[f"layers.{i}.indexer.k_cache" for i in range(layers)],
        layer_stride=page_bytes,
        block_stride=block_bytes,
    )
    return KVCacheConfig(num_blocks, [tensor], [])


@pytest.mark.parametrize("extensible", [False, True])
def test_kv_cache_granules_live_in_their_planned_domain(domains, extensible):
    """Every committed granule of the KV cache is in the locality domain of
    its midpoint's run, and the page-to-domain table follows the plan."""
    from cuda.bindings import driver as cu

    from vllm.utils.extensible_tensor import granule_size
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache

    device = _device()
    page_rows, row_bytes = 64, 132
    config = _cache_config(6000, page_rows * row_bytes, 4)
    size = config.kv_cache_tensors[0].size
    kv_cache = None
    allocate = None
    if extensible:
        kv_cache = ExtensibleKVCache(config, device, KVCacheLayout.BLHNC)
        allocate = kv_cache.allocate
    allocate = placement.localized_kv_allocate(config, device, allocate)
    assert allocate is not None
    buf = allocate(size)
    if kv_cache is not None:
        kv_cache.commit(config.num_blocks)
    found = placement.find_buffer(buf.data_ptr())
    assert found is not None
    base, _, run_bytes, _ = found
    block_bytes = config.kv_cache_tensors[0].block_stride
    assert run_bytes % block_bytes == 0

    granule = granule_size(device.index)
    for g in range(size // granule):
        handle = cu.cuMemRetainAllocationHandle(base + g * granule + granule // 2)[1]
        prop = cu.cuMemGetAllocationPropertiesFromHandle(handle)[1]
        cu.cuMemRelease(handle)
        assert (
            prop.location.type
            == cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        planned = ((g * granule + granule // 2) // run_bytes) % 2
        assert prop.location.localized.localityDomainId == planned, g

    view = torch.as_strided(
        buf.view(torch.uint8),
        (config.num_blocks, page_rows, row_bytes),
        (block_bytes, row_bytes, 1),
        storage_offset=page_rows * row_bytes,
    )
    table = placement.page_domain_table(view)
    assert table is not None
    mid = view.data_ptr() - base + torch.arange(config.num_blocks) * block_bytes
    mid += page_rows * row_bytes // 2
    assert torch.equal(table.cpu(), ((mid // run_bytes) % 2).to(torch.int8))
    q0, k = placement.page_domain_plan(view)
    pages = torch.arange(config.num_blocks, device=device)
    assert k > 0 and torch.equal((((q0 + pages) // k) % 2).to(torch.int8), table)
