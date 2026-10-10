# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the vLLM project
"""copy_kv_cache_blocks_inplace must handle tuple-valued layer caches.

Hybrid models (e.g. GLM-5-style MoE hybrids with an MLA latent cache plus
auxiliary indexer/state pools) can bind one attention layer's KV cache as a
tuple of tensors — one entry per cache role. Prefix-cache block copies used
to call ``cache.device`` on the tuple itself and crash with
``AttributeError: 'tuple' object has no attribute 'device'``. The fix
flattens tuple/list entries before the per-tensor dedup/copy loop; these
tests pin that behavior on the current accelerator device.
"""

import pytest
import torch

import vllm.utils.torch_utils as torch_utils
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

# The copy helper uploads the index tensor through async_tensor_h2d, whose
# pinned-memory fast path is accelerator-specific. Disable it so the test is
# valid on every device type.
torch_utils.PIN_MEMORY = False

BLOCKS = 8


def _make_cache(num_blocks: int, fill: float) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.full((num_blocks, 4, 8, 16), fill, dtype=torch.float32, device=device)


def _copies() -> list[KVCacheBlockCopy]:
    # Copy block 0 -> 1 and 2 -> 3.
    return [KVCacheBlockCopy(0, 1), KVCacheBlockCopy(2, 3)]


def test_tuple_valued_layer_caches_are_copied():
    """Each tensor inside a tuple cache must receive the block copies."""
    latent = _make_cache(BLOCKS, fill=7.0)
    indexer = _make_cache(BLOCKS, fill=9.0)
    plain = _make_cache(BLOCKS, fill=11.0)

    copy_kv_cache_blocks_inplace(
        [(latent, indexer), plain],
        num_blocks=BLOCKS,
        kv_cache_block_copies=_copies(),
    )

    for cache, fill in ((latent, 7.0), (indexer, 9.0), (plain, 11.0)):
        assert torch.all(cache[1] == fill), "dst block 1 must hold src block 0"
        assert torch.all(cache[3] == fill), "dst block 3 must hold src block 2"
        assert torch.all(cache[4] == fill), "untouched blocks must be intact"


def test_plain_tensor_caches_still_work():
    plain_a = _make_cache(BLOCKS, fill=3.0)
    copy_kv_cache_blocks_inplace(
        [plain_a],
        num_blocks=BLOCKS,
        kv_cache_block_copies=_copies(),
    )
    assert torch.all(plain_a[1] == 3.0)
    assert torch.all(plain_a[3] == 3.0)


def test_empty_copies_is_a_noop():
    cache = _make_cache(BLOCKS, fill=5.0)
    copy_kv_cache_blocks_inplace(
        [(cache, cache), cache],
        num_blocks=BLOCKS,
        kv_cache_block_copies=[],
    )
    assert torch.all(cache == 5.0)


def test_shared_storage_is_copied_once():
    """Two views of one storage must not double-apply the copy.

    Block 0 -> 1 twice on the same storage would copy the already-copied
    dst back over itself; with dedup by (device, data_ptr) the second
    application is skipped and the result is stable.
    """
    base = _make_cache(BLOCKS, fill=1.0)
    view = base.view_as(base)
    copy_kv_cache_blocks_inplace(
        [(base, view)],
        num_blocks=BLOCKS,
        kv_cache_block_copies=[KVCacheBlockCopy(0, 1)],
    )
    assert torch.all(base[1] == 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
