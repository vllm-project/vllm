import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.mediums import CXLLoadStoreSpec, DRAMLoadStoreSpec
from vllm.weave.cxl_backend import WeaveCXLBackend
from vllm.weave.dram_backend import WeaveDRAMBackend
from vllm.weave.two_tier_manager import TwoTierOffloadingManager

pytestmark = pytest.mark.skip_global_cleanup


def _bh(i: int) -> BlockHash:
    return BlockHash(i.to_bytes(32, "big", signed=False))


def test_store_to_dram_then_flush_to_cxl_and_dedup() -> None:
    dram_backend = WeaveDRAMBackend(block_size=16, num_blocks=4)
    cxl_backend = WeaveCXLBackend(block_size=16, num_blocks=4)
    mgr = TwoTierOffloadingManager(dram_backend=dram_backend, cxl_backend=cxl_backend)

    b0 = _bh(1)

    out = mgr.prepare_store([b0])
    assert out is not None
    assert out.block_hashes_to_store == [b0]
    assert out.store_spec.medium() == DRAMLoadStoreSpec.medium()

    mgr.complete_store([b0], success=True)

    flush = mgr.prepare_flush([b0])
    assert flush is not None
    src, dst = flush
    assert src.medium() == DRAMLoadStoreSpec.medium()
    assert dst.medium() == CXLLoadStoreSpec.medium()

    mgr.complete_flush([b0], success=True)

    assert mgr.probe_cxl([b0]) == [True]

    # Dedup: once committed in CXL, store path should no-op (no new DRAM store).
    out2 = mgr.prepare_store([b0])
    assert out2 is not None
    assert out2.block_hashes_to_store == []


def test_promotion_requires_committed_cxl() -> None:
    dram_backend = WeaveDRAMBackend(block_size=16, num_blocks=4)
    cxl_backend = WeaveCXLBackend(block_size=16, num_blocks=4)
    mgr = TwoTierOffloadingManager(dram_backend=dram_backend, cxl_backend=cxl_backend)

    b0 = _bh(2)

    # Not in CXL yet
    assert mgr.prepare_promotion([b0]) is None

    # Store -> DRAM
    out = mgr.prepare_store([b0])
    assert out is not None
    mgr.complete_store([b0])

    # Flush -> CXL + commit
    flush = mgr.prepare_flush([b0])
    assert flush is not None
    mgr.complete_flush([b0], success=True)

    # Promotion should be blocked if DRAM already has a copy.
    assert mgr.prepare_promotion([b0]) is None
