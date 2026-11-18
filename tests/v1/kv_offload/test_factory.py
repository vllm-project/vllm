# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for native offloading specs and their factory."""

import glob
import logging
import os
import time
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import (
    OffloadingGroupConfig,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingInfo
from vllm.v1.kv_offload.cpu.shared_offload_region import (
    REGION_SEGMENTS,
    SharedOffloadRegion,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


@pytest.fixture(autouse=True)
def restore_registry():
    original = dict(OffloadingSpecFactory._registry)
    yield
    OffloadingSpecFactory._registry = original


def _create_spec(**kwargs: Any) -> OffloadingSpec:
    return OffloadingSpecFactory.create_spec(make_offloading_config(**kwargs))


def _capacity_tokens(spec: OffloadingSpec) -> str | int | float | bool:
    """The token capacity the manager of the spec publishes as an info label."""
    (info,) = spec.get_manager().config_info()
    return info["cpu_kv_cache_size_tokens"]


class SingleArgExternalOffloadingSpec(OffloadingSpec):
    def get_manager(self) -> OffloadingManager:
        raise NotImplementedError

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError


def test_pre_registered_specs_can_be_imported():
    for name in OffloadingSpecFactory._registry:
        cls = OffloadingSpecFactory._registry[name]()
        assert issubclass(cls, OffloadingSpec)


def test_cpu_spec_registered():
    cls = OffloadingSpecFactory._registry["CPUOffloadingSpec"]()
    assert cls is CPUOffloadingSpec


def test_tiering_spec_registered():
    cls = OffloadingSpecFactory._registry["TieringOffloadingSpec"]()
    assert cls is TieringOffloadingSpec


def test_get_spec_cls_returns_registered_class():
    spec_cls = OffloadingSpecFactory.get_spec_cls(make_offloading_config().extra_config)
    assert spec_cls is CPUOffloadingSpec


def test_get_spec_cls_defaults_to_cpu():
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        make_offloading_config(spec_name=None).extra_config
    )
    assert spec_cls is CPUOffloadingSpec


def test_create_cpu_offloading_spec():
    spec = _create_spec()
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks > 0


@pytest.mark.parametrize("shared_region", [False, True])
def test_cpu_spec_sizes_normalized_worker_layout(monkeypatch, shared_region):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(
        CPUOffloadingSpec, "_uses_shared_region", lambda _: shared_region
    )
    log_info = MagicMock()
    monkeypatch.setattr(cpu_spec_module.logger, "info", log_info)
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    budget = alignment * 18
    spec = _create_spec(
        cpu_bytes_to_use=budget,
        worker_kv_bytes_per_block=16,
        blocks_per_chunk=2,
        world_size=6,
        tp_size=3,
        pp_size=2,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 32
    assert spec.kv_bytes_per_chunk == alignment * (6 if shared_region else 1)
    assert spec.num_chunks == (3 if shared_region else 18)
    assert spec.num_chunks * spec.kv_bytes_per_chunk <= budget
    # The chunks lost to per-rank padding are reported.
    assert log_info.called is shared_region
    if shared_region:
        assert log_info.call_args.args[1:] == (3, 15)


def test_cpu_spec_zero_worker_bytes_produces_empty_cache():
    spec = _create_spec(worker_kv_bytes_per_block=0, world_size=4)

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 0
    assert spec.kv_bytes_per_chunk == 0
    assert spec.num_chunks == 0


@pytest.mark.parametrize("blocks_per_chunk", [1, 2, 4])
def test_cpu_config_info_converts_slots_to_tokens(blocks_per_chunk: int):
    """One uncapped group makes the capacity the slot count in KV tokens.

    A slot holds blocks_per_chunk blocks of tokens_per_block tokens each, so
    dropping either factor understates the tier.
    """
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    tokens_per_block = 8
    spec = _create_spec(
        cpu_bytes_to_use=alignment * 12,
        worker_kv_bytes_per_block=alignment,
        blocks_per_chunk=blocks_per_chunk,
        groups=(OffloadingGroupConfig(tokens_per_block, ("layer",), 0),),
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks == 12 // blocks_per_chunk
    assert _capacity_tokens(spec) == (
        spec.num_chunks * blocks_per_chunk * tokens_per_block
    )


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_cpu_config_info_capacity_accounts_for_tensor_parallel_copies(
    world_size: int,
):
    """A slot holds every worker's copy of the block, so capacity divides by TP.

    Without the num_copies factor a TP=4 tier would be reported at four times
    the tokens it can hold.
    """
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    tokens_per_block = 8
    spec = _create_spec(
        cpu_bytes_to_use=alignment * 12,
        worker_kv_bytes_per_block=alignment,
        world_size=world_size,
        groups=(OffloadingGroupConfig(tokens_per_block, ("layer",), 0),),
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks == 12 // world_size
    assert _capacity_tokens(spec) == spec.num_chunks * tokens_per_block


def test_cpu_config_info_capacity_dedups_a_replicated_layout(monkeypatch):
    """A replicated layout stores one copy, so capacity does not divide by TP.

    Same TP=4 sizing as the test above, which yields 3 slots; deduplicating to a
    single copy yields 12.
    """
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    # Pin the ROCm answer too: on a ROCm runner the shared-region path is off,
    # and the dedup under test never engages.
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: False)
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    tokens_per_block = 8
    spec = _create_spec(
        cpu_bytes_to_use=alignment * 12,
        worker_kv_bytes_per_block=alignment,
        world_size=4,
        replicated_layout=True,
        groups=(OffloadingGroupConfig(tokens_per_block, ("layer",), 0),),
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks == 12
    assert _capacity_tokens(spec) == 12 * tokens_per_block


def test_cpu_config_info_mirrors_spec_sizing():
    """The published labels are the facts of the spec, not a second derivation.

    The frontend turns these keys into Prometheus label names, so a renamed key
    renames a label. The "cpu_" prefix keeps one name for the cache, standalone
    or as the primary tier of a tiering manager.
    """
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    tokens_per_block = 16
    spec = _create_spec(
        cpu_bytes_to_use=alignment * 5,
        worker_kv_bytes_per_block=alignment,
        blocks_per_chunk=1,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.get_manager().config_info() == [
        {
            "cpu_num_chunks": spec.num_chunks,
            "cpu_blocks_per_chunk": spec.blocks_per_chunk,
            "cpu_kv_cache_size_bytes": spec.num_chunks * spec.kv_bytes_per_chunk,
            "cpu_kv_cache_size_tokens": spec.num_chunks * tokens_per_block,
        }
    ]


def test_cpu_config_info_zero_capacity_is_exact_not_unknown():
    """A tier sized to nothing holds zero tokens; that is known, not unknown."""
    spec = _create_spec(
        worker_kv_bytes_per_block=0,
        groups=(OffloadingGroupConfig(16, ("layer",), 0),),
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks == 0
    assert _capacity_tokens(spec) == 0


def test_cpu_config_info_capacity_shrinks_when_a_second_group_shares_slots():
    """Two groups take two chunks for one request, so the capacity halves."""
    one_group = _create_spec(groups=(OffloadingGroupConfig(16, ("full_layer",), 0),))
    two_groups = _create_spec(
        groups=(
            OffloadingGroupConfig(16, ("full_layer",), 0),
            OffloadingGroupConfig(16, ("swa_layer",), 1),
        ),
    )

    assert isinstance(one_group, CPUOffloadingSpec)
    assert isinstance(two_groups, CPUOffloadingSpec)
    assert two_groups.num_chunks == one_group.num_chunks
    one_capacity = _capacity_tokens(one_group)
    assert isinstance(one_capacity, int)
    assert _capacity_tokens(two_groups) == one_capacity // 2


def test_cpu_config_info_no_token_capacity_without_a_kv_cache_group():
    """A model with no KV cache has no group whose chunk size to apply."""
    spec = _create_spec(groups=())

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks > 0
    assert _capacity_tokens(spec) == "None"


def test_cpu_config_info_no_token_capacity_without_max_model_len():
    """The capacity holds at max_model_len, so an unknown length gives None."""
    spec = _create_spec(max_model_len=0)

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks > 0
    assert _capacity_tokens(spec) == "None"


def test_tiering_primary_tier_publishes_the_cpu_labels():
    """The primary tier must keep the labels, because it renders them itself.

    TieringOffloadingManager.config_info() does not touch a primary tier label,
    so a CPU fact keeps one name in both managers. The tiering spec passes the
    same config info as the CPU spec.
    """
    primary_tier = CPUPrimaryTierOffloadingManager(
        num_chunks=5,
        mmap_region=MagicMock(),
        info=CPUOffloadingInfo.from_config(
            make_offloading_config(blocks_per_chunk=2, max_model_len=1024),
            num_chunks=5,
            kv_bytes_per_chunk=4096,
        ),
    )

    assert primary_tier.config_info() == [
        {
            "cpu_num_chunks": 5,
            "cpu_blocks_per_chunk": 2,
            "cpu_kv_cache_size_bytes": 5 * 4096,
            # 5 slots of 32 tokens each, over a request of 32 chunks.
            "cpu_kv_cache_size_tokens": 160,
        }
    ]


@pytest.mark.skip_global_cleanup
@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="requires /dev/shm")
def test_tiering_spec_passes_the_cpu_facts_to_the_primary_tier():
    """The tiering spec must give the primary tier the facts that it publishes.

    The spec builds a real shared region before the primary tier, so this test
    needs /dev/shm. It is the only test that runs the argument list of
    TieringOffloadingSpec.get_manager(). Both the chunk size and the
    configuration have a default, so a dropped argument gives no type error.
    A unique engine_id keeps the region path of this test separate.
    """
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        engine_id=f"test-tiering-{uuid.uuid4().hex}",
        blocks_per_chunk=2,
        max_model_len=1024,
    )
    assert isinstance(spec, TieringOffloadingSpec)

    manager = spec.get_manager()
    try:
        (primary_info,) = manager.config_info()
        assert primary_info == {
            "tier": "0:primary",
            "cpu_num_chunks": spec.num_chunks,
            "cpu_blocks_per_chunk": 2,
            "cpu_kv_cache_size_bytes": spec.num_chunks * spec.kv_bytes_per_chunk,
            # 32 tokens for each chunk, over a request of 32 chunks.
            "cpu_kv_cache_size_tokens": spec.num_chunks * 32,
        }
    finally:
        manager.shutdown()


def test_tiering_spec_aligns_row_size():
    alignment = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=alignment * 3,
        worker_kv_bytes_per_block=16,
        blocks_per_chunk=2,
        world_size=6,
        tp_size=3,
        pp_size=2,
    )

    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.cpu_page_size_per_worker == 32
    assert spec.kv_bytes_per_chunk == alignment
    assert spec.num_chunks == 3


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_tiering_spec_replicated_sizing_removes_world_factor(world_size: int):
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=True,
    )

    assert isinstance(spec, TieringOffloadingSpec)
    assert spec.replicated_layout is True
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block
    assert spec.num_chunks == 8


def test_tiering_spec_create_worker_uses_single_slot_for_replicated_layout(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        rank=1,
        world_size=4,
        replicated_layout=True,
    )
    assert isinstance(spec, TieringOffloadingSpec)

    region = MagicMock()
    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return region

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", fake_worker_ctor)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 5)

    kv_caches = MagicMock()
    spec.create_worker(kv_caches)

    assert region_calls[0]["rank"] == 0
    # Workers share the slot, so each pre-faults only its share of the region.
    assert region_calls[0]["populate_shard"] == (1, 4)
    assert region_calls[0]["kv_bytes_per_chunk"] == worker_kv_bytes_per_block
    assert worker_calls[0]["kv_caches"] is kv_caches
    assert worker_calls[0]["mmap_region"] is region


def test_tiering_spec_create_worker_uses_worker_rank_for_sharded_layout(monkeypatch):
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        worker_kv_bytes_per_block=4096,
        rank=1,
        world_size=4,
    )
    assert isinstance(spec, TieringOffloadingSpec)

    region_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(tiering_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(tiering_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(
        torch.accelerator,
        "current_device_index",
        lambda: 6,
    )

    spec.create_worker(MagicMock())

    assert region_calls[0]["rank"] == 1
    assert region_calls[0]["populate_shard"] is None


@pytest.mark.parametrize(
    "secondary_tiers,canonical_layout,is_parallelism_agnostic,replicated_layout,warns",
    [
        ([{"type": "example"}], False, True, False, True),
        ([{"type": "example"}], False, False, False, True),
        ([{"type": "example"}], True, False, False, True),
        ([{"type": "example"}], True, True, False, False),
        ([{"type": "example"}], False, False, True, False),
        ([], False, False, False, False),
    ],
)
def test_tiering_spec_warns_when_secondary_tiers_are_topology_bound(
    caplog_vllm,
    monkeypatch,
    secondary_tiers,
    canonical_layout,
    is_parallelism_agnostic,
    replicated_layout,
    warns,
):
    """Direct-layout KV warns even when it is flagged parallelism-agnostic,
    unless it is replicated MLA, which FileMapper shares across TP sizes."""
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    monkeypatch.setattr(
        tiering_spec_module, "CPUPrimaryTierOffloadingManager", MagicMock()
    )
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        engine_id=f"warns-{uuid.uuid4().hex[:8]}",
        worker_kv_bytes_per_block=4096,
        world_size=2 if replicated_layout else 1,
        is_parallelism_agnostic=is_parallelism_agnostic,
        replicated_layout=replicated_layout,
        canonical_layout=canonical_layout,
        extra_config={"secondary_tiers": secondary_tiers},
    )
    assert isinstance(spec, TieringOffloadingSpec)
    # The scheduler joins the region that worker rank 0 created.
    worker_region = _worker_rank_0_region(spec)
    try:
        spec.get_manager().shutdown()
    finally:
        worker_region.cleanup()

    warnings = [
        record
        for record in caplog_vllm.records
        if record.name == tiering_spec_module.__name__
        and record.levelno == logging.WARNING
    ]
    assert len(warnings) == int(warns)


def _worker_rank_0_region(spec: TieringOffloadingSpec) -> SharedOffloadRegion:
    """The shared region that worker rank 0 of the spec's engine creates."""
    return SharedOffloadRegion(
        engine_id=spec.config.engine_id,
        num_chunks=spec.num_chunks,
        rank=0,
        kv_bytes_per_chunk=spec.kv_bytes_per_chunk,
        cpu_page_size=spec.cpu_page_size_per_worker,
        num_segments=REGION_SEGMENTS,
    )


def test_tiering_scheduler_requires_worker_rank_0_region():
    """Secondary tiers read the region worker rank 0 created, into which the
    mirror copies the remote slots: a scheduler that has to create the region
    itself is not on worker rank 0's node, and must not leave it behind."""
    engine_id = f"no-worker-{uuid.uuid4().hex[:8]}"
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        engine_id=engine_id,
        worker_kv_bytes_per_block=4096,
        world_size=2,
        extra_config={"secondary_tiers": _TIERS},
    )

    with pytest.raises(RuntimeError, match="share a node with worker rank 0"):
        spec.get_manager()
    assert glob.glob(f"/dev/shm/vllm_offload_{engine_id}.mmap*") == []


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_sizing_on_shared_region(monkeypatch, world_size: int):
    # On CUDA the default spec honors
    # replicated layout: a single MLA copy (num_copies=1), matching tiering.
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: False)
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=True,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is True
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block
    assert spec.num_chunks == 8


@pytest.mark.parametrize("rocm", [False, True], ids=["non-cuda-alike", "rocm"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_cpu_spec_replicated_disabled_without_shared_region(
    monkeypatch, world_size: int, rocm: bool
):
    # Data-loss guard: platforms without shared regions keep a private pinned
    # tensor (no shared medium), so replicated layout MUST stay off. Otherwise
    # the rank-0 writer gate would ack rank>0 stores without writing, leaving
    # those private buffers empty and corrupting subsequent loads.
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: rocm)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: rocm)
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * world_size * 2,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=world_size,
        replicated_layout=True,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is False
    assert spec.cpu_page_size_per_worker == worker_kv_bytes_per_block
    assert spec.kv_bytes_per_chunk == worker_kv_bytes_per_block * world_size
    assert spec.num_chunks == 2


@pytest.mark.parametrize("config_replicated", [True, False])
@pytest.mark.parametrize(
    ("cuda_alike", "rocm", "shared_region"),
    [(True, False, True), (True, True, False), (False, False, False)],
    ids=["cuda", "rocm", "non-cuda-alike"],
)
def test_cpu_spec_replicated_layout_truth_matrix(
    monkeypatch,
    cuda_alike: bool,
    rocm: bool,
    shared_region: bool,
    config_replicated: bool,
):
    # replicated_layout is enabled iff the config gate passes AND the deployment
    # actually allocates on the shared region.
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        cpu_spec_module.current_platform, "is_cuda_alike", lambda: cuda_alike
    )
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: rocm)
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=4,
        replicated_layout=config_replicated,
    )

    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.replicated_layout is (shared_region and config_replicated)


def test_cpu_spec_create_worker_uses_mmap_on_cuda(monkeypatch):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        world_size=4,
    )
    assert isinstance(spec, CPUOffloadingSpec)

    region = MagicMock()
    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return region

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", fake_worker_ctor)
    monkeypatch.setattr(cpu_spec_module, "check_shm_free_space", MagicMock())
    monkeypatch.setattr(cpu_spec_module, "_all_workers_barrier", MagicMock())
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 5)

    kv_caches = MagicMock()
    spec.create_worker(kv_caches)

    # The logical rank keys its private region, regardless of the CUDA device.
    assert region_calls[0]["engine_id"] == "test-engine_0"
    assert region_calls[0]["rank"] == 0
    assert region_calls[0]["kv_bytes_per_chunk"] == worker_kv_bytes_per_block
    assert worker_calls[0]["kv_caches"] is kv_caches
    assert worker_calls[0]["mmap_region"] is region


@pytest.mark.parametrize("rocm", [False, True], ids=["non-cuda-alike", "rocm"])
def test_cpu_spec_create_worker_uses_tensor_path_without_shared_region(
    monkeypatch, rocm
):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    spec = _create_spec(worker_kv_bytes_per_block=4096, world_size=4)
    assert isinstance(spec, CPUOffloadingSpec)

    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    def fake_worker_ctor(**kwargs):
        worker_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: rocm)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: rocm)
    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", fake_worker_ctor)

    spec.create_worker(MagicMock())

    # ROCm and non-CUDA-alike platforms use the per-rank pinned-tensor path.
    assert region_calls == []
    assert worker_calls[0]["mmap_region"] is None


def test_cpu_spec_create_worker_skips_mmap_for_empty_cache(monkeypatch):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    # worker_kv_bytes_per_block=0 yields num_chunks=0; a zero-byte region cannot
    # be mmap'd, so even on CUDA this must fall back to the tensor path.
    spec = _create_spec(worker_kv_bytes_per_block=0, world_size=4)
    assert isinstance(spec, CPUOffloadingSpec)
    assert spec.num_chunks == 0

    region_calls: list[dict[str, Any]] = []
    worker_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        cpu_spec_module,
        "SharedOffloadRegion",
        lambda **kwargs: region_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cpu_spec_module,
        "CPUOffloadingWorker",
        lambda **kwargs: worker_calls.append(kwargs),
    )

    spec.create_worker(MagicMock())

    assert region_calls == []
    assert worker_calls[0]["mmap_region"] is None


@pytest.mark.parametrize(
    (
        "replicated_layout",
        "rank",
        "device_index",
        "world_size",
        "local_world_size",
        "expected_engine_id",
        "capacity_error",
    ),
    [
        (True, 1, 5, 4, 4, "test-engine", False),
        (True, 0, 0, 4, 4, "test-engine", False),
        (False, 1, 5, 4, 4, "test-engine_1", False),
        (False, 0, 5, 4, 4, "test-engine_0", False),
        (False, 3, 6, 4, 4, "test-engine_3", False),
        # Ranks on other nodes do not allocate in this node's /dev/shm.
        (False, 3, 1, 4, 2, "test-engine_3", False),
        (False, 0, 5, 4, 4, "test-engine_0", True),
    ],
)
def test_cpu_spec_create_worker_region_assignment(
    monkeypatch,
    replicated_layout,
    rank,
    device_index,
    world_size,
    local_world_size,
    expected_engine_id,
    capacity_error,
):
    import vllm.v1.kv_offload.cpu.spec as cpu_spec_module

    monkeypatch.setattr(cpu_spec_module.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(cpu_spec_module.current_platform, "is_rocm", lambda: False)
    worker_kv_bytes_per_block = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    spec = _create_spec(
        cpu_bytes_to_use=worker_kv_bytes_per_block * 8,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        replicated_layout=replicated_layout,
    )
    assert isinstance(spec, CPUOffloadingSpec)

    region_calls: list[dict[str, Any]] = []
    memory_check = MagicMock()
    barrier = MagicMock()

    def fake_region_ctor(**kwargs):
        region_calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(cpu_spec_module, "SharedOffloadRegion", fake_region_ctor)
    monkeypatch.setattr(cpu_spec_module, "CPUOffloadingWorker", MagicMock())
    monkeypatch.setattr(cpu_spec_module, "check_shm_free_space", memory_check)
    monkeypatch.setattr(cpu_spec_module, "_all_workers_barrier", barrier)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: device_index)

    if capacity_error:
        memory_check.side_effect = RuntimeError("Insufficient space in /dev/shm")
        with pytest.raises(RuntimeError, match="Insufficient space"):
            spec.create_worker(MagicMock())
        # Peers waiting at the barrier are released even when the check fails.
        barrier.assert_called_once_with()
        assert region_calls == []
        return

    spec.create_worker(MagicMock())

    assert region_calls[0]["engine_id"] == expected_engine_id
    assert region_calls[0]["rank"] == 0
    assert region_calls[0]["populate_shard"] == (
        (rank, world_size) if replicated_layout else None
    )
    assert region_calls[0]["private"] is not replicated_layout
    if replicated_layout:
        assert region_calls[0]["barrier"] is barrier
        memory_check.assert_not_called()
        barrier.assert_not_called()
    else:
        assert region_calls[0]["barrier"] is None
        region_bytes = spec.num_chunks * region_calls[0]["kv_bytes_per_chunk"]
        memory_check.assert_called_once_with(
            region_bytes * local_world_size,
            allocation_name="CPU KV offload regions in /dev/shm",
        )
        barrier.assert_called_once_with()


def test_offloading_spec_has_replicated_layout_default():
    spec = SingleArgExternalOffloadingSpec(make_offloading_config())
    assert spec.replicated_layout is False


def test_offloading_spec_uses_normalized_chunk_geometry():
    groups = (
        OffloadingGroupConfig(12, ("full_layer",), 0),
        OffloadingGroupConfig(16, ("mla_layer",), 1),
    )
    spec = _create_spec(
        groups=groups,
        tokens_per_hash=4,
        blocks_per_chunk=2,
    )

    assert spec.tokens_per_block == (12, 16)
    assert spec.tokens_per_hash == 4
    assert spec.blocks_per_chunk == 2


def test_create_dynamic_spec_receives_config():
    config = make_offloading_config(
        spec_name="SingleArgExternalOffloadingSpec",
        extra_config={"spec_module_path": "tests.v1.kv_offload.test_factory"},
    )

    spec = OffloadingSpecFactory.create_spec(config)

    assert isinstance(spec, SingleArgExternalOffloadingSpec)
    assert spec.config is config


def test_dynamic_load_via_spec_module_path():
    del OffloadingSpecFactory._registry["CPUOffloadingSpec"]
    config = make_offloading_config(
        extra_config={"spec_module_path": "vllm.v1.kv_offload.cpu.spec"}
    )

    spec_cls = OffloadingSpecFactory.get_spec_cls(config.extra_config)

    assert spec_cls is CPUOffloadingSpec


def test_unregistered_spec_without_module_path_raises():
    config = make_offloading_config(spec_name="NonexistentSpec")
    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.get_spec_cls(config.extra_config)

    with pytest.raises(ValueError, match="Unsupported spec type"):
        OffloadingSpecFactory.create_spec(config)


def test_cpu_spec_missing_cpu_bytes_to_use_raises():
    with pytest.raises(Exception, match="cpu_bytes_to_use must be specified"):
        _create_spec(cpu_bytes_to_use=None)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="is already registered"):
        OffloadingSpecFactory.register_spec(
            "CPUOffloadingSpec", "some.module", "SomeClass"
        )


def test_build_metric_definitions_below_threshold():
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    extra_config = {"store_threshold": 1}
    spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "CPUOffloadingSpec"})
    metrics = spec_cls.build_metric_definitions(extra_config)

    assert CPUOffloadingMetrics.STORES_SKIPPED not in metrics
    assert CPUOffloadingMetrics.CPU_ALLOCATION_SIZE in metrics


def test_build_metric_definitions_allocation_size_histogram():
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "CPUOffloadingSpec"})
    metrics = spec_cls.build_metric_definitions({})
    metadata = metrics[CPUOffloadingMetrics.CPU_ALLOCATION_SIZE]

    assert isinstance(metadata, OffloadingHistogramMetadata)
    assert metadata.buckets == (
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
    )


def test_build_metric_definitions_returns_counter_at_threshold():
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    extra_config = {"store_threshold": 2}
    spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "CPUOffloadingSpec"})
    metrics = spec_cls.build_metric_definitions(extra_config)

    assert CPUOffloadingMetrics.STORES_SKIPPED in metrics


def _join_engine(
    rank: int, init_method: str, nodes: str, tp: int, pp: int = 1, pcp: int = 1
) -> None:
    """Make this process rank `rank` of a world whose rank r sits on node
    `nodes[r]`, holding one engine or, with more ranks than tp * pp * pcp,
    several (external launchers).

    The world, TP, PP and PCP groups are vLLM `GroupCoordinator`s on Gloo
    without device communicators, so no GPU is needed. Node placement is
    simulated where `in_the_same_node_as` probes it: a shared memory segment
    created on one node cannot be opened from another.
    """
    from datetime import timedelta
    from multiprocessing import shared_memory

    import torch.distributed as dist

    import vllm.distributed.parallel_state as parallel_state

    node = nodes[rank]
    open_shm = shared_memory.SharedMemory

    def node_shared_memory(name=None, create=False, size=0):
        if create:
            name = f"node{node}_{uuid.uuid4().hex[:16]}"
            return open_shm(name=name, create=True, size=size)
        if not name.startswith(f"node{node}_"):
            raise FileNotFoundError(name)  # created on another node
        return open_shm(name=name)

    shared_memory.SharedMemory = node_shared_memory  # type: ignore[misc]

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=len(nodes),
        timeout=timedelta(seconds=60),
    )

    def coordinator(group_ranks: list[list[int]], name: str):
        return parallel_state.GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=0,
            torch_distributed_backend="gloo",
            use_device_communicator=False,
            group_name=name,
        )

    # vLLM's rank grid: TP is the innermost axis, then PCP, then PP.
    grid = torch.arange(len(nodes)).reshape(-1, 1, pp, pcp, tp)
    parallel_state._WORLD = coordinator([list(range(len(nodes)))], "world")
    parallel_state._TP = coordinator(grid.reshape(-1, tp).tolist(), "tp")
    parallel_state._PCP = coordinator(
        grid.transpose(3, 4).reshape(-1, pcp).tolist(), "pcp"
    )
    parallel_state._PP = coordinator(
        grid.transpose(2, 4).reshape(-1, pp).tolist(), "pp"
    )


def _run_ranks(fn, args: tuple, nprocs: int, timeout_s: float = 180) -> None:
    """Run `fn(rank, *args)` on `nprocs` spawned ranks and wait for them. A
    rank that raises fails the test; one still running at the deadline, even
    outside a collective, is terminated and fails it too."""
    import torch.multiprocessing as mp

    context = mp.start_processes(
        fn, args=args, nprocs=nprocs, join=False, start_method="spawn"
    )
    assert context is not None
    deadline = time.monotonic() + timeout_s
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            for process in context.processes:
                process.join(5)
            pytest.fail(f"ranks still running after {timeout_s} s")


def _span_nodes_on_rank(rank: int, init_method: str, results) -> None:
    """Run `_workers_span_nodes` on one rank of a TP=2, PP=2 engine whose
    ranks 0-2 share a node and rank 3 sits on another."""
    import torch.distributed as dist

    from vllm.v1.kv_offload.tiering.spec import _workers_span_nodes

    _join_engine(rank, init_method, "aaab", tp=2, pp=2)
    results.put((rank, _workers_span_nodes()))
    dist.destroy_process_group()


def test_workers_span_nodes_agrees_across_ranks(tmp_path):
    """Rank 0's TP and PP groups share its node, but rank 3's do not; ranks
    that disagreed would split between the mirror's collectives and hang."""
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    results = context.SimpleQueue()
    init_method = f"file://{tmp_path / 'store'}"
    _run_ranks(_span_nodes_on_rank, (init_method, results), 4)
    assert sorted(results.get() for _ in range(4)) == [
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]


class _FakeNixlAgent:
    """Stands in for `nixl_utils.NixlWrapper`, the NIXL agent: the RDMA
    library and the memory of peers on other nodes are outside this host.
    Records what each agent registers and which peers it adds."""

    created: list["_FakeNixlAgent"] = []

    def __init__(self, name: str, config: Any):
        self.name = name
        self.registered: list[tuple[str, list[tuple[int, int]]]] = []
        self.peers: list[str] = []
        _FakeNixlAgent.created.append(self)

    def create_backend(self, backend: str, params: dict) -> None:
        assert backend == "UCX"

    def register_memory(self, descs: list, mem_type: str):
        registration = (mem_type, [(addr, size) for addr, size, _, _ in descs])
        self.registered.append(registration)
        return registration

    def deregister_memory(self, registration) -> None:
        self.registered.remove(registration)

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        self.peers.append(metadata.decode())
        return self.peers[-1]

    def remove_remote_agent(self, peer: str) -> None:
        self.peers.remove(peer)


def _region_mapping(engine_id: str) -> tuple[int, int, str] | None:
    """(start, end, path) of this process's mapping of the engine's region."""
    with open("/proc/self/maps") as maps:
        for line in maps:
            if f"vllm_offload_{engine_id}.mmap" in line:
                fields = line.split()
                start, end = (int(a, 16) for a in fields[0].split("-"))
                return start, end, fields[-1]
    return None


def _create_worker_on_rank(
    rank: int,
    init_method: str,
    nodes: str,
    parallel: dict[str, int],
    secondary_tiers: list[dict],
    engine_id: str,
    results,
) -> None:
    """Build this rank's offloading worker through `TieringOffloadingSpec`
    and report what it allocated and registered."""
    import torch.distributed as dist

    from vllm.distributed import nixl_utils
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCacheTensor

    _join_engine(rank, init_method, nodes, **parallel)
    nixl_utils.NixlWrapper = _FakeNixlAgent
    nixl_utils.nixl_agent_config = dict
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        engine_id=engine_id,
        worker_kv_bytes_per_block=4096,
        rank=rank,
        world_size=len(nodes),
        tp_size=parallel["tp"],
        pp_size=parallel.get("pp", 1),
        pcp_size=parallel.get("pcp", 1),
        extra_config={"secondary_tiers": secondary_tiers},
    )
    assert isinstance(spec, TieringOffloadingSpec)
    blocks = torch.zeros((8, 4096), dtype=torch.int8)
    kv_caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=blocks, page_size_bytes=4096)],
        group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=4096)]],
    )

    worker = spec.create_worker(kv_caches)
    # Every rank has built its worker, pre-faulting its share of the region.
    dist.barrier()

    mapping = _region_mapping(engine_id)
    names = {blocks.data_ptr(): "kv"}
    region = None
    if mapping is not None:
        start, end, path = mapping
        names[start] = "region"
        region = (end - start, os.stat(path).st_blocks * 512)
    prefix = f"kv-offload-mirror-{engine_id}-"
    results.put(
        (
            rank,
            type(worker).__name__,
            region,
            [
                (
                    agent.name.removeprefix(prefix),
                    [
                        (mem, [(names.get(addr, addr), size) for addr, size in descs])
                        for mem, descs in agent.registered
                    ],
                    [peer.removeprefix(prefix) for peer in agent.peers],
                )
                for agent in _FakeNixlAgent.created
            ],
        )
    )
    # The region's file goes away when its creator shuts down.
    dist.barrier()
    worker.shutdown()
    dist.destroy_process_group()


_TIERS = [{"type": "example"}]


@pytest.mark.parametrize(
    ("nodes", "parallel", "secondary_tiers", "placement"),
    [
        # H: rank 0 registering the head region, L: a CPU worker on its node's
        # region, R: off the head node, moving its KV to its slot over NIXL.
        pytest.param("aaab", {"tp": 2, "pp": 2}, [], "LLLL", id="no-tiers"),
        pytest.param("aaaa", {"tp": 2, "pp": 2}, _TIERS, "LLLL", id="one-node"),
        pytest.param("abab", {"tp": 2, "pp": 2}, _TIERS, "HRLR", id="tp-spans"),
        pytest.param("aabb", {"tp": 2, "pp": 2}, _TIERS, "HLRR", id="pp-spans"),
        pytest.param("aabb", {"tp": 2, "pcp": 2}, _TIERS, "HLRR", id="pcp-spans"),
        # Only rank 3's groups span nodes; three head ranks pre-fault.
        pytest.param("aaab", {"tp": 2, "pp": 2}, _TIERS, "HLLR", id="rank3-off"),
        pytest.param("abbb", {"tp": 2, "pp": 2}, _TIERS, "HRRR", id="rank0-alone"),
    ],
)
def test_tiering_spec_mirrors_secondary_tiers_across_nodes(
    tmp_path, nodes, parallel, secondary_tiers, placement
):
    """Secondary tiers on an engine that spans nodes get a region mirror: the
    head node's workers map and pre-fault the whole region between them,
    rank 0 registers it for the others, which map no region and register
    their KV cache instead."""
    import torch.multiprocessing as mp

    engine_id = f"mirror-{uuid.uuid4().hex[:8]}"
    context = mp.get_context("spawn")
    results = context.SimpleQueue()
    _run_ranks(
        _create_worker_on_rank,
        (
            f"file://{tmp_path / 'store'}",
            nodes,
            parallel,
            secondary_tiers,
            engine_id,
            results,
        ),
        len(nodes),
    )

    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        worker_kv_bytes_per_block=4096,
        world_size=len(nodes),
        tp_size=parallel["tp"],
        extra_config={"secondary_tiers": secondary_tiers},
    )
    assert isinstance(spec, TieringOffloadingSpec)
    region_bytes = spec.num_chunks * spec.kv_bytes_per_chunk
    # Mapped bytes, and bytes allocated once every rank has pre-faulted.
    whole_region = (region_bytes, region_bytes)
    remote = [str(rank) for rank, kind in enumerate(placement) if kind == "R"]

    def expected(rank: int, kind: str) -> tuple:
        if kind == "H":
            agent = ("0", [("DRAM", [("region", region_bytes)])], remote)
            return rank, "RegionMirrorHeadWorker", whole_region, [agent]
        if kind == "R":
            agent = (str(rank), [("VRAM", [("kv", 8 * 4096)])], ["0"])
            return rank, "RegionMirrorWorker", None, [agent]
        return rank, "CPUOffloadingWorker", whole_region, []

    assert sorted(results.get() for _ in nodes) == [
        expected(rank, kind) for rank, kind in enumerate(placement)
    ]


def _worker_error_on_rank(
    rank: int,
    init_method: str,
    nodes: str,
    parallel: dict[str, int],
    change: dict[str, Any],
    engine_id: str,
    results,
) -> None:
    """Build this rank's worker with `change` applied to its offloading config
    and report what it raised and left. `pages` gives each engine rank's KV
    bytes per block."""
    import torch.distributed as dist

    from vllm.distributed import nixl_utils
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCacheTensor

    _join_engine(rank, init_method, nodes, **parallel)
    nixl_utils.NixlWrapper = _FakeNixlAgent
    nixl_utils.nixl_agent_config = dict
    change = dict(change)
    engine_size = parallel["tp"] * parallel.get("pp", 1) * parallel.get("pcp", 1)
    engine_rank = rank % engine_size
    if engine_size < len(nodes):
        # One region name per engine, as each external replica has its own.
        engine_id = f"{engine_id}-{rank // engine_size}"
    page = change.pop("pages", [4096] * engine_size)[engine_rank]
    tiers = change.pop("secondary_tiers", _TIERS)
    spec = _create_spec(
        spec_name="TieringOffloadingSpec",
        engine_id=engine_id,
        worker_kv_bytes_per_block=page,
        rank=engine_rank,
        world_size=engine_size,
        tp_size=parallel["tp"],
        pp_size=parallel.get("pp", 1),
        pcp_size=parallel.get("pcp", 1),
        extra_config={"secondary_tiers": tiers},
        **change,
    )
    assert isinstance(spec, TieringOffloadingSpec)
    kv_caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(
                tensor=torch.zeros((8, page), dtype=torch.int8), page_size_bytes=page
            )
        ],
        group_data_refs=[[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page)]],
    )
    try:
        spec.create_worker(kv_caches)
        error = "no error"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    results.put(
        (
            rank,
            error.split(";")[0],
            _region_mapping(engine_id) is not None,
            [agent.registered for agent in _FakeNixlAgent.created],
        )
    )
    dist.destroy_process_group()


def _worker_errors(
    tmp_path, nodes: str, parallel: dict[str, int], change: dict[str, Any]
) -> tuple[list[tuple], str]:
    """Each rank's outcome, once every rank has built its worker or failed. A
    rank left in a collective fails on the Gloo timeout, and any rank still
    running at the deadline fails the test."""
    import torch.multiprocessing as mp

    engine_id = f"mirror-{uuid.uuid4().hex[:8]}"
    results = mp.get_context("spawn").SimpleQueue()
    _run_ranks(
        _worker_error_on_rank,
        (
            f"file://{tmp_path / 'store'}",
            nodes,
            parallel,
            change,
            engine_id,
            results,
        ),
        len(nodes),
    )
    return sorted(results.get() for _ in nodes), engine_id


def test_tiering_spec_rejects_mismatched_mirror_layouts_on_every_rank(tmp_path):
    """Pipeline stages of unequal KV size derive different head layouts, which
    would address the wrong bytes of each row. Every rank must raise, none may
    be left in a collective, and rank 0 must release the region and its NIXL
    registration."""
    outcomes, engine_id = _worker_errors(
        tmp_path, "ab", {"tp": 1, "pp": 2}, {"pages": [4096, 8192]}
    )
    mismatch = (
        "ValueError: The KV offload mirror needs the same CPU region geometry "
        "on every worker"
    )
    # Rank 0 registered the head region and released it; rank 1, off the head
    # node, never got as far as registering its KV cache.
    assert outcomes == [(0, mismatch, False, [[]]), (1, mismatch, False, [])]
    assert not os.path.exists(f"/dev/shm/vllm_offload_{engine_id}.mmap")


@pytest.mark.parametrize(
    ("nodes", "change", "error"),
    [
        ("ab", {"data_parallel_size": 2}, NotImplementedError),
        # A torchrun launcher's world group holds every external replica: two
        # engines of two workers each.
        ("abab", {}, NotImplementedError),
        ("ab", {"replicated_layout": True}, ValueError),
        ("ab", {"secondary_tiers": [{"type": "kvcr"}]}, ValueError),
    ],
)
def test_tiering_spec_rejects_unmirrorable_engines_across_nodes(
    tmp_path, nodes, change, error
):
    """Setups the mirror cannot serve are rejected on every rank, and no rank
    is left with a region mapped or memory registered."""
    if "secondary_tiers" in change:
        pytest.importorskip("kvcr")
    outcomes, _ = _worker_errors(tmp_path, nodes, {"tp": 2}, change)
    raised = {message for _, message, _, _ in outcomes}
    assert len(raised) == 1, outcomes
    assert raised.pop().startswith(f"{error.__name__}: "), outcomes
    assert [outcome[2:] for outcome in outcomes] == [(False, [])] * len(nodes)
