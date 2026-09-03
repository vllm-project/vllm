# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache event emission tests for SimpleCPUOffloadScheduler.

Verifies that BlockStored/BlockRemoved events carry the correct storage
medium (MEDIUM_CPU / MEDIUM_STORAGE) and per-group metadata when
``enable_kv_cache_events`` is set, and that no events are emitted when it
is unset.
"""

from __future__ import annotations

from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    DTYPE,
    HEAD_SIZE,
    NUM_KV_HEADS,
    SchedulerFixture,
    _alloc_and_register,
    _allocate_cp_gpu_blocks,
    _make_cp_request,
    _make_kv_cache_config,
    _make_vllm_config,
    make_request,
    make_scheduler_output,
    simulate_store_completion,
)
from vllm import SamplingParams
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_STORAGE,
    BlockRemoved,
    BlockStored,
)
from vllm.lora.request import LoRARequest
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_request_block_hasher,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.manager import (
    SimpleCPUOffloadScheduler,
    StoreRequestState,
)


def _make_mixed_kv_cache_config(
    num_blocks: int,
    sliding_window: int | None = None,
    with_mamba: bool = False,
    mamba_block_size: int = BLOCK_SIZE,
) -> KVCacheConfig:
    """Build a KVCacheConfig with a FullAttention group plus optional
    SlidingWindow and Mamba groups for per-group metadata tests."""
    register_all_kvcache_specs(vllm_config=None)
    groups = []
    tensors = []

    # Group 0 is always FullAttention (the scheduler requires one FA group).
    fa_layers = ["layer_fa"]
    fa_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups.append(KVCacheGroupSpec(fa_layers, fa_spec))
    fa_bytes = _BYTES_PER_BLOCK * num_blocks
    tensors.append(
        KVCacheTensor(
            size=fa_bytes,
            layers=fa_layers,
            layer_stride=fa_bytes,
            block_stride=_BYTES_PER_BLOCK,
        )
    )

    if sliding_window is not None:
        sw_layers = ["layer_sw"]
        sw_spec = SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
            sliding_window=sliding_window,
        )
        groups.append(KVCacheGroupSpec(sw_layers, sw_spec))
        sw_bytes = _BYTES_PER_BLOCK * num_blocks
        tensors.append(
            KVCacheTensor(
                size=sw_bytes,
                layers=sw_layers,
                layer_stride=sw_bytes,
                block_stride=_BYTES_PER_BLOCK,
            )
        )

    if with_mamba:
        m_layers = ["layer_m"]
        m_spec = MambaSpec(
            block_size=mamba_block_size,
            shapes=((NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE),),
            dtypes=(DTYPE,),
            mamba_cache_mode="align",
        )
        groups.append(KVCacheGroupSpec(m_layers, m_spec))
        m_bytes = _BYTES_PER_BLOCK * num_blocks
        tensors.append(
            KVCacheTensor(
                size=m_bytes,
                layers=m_layers,
                layer_stride=m_bytes,
                block_stride=_BYTES_PER_BLOCK,
            )
        )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def make_events_scheduler(
    num_cpu_blocks: int = 8,
    num_gpu_blocks: int = 16,
    num_groups: int = 1,
    lazy: bool = False,
    disk_capacity_bytes: int = 0,
    dcp_world_size: int = 1,
    kv_cache_config: KVCacheConfig | None = None,
    enable_events: bool = True,
) -> SchedulerFixture:
    """Build a SimpleCPUOffloadScheduler with kv cache events enabled."""
    if kv_cache_config is None:
        kv_cache_config = _make_kv_cache_config(num_gpu_blocks, num_groups)
    vllm_config = _make_vllm_config()
    if enable_events:
        vllm_config.kv_events_config = KVEventsConfig(enable_kv_cache_events=True)
    vllm_config.parallel_config.decode_context_parallel_size = dcp_world_size

    virtual_block_size = BLOCK_SIZE * dcp_world_size
    cpu_capacity_bytes = _BYTES_PER_BLOCK * num_cpu_blocks * num_groups

    sched = SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=cpu_capacity_bytes,
        scheduler_block_size=virtual_block_size,
        hash_block_size=virtual_block_size,
        lazy_offload=lazy,
        disk_capacity_bytes=disk_capacity_bytes,
    )

    gpu_block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=virtual_block_size,
    )
    sched.bind_gpu_block_pool(gpu_block_pool)
    return SchedulerFixture(
        scheduler=sched,
        gpu_block_pool=gpu_block_pool,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        num_groups=num_groups,
    )


def _do_eager_store(
    fix: SchedulerFixture,
    num_blocks: int = 2,
    lora_request: LoRARequest | None = None,
) -> tuple[object, Request]:
    """Run an eager store of ``num_blocks`` and complete it.

    Returns (meta, req) for further assertions.
    """
    sched = fix.scheduler
    if lora_request is not None:
        # lora_request must be set in the constructor so block_hashes
        # (computed in __init__) include the lora extra_keys.
        req = Request(
            request_id="req-lora",
            prompt_token_ids=list(range(num_blocks * BLOCK_SIZE + 1)),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            mm_features=None,
            lora_request=lora_request,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
    else:
        req = make_request(num_blocks=num_blocks)
    kv_blocks = _alloc_and_register(fix, req, num_blocks)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    block_ids = kv_blocks.get_block_ids()
    sched_out = make_scheduler_output(
        {req.request_id: num_blocks * BLOCK_SIZE},
        new_reqs={req.request_id: block_ids},
    )
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)
    return meta, req


# ---------------------------------------------------------------------------
# events disabled -> no events
# ---------------------------------------------------------------------------
def test_events_disabled_no_events() -> None:
    """When enable_kv_cache_events is False, take_events returns nothing."""
    fix = make_events_scheduler(enable_events=False)
    sched = fix.scheduler
    assert sched.enable_kv_cache_events is False
    _do_eager_store(fix, num_blocks=2)
    events = list(sched.take_events())
    assert events == [], f"expected no events, got {events}"


# ---------------------------------------------------------------------------
# eager store emits BlockStored with per-group metadata
# ---------------------------------------------------------------------------
def test_block_stored_per_group_metadata_full_attention() -> None:
    """BlockStored carries medium, group_idx, kind, sliding_window, block_size,
    and the correct block hash."""
    fix = make_events_scheduler()
    sched = fix.scheduler
    assert sched.kv_event_medium == MEDIUM_CPU
    _, req = _do_eager_store(fix, num_blocks=2)
    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 2
    ev = stored[0]
    assert ev.medium == MEDIUM_CPU
    assert ev.group_idx == 0
    assert ev.kv_cache_spec_kind == "full_attention"
    assert ev.kv_cache_spec_sliding_window is None
    assert ev.block_size == BLOCK_SIZE
    assert ev.locality == "LOCAL"
    expected_hash_0 = maybe_convert_block_hash(
        get_block_hash(make_block_hash_with_group_id(req.block_hashes[0], 0))
    )
    assert ev.block_hashes == [expected_hash_0]
    assert ev.parent_block_hash is None
    assert ev.token_ids == req.prompt_token_ids[0:BLOCK_SIZE]

    # Second block's parent_block_hash equals the first block's hash.
    ev1 = stored[1]
    expected_hash_1 = maybe_convert_block_hash(
        get_block_hash(make_block_hash_with_group_id(req.block_hashes[1], 0))
    )
    assert ev1.block_hashes == [expected_hash_1]
    assert ev1.parent_block_hash == expected_hash_0
    assert ev1.token_ids == req.prompt_token_ids[BLOCK_SIZE : 2 * BLOCK_SIZE]


def test_eager_store_lora_metadata() -> None:
    """BlockStored carries lora_id and lora_name from req.lora_request."""
    fix = make_events_scheduler()
    sched = fix.scheduler

    _do_eager_store(
        fix,
        num_blocks=1,
        lora_request=LoRARequest(
            lora_name="test-lora",
            lora_int_id=42,
            lora_path="/fake/lora/path",
        ),
    )

    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 1
    ev = stored[0]
    assert ev.lora_id == 42, f"expected lora_id=42, got {ev.lora_id}"
    assert ev.lora_name == "test-lora", (
        f"expected lora_name='test-lora', got {ev.lora_name!r}"
    )
    assert ev.extra_keys is not None, "expected extra_keys for lora request"
    assert any("test-lora" in keys for keys in ev.extra_keys), (
        f"expected 'test-lora' in extra_keys, got {ev.extra_keys}"
    )
    assert len(ev.token_ids) == BLOCK_SIZE


# ---------------------------------------------------------------------------
# disk mode -> MEDIUM_STORAGE
# ---------------------------------------------------------------------------
def test_disk_mode_medium_storage_on_block_stored() -> None:
    """disk_capacity_bytes>0 sets kv_event_medium=STORAGE on BlockStored."""
    fix = make_events_scheduler(
        disk_capacity_bytes=_BYTES_PER_BLOCK * 8,
        enable_events=True,
    )
    sched = fix.scheduler
    assert sched.kv_event_medium == MEDIUM_STORAGE
    _do_eager_store(fix, num_blocks=1)
    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 1
    assert stored[0].medium == MEDIUM_STORAGE
    assert stored[0].locality == "LOCAL"


def test_finished_eager_store_emits_all_storage_events() -> None:
    """Finished stores preserve and announce every registered hash."""
    fix = make_events_scheduler(
        disk_capacity_bytes=_BYTES_PER_BLOCK * 8,
        enable_events=True,
    )
    sched = fix.scheduler
    req = make_request(num_blocks=2)
    kv_blocks = _alloc_and_register(fix, req, num_blocks=2)
    gpu_blocks = kv_blocks.blocks[0]
    primary_hashes = []
    for gpu_block in gpu_blocks:
        assert gpu_block.block_hash is not None
        primary_hashes.append(gpu_block.block_hash)

    fine_grained_req = Request(
        request_id="req-finished-fine-grained-hash",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE // 2, sha256),
    )
    secondary_hash = make_block_hash_with_group_id(fine_grained_req.block_hashes[0], 0)
    fix.gpu_block_pool._insert_block_hash(
        secondary_hash, gpu_blocks[0], num_tokens=BLOCK_SIZE // 2
    )

    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched.request_finished_all_groups(req, kv_blocks.get_block_ids())
    meta = sched.build_connector_meta(make_scheduler_output({}))
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)

    primary_cpu_block = sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
        primary_hashes[0]
    )
    secondary_cpu_block = sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
        secondary_hash
    )
    assert primary_cpu_block is not None
    assert secondary_cpu_block is primary_cpu_block

    stored = [e for e in sched.take_events() if isinstance(e, BlockStored)]
    primary_event_hashes = [
        maybe_convert_block_hash(get_block_hash(block_hash))
        for block_hash in primary_hashes
    ]
    secondary_event_hash = maybe_convert_block_hash(get_block_hash(secondary_hash))
    by_hash = {event.block_hashes[0]: event for event in stored}
    assert set(by_hash) == {*primary_event_hashes, secondary_event_hash}
    assert (
        by_hash[primary_event_hashes[0]].token_ids == req.prompt_token_ids[:BLOCK_SIZE]
    )
    assert (
        by_hash[primary_event_hashes[1]].token_ids
        == req.prompt_token_ids[BLOCK_SIZE : 2 * BLOCK_SIZE]
    )
    assert by_hash[primary_event_hashes[1]].parent_block_hash == primary_event_hashes[0]
    secondary_event = by_hash[secondary_event_hash]
    assert secondary_event.medium == MEDIUM_STORAGE
    assert secondary_event.block_size == 0
    assert secondary_event.token_ids == []
    assert secondary_event.parent_block_hash is None

    sched.cpu_block_pool.evict_blocks({primary_cpu_block.block_id})
    removed = [
        event for event in sched.take_events() if isinstance(event, BlockRemoved)
    ]
    assert {event.block_hashes[0] for event in removed} == {
        primary_event_hashes[0],
        secondary_event_hash,
    }
    assert all(event.medium == MEDIUM_STORAGE for event in removed)


def test_secondary_hash_block_stored_metadata() -> None:
    """Secondary hashes carry their own token range and block size."""
    fix = make_events_scheduler(
        disk_capacity_bytes=_BYTES_PER_BLOCK * 8,
        enable_events=True,
    )
    sched = fix.scheduler
    sched.hash_block_size = BLOCK_SIZE // 2
    req = Request(
        request_id="req-secondary-hash",
        prompt_token_ids=list(range(BLOCK_SIZE + 1)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE // 2, sha256),
    )
    req.num_computed_tokens = BLOCK_SIZE
    gpu_block = fix.gpu_block_pool.get_new_blocks(1)[0]
    primary_hash = make_block_hash_with_group_id(req.block_hashes[1], 0)
    secondary_hash = make_block_hash_with_group_id(req.block_hashes[0], 0)
    fix.gpu_block_pool._insert_block_hash(
        primary_hash, gpu_block, num_tokens=BLOCK_SIZE
    )
    fix.gpu_block_pool._insert_block_hash(
        secondary_hash, gpu_block, num_tokens=BLOCK_SIZE // 2
    )
    state = StoreRequestState(
        request=req,
        block_ids=([gpu_block.block_id],),
        num_stored_blocks=[0],
    )

    gpu_block_ids, advanced_per_group, block_meta = sched._select_eager_blocks_to_store(
        state,
        ([gpu_block.block_id],),
    )
    assert gpu_block_ids == [gpu_block.block_id]
    assert advanced_per_group == [1]
    assert block_meta is not None
    assert set(block_meta[0]) == {primary_hash, secondary_hash}

    cpu_block = sched.cpu_block_pool.get_new_blocks(1)[0]
    sched._process_store_completion(
        gpu_block_ids,
        [cpu_block.block_id],
        block_meta,
    )

    primary_event_hash = maybe_convert_block_hash(get_block_hash(primary_hash))
    secondary_event_hash = maybe_convert_block_hash(get_block_hash(secondary_hash))
    stored = [event for event in sched.take_events() if isinstance(event, BlockStored)]
    by_hash = {event.block_hashes[0]: event for event in stored}
    assert set(by_hash) == {primary_event_hash, secondary_event_hash}
    assert by_hash[primary_event_hash].block_size == BLOCK_SIZE
    assert by_hash[primary_event_hash].token_ids == req.prompt_token_ids[:BLOCK_SIZE]
    assert by_hash[secondary_event_hash].block_size == BLOCK_SIZE // 2
    assert (
        by_hash[secondary_event_hash].token_ids
        == req.prompt_token_ids[: BLOCK_SIZE // 2]
    )

    primary_cpu_block = sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
        primary_hash
    )
    secondary_cpu_block = sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
        secondary_hash
    )
    assert primary_cpu_block is not None
    assert secondary_cpu_block is primary_cpu_block


# ---------------------------------------------------------------------------
# BlockRemoved relabeled to correct medium
# ---------------------------------------------------------------------------
def test_block_removed_relabeled_medium_cpu() -> None:
    """Evicted BlockRemoved (emitted as GPU by BlockPool) is relabeled CPU."""
    # 5 total = 4 usable (null_block takes 1). Fill 4, then store 2 more to
    # evict 2 LRU blocks, producing BlockRemoved events.
    fix = make_events_scheduler(num_cpu_blocks=5, num_gpu_blocks=16)
    sched = fix.scheduler

    # Fill CPU with 4 blocks (2 + 2).
    _do_eager_store(fix, num_blocks=2)
    # Drain store events so they don't clutter the eviction check.
    list(sched.take_events())
    _do_eager_store(fix, num_blocks=2)
    list(sched.take_events())

    # Now store 2 more -> evicts 2 LRU blocks -> BlockRemoved emitted by pool
    # with MEDIUM_GPU, then relabeled by scheduler.take_events().
    _do_eager_store(fix, num_blocks=2)
    events = list(sched.take_events())

    removed = [e for e in events if isinstance(e, BlockRemoved)]
    assert len(removed) >= 2, (
        f"expected >=2 BlockRemoved from eviction, got {len(removed)}"
    )
    for ev in removed:
        assert ev.medium == MEDIUM_CPU, (
            f"BlockRemoved should be relabeled CPU, got {ev.medium}"
        )
        assert ev.locality == "LOCAL", (
            f"BlockRemoved should be LOCAL, got {ev.locality}"
        )
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 2


# ---------------------------------------------------------------------------
# lazy store emits BlockStored
# ---------------------------------------------------------------------------
def test_lazy_store_emits_block_stored() -> None:
    """Lazy-mode store completion also emits BlockStored with MEDIUM_CPU."""
    fix = make_events_scheduler(num_cpu_blocks=8, num_gpu_blocks=8, lazy=True)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    # Allocate, hash, free -> hashed blocks in free queue.
    req = make_request(num_blocks=2)
    gpu_blocks = []
    for blk in gpu_pool.get_new_blocks(2):
        gpu_blocks.append(blk)
    gpu_pool.cache_full_blocks(
        request=req,
        blocks=gpu_blocks,
        num_cached_blocks=0,
        num_full_blocks=2,
        block_size=BLOCK_SIZE,
        kv_cache_group_id=0,
    )
    gpu_pool.free_blocks(gpu_blocks)

    # Push hashed blocks to LRU head (8 total - 1 null = 7 usable; 2 freed
    # hashed + 5 other free -> consume 5 fillers).
    fillers = gpu_pool.get_new_blocks(5)

    meta = sched.build_connector_meta(make_scheduler_output({}))
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)
    gpu_pool.free_blocks(fillers)

    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 2, f"expected 2 BlockStored, got {len(stored)}"
    for ev in stored:
        assert ev.medium == MEDIUM_CPU
        assert ev.group_idx == 0
        assert ev.kv_cache_spec_kind == "full_attention"
        assert ev.locality == "LOCAL"
        assert ev.token_ids == []
        assert ev.parent_block_hash is None


# ---------------------------------------------------------------------------
# cp_world_size scales block_size for non-Mamba
# ---------------------------------------------------------------------------
def test_cp_world_size_scales_block_size() -> None:
    """block_size on BlockStored = spec.block_size * cp_world_size (FA)."""
    fix = make_events_scheduler(dcp_world_size=2)
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool
    vbs = BLOCK_SIZE * 2
    assert sched.cp_world_size == 2

    req = _make_cp_request(num_blocks=1, virtual_block_size=vbs)
    gpu_blocks = _allocate_cp_gpu_blocks(gpu_pool, req, 1, vbs)
    kv_blocks = KVCacheBlocks(blocks=(gpu_blocks,))
    req.num_computed_tokens = vbs
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output(
        {req.request_id: vbs},
        new_reqs={req.request_id: kv_blocks.get_block_ids()},
    )
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)

    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 1
    # spec.block_size=BLOCK_SIZE, cp_world_size=2 -> BLOCK_SIZE * 2
    assert stored[0].block_size == BLOCK_SIZE * 2
    assert stored[0].locality == "LOCAL"


# ---------------------------------------------------------------------------
# Mamba block_size NOT scaled by cp_world_size
# ---------------------------------------------------------------------------
def test_mamba_block_size_not_scaled() -> None:
    """Mamba group: block_size = spec.block_size (no cp scaling).

    With cp_world_size=2, scheduler_block_size=hash_block_size=32. The
    coordinator constraints force Mamba spec.block_size=32 (must divide
    scheduler_block_size and be a multiple of hash_block_size). The event
    block_size for Mamba must be 32 (spec.block_size * 1), NOT 64
    (spec.block_size * cp_world_size) -- that is the bug this guards.
    """
    kv_cfg = _make_mixed_kv_cache_config(
        num_blocks=16, with_mamba=True, mamba_block_size=32
    )
    fix = make_events_scheduler(
        num_cpu_blocks=8,
        num_gpu_blocks=16,
        kv_cache_config=kv_cfg,
        dcp_world_size=2,
        num_groups=2,
    )
    sched = fix.scheduler

    cpu_pool = sched.cpu_block_pool
    gpu_pool = fix.gpu_block_pool

    cpu_blk = cpu_pool.get_new_blocks(1)[0]
    gpu_blk = gpu_pool.get_new_blocks(1)[0]

    raw_hash = make_request(num_blocks=1).block_hashes[0]
    mamba_hash = make_block_hash_with_group_id(raw_hash, group_id=1)
    gpu_blk._block_hash = mamba_hash  # type: ignore[attr-defined]

    sched._process_store_completion(
        gpu_block_ids=[gpu_blk.block_id],
        cpu_block_ids=[cpu_blk.block_id],
    )

    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 1
    ev = stored[0]
    assert ev.group_idx == 1
    assert ev.kv_cache_spec_kind == "mamba"
    assert ev.kv_cache_spec_sliding_window is None
    assert ev.locality == "LOCAL"
    assert ev.block_size == 32, (
        f"Mamba block_size should be unscaled (32), got {ev.block_size}"
    )


# ---------------------------------------------------------------------------
# multi-group FA+SW stores carry correct group_idx and kind
# ---------------------------------------------------------------------------
def test_multi_group_fa_block_stored_group_idx() -> None:
    """Two distinct groups (FA + SW): each BlockStored carries its group_idx
    and kind. HybridKVCacheCoordinator requires >=2 distinct spec groups, so
    we use FA (group 0) + SlidingWindow (group 1)."""
    kv_cfg = _make_mixed_kv_cache_config(num_blocks=16, sliding_window=32)
    fix = make_events_scheduler(
        num_cpu_blocks=8,
        num_gpu_blocks=16,
        kv_cache_config=kv_cfg,
        num_groups=2,
    )
    sched = fix.scheduler

    cpu_pool = sched.cpu_block_pool
    gpu_pool = fix.gpu_block_pool
    raw_hash = make_request(num_blocks=1).block_hashes[0]

    # Store one block for each group via direct _process_store_completion.
    for gidx in (0, 1):
        cpu_blk = cpu_pool.get_new_blocks(1)[0]
        gpu_blk = gpu_pool.get_new_blocks(1)[0]
        gpu_blk._block_hash = make_block_hash_with_group_id(  # type: ignore
            raw_hash, group_id=gidx
        )
        sched._process_store_completion(
            gpu_block_ids=[gpu_blk.block_id],
            cpu_block_ids=[cpu_blk.block_id],
        )

    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 2
    by_group = {ev.group_idx: ev for ev in stored}
    assert set(by_group) == {0, 1}
    assert by_group[0].kv_cache_spec_kind == "full_attention"
    assert by_group[0].kv_cache_spec_sliding_window is None
    assert by_group[1].kv_cache_spec_kind == "sliding_window"
    assert by_group[1].kv_cache_spec_sliding_window == 32
    for ev in stored:
        assert ev.locality == "LOCAL"


def test_disk_mode_block_removed_relabeled_storage() -> None:
    """Disk mode: evicted BlockRemoved relabeled to MEDIUM_STORAGE."""
    fix = make_events_scheduler(
        num_cpu_blocks=5,
        num_gpu_blocks=16,
        disk_capacity_bytes=_BYTES_PER_BLOCK * 5,
    )
    sched = fix.scheduler
    assert sched.kv_event_medium == MEDIUM_STORAGE

    # Fill disk with 4 blocks (2 + 2).
    _do_eager_store(fix, num_blocks=2)
    list(sched.take_events())
    _do_eager_store(fix, num_blocks=2)
    list(sched.take_events())

    # Store 2 more -> evicts 2 LRU blocks -> BlockRemoved relabeled STORAGE.
    _do_eager_store(fix, num_blocks=2)
    events = list(sched.take_events())

    removed = [e for e in events if isinstance(e, BlockRemoved)]
    assert len(removed) >= 2, (
        f"expected >=2 BlockRemoved from eviction, got {len(removed)}"
    )
    for ev in removed:
        assert ev.medium == MEDIUM_STORAGE, (
            f"BlockRemoved should be relabeled STORAGE, got {ev.medium}"
        )
        assert ev.locality == "LOCAL", (
            f"BlockRemoved should be LOCAL, got {ev.locality}"
        )
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 2
    for ev in stored:
        assert ev.medium == MEDIUM_STORAGE


def test_mamba_dcp_metadata_guard() -> None:
    """Mamba+DCP: token_ids guard emits [] when capture and emission sizes differ.

    The metadata capture at _prepare_eager_store_specs uses
    g_block_size = spec.block_size * cp_world_size = 32 * 2 = 64 to slice
    tokens. The event emission uses block_size = spec.block_size * 1 = 32
    for Mamba (SSM state is not CP-sharded). The guard in
    _process_store_completion detects len(token_ids) != block_size and emits
    token_ids=[] instead of misleading data. When #49962 fixes the capture
    path, the guard stops triggering and correct token_ids flow through.
    """
    vbs = BLOCK_SIZE * 2
    kv_cfg = _make_mixed_kv_cache_config(
        num_blocks=16, with_mamba=True, mamba_block_size=32
    )
    fix = make_events_scheduler(
        num_cpu_blocks=8,
        num_gpu_blocks=16,
        kv_cache_config=kv_cfg,
        dcp_world_size=2,
        num_groups=2,
    )
    sched = fix.scheduler
    gpu_pool = fix.gpu_block_pool

    # Mamba g_block_size = 32 * 2 = 64, so 2 Mamba blocks need 128 computed
    # tokens to be "ready"; FA g_block_size = 16 * 2 = 32 needs only 64.
    req = _make_cp_request(num_blocks=4, virtual_block_size=vbs)
    fa_blocks = _allocate_cp_gpu_blocks(gpu_pool, req, 2, vbs, group_id=0)
    mamba_blocks = _allocate_cp_gpu_blocks(gpu_pool, req, 2, vbs, group_id=1)
    kv_blocks = KVCacheBlocks(blocks=(fa_blocks, mamba_blocks))
    req.num_computed_tokens = 4 * vbs
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)

    block_ids = kv_blocks.get_block_ids()
    sched_out = make_scheduler_output(
        {req.request_id: 4 * vbs},
        new_reqs={req.request_id: block_ids},
    )
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0
    simulate_store_completion(sched, meta.store_event)

    events = list(sched.take_events())
    stored = [e for e in events if isinstance(e, BlockStored)]
    assert len(stored) == 4, (
        f"expected 4 BlockStored (2 FA + 2 Mamba), got {len(stored)}"
    )

    by_group: dict[int, list[BlockStored]] = {}
    for ev in stored:
        by_group.setdefault(ev.group_idx, []).append(ev)

    # FA group: capture size (64) == emission size (64) → token_ids present.
    fa_events = by_group[0]
    assert len(fa_events) == 2
    for ev in fa_events:
        assert ev.block_size == vbs
        assert len(ev.token_ids) == vbs, (
            f"FA token_ids should be {vbs}, got {len(ev.token_ids)}"
        )

    # Mamba group: capture size (64) != emission size (32) → guard emits [].
    mamba_events = by_group[1]
    assert len(mamba_events) == 2
    for ev in mamba_events:
        assert ev.block_size == 32
        assert ev.token_ids == [], (
            f"Mamba token_ids should be [] (guard), got {ev.token_ids}"
        )
        assert ev.parent_block_hash is None, (
            "Mamba parent_block_hash should be None (guard)"
        )
        assert ev.extra_keys is None, "Mamba extra_keys should be None (guard)"
