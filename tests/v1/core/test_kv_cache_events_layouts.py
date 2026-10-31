# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV event payloads for MLA, GDN and dense partial-hit cache layouts.

The existing event tests cover full attention, sliding window and the bare
`BlockPool.cache_partial_block` primitive. These pin what a consumer sees for
the remaining P1 layouts in #63149: the emitted spec kind, the hash chain that
identifies the reusable prefix, and the token span each event describes.

Deliberately not pinned here: the zero-hash `BlockStored` a recurrent group
emits for its own incremental chunks (the sparse hash-to-token mapping tracked
by #44488), off-grid single-chunk prefills, and DCP > 1. All cases below are
DCP-1 and drive the manager directly rather than through the scheduler.
"""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.distributed.kv_events import MEDIUM_GPU, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import (
    ExternalBlockHash,
    get_request_block_hasher,
    init_none_hash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
HASH_BLOCK_SIZE = 8


@pytest.fixture(autouse=True)
def _init_hash_fn():
    init_none_hash(sha256)


def _full_attention(block_size: int = BLOCK_SIZE) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def _mla(block_size: int = BLOCK_SIZE) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )


def _gdn(
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.GDN_ATTN,
) -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_type=mamba_type,
        mamba_cache_mode="align",
    )


def _make_manager(*specs: KVCacheSpec, hash_block_size: int, num_blocks: int = 64):
    """Build a manager for a DCP-1 config, one group per spec."""
    return make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
            ],
        ),
        max_model_len=8192,
        enable_caching=True,
        enable_kv_cache_events=True,
        hash_block_size=hash_block_size,
    )


def _make_full_report_request(
    request_id: str, prompt_token_ids: list[int], hash_block_size: int
) -> Request:
    sampling_params = SamplingParams(
        max_tokens=1, extra_args={"kv_cache_report_mode": "full"}
    )
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def _block_stored_events(events) -> list[BlockStored]:
    return [event for event in events if isinstance(event, BlockStored)]


def _boundary_hash(
    req: Request, num_tokens: int, hash_block_size: int
) -> ExternalBlockHash:
    """External hash of the prefix ending at ``num_tokens``."""
    return maybe_convert_block_hash(req.block_hashes[num_tokens // hash_block_size - 1])


def _assert_dense_block_stored(
    event: BlockStored,
    *,
    req: Request,
    hash_block_size: int,
    group_idx: int,
    kind: str,
    token_start: int,
    token_end: int,
    parent_token_end: int | None,
    block_size: int = BLOCK_SIZE,
) -> None:
    """Check one dense ``BlockStored`` against ``req``'s hash chain.

    The event must cover tokens ``[token_start, token_end)``: one hash per
    cache block ending at those positions, the matching token slice, and a
    parent rooted at ``parent_token_end`` (or no parent at the prefix head).
    """
    assert event.group_idx == group_idx
    assert event.kv_cache_spec_kind == kind
    assert event.kv_cache_spec_sliding_window is None
    assert event.block_size == block_size
    assert event.medium == MEDIUM_GPU
    assert event.lora_id is None
    assert event.lora_name is None
    assert event.token_ids == req.all_token_ids[token_start:token_end]
    assert event.block_hashes == [
        _boundary_hash(req, num_tokens, hash_block_size)
        for num_tokens in range(token_start + block_size, token_end + 1, block_size)
    ]
    assert event.extra_keys is not None
    assert len(event.extra_keys) == len(event.block_hashes)
    if parent_token_end is None:
        assert event.parent_block_hash is None
    else:
        assert event.parent_block_hash == _boundary_hash(
            req, parent_token_end, hash_block_size
        )


def _assert_checkpoint_block_stored(
    event: BlockStored,
    *,
    req: Request,
    hash_block_size: int,
    group_idx: int,
    token_end: int,
    parent_token_end: int | None = None,
) -> None:
    """Check a recurrent group's checkpoint event.

    A recurrent state is a single checkpoint, not a block table: one hash
    identifies the state at ``token_end`` however many tokens it covers, so the
    hash count is smaller than the token span by design.
    """
    assert event.group_idx == group_idx
    assert event.kv_cache_spec_kind == KVCacheSpecKind.MAMBA.value
    assert event.block_size == BLOCK_SIZE
    assert event.medium == MEDIUM_GPU
    assert event.token_ids == req.all_token_ids[:token_end]
    assert event.block_hashes == [_boundary_hash(req, token_end, hash_block_size)]
    assert event.extra_keys == [None]
    if parent_token_end is None:
        assert event.parent_block_hash is None
    else:
        assert event.parent_block_hash == _boundary_hash(
            req, parent_token_end, hash_block_size
        )


def test_mla_store_and_reuse_events():
    """An MLA group reports its kind and the exact reusable prefix."""
    manager = _make_manager(_mla(), hash_block_size=BLOCK_SIZE)
    producer = make_request("producer", list(range(64)), BLOCK_SIZE, sha256)
    assert manager.allocate_slots(producer, 64) is not None

    [stored] = _block_stored_events(manager.take_events())
    _assert_dense_block_stored(
        stored,
        req=producer,
        hash_block_size=BLOCK_SIZE,
        group_idx=0,
        kind=KVCacheSpecKind.MLA_ATTENTION.value,
        token_start=0,
        token_end=64,
        parent_token_end=None,
    )

    manager.free(producer)
    manager.new_step_starts()

    # Reusing an identical prompt recomputes the last block for its logits, so
    # the reported prefix ends one block short.
    reuse = make_request("reuse", list(range(64)), BLOCK_SIZE, sha256)
    _, num_computed, _ = manager.get_computed_blocks(reuse)
    assert num_computed == 64 - BLOCK_SIZE

    # Extending the hit with fresh tokens continues the chain from token 64.
    extended = make_request("extended", list(range(96)), BLOCK_SIZE, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(extended)
    assert num_computed == 64
    assert manager.allocate_slots(extended, 96, num_computed, blocks) is not None

    [stored] = _block_stored_events(manager.take_events())
    _assert_dense_block_stored(
        stored,
        req=extended,
        hash_block_size=BLOCK_SIZE,
        group_idx=0,
        kind=KVCacheSpecKind.MLA_ATTENTION.value,
        token_start=64,
        token_end=96,
        parent_token_end=64,
    )


def test_mla_full_report_publishes_the_reused_prefix():
    """``kv_cache_report_mode="full"`` republishes the MLA hit rooted at None."""
    manager = _make_manager(_mla(), hash_block_size=BLOCK_SIZE)
    producer = make_request("producer", list(range(64)), BLOCK_SIZE, sha256)
    assert manager.allocate_slots(producer, 64) is not None
    manager.take_events()
    manager.free(producer)
    manager.new_step_starts()

    consumer = _make_full_report_request("consumer", list(range(64)), BLOCK_SIZE)
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 64 - BLOCK_SIZE

    # The republished chain is the consumer's own, rooted at the head.
    [reported] = _block_stored_events(manager.take_events())
    _assert_dense_block_stored(
        reported,
        req=consumer,
        hash_block_size=BLOCK_SIZE,
        group_idx=0,
        kind=KVCacheSpecKind.MLA_ATTENTION.value,
        token_start=0,
        token_end=num_computed,
        parent_token_end=None,
    )
    assert manager.allocate_slots(consumer, 64, num_computed, blocks) is not None


@pytest.mark.parametrize(
    "mamba_type",
    [MambaAttentionBackendEnum.MAMBA2, MambaAttentionBackendEnum.GDN_ATTN],
)
def test_recurrent_group_events_report_one_state_hash_per_reusable_span(
    mamba_type: MambaAttentionBackendEnum,
):
    """GDN and Mamba2 report the same shape: one checkpoint hash per span."""
    manager = _make_manager(
        _full_attention(), _gdn(mamba_type), hash_block_size=BLOCK_SIZE
    )
    producer = make_request("producer", list(range(64)), BLOCK_SIZE, sha256)
    assert manager.allocate_slots(producer, 64) is not None

    attention_event, mamba_event = _block_stored_events(manager.take_events())

    _assert_dense_block_stored(
        attention_event,
        req=producer,
        hash_block_size=BLOCK_SIZE,
        group_idx=0,
        kind=KVCacheSpecKind.FULL_ATTENTION.value,
        token_start=0,
        token_end=64,
        parent_token_end=None,
    )
    _assert_checkpoint_block_stored(
        mamba_event, req=producer, hash_block_size=BLOCK_SIZE, group_idx=1, token_end=64
    )

    manager.free(producer)
    manager.new_step_starts()

    consumer = make_request("consumer", list(range(96)), BLOCK_SIZE, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == 64
    assert manager.allocate_slots(consumer, 96, num_computed, blocks) is not None

    # Only the dense group is asserted: the recurrent group's chunk event has
    # an empty hash list, a shape #44488 is still defining.
    attention_event = next(
        event
        for event in _block_stored_events(manager.take_events())
        if event.group_idx == 0
    )
    _assert_dense_block_stored(
        attention_event,
        req=consumer,
        hash_block_size=BLOCK_SIZE,
        group_idx=0,
        kind=KVCacheSpecKind.FULL_ATTENTION.value,
        token_start=64,
        token_end=96,
        parent_token_end=64,
    )


def test_dense_partial_hit_events_describe_the_reusable_prefix():
    """A sub-block tail is reported at hash granularity and is reusable."""
    manager = _make_manager(_full_attention(), _gdn(), hash_block_size=HASH_BLOCK_SIZE)
    assert manager.coordinator.enable_partial_hash_hits

    # Mirrors how the scheduler splits this prompt: a full 64-token chunk and
    # then the 8-token tail ending on a hash boundary.
    tail_end = 64 + HASH_BLOCK_SIZE
    producer = make_request(
        "producer", list(range(tail_end + 1)), HASH_BLOCK_SIZE, sha256
    )
    assert manager.allocate_slots(producer, 64) is not None
    _block_stored_events(manager.take_events())
    producer.num_computed_tokens = 64
    manager.new_step_starts()
    assert manager.allocate_slots(producer, HASH_BLOCK_SIZE) is not None

    attention_tail, mamba_tail = _block_stored_events(manager.take_events())

    for event, group_idx, kind in (
        (attention_tail, 0, KVCacheSpecKind.FULL_ATTENTION.value),
        (mamba_tail, 1, KVCacheSpecKind.MAMBA.value),
    ):
        assert event.group_idx == group_idx
        assert event.kv_cache_spec_kind == kind
        # The tail is reported at hash granularity, not at the cache block size.
        assert event.block_size == HASH_BLOCK_SIZE
        assert event.token_ids == producer.all_token_ids[64:tail_end]
        assert event.parent_block_hash == _boundary_hash(producer, 64, HASH_BLOCK_SIZE)
        # The last hash identifies the tail boundary; a producer may publish the
        # intermediate boundaries too (see #62195), which changes nothing about
        # where the prefix ends.
        assert event.block_hashes[-1] == _boundary_hash(
            producer, tail_end, HASH_BLOCK_SIZE
        )

    manager.free(producer)
    manager.new_step_starts()

    # The reported tail is exactly what a sibling can resume from.
    consumer = make_request(
        "consumer", list(range(tail_end + 1)), HASH_BLOCK_SIZE, sha256
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        consumer.block_hashes, consumer.num_tokens - 1
    )
    assert per_group_hits == (tail_end, tail_end)
    _, num_computed, _ = manager.get_computed_blocks(consumer)
    assert num_computed == tail_end


def test_attention_partial_tail_does_not_imply_a_reusable_hybrid_prefix():
    """An attention tail event past the last recurrent checkpoint is not reusable.

    A 64-token prompt leaves the full-attention group with a sub-block tail
    ending at token 56, while the recurrent group only has a state at 64. The
    tail event is published, but a sibling sharing those 56 tokens does not
    match the recurrent group, so the reconciled hit is empty. The event stream
    therefore under-describes nothing but over-promises per group: a consumer
    must intersect attention coverage with recurrent checkpoints rather than
    trusting the attention tail on its own.
    """
    manager = _make_manager(_full_attention(), _gdn(), hash_block_size=HASH_BLOCK_SIZE)
    assert manager.coordinator.enable_partial_hash_hits

    prompt_tokens = 4 * BLOCK_SIZE
    tail_end = prompt_tokens - HASH_BLOCK_SIZE
    producer = make_request(
        "producer", list(range(prompt_tokens)), HASH_BLOCK_SIZE, sha256
    )
    assert manager.allocate_slots(producer, prompt_tokens) is not None

    attention_tail = next(
        event
        for event in _block_stored_events(manager.take_events())
        if event.group_idx == 0 and event.block_size == HASH_BLOCK_SIZE
    )
    _assert_dense_block_stored(
        attention_tail,
        req=producer,
        hash_block_size=HASH_BLOCK_SIZE,
        group_idx=0,
        kind=KVCacheSpecKind.FULL_ATTENTION.value,
        token_start=tail_end - HASH_BLOCK_SIZE,
        token_end=tail_end,
        parent_token_end=tail_end - HASH_BLOCK_SIZE,
        block_size=HASH_BLOCK_SIZE,
    )

    manager.free(producer)
    manager.new_step_starts()

    sharing_tail = make_request(
        "sharing_tail",
        list(range(tail_end)) + list(range(900, 916)),
        HASH_BLOCK_SIZE,
        sha256,
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        sharing_tail.block_hashes, sharing_tail.num_tokens - 1
    )
    assert per_group_hits == (tail_end, 0)
    _, num_computed, _ = manager.get_computed_blocks(sharing_tail)
    assert num_computed == 0

    # The same sibling starting at the checkpoint is a full hybrid hit.
    sharing_checkpoint = make_request(
        "sharing_checkpoint",
        list(range(prompt_tokens)) + list(range(900, 916)),
        HASH_BLOCK_SIZE,
        sha256,
    )
    _, per_group_hits = manager.coordinator.find_longest_cache_hit_per_group(
        sharing_checkpoint.block_hashes, sharing_checkpoint.num_tokens - 1
    )
    assert per_group_hits == (prompt_tokens, prompt_tokens)
    _, num_computed, _ = manager.get_computed_blocks(sharing_checkpoint)
    assert num_computed == prompt_tokens
