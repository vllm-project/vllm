# SPDX-License-Identifier: Apache-2.0
"""The eager scratch pool's template families must not alias each other.

The pool used to size its aux storage with `max()` across the FP4-indexer,
global-topk and compressor families and carve all three from offset 0, on the
reasoning that they are temporally disjoint (FP4 is C4-only, compressor is
C128-only, global mapping runs after FP4). That is an assumption about ordering,
and it does not hold once the attention eager break runs the indexer and
compressor on parallel aux streams: under concurrent mixed prefill+decode one
consumer clobbers another mid-read, and whichever request is co-batched gets
garbled top-k indices -> attention over the wrong KV -> corrupted output.

Reported with a clean bisect on vllm-project/vllm#41834 (TP=4 SM12x, 1M
context): pool active 7/7 rounds corrupt, same build with the pool disabled 0/2,
pre-pool build 0/2.

These tests run on CPU -- the defect is in address arithmetic, not in any
kernel, so it needs no GPU and no model.
"""

import pytest
import torch

from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool

# Small but structurally faithful: index_q_head_dim must divide the MXFP4 block
# size, and index_topk drives the global-mapping family.
POOL_KWARGS = dict(
    max_num_tokens=64,
    padded_q_heads=16,
    q_head_dim=576,
    index_q_heads=8,
    index_q_head_dim=128,
    index_topk=512,
    device="cpu",
)


def _byte_range(pool: DeepseekV4EagerScratchPool, t: torch.Tensor) -> tuple[int, int]:
    """Half-open [start, end) of ``t`` inside the pool's storage, in bytes."""
    base = pool._storage.data_ptr()
    start = t.data_ptr() - base
    return start, start + t.numel() * t.element_size()


def _families(pool: DeepseekV4EagerScratchPool) -> dict[str, list[torch.Tensor]]:
    return {
        "fp4": list(pool._fp4_template),
        "global": list(pool._global_template),
        "compressor": [pool._compressor_template],
    }


def test_template_families_do_not_overlap():
    """The bug, stated directly: no two families may share a byte."""
    pool = DeepseekV4EagerScratchPool(**POOL_KWARGS, allocate_q=False)
    fams = _families(pool)

    spans = {
        name: (
            min(_byte_range(pool, t)[0] for t in tensors),
            max(_byte_range(pool, t)[1] for t in tensors),
        )
        for name, tensors in fams.items()
    }

    names = sorted(spans)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            a0, a1 = spans[a]
            b0, b1 = spans[b]
            assert a1 <= b0 or b1 <= a0, (
                f"{a} [{a0},{a1}) overlaps {b} [{b0},{b1}) -- the families alias, "
                "which is the #41834 corruption"
            )


def test_writing_one_family_does_not_disturb_the_others():
    """Behavioural counterpart: aliasing is observable, so observe it.

    Fills each family with a distinct sentinel in turn and checks the others
    still read back what they were given. Under the old `max()` sizing the
    second fill would visibly rewrite the first.
    """
    pool = DeepseekV4EagerScratchPool(**POOL_KWARGS, allocate_q=False)
    fams = _families(pool)

    for i, tensors in enumerate(fams.values(), start=1):
        for t in tensors:
            t.fill_(i if t.dtype != torch.uint8 else i % 256)

    for i, (name, tensors) in enumerate(fams.items(), start=1):
        want = i if tensors[0].dtype != torch.uint8 else i % 256
        for t in tensors:
            expected = i if t.dtype != torch.uint8 else i % 256
            assert bool((t == expected).all()), (
                f"{name} was clobbered by a later family's write "
                f"(expected {expected}); the families alias"
            )
        del want


def test_storage_is_sum_not_max_of_the_families():
    """Guards the sizing itself, so a future `max()` cannot creep back in."""
    pool = DeepseekV4EagerScratchPool(**POOL_KWARGS, allocate_q=False)
    total = sum(
        max(_byte_range(pool, t)[1] for t in tensors)
        - min(_byte_range(pool, t)[0] for t in tensors)
        for tensors in _families(pool).values()
    )
    assert pool._storage.numel() >= total, (
        f"storage {pool._storage.numel()} B cannot hold the three families "
        f"({total} B); it is sized with max() rather than sum()"
    )


@pytest.mark.parametrize("allocate_q", [True, False])
def test_q_buffer_is_separate_storage(allocate_q: bool):
    """The Q buffer is its own allocation and must never fall inside the pool."""
    pool = DeepseekV4EagerScratchPool(**POOL_KWARGS, allocate_q=allocate_q)
    if not allocate_q:
        assert pool._q is None
        with pytest.raises(RuntimeError):
            pool.q_out(1)
        return
    lo = pool._storage.data_ptr()
    hi = lo + pool._storage.numel()
    assert not (lo <= pool._q.data_ptr() < hi)
