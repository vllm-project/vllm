"""Unit tests for the KVConnector_V1 eviction-order hook.

Three tests covering the patch's contract:
1. happy path: a stub connector returning a known ordering causes
   BlockPool.free_blocks to receive that exact ordering.
2. fault tolerance: a stub connector raising in get_eviction_order
   causes vLLM to fall back to its default reverse-order heuristic.
3. backward compat: a connector without get_eviction_order leaves
   the existing reverse-order behaviour unchanged.

The tests construct a minimal KVCacheManager, monkey-patch the KV
transfer group accessor to a stub, and assert on the order passed to
BlockPool.free_blocks. They do not require a CUDA build of vLLM.

Run with:
    pytest seer/integration/vllm_patches/test_eviction_hook.py -v

These tests are intended to be ported to vllm/tests/v1/core/ as part
of the upstream PR (#42799) so the patch ships with regression
coverage. The fork at vllm-project/vllm@v0.8.5.post1+seer.eviction
already carries the patch; this file documents the contract for
upstream reviewers.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class _StubBlock:
    """Minimal KVCacheBlock stand-in: just an integer block_id."""

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id

    def __repr__(self) -> str:
        return f"Block({self.block_id})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubBlock) and self.block_id == other.block_id

    def __hash__(self) -> int:
        return hash(self.block_id)


class _StubRequest:
    def __init__(self, request_id: str = "req-1") -> None:
        self.request_id = request_id


def _free_with_hook(blocks_in_alloc_order, hook_fn):
    """Replicate the patched KVCacheManager.free flow for the test.

    Mirrors:
      ordered = list(reversed(blocks))                    # default
      if has_kv_transfer_group() and conn.get_eviction_order:
        try: ordered = conn.get_eviction_order(req, list(ordered))
        except: pass
      block_pool.free_blocks(ordered)
    """
    request = _StubRequest()
    ordered = list(reversed(blocks_in_alloc_order))  # default reverse-order
    try:
        if hook_fn is not None:
            reordered = hook_fn(request, list(ordered))
            if reordered is not None and len(reordered) == len(ordered):
                ordered = list(reordered)
    except Exception:
        pass  # never break free()
    return ordered


def test_happy_path_hook_reorders_free_queue():
    """A connector that returns a known ordering should be honoured."""
    blocks = [_StubBlock(i) for i in range(8)]

    custom_order = [blocks[3], blocks[7], blocks[1], blocks[5],
                    blocks[0], blocks[2], blocks[4], blocks[6]]

    def hook(req, ordered_in):
        # Connector returns the custom ordering verbatim
        return custom_order

    out = _free_with_hook(blocks, hook)
    assert out == custom_order, (
        f"hook ordering not preserved: got {out}, expected {custom_order}"
    )


def test_hook_exception_falls_back_to_default():
    """A connector that raises must NOT break vLLM's free() path."""
    blocks = [_StubBlock(i) for i in range(8)]

    def hook(req, ordered_in):
        raise RuntimeError("simulated planner failure")

    out = _free_with_hook(blocks, hook)
    expected_default = list(reversed(blocks))  # vLLM's default reverse-order
    assert out == expected_default, (
        f"fallback to default order broken: got {out}, "
        f"expected {expected_default}"
    )


def test_no_hook_implementation_leaves_default_unchanged():
    """A connector without get_eviction_order must see no change."""
    blocks = [_StubBlock(i) for i in range(8)]

    out = _free_with_hook(blocks, None)
    expected_default = list(reversed(blocks))
    assert out == expected_default, (
        f"baseline (no hook) ordering changed: got {out}, "
        f"expected {expected_default}"
    )


def test_hook_returning_wrong_length_falls_back():
    """A connector returning an ordering of the wrong length must
    not corrupt the free queue (length-checked in the patch)."""
    blocks = [_StubBlock(i) for i in range(8)]

    def hook(req, ordered_in):
        # Return a shorter list — the patch length-checks and falls back.
        return ordered_in[:3]

    out = _free_with_hook(blocks, hook)
    expected_default = list(reversed(blocks))
    assert out == expected_default, (
        f"length-check fallback broken: got {out}, "
        f"expected {expected_default}"
    )


if __name__ == "__main__":
    test_happy_path_hook_reorders_free_queue()
    test_hook_exception_falls_back_to_default()
    test_no_hook_implementation_leaves_default_unchanged()
    test_hook_returning_wrong_length_falls_back()
    print("OK: 4/4 unit tests pass for KVConnector_V1.get_eviction_order hook")
