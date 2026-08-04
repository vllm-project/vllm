# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the paged-Q ring buffer's block accounting and scatter.
Covers QRingBuffer's allocator invariants, scatter semantics, and attention
layer selection.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.qringbuffer import (
    QRingBuffer,
    attention_layer_names_from_vllm,
    get_tensor,
)

BLOCK_SIZE = 8
HIDDEN_DIM = 16


def _ring(
    num_layers: int = 2, num_blocks: int = 4, dtype: torch.dtype = torch.float32
) -> QRingBuffer:
    """A small CPU ring: 2 layers x 4 blocks x 8 tokens x 16 features."""
    return QRingBuffer(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        hidden_dim=HIDDEN_DIM,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def test_allocate_zero_returns_empty_without_consuming() -> None:
    """Check that a zero-length allocation returns an empty list and does not
    reduce the free count."""
    ring = _ring()

    assert ring.allocate(0) == []
    assert ring.num_free_blocks() == 4


def test_allocate_debits_distinct_blocks() -> None:
    """Check that allocated ids are distinct, in range, and reducing free
    count."""
    ring = _ring()

    first = ring.allocate(3)

    assert first is not None
    assert len(set(first)) == 3
    assert all(0 <= b < 4 for b in first)
    assert ring.num_free_blocks() == 1
    assert set(ring.allocate(1) or []).isdisjoint(first)


def test_allocate_beyond_capacity_leaves_free_list_intact() -> None:
    """Check that if the request exceeds the number of free blocks,
    None is returned and the free list is unchanged."""
    ring = _ring()

    assert ring.allocate(5) is None
    assert ring.num_free_blocks() == 4
    assert ring.allocate(4) is not None  # capacity was not consumed


def test_free_drops_double_and_out_of_range_frees() -> None:
    """Check that freeing a block that is already free or does not exist is
    logged and dropped."""
    ring = _ring()
    blocks = ring.allocate(2)
    assert blocks is not None
    ring.free(blocks)

    ring.free([blocks[0]])  # one id, it's already free
    ring.free([99])  # id out of range
    ring.free(blocks)  # both ids free
    assert ring.num_free_blocks() == 4


def test_scatter_writes_only_valid_slots() -> None:
    """Checks that the ring's scatter method writes only to the slots that are
    not -1."""
    ring = _ring()
    ring.tensors["lmcache_q_layer_0"].zero_()
    query = torch.arange(3 * HIDDEN_DIM, dtype=torch.float32).reshape(
        3, 2, HIDDEN_DIM // 2
    )
    slots = torch.tensor([0, -1, BLOCK_SIZE + 2], dtype=torch.int64)
    ring.scatter(0, query, slots)
    layer = ring.tensors["lmcache_q_layer_0"]
    assert torch.equal(layer[0, 0], query[0].reshape(-1))
    assert torch.equal(layer[1, 2], query[2].reshape(-1))
    assert layer[0, 1].abs().sum() == 0  # the dropped row wrote nothing


def test_scatter_all_dropped_leaves_ring_untouched() -> None:
    """An all -1 mapping returns early without touching the ring."""
    ring = _ring()
    ring.tensors["lmcache_q_layer_0"].zero_()
    before = ring.tensors["lmcache_q_layer_0"].clone()
    ring.scatter(0, torch.ones(2, HIDDEN_DIM), torch.tensor([-1, -1]))
    assert torch.equal(ring.tensors["lmcache_q_layer_0"], before)


def test_scatter_width_mismatch_raises() -> None:
    """Check that for query tensors that flatten to a width different
    from the ring's hidden_dim, a ValueError is raised."""
    ring = _ring()
    query = torch.ones(2, 4, HIDDEN_DIM)  # flattens to 4 * 16
    with pytest.raises(ValueError, match="flattens to width"):
        ring.scatter(0, query, torch.tensor([0, 1]))


def test_scatter_casts_query_dtype_to_ring_dtype() -> None:
    """Check that the ring's scatter method casts the query tensor to the
    ring's dtype before writing. See QRingBuffer.scatter."""
    ring = _ring(num_layers=1, num_blocks=1, dtype=torch.bfloat16)
    query = torch.full((1, HIDDEN_DIM), 0.5, dtype=torch.float32)

    ring.scatter(0, query, torch.tensor([0]))

    stored = ring.tensors["lmcache_q_layer_0"][0, 0]
    assert stored.dtype == torch.bfloat16
    assert torch.allclose(stored.float(), query[0], atol=1e-2)


def test_scatter_accepts_flat_and_head_shaped_query() -> None:
    """Check that both [T, hidden] and [T, heads, head_size] shaped query
    tensors are accepted by the ring's scatter."""
    ring = _ring()
    flat = torch.ones(1, HIDDEN_DIM)
    shaped = torch.ones(1, 2, HIDDEN_DIM // 2)

    ring.scatter(0, flat, torch.tensor([0]))
    ring.scatter(0, shaped, torch.tensor([1]))

    layer = ring.tensors["lmcache_q_layer_0"]
    assert torch.equal(layer[0, 0], layer[0, 1])


def test_get_tensor_prefers_the_first_matching_name() -> None:
    """Checks that kv_transfer_utils.py in vLLM uses the known query tensor
    names, which is leveraged by get_tensor to find the matching tensor."""
    q, query = torch.zeros(1), torch.ones(1)

    assert get_tensor({"q": q, "query": query}, ["q", "query"]) is q
    assert get_tensor({"query": query}, ["q", "query"]) is query
    assert get_tensor({"k": q}, ["q", "query"]) is None


def _attention_spec() -> object:
    """Registers a fake attention spec for testing
    attention_layer_names_from_vllm()."""
    kv_iface = pytest.importorskip("vllm.v1.kv_cache_interface")
    return kv_iface.FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=torch.bfloat16
    )


def _caches(*names: str) -> dict[str, torch.Tensor]:
    """Registers a fake kv_caches dict for testing
    attention_layer_names_from_vllm()."""
    return {name: torch.zeros(1) for name in names}


@pytest.mark.parametrize("config", [None, SimpleNamespace(kv_cache_groups=[])])
def test_layer_names_fall_back_to_kv_cache_order(config) -> None:
    """Check that if the vLLM config has no attention spec, the layer names are
    taken from the kv_caches dict in order."""
    caches = _caches("layer.2", "layer.0", "layer.1")

    assert attention_layer_names_from_vllm(config, caches) == list(caches)


def test_only_attention_layers_selected_in_kv_cache_order() -> None:
    """Check that only layers with an attention spec are selected,
    returned in the order they appear in the kv_caches dict."""
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.2"], kv_cache_spec=_attention_spec()),
            SimpleNamespace(layer_names=["layer.1"], kv_cache_spec=object()),
            SimpleNamespace(layer_names=["layer.0"], kv_cache_spec=_attention_spec()),
        ]
    )
    names = attention_layer_names_from_vllm(
        config, _caches("layer.0", "layer.1", "layer.2")
    )
    assert names == ["layer.0", "layer.2"]
