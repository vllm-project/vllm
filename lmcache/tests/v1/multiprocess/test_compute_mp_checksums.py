# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.http_apis.cache_api import _compute_block_checksums


def _make_5d_kv(
    num_layers: int = 2,
    kv_size: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Create deterministic 5D CPU KV tensors: ``[kv, NB, BS, NH, HS]``.

    Matches the ``NL_X_TWO_NB_BS_NH_HS`` layout (block_axis=1).
    ``kv_size=1`` mimics the MLA-as-5D edge case.
    """
    torch.manual_seed(0)
    return [
        torch.randn(kv_size, num_blocks, block_size, num_heads, head_size, dtype=dtype)
        for _ in range(num_layers)
    ]


def _make_3d_mla_kv(
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    head_size: int = 8,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Create MLA-style 3D CPU tensors: ``[NB, BS, HS]`` (block_axis=0)."""
    torch.manual_seed(0)
    return [
        torch.randn(num_blocks, block_size, head_size, dtype=dtype)
        for _ in range(num_layers)
    ]


class TestComputeBlockChecksums5DLayout:
    """Checksum computation on the 5D ``[2, NB, BS, NH, HS]`` layout."""

    def test_non_layerwise_shape(self):
        kv = _make_5d_kv()
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2, 3],
            block_axes=[1] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert result["layerwise"] is False
        assert result["chunk_size"] == 2
        assert result["num_chunks"] == 2
        assert len(result["chunk_checksums"]) == 2

    def test_layerwise_shape(self):
        kv = _make_5d_kv(num_layers=3)
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2, 3],
            block_axes=[1] * len(kv),
            chunk_size=4,
            layerwise=True,
        )
        assert result["layerwise"] is True
        assert result["num_chunks"] == 1
        per_layer = result["chunk_checksums"]
        assert set(per_layer.keys()) == {"layer_0", "layer_1", "layer_2"}
        for digests in per_layer.values():
            assert len(digests) == 1

    def test_partial_last_chunk(self):
        kv = _make_5d_kv()
        # 3 blocks with chunk_size=2 -> 2 chunks (2 + 1).
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2],
            block_axes=[1] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        assert result["num_chunks"] == 2
        assert len(result["chunk_checksums"]) == 2

    def test_deterministic(self):
        kv = _make_5d_kv()
        args = dict(
            block_ids=[0, 1], block_axes=[1] * len(kv), chunk_size=2, layerwise=False
        )
        r1 = _compute_block_checksums(kv, **args)
        r2 = _compute_block_checksums(kv, **args)
        assert r1["chunk_checksums"] == r2["chunk_checksums"]

    def test_md5_hex_format(self):
        kv = _make_5d_kv()
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1],
            block_axes=[1] * len(kv),
            chunk_size=1,
            layerwise=False,
        )
        for digest in result["chunk_checksums"]:
            assert len(digest) == 32
            int(digest, 16)

    def test_bfloat16_does_not_crash(self):
        kv = _make_5d_kv(dtype=torch.bfloat16)
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2, 3],
            block_axes=[1] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert len(result["chunk_checksums"]) == 2

    def test_mla_as_5d_kv_size_one(self):
        """5D layout with ``kv_size=1`` (MLA-in-5D) hashes cleanly."""
        kv = _make_5d_kv(kv_size=1)
        assert kv[0].shape[0] == 1
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2, 3],
            block_axes=[1] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert len(result["chunk_checksums"]) == 2


class TestComputeBlockChecksums3DMLA:
    """Checksum computation on the native 3D MLA layout ``[NB, BS, HS]``."""

    def test_non_layerwise_shape(self):
        kv = _make_3d_mla_kv()
        assert kv[0].ndim == 3
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2, 3],
            block_axes=[0] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert result["num_chunks"] == 2

    def test_layerwise_shape(self):
        kv = _make_3d_mla_kv()
        result = _compute_block_checksums(
            kv,
            block_ids=[0, 1, 2, 3],
            block_axes=[0] * len(kv),
            chunk_size=2,
            layerwise=True,
        )
        assert result["layerwise"] is True
        assert set(result["chunk_checksums"].keys()) == {"layer_0", "layer_1"}

    def test_matches_manual_md5(self):
        """Full oracle: recompute MD5 by hand and compare."""
        torch.manual_seed(42)
        kv = [torch.randn(4, 2, 3, dtype=torch.float32)]
        block_ids = [0, 2, 3]
        chunk_size = 2
        result = _compute_block_checksums(
            kv,
            block_ids=block_ids,
            block_axes=[0] * len(kv),
            chunk_size=chunk_size,
            layerwise=False,
        )
        gathered = kv[0].index_select(0, torch.tensor(block_ids)).contiguous()
        expected = []
        for ci in range((len(block_ids) + chunk_size - 1) // chunk_size):
            s = ci * chunk_size
            e = min(s + chunk_size, len(block_ids))
            chunk = gathered.narrow(0, s, e - s).contiguous()
            md5 = hashlib.md5()
            md5.update(chunk.numpy().tobytes())
            expected.append(md5.hexdigest())
        assert result["chunk_checksums"] == expected


class TestBlockSelectionSemantics:
    """The endpoint is block-addressed: duplicated / reordered block IDs
    must visibly change the checksum, and selecting disjoint block sets
    must yield different digests."""

    def test_reordered_blocks_change_checksum(self):
        kv = _make_3d_mla_kv()
        r1 = _compute_block_checksums(
            kv,
            block_ids=[0, 1],
            block_axes=[0] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        r2 = _compute_block_checksums(
            kv,
            block_ids=[1, 0],
            block_axes=[0] * len(kv),
            chunk_size=2,
            layerwise=False,
        )
        assert r1["chunk_checksums"] != r2["chunk_checksums"]

    def test_different_block_sets_differ(self):
        kv = _make_3d_mla_kv()
        r_a = _compute_block_checksums(
            kv,
            block_ids=[0, 1],
            block_axes=[0] * len(kv),
            chunk_size=1,
            layerwise=False,
        )
        r_b = _compute_block_checksums(
            kv,
            block_ids=[2, 3],
            block_axes=[0] * len(kv),
            chunk_size=1,
            layerwise=False,
        )
        assert r_a["chunk_checksums"] != r_b["chunk_checksums"]

    def test_out_of_range_block_axis_raises(self):
        """A block axis out of range for the tensor ndim must bubble up
        from ``index_select`` as ``IndexError``."""
        kv = _make_3d_mla_kv()
        with pytest.raises(IndexError):
            _compute_block_checksums(
                kv,
                block_ids=[0],
                block_axes=[9] * len(kv),
                chunk_size=1,
                layerwise=False,
            )
