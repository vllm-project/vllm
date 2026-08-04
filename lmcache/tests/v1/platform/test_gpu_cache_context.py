# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the temp-GPU-buffer machinery in
``lmcache.v1.platform.cuda.cache_context``.

Two layers are exercised:

* ``_TempGPUBuffer`` -- the standalone buffer manager. It is built directly
  from a real :class:`KVLayerGroupsManager` (its constructor is fully public),
  so the layout invariants (per-kernel-group shape/dtype, per-object-group flat
  views, non-overlap, write isolation, byte sizing) are tested in isolation.

* ``GPUCacheContext`` -- the higher-level context that owns a ``_TempGPUBuffer``
  and exposes the per-kernel-group / per-object-group buffer accessors plus
  ``get_kernel_group_kv_pointers``, ``calculate_num_blocks``,
  ``kv_layer_groups_manager`` and ``report_status``. It is built through its
  real public constructor using a lightweight ``to_tensor`` test double in place
  of ``CudaIPCWrapper`` (same-process CUDA IPC cannot reimport its own handle).
"""

# Standard
from collections.abc import Sequence

# Third Party
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# First Party
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager  # noqa: E402
from lmcache.v1.multiprocess.group_view import EngineGroupInfo  # noqa: E402
from lmcache.v1.platform.cuda.cache_context import (  # noqa: E402
    GPUCacheContext,
    _TempGPUBuffer,
)
import lmcache.c_ops as lmc_ops  # noqa: E402

_DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GroupSpec:
    """Description of one homogeneous block of KV layers used to build the
    synthetic ``[2, NB, BS, NH, HS]`` (non-MLA) tensors fed to the manager."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int = 8,
        head_size: int = 64,
        block_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.dtype = dtype


def _make_kv_tensors(
    specs: Sequence[_GroupSpec],
    num_blocks: int = 4,
) -> list[torch.Tensor]:
    """Build non-MLA per-layer KV tensors shaped ``[2, NB, BS, NH, HS]``."""
    tensors: list[torch.Tensor] = []
    for spec in specs:
        for _ in range(spec.num_layers):
            tensors.append(
                torch.empty(
                    2,
                    num_blocks,
                    spec.block_size,
                    spec.num_heads,
                    spec.head_size,
                    dtype=spec.dtype,
                    device=_DEVICE,
                )
            )
    return tensors


def _build_manager(
    tensors: list[torch.Tensor],
    engine_kv_format: "lmc_ops.EngineKVFormat" = (
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    ),
    engine_group_infos: Sequence[EngineGroupInfo] = (),
    lmcache_tokens_per_chunk: int = 256,
) -> KVLayerGroupsManager:
    """Build a real :class:`KVLayerGroupsManager` from synthetic tensors."""
    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[engine_kv_format] * len(tensors),
        engine_group_infos=engine_group_infos,
        lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
    )


def _make_temp_buffer(
    specs: Sequence[_GroupSpec],
    chunk_size: int = 256,
    max_batch_size: int = 4,
    num_blocks: int = 4,
    engine_group_infos: Sequence[EngineGroupInfo] = (),
) -> _TempGPUBuffer:
    """Build a ``_TempGPUBuffer`` backed by a real manager."""
    tensors = _make_kv_tensors(specs, num_blocks=num_blocks)
    manager = _build_manager(
        tensors,
        engine_group_infos=engine_group_infos,
        lmcache_tokens_per_chunk=chunk_size,
    )
    return _TempGPUBuffer(
        kv_layer_groups_manager=manager,
        lmcache_tokens_per_chunk=chunk_size,
        device=_DEVICE,
        max_batch_size=max_batch_size,
    )


def _expected_kernel_group_shape(
    manager: KVLayerGroupsManager, num_tokens: int, kernel_group_idx: int
) -> torch.Size:
    """Compute the expected kernel-group buffer shape from the manager's
    public metadata (kv_size, num_layers, slots, hidden_dim)."""
    group = manager.kernel_groups[kernel_group_idx]
    num_slots = num_tokens * group.slots_per_block // group.tokens_per_block
    return torch.Size(
        (
            group.shape_desc.kv_size,
            group.num_layers,
            num_slots,
            group.hidden_dim_size,
        )
    )


def _expected_kernel_group_bytes(
    manager: KVLayerGroupsManager, chunk_size: int, kernel_group_idx: int
) -> int:
    """Byte size of one kernel group's per-chunk buffer."""
    group = manager.kernel_groups[kernel_group_idx]
    shape = _expected_kernel_group_shape(manager, chunk_size, kernel_group_idx)
    return shape.numel() * group.dtype.itemsize


def _byte_region(buf: torch.Tensor) -> tuple[int, int]:
    """Return ``(start_ptr, end_ptr)`` covering a tensor's bytes."""
    start = buf.data_ptr()
    return start, start + buf.nelement() * buf.element_size()


def _assert_disjoint(regions: list[tuple[int, int, str]]) -> None:
    """Assert that no two ``(start, end, label)`` byte ranges overlap."""
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            s_i, e_i, label_i = regions[i]
            s_j, e_j, label_j = regions[j]
            assert e_i <= s_j or e_j <= s_i, f"Overlap between {label_i} and {label_j}"


class _FakeIPCWrapper:
    """Test-only stand-in for ``CudaIPCWrapper``.

    ``GPUCacheContext`` only needs ``to_tensor()`` from each entry of its
    ``kv_caches`` argument. Same-process CUDA IPC cannot reopen its own handle,
    so this test double simply hands back a locally allocated CUDA tensor,
    letting the real ``GPUCacheContext`` constructor run end to end.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Return the wrapped local CUDA tensor (test-only)."""
        return self._tensor


def _make_context(
    specs: Sequence[_GroupSpec],
    chunk_size: int = 256,
    num_blocks: int = 4,
    engine_group_infos: Sequence[EngineGroupInfo] = (),
) -> GPUCacheContext:
    """Build a real ``GPUCacheContext`` via its public constructor."""
    tensors = _make_kv_tensors(specs, num_blocks=num_blocks)
    kv_caches = [_FakeIPCWrapper(t) for t in tensors]
    return GPUCacheContext(
        kv_caches,  # type: ignore
        lmcache_tokens_per_chunk=chunk_size,
        engine_group_infos=engine_group_infos,
    )


# Common group layouts reused across tests.
_SINGLE_GROUP = [_GroupSpec(num_layers=4, num_heads=8, head_size=64)]
_MULTI_GROUP = [
    _GroupSpec(num_layers=4, num_heads=8, head_size=64, dtype=torch.bfloat16),
    _GroupSpec(num_layers=2, num_heads=16, head_size=64, dtype=torch.float16),
]


# ---------------------------------------------------------------------------
# _TempGPUBuffer tests
# ---------------------------------------------------------------------------


class TestTempGPUBufferConstruction:
    def test_max_batch_size_property(self) -> None:
        buf = _make_temp_buffer(_SINGLE_GROUP, max_batch_size=3)
        assert buf.max_batch_size == 3


class TestTempGPUBufferKernelGroupBuffer:
    def test_shape_and_dtype(self) -> None:
        tensors = _make_kv_tensors(_MULTI_GROUP)
        manager = _build_manager(tensors)
        buf = _TempGPUBuffer(manager, 256, _DEVICE)
        for kg in range(manager.num_kernel_groups):
            tensor = buf.get_temp_kernel_group_buffer(0, kg)
            assert tensor.shape == _expected_kernel_group_shape(manager, 256, kg)
            assert tensor.dtype == manager.kernel_groups[kg].dtype

    def test_contiguous(self) -> None:
        buf = _make_temp_buffer(_SINGLE_GROUP)
        assert buf.get_temp_kernel_group_buffer(0, 0).is_contiguous()

    def test_repeated_calls_same_ptr(self) -> None:
        buf = _make_temp_buffer(_SINGLE_GROUP)
        first = buf.get_temp_kernel_group_buffer(1, 0)
        second = buf.get_temp_kernel_group_buffer(1, 0)
        assert first.data_ptr() == second.data_ptr()

    def test_invalid_batch_idx_raises(self) -> None:
        buf = _make_temp_buffer(_SINGLE_GROUP, max_batch_size=4)
        with pytest.raises(ValueError, match="Invalid batch_idx"):
            buf.get_temp_kernel_group_buffer(4, 0)

    def test_invalid_kernel_group_idx_raises(self) -> None:
        buf = _make_temp_buffer(_SINGLE_GROUP)
        with pytest.raises(ValueError, match="kernel_group_idx"):
            buf.get_temp_kernel_group_buffer(0, 99)

    def test_buffers_non_overlapping(self) -> None:
        """Every (batch, kernel_group) buffer occupies disjoint memory."""
        tensors = _make_kv_tensors(_MULTI_GROUP)
        manager = _build_manager(tensors)
        max_batch_size = 4
        buf = _TempGPUBuffer(manager, 256, _DEVICE, max_batch_size=max_batch_size)
        regions: list[tuple[int, int, str]] = []
        for batch in range(max_batch_size):
            for kg in range(manager.num_kernel_groups):
                tensor = buf.get_temp_kernel_group_buffer(batch, kg)
                start, end = _byte_region(tensor)
                regions.append((start, end, f"batch={batch},kg={kg}"))
        _assert_disjoint(regions)

    def test_write_isolation(self) -> None:
        """Writing to one batch slot must not corrupt another."""
        buf = _make_temp_buffer(
            [_GroupSpec(num_layers=2, num_heads=2, head_size=16)],
            chunk_size=32,
            max_batch_size=4,
        )
        for batch in range(4):
            buf.get_temp_kernel_group_buffer(batch, 0).fill_(float(batch + 1))
        for batch in range(4):
            tensor = buf.get_temp_kernel_group_buffer(batch, 0).to(torch.float32)
            assert tensor.min().item() == pytest.approx(batch + 1, rel=1e-3)
            assert tensor.max().item() == pytest.approx(batch + 1, rel=1e-3)


class TestTempGPUBufferObjectGroupBuffer:
    def test_flat_uint8(self) -> None:
        buf = _make_temp_buffer(_MULTI_GROUP)
        tensor = buf.get_temp_object_group_buffer(0, 0)
        assert tensor.dtype == torch.uint8
        assert tensor.dim() == 1
        assert tensor.is_contiguous()

    def test_size_covers_all_kernel_groups(self) -> None:
        """The single object group's flat buffer spans every kernel group's
        bytes for one chunk."""
        tensors = _make_kv_tensors(_MULTI_GROUP)
        manager = _build_manager(tensors)
        chunk_size = 256
        buf = _TempGPUBuffer(manager, chunk_size, _DEVICE)
        obj_group = manager.object_groups[0]
        expected_bytes = sum(
            _expected_kernel_group_bytes(manager, chunk_size, kg)
            for kg in obj_group.kernel_group_indices
        )
        assert buf.get_temp_object_group_buffer(0, 0).numel() == expected_bytes

    def test_starts_at_first_kernel_group(self) -> None:
        """The object-group flat view aliases the same memory as its first
        kernel group's buffer."""
        tensors = _make_kv_tensors(_MULTI_GROUP)
        manager = _build_manager(tensors)
        buf = _TempGPUBuffer(manager, 256, _DEVICE)
        first_kg = manager.object_groups[0].kernel_group_indices[0]
        obj_buf = buf.get_temp_object_group_buffer(0, 0)
        kg_buf = buf.get_temp_kernel_group_buffer(0, first_kg)
        assert obj_buf.data_ptr() == kg_buf.data_ptr()

    def test_invalid_indices_raise(self) -> None:
        buf = _make_temp_buffer(_SINGLE_GROUP, max_batch_size=4)
        with pytest.raises(ValueError, match="object_group_idx"):
            buf.get_temp_object_group_buffer(0, 99)
        with pytest.raises(ValueError, match="batch_idx"):
            buf.get_temp_object_group_buffer(4, 0)

    def test_contains_kernel_group_data(self) -> None:
        """Bytes written through kernel-group views are visible through the
        object-group flat view at matching offsets."""
        tensors = _make_kv_tensors(_MULTI_GROUP)
        chunk_size = 64
        manager = _build_manager(tensors, lmcache_tokens_per_chunk=chunk_size)
        buf = _TempGPUBuffer(manager, chunk_size, _DEVICE)
        obj_group = manager.object_groups[0]

        for offset_kg, kg in enumerate(obj_group.kernel_group_indices):
            buf.get_temp_kernel_group_buffer(0, kg).view(torch.uint8).fill_(
                offset_kg + 1
            )

        flat = buf.get_temp_object_group_buffer(0, 0)
        cursor = 0
        for offset_kg, kg in enumerate(obj_group.kernel_group_indices):
            size = _expected_kernel_group_bytes(manager, chunk_size, kg)
            region = flat[cursor : cursor + size]
            assert region.min().item() == offset_kg + 1
            assert region.max().item() == offset_kg + 1
            cursor += size

    def test_object_groups_non_overlapping(self) -> None:
        """Object-group buffers across batch slots occupy disjoint memory."""
        tensors = _make_kv_tensors(_MULTI_GROUP)
        manager = _build_manager(tensors)
        max_batch_size = 4
        buf = _TempGPUBuffer(manager, 256, _DEVICE, max_batch_size=max_batch_size)
        regions: list[tuple[int, int, str]] = []
        for batch in range(max_batch_size):
            for og in range(manager.num_object_groups):
                start, end = _byte_region(buf.get_temp_object_group_buffer(batch, og))
                regions.append((start, end, f"batch={batch},og={og}"))
        _assert_disjoint(regions)


class TestTempGPUBufferShapeDtype:
    def test_shape_scales_with_num_tokens(self) -> None:
        # num_tokens must be a whole number of chunks; the shape scales
        # linearly with the chunk count.
        tensors = _make_kv_tensors(_SINGLE_GROUP)
        manager = _build_manager(tensors)
        buf = _TempGPUBuffer(manager, 256, _DEVICE)
        for num_tokens in (256, 512, 768):
            shape, dtype = buf.get_kernel_group_shape_dtype(num_tokens, 0)
            assert shape == _expected_kernel_group_shape(manager, num_tokens, 0)
            assert dtype == manager.kernel_groups[0].dtype

    def test_shape_compressed_group(self) -> None:
        """For a compressed group, the token dim is divided by compress_ratio."""
        tensors = _make_kv_tensors([_GroupSpec(num_layers=2, block_size=8)])
        manager = _build_manager(
            tensors,
            engine_group_infos=[EngineGroupInfo(0, (0, 1), tokens_per_block=16)],
        )
        kg = manager.kernel_groups[0]
        assert kg.tokens_per_block // kg.slots_per_block == 2
        buf = _TempGPUBuffer(manager, 256, _DEVICE)
        shape, _ = buf.get_kernel_group_shape_dtype(256, 0)
        assert shape[2] == 256 // 2

    def test_not_whole_chunks_raises(self) -> None:
        tensors = _make_kv_tensors([_GroupSpec(num_layers=2, block_size=8)])
        manager = _build_manager(
            tensors,
            engine_group_infos=[EngineGroupInfo(0, (0, 1), tokens_per_block=16)],
        )
        buf = _TempGPUBuffer(manager, 256, _DEVICE)
        with pytest.raises(ValueError, match="must be a multiple of"):
            buf.get_kernel_group_shape_dtype(255, 0)


class TestTempGPUBufferCacheSize:
    def test_cache_size_per_token(self) -> None:
        tensors = _make_kv_tensors(_MULTI_GROUP)
        manager = _build_manager(tensors)
        chunk_size = 256
        buf = _TempGPUBuffer(manager, chunk_size, _DEVICE)
        expected = (
            sum(
                _expected_kernel_group_bytes(manager, chunk_size, kg)
                for kg in range(manager.num_kernel_groups)
            )
            // chunk_size
        )
        assert buf.get_cache_size_per_token() == expected

    def test_cache_size_per_token_compressed(self) -> None:
        """Compression halves per-physical-slot bytes, so the per-logical-token
        size of a 2x-compressed group is half its uncompressed counterpart."""
        uncompressed = _make_temp_buffer([_GroupSpec(num_layers=2, block_size=16)])
        compressed = _make_temp_buffer(
            [_GroupSpec(num_layers=2, block_size=8)],
            engine_group_infos=[EngineGroupInfo(0, (0, 1), tokens_per_block=16)],
        )
        assert (
            compressed.get_cache_size_per_token() * 2
            == uncompressed.get_cache_size_per_token()
        )


# ---------------------------------------------------------------------------
# GPUCacheContext tests
# ---------------------------------------------------------------------------


class TestGPUCacheContextBuffers:
    def test_max_batch_size(self) -> None:
        ctx = _make_context(_SINGLE_GROUP)
        assert ctx.max_batch_size == 4

    def test_kv_layer_groups_manager(self) -> None:
        ctx = _make_context(_MULTI_GROUP)
        manager = ctx.kv_layer_groups_manager
        assert isinstance(manager, KVLayerGroupsManager)
        assert manager.num_kernel_groups == 2

    def test_get_temp_kernel_group_buffer(self) -> None:
        ctx = _make_context(_MULTI_GROUP)
        manager = ctx.kv_layer_groups_manager
        for kg in range(manager.num_kernel_groups):
            tensor = ctx.get_temp_kernel_group_buffer(0, kg)
            assert tensor.shape == _expected_kernel_group_shape(manager, 256, kg)
            assert tensor.dtype == manager.kernel_groups[kg].dtype

    def test_get_temp_object_group_buffer(self) -> None:
        ctx = _make_context(_MULTI_GROUP)
        tensor = ctx.get_temp_object_group_buffer(0, 0)
        assert tensor.dtype == torch.uint8
        assert tensor.dim() == 1

    def test_get_kernel_group_shape_dtype(self) -> None:
        ctx = _make_context(_SINGLE_GROUP)
        manager = ctx.kv_layer_groups_manager
        shape, dtype = ctx.get_kernel_group_shape_dtype(256, 0)
        assert shape == _expected_kernel_group_shape(manager, 256, 0)
        assert dtype == manager.kernel_groups[0].dtype


class TestGPUCacheContextPointers:
    def test_get_kernel_group_kv_pointers(self) -> None:
        ctx = _make_context(_MULTI_GROUP)
        manager = ctx.kv_layer_groups_manager
        for kg in range(manager.num_kernel_groups):
            pointers = ctx.get_kernel_group_kv_pointers(kg)
            assert pointers.dtype == torch.long
            # One pointer per layer in the group.
            assert pointers.numel() == manager.kernel_groups[kg].num_layers


class TestGPUCacheContextBlocks:
    def test_calculate_num_blocks_uncompressed(self) -> None:
        # block_size=16, compress_ratio=1 -> 256 tokens span 16 blocks.
        ctx = _make_context([_GroupSpec(num_layers=2, block_size=16)])
        assert ctx.calculate_num_blocks(256, 0) == 16

    def test_calculate_num_blocks_matches_manager(self) -> None:
        ctx = _make_context(_MULTI_GROUP)
        manager = ctx.kv_layer_groups_manager
        for kg in range(manager.num_kernel_groups):
            assert ctx.calculate_num_blocks(256, kg) == manager.calculate_num_blocks(
                kg, 256
            )


class TestGPUCacheContextReportStatus:
    _TOP_LEVEL_KEYS = {
        "num_layers",
        "num_blocks",
        "cache_size_per_token",
        "kernel_groups",
    }
    _GROUP_KEYS = {
        "kernel_group_idx",
        "engine_group_idx",
        "object_group_idx",
        "num_layers",
        "layer_indices",
        "tokens_per_block",
        "slots_per_block",
        "dtype",
        "engine_kv_concrete_shape",
        "is_mla",
        "engine_kv_format",
        "engine_kv_shape",
        "attention_backend",
    }

    def test_report_status_fields(self) -> None:
        ctx = _make_context(_SINGLE_GROUP)
        status = ctx.report_status()

        assert set(status.keys()) == self._TOP_LEVEL_KEYS
        assert status["num_layers"] == 4
        assert status["cache_size_per_token"] == ctx.cache_size_per_token()

        assert len(status["kernel_groups"]) == 1
        group = status["kernel_groups"][0]
        assert set(group.keys()) == self._GROUP_KEYS
        assert group["kernel_group_idx"] == 0
        assert group["num_layers"] == 4
        assert group["layer_indices"] == [0, 1, 2, 3]
        assert group["is_mla"] is False
        assert group["engine_kv_format"] == "NL_X_TWO_NB_BS_NH_HS"
        assert group["dtype"] == str(ctx.kv_tensors[0].dtype)

    def test_report_status_multi_group(self) -> None:
        ctx = _make_context(_MULTI_GROUP)
        manager = ctx.kv_layer_groups_manager
        status = ctx.report_status()
        assert status["num_layers"] == 6
        assert len(status["kernel_groups"]) == manager.num_kernel_groups

        # Group reports enumerate in order and stay self-consistent with the
        # manager's kernel groups.
        for kg_idx, (group, kernel_group) in enumerate(
            zip(status["kernel_groups"], manager.kernel_groups, strict=False)
        ):
            assert set(group.keys()) == self._GROUP_KEYS
            assert group["kernel_group_idx"] == kg_idx
            assert group["engine_group_idx"] == kernel_group.engine_group_idx
            assert group["num_layers"] == kernel_group.num_layers
            assert group["slots_per_block"] == kernel_group.slots_per_block
            assert group["tokens_per_block"] == kernel_group.tokens_per_block
            assert 0 <= group["object_group_idx"] < manager.num_object_groups


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
