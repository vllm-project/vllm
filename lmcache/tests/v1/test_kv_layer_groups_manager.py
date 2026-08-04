# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.kv_layer_groups import (
    EXCLUDED_ENGINE_GROUP,
    KernelGroupIdentity,
    KernelGroupInfo,
    KVLayerGroupInfo,
    KVLayerGroupsManager,
    LayerGroupIdentity,
    ObjectGroupInfo,
    format_kvcache_shape_spec,
    group_layers_by_identity,
    parse_kvcache_shape_spec,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PageBufferShapeDesc requires CUDA build"
)


def _build_manager(
    tensors: list[torch.Tensor],
    *,
    engine_group_infos: Sequence[EngineGroupInfo] = (),
    separate_object_groups: bool = False,
) -> KVLayerGroupsManager:
    """Build a manager using the per-layer NHD format.

    Tensors in these tests have shape ``[2, NB, BS, NH, HS]`` — the
    canonical vLLM flash-attention per-layer NHD layout matched by
    ``GPUKVFormat.NL_X_TWO_NB_BS_NH_HS``. ``bs`` and ``nb`` are discovered
    per-layer from the tensor shapes, so callers pass neither.
    """
    # First Party
    import lmcache.c_ops as lmc_ops

    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS] * len(tensors),
        engine_group_infos=engine_group_infos,
        separate_object_groups=separate_object_groups,
    )


class TestKVLayerGroupsManager:
    """Tests for KVLayerGroupsManager construction and lookups."""

    def test_build_empty(self):
        manager = _build_manager([])
        assert manager.kernel_groups == []

    def test_build_single_layer(self):
        tensors = [torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)]
        manager = _build_manager(tensors)

        assert len(manager.kernel_groups) == 1
        group = manager.kernel_groups[0]
        assert isinstance(group, KVLayerGroupInfo)
        assert group.layer_indices == [0]
        assert group.shape_desc.kv_size == 2
        assert group.shape_desc.nh == 8
        assert group.shape_desc.hs == 64
        assert group.shape_desc.nl == 1
        assert group.shape_desc.nb == 32
        assert group.shape_desc.bs == 256
        assert group.dtype == torch.float16

    def test_build_mixed_formats_per_group(self):
        """Mixed-format shape: a K+V group and a key-only MLA group are shaped
        with their own per-layer formats (kv_size 2 and 1), not one shared
        format -- the server-side per-group path."""
        # First Party
        import lmcache.c_ops as lmc_ops

        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.bfloat16),  # K+V (rank-5)
            torch.randn(32, 256, 128, dtype=torch.bfloat16),  # MLA key-only (rank-3)
        ]
        manager = KVLayerGroupsManager(
            tensors,
            engine_kv_formats=[
                lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
                lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
            ],
            engine_group_infos=[
                EngineGroupInfo(0, (0,)),
                EngineGroupInfo(1, (1,)),
            ],
        )

        groups = manager.kernel_groups
        assert len(groups) == 2
        by_group = {g.engine_group_idx: g for g in groups}
        assert by_group[0].shape_desc.kv_size == 2  # K+V main cache
        assert by_group[0].shape_desc.nh == 8
        assert by_group[1].shape_desc.kv_size == 1  # key-only MLA index cache
        assert by_group[1].shape_desc.nh == 1
        assert by_group[1].shape_desc.hs == 128
        # Each kernel group persists its own format for the transfer path.
        assert (
            by_group[0].engine_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
        )
        assert by_group[1].engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_BS_HS

    def test_build_multiple_layers_same_shape(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(3)
        ]
        manager = _build_manager(tensors)

        assert len(manager.kernel_groups) == 1
        group = manager.kernel_groups[0]
        assert group.layer_indices == [0, 1, 2]
        assert group.shape_desc.nl == 3
        assert group.shape_desc.nh == 8
        assert group.engine_group_idx == 0

    def test_build_splits_same_shape_by_engine_group_idx(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(4)
        ]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0, 2)),
                EngineGroupInfo(1, (1, 3)),
            ],
        )

        assert len(manager.kernel_groups) == 2
        groups_by_engine_group_idx = {
            group.engine_group_idx: group for group in manager.kernel_groups
        }
        assert groups_by_engine_group_idx[0].layer_indices == [0, 2]
        assert groups_by_engine_group_idx[1].layer_indices == [1, 3]

    def test_build_rejects_bad_engine_group_infos(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="outside registered layer"):
            _build_manager(
                tensors,
                engine_group_infos=[EngineGroupInfo(0, (2,))],
            )

    def test_build_rejects_coarse_engine_group_infos(self):
        # One info covering two layers that split into two kernel groups
        # (different num_heads) violates the one-info-per-kernel-group
        # contract.
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        ]
        with pytest.raises(ValueError, match="engine group info"):
            _build_manager(
                tensors,
                engine_group_infos=[EngineGroupInfo(0, (0, 1))],
            )

    def test_build_different_shapes(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors)
        assert len(manager.kernel_groups) == 2
        group1, group2 = manager.kernel_groups
        assert group1.layer_indices == [0, 2]
        assert group1.shape_desc.nh == 8
        assert group2.layer_indices == [1]
        assert group2.shape_desc.nh == 16

    def test_build_different_dtypes(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors)
        assert len(manager.kernel_groups) == 2
        group1, group2 = manager.kernel_groups
        assert group1.layer_indices == [0, 2]
        assert group1.dtype == torch.float16
        assert group2.layer_indices == [1]
        assert group2.dtype == torch.float32

    def test_build_mixed_differences(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # nh=8, f16
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),  # nh=8, f32
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),  # nh=16, f16
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),  # nh=8, f16
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float32),  # nh=16, f32
        ]
        manager = _build_manager(tensors)
        assert len(manager.kernel_groups) == 4

        groups_by_key = {(g.shape_desc.nh, g.dtype): g for g in manager.kernel_groups}
        assert groups_by_key[(8, torch.float16)].layer_indices == [0, 3]
        assert groups_by_key[(8, torch.float32)].layer_indices == [1]
        assert groups_by_key[(16, torch.float16)].layer_indices == [2]
        assert groups_by_key[(16, torch.float32)].layer_indices == [4]

    def test_get_shape_desc_by_group_idx(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors)

        sd0 = manager.get_shape_desc(0)
        assert sd0.nh == 8
        assert sd0.hs == 64
        assert sd0.nl == 1

        sd1 = manager.get_shape_desc(1)
        assert sd1.nh == 16
        assert sd1.hs == 64


class TestParseKvcacheShapeSpec:
    """Test cases for parse_kvcache_shape_spec function."""

    def test_single_group(self):
        """Test parsing a single group spec."""
        groups = parse_kvcache_shape_spec("(2,1024,16,8,128):float16:32")
        assert len(groups) == 1
        g = groups[0]
        assert g.num_layers == 32
        assert g.shape_desc.kv_size == 2
        assert g.shape_desc.nb == 1024
        assert g.shape_desc.bs == 16
        assert g.shape_desc.nh == 8
        assert g.shape_desc.hs == 128
        assert g.shape_desc.nl == 32
        assert g.dtype == torch.float16
        assert g.layer_indices == list(range(32))
        # Bench bookkeeping groups carry no format (the server re-detects); the
        # spec has no format enum and these never drive a transfer.
        assert g.engine_kv_format is None

    def test_multiple_groups(self):
        """Test parsing multiple groups separated by semicolons."""
        spec = "(2,1024,16,8,128):float16:30;(2,1024,16,4,64):bfloat16:2"
        groups = parse_kvcache_shape_spec(spec)
        assert len(groups) == 2

        # First group: 30 layers
        assert groups[0].num_layers == 30
        assert groups[0].dtype == torch.float16
        assert groups[0].layer_indices == list(range(30))

        # Second group: 2 layers, offset by 30
        assert groups[1].num_layers == 2
        assert groups[1].dtype == torch.bfloat16
        assert groups[1].shape_desc.nh == 4
        assert groups[1].shape_desc.hs == 64
        assert groups[1].layer_indices == [30, 31]

    def test_empty_spec_raises(self):
        """Test that empty spec raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_kvcache_shape_spec("")

    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid group spec"):
            parse_kvcache_shape_spec("bad_format")

    def test_unrecognized_dtype_raises(self):
        """Test that unrecognized dtype raises with helpful message."""
        with pytest.raises(ValueError, match="Unrecognized dtype"):
            parse_kvcache_shape_spec("(2,1024,16,8,128):float64:32")

    def test_invalid_number_raises(self):
        """Test that non-numeric shape values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid number"):
            parse_kvcache_shape_spec("(2,abc,16,8,128):float16:32")

    def test_whitespace_handling(self):
        """Test that whitespace around group separators is handled."""
        groups = parse_kvcache_shape_spec(
            " (2,1024,16,8,128):float16:4 ; (2,1024,16,4,64):bfloat16:2 "
        )
        assert len(groups) == 2
        assert groups[0].num_layers == 4
        assert groups[1].num_layers == 2

    def test_no_valid_groups_raises(self):
        """Test that spec with only separators raises."""
        with pytest.raises(ValueError, match="No valid layer groups"):
            parse_kvcache_shape_spec(";;;")


class TestFormatKvcacheShapeSpec:
    """Test cases for format_kvcache_shape_spec function."""

    def test_single_group(self):
        spec = "(2,1024,16,8,128):float16:32"
        groups = parse_kvcache_shape_spec(spec)
        assert format_kvcache_shape_spec(groups) == spec

    def test_multiple_groups(self):
        spec = "(2,1024,16,8,128):float16:30;(1,512,8,4,64):bfloat16:2"
        groups = parse_kvcache_shape_spec(spec)
        assert format_kvcache_shape_spec(groups) == spec

    def test_uint8_dtype(self):
        spec = "(2,1024,16,8,128):uint8:32"
        groups = parse_kvcache_shape_spec(spec)
        assert format_kvcache_shape_spec(groups) == spec

    def test_round_trip_normalizes_whitespace(self):
        """format() always produces the canonical (whitespace-free) form."""
        messy = " (2,1024,16,8,128):float16:4 ; (2,1024,16,4,64):bfloat16:2 "
        canonical = "(2,1024,16,8,128):float16:4;(2,1024,16,4,64):bfloat16:2"
        assert format_kvcache_shape_spec(parse_kvcache_shape_spec(messy)) == canonical

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="empty"):
            format_kvcache_shape_spec([])


class TestValidateBlockChunkSizeConfig:
    """Construction-time validation of the block/chunk size configuration:
    ``tokens_per_block`` (engine KV cache spec) must pack whole
    ``slots_per_block`` (registered tensor batch dimension), an LMCache chunk
    must span whole paged blocks, and a sub-chunk sliding window must cover
    whole paged blocks.
    """

    def _validate(
        self, slots: int, tokens: int, chunk: int = 256, sw: int = -1
    ) -> None:
        KVLayerGroupsManager._validate_block_chunk_size_config(
            group_idx=0,
            slots_per_block=slots,
            tokens_per_block=tokens,
            lmcache_tokens_per_chunk=chunk,
            sw_size_tokens=sw,
        )

    def test_valid_configs_pass(self):
        self._validate(slots=16, tokens=16)
        # slots=8 packs 2 logical tokens per physical slot (DeepSeek V4 style).
        self._validate(slots=8, tokens=16)
        # Sub-chunk window aligned to whole paged blocks.
        self._validate(slots=16, tokens=16, sw=64)
        # Big window (>= chunk) needs no sub-chunk alignment.
        self._validate(slots=16, tokens=16, sw=1000)

    def test_not_divisible_raises(self):
        # Divisibility is enforced loudly (e.g. slots=6 does not divide 16).
        with pytest.raises(ValueError, match="must be a multiple of"):
            self._validate(slots=6, tokens=16)

    def test_chunk_not_divisible_by_ratio_raises(self):
        with pytest.raises(ValueError, match="lmcache_tokens_per_chunk"):
            self._validate(slots=1, tokens=96, chunk=256)

    def test_subchunk_window_not_block_aligned_raises(self):
        # A sub-chunk window of 100 tokens does not cover whole 16-token
        # blocks, so the transfer slot count would disagree with the kept
        # block IDs.
        with pytest.raises(ValueError, match="sliding window"):
            self._validate(slots=16, tokens=16, sw=100)


class TestKernelGroupIdentity:
    """The grouping key is a named tuple; ``LayerGroupIdentity`` is its alias."""

    def test_fields_and_alias(self):
        # First Party
        import lmcache.c_ops as lmc_ops

        fmt = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
        ident = KernelGroupIdentity(
            kv_size=2,
            num_heads=8,
            head_size=64,
            block_size=16,
            engine_group_idx=0,
            dtype=torch.float16,
            engine_kv_format=fmt,
        )
        assert ident.kv_size == 2
        assert ident.num_heads == 8
        assert ident.head_size == 64
        assert ident.block_size == 16
        assert ident.engine_group_idx == 0
        assert ident.dtype == torch.float16
        assert ident.engine_kv_format == fmt
        assert LayerGroupIdentity is KernelGroupIdentity

    def test_hashable_as_dict_key(self):
        # First Party
        import lmcache.c_ops as lmc_ops

        fmt = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
        ident = KernelGroupIdentity(2, 8, 64, 16, 0, torch.float16, fmt)
        assert {ident: "x"}[ident] == "x"

    def test_excluded_engine_group_sentinel(self):
        assert EXCLUDED_ENGINE_GROUP == -1

    def test_format_in_identity_splits_same_geometry(self):
        """Two layers with identical geometry but different layouts (NHD vs HND,
        num_heads == block_size) must not merge into one kernel group: format is
        part of the identity, so each gets its own kernel with the correct
        layout instead of one transferring the other with the wrong axis order.
        """
        # First Party
        import lmcache.c_ops as lmc_ops

        # NH == BS == 16, so NHD [.., BS, NH, ..] and HND [.., NH, BS, ..] yield
        # the same kv_size/num_heads/head_size/block_size; only axis order differs.
        tensors = [
            torch.randn(2, 32, 16, 16, 64, dtype=torch.float16),
            torch.randn(2, 32, 16, 16, 64, dtype=torch.float16),
        ]
        groups = group_layers_by_identity(
            tensors,
            [
                lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # NHD
                lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,  # HND
            ],
        )
        # Without the format in the identity these share one geometry and would
        # have merged into a single group; with it they split into two.
        assert len(groups) == 2
        assert {idxs[0] for _, idxs in groups} == {0, 1}


class TestKernelAndObjectGroups:
    """Kernel-group accessors, deprecated aliases, and the (currently single)
    object-group layout."""

    def test_kernel_groups_match_deprecated_alias(self):
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(3)
        ]
        manager = _build_manager(tensors)
        # The deprecated alias must still return the live list, not a bound
        # method (regression guard for the @property/@deprecate ordering).
        assert isinstance(manager.kv_layer_groups, list)
        assert manager.kernel_groups is manager.kv_layer_groups
        assert manager.num_kernel_groups == manager.num_groups
        assert manager.num_kernel_groups == len(manager.kernel_groups)
        assert all(isinstance(g, KernelGroupInfo) for g in manager.kernel_groups)

    def test_single_object_group_covers_all_kernel_groups(self):
        # Two distinct kernel groups (different num_heads) still share one
        # object group under the current single-object-group assumption.
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors)
        assert manager.num_kernel_groups == 2
        assert manager.num_object_groups == 1
        obj = manager.object_groups[0]
        assert isinstance(obj, ObjectGroupInfo)
        assert obj.kernel_group_indices == list(range(manager.num_kernel_groups))
        assert obj.sw_size_chunks == -1
        assert manager.get_attn_desc().num_chunks_in_sw == [-1]

    def test_object_group_separation_disabled_merges_groups(self):
        # With separation off (the default), a full-attention group and a
        # sliding-window group still collapse into one full-attention object
        # group, and get_attn_desc reports full attention.
        tensors = [torch.randn(2, 32, 32, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,)),
                EngineGroupInfo(1, (1,), sw_size_tokens=64),
            ],
            separate_object_groups=False,
        )
        assert manager.num_kernel_groups == 2
        assert manager.num_object_groups == 1
        assert manager.object_groups[0].kernel_group_indices == [0, 1]
        assert manager.get_attn_desc().num_chunks_in_sw == [-1]

    def test_object_group_separation_enabled_buckets_by_window(self):
        # With separation on, the full-attention and sliding-window kernel groups
        # land in distinct object groups, ordered by first kernel group index,
        # and get_attn_desc reports each group's real window.
        tensors = [torch.randn(2, 32, 32, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,)),
                EngineGroupInfo(1, (1,), sw_size_tokens=64),
            ],
            separate_object_groups=True,
        )
        assert manager.num_kernel_groups == 2
        assert manager.num_object_groups == 2
        # Group 0: full attention (kernel group 0). Group 1: sliding window.
        assert manager.object_groups[0].kernel_group_indices == [0]
        assert manager.object_groups[0].sw_size_chunks == -1
        attn_desc = manager.get_attn_desc()
        assert attn_desc.num_chunks_in_sw[0] == -1
        assert manager.object_groups[1].kernel_group_indices == [1]
        assert manager.object_groups[1].sw_size_chunks >= 1
        assert attn_desc.num_chunks_in_sw[1] == manager.object_groups[1].sw_size_chunks

    def test_object_group_separation_enabled_non_hybrid_single_group(self):
        # Even with separation on, a non-hybrid model (no sliding-window groups)
        # yields a single full-attention object group.
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        ]
        manager = _build_manager(tensors, separate_object_groups=True)
        assert manager.num_object_groups == 1
        assert manager.get_attn_desc().num_chunks_in_sw == [-1]

    def test_kernel_groups_carry_sw_size_tokens(self):
        # Same-shape layers split by engine group; the sliding-window group's
        # window size lands on its kernel group, the other stays -1.
        tensors = [torch.randn(2, 32, 32, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,)),
                EngineGroupInfo(1, (1,), sw_size_tokens=64),
            ],
        )
        assert [g.sw_size_tokens for g in manager.kernel_groups] == [-1, 64]

    def test_subchunk_window_not_block_aligned_rejected(self):
        # A 64-token window over 256-slot blocks does not cover whole blocks;
        # construction fails loudly instead of mistransferring.
        tensors = [torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)]
        with pytest.raises(ValueError, match="sliding window"):
            _build_manager(
                tensors,
                engine_group_infos=[EngineGroupInfo(0, (0,), sw_size_tokens=64)],
            )

    def test_subchunk_sw_size_tokens(self):
        # lmcache chunk size is 256 (default), 32-slot blocks. Sub-chunk
        # window (64) is returned as-is; non-SW (-1) and big-SW (512) return
        # the chunk size.
        tensors = [
            torch.randn(2, 32, 32, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 32, 16, 64, dtype=torch.float16),
            torch.randn(2, 32, 32, 32, 64, dtype=torch.float16),
        ]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,)),
                EngineGroupInfo(0, (1,), sw_size_tokens=64),
                EngineGroupInfo(0, (2,), sw_size_tokens=512),
            ],
        )
        assert manager.get_subchunk_sw_size_tokens(0) == 256
        assert manager.get_subchunk_sw_size_tokens(1) == 64
        assert manager.get_subchunk_sw_size_tokens(2) == 256
        # Transfer slots follow the sub-chunk window (ratio 1 here).
        assert manager.get_slots_per_chunk_in_sw(0) == 256
        assert manager.get_slots_per_chunk_in_sw(1) == 64
        assert manager.get_slots_per_chunk_in_sw(2) == 256

    def test_mixed_sw_kernel_groups_share_single_object_group(self):
        # Object-level bucketing by sliding window size is not enabled yet:
        # kernel groups with differing window sizes still land in ONE object
        # group and get_attn_desc stays full attention.
        tensors = [
            torch.randn(2, 32, 32, 8, 64, dtype=torch.float16),
            torch.randn(2, 32, 32, 16, 64, dtype=torch.float16),
            torch.randn(2, 32, 32, 32, 64, dtype=torch.float16),
        ]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,)),
                EngineGroupInfo(0, (1,), sw_size_tokens=64),
                EngineGroupInfo(0, (2,), sw_size_tokens=512),
            ],
        )
        assert manager.num_object_groups == 1
        obj = manager.object_groups[0]
        assert obj.kernel_group_indices == list(range(manager.num_kernel_groups))
        assert obj.sw_size_chunks == -1
        assert manager.get_attn_desc().num_chunks_in_sw == [-1]

    def test_empty_manager_has_no_groups(self):
        # Empty registration returns early in __init__; both group lists must
        # still be initialized (regression guard for missing _object_groups).
        manager = _build_manager([])
        assert manager.kernel_groups == []
        assert manager.num_kernel_groups == 0
        assert manager.object_groups == []
        assert manager.num_object_groups == 0

    def test_excluded_layer_left_out_of_all_groups(self):
        # Layer 2 is referenced by no engine group info, so it is excluded entirely.
        tensors = [
            torch.randn(2, 32, 256, 8, 64, dtype=torch.float16) for _ in range(3)
        ]
        manager = _build_manager(
            tensors,
            engine_group_infos=[EngineGroupInfo(0, (0, 1))],
        )
        grouped = sorted(
            idx for group in manager.kernel_groups for idx in group.layer_indices
        )
        assert grouped == [0, 1]

    def test_calculate_num_blocks_uncompressed(self):
        # bs=16, compress_ratio=1 -> 256 tokens span 16 blocks.
        tensors = [torch.randn(2, 32, 16, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(tensors)
        assert manager.calculate_num_blocks(0, 256) == 16

    def test_dsv4_flash_style_mixed_compression(self):
        # Mirrors DeepSeek-V4-Flash: one 256-token engine group whose layers
        # have 64- and 2-slot pages (declared compress ratios 4 and 128), one
        # 64-token SWA group and one 4-token compressor-state group (ratio 1).
        tensors = [
            torch.randn(2, 8, 64, 1, 64, dtype=torch.float16),
            torch.randn(2, 8, 2, 1, 64, dtype=torch.float16),
            torch.randn(2, 8, 64, 1, 32, dtype=torch.float16),
            torch.randn(2, 8, 4, 1, 128, dtype=torch.float32),
        ]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0,), tokens_per_block=256),
                EngineGroupInfo(0, (1,), tokens_per_block=256),
                EngineGroupInfo(1, (2,), tokens_per_block=64),
                EngineGroupInfo(2, (3,), tokens_per_block=4),
            ],
        )
        by_layer = {g.layer_indices[0]: g for g in manager.kernel_groups}
        assert by_layer[0].tokens_per_block // by_layer[0].slots_per_block == 4
        assert by_layer[1].tokens_per_block // by_layer[1].slots_per_block == 128
        assert by_layer[2].tokens_per_block // by_layer[2].slots_per_block == 1
        assert by_layer[3].tokens_per_block // by_layer[3].slots_per_block == 1
        # 256-token LMCache chunk -> 2 physical slots in the ratio-128 group.
        assert by_layer[1].calculate_slots(256) == 2
        assert by_layer[0].calculate_slots(256) == 64

    def test_calculate_num_blocks_compressed(self):
        # slots_per_block=8 (tensor), tokens_per_block=16 (engine spec) ->
        # compress_ratio=2; 256 logical tokens -> 128 physical slots ->
        # 128 // 8 = 16 blocks.
        tensors = [torch.randn(2, 32, 8, 8, 64, dtype=torch.float16) for _ in range(2)]
        manager = _build_manager(
            tensors,
            engine_group_infos=[
                EngineGroupInfo(0, (0, 1), tokens_per_block=16),
            ],
        )
        group = manager.kernel_groups[0]
        assert group.tokens_per_block == 16
        assert group.slots_per_block == 8
        assert group.tokens_per_block // group.slots_per_block == 2
        assert manager.calculate_num_blocks(0, 256) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
