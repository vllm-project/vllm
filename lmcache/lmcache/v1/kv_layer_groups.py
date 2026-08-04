# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import lmcache_deprecate
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.platform.ops_types import set_shape_desc_dtype
import lmcache.c_ops as lmc_ops

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import DiscoverableKVCache
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

DEFAULT_LAYER_NAME_PREFIX = "model.layers."

# ------------------------------------------------------------------ #
#  dtype mapping                                                       #
# ------------------------------------------------------------------ #

DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


# The tuple that uniquely identifies a set of kernel-equivalent layers; one
# distinct identity becomes one LMCache KV group:
# ``(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype,
# engine_kv_format)``.
# ``engine_group_idx`` is the engine group id (one paged-block address space):
# block IDs are only meaningful within one group, so layers from different
# groups must not share an LMCache group even if shape and dtype match.
# ``engine_kv_format`` keeps layouts that share an engine group from merging --
# load-bearing when one ``engine_group_idx`` mixes layouts (a rank-5 K/V group
# alongside a rank-3 key-only indexer cache): this field is what splits them
# into separate kernel groups, each transferred with its own layout.
class KernelGroupIdentity(NamedTuple):
    kv_size: int
    num_heads: int
    head_size: int
    block_size: int
    engine_group_idx: int
    dtype: torch.dtype
    engine_kv_format: "lmc_ops.EngineKVFormat"


LayerGroupIdentity = KernelGroupIdentity  # Alias for compatibility


# Sentinel ``per_layer_engine_group_idx`` value: a KV tensor tagged with it is
# excluded from every LMCache group (used for cross-layer KV-sharing layers; see
# ``create_engine_group_infos_from_vllm``).
EXCLUDED_ENGINE_GROUP = -1


def group_layers_by_identity(
    kv_caches: "DiscoverableKVCache",
    engine_kv_formats: "Sequence[lmc_ops.EngineKVFormat]",
    per_layer_engine_group_idx: Sequence[int] | None = None,
) -> list[tuple[LayerGroupIdentity, list[int]]]:
    """Partition layer indices by :data:`LayerGroupIdentity`.

    Args:
        kv_caches: Registered KV cache structure, one entry per layer.
        engine_kv_formats: One Engine KV format per registered tensor; its length
            is the layer count. Homogeneous models repeat one shared format; a
            model that mixes formats across engine groups -- e.g. a K+V main cache
            (``kv_size=2``) plus a key-only MLA index cache (``kv_size=1``) --
            supplies the differing per-layer formats.
        per_layer_engine_group_idx: Optional engine block group id per layer.
            When ``None`` every layer is treated as block group 0 (non-hybrid);
            when present, layers from different engine block groups never share an
            identity even if their tensor shapes match. Layers whose value is
            ``EXCLUDED_ENGINE_GROUP`` are left out of all groups (e.g. cross-layer
            KV-sharing layers whose KV lives in their target owner's blocks).

    Returns:
        A list of ``(identity, layer_indices)`` pairs sorted by each group's
        first layer index, so the group order is deterministic.

    Raises:
        ValueError: If ``per_layer_engine_group_idx`` is given but its length
            does not match ``engine_kv_formats``.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_dtype,
        get_head_size,
        get_num_heads,
        is_mla,
    )

    num_layers = len(engine_kv_formats)
    if (
        per_layer_engine_group_idx is not None
        and len(per_layer_engine_group_idx) != num_layers
    ):
        raise ValueError(
            f"per_layer_engine_group_idx has {len(per_layer_engine_group_idx)} "
            f"entries but engine_kv_formats has {num_layers} layers"
        )

    groups_dict: dict[LayerGroupIdentity, list[int]] = defaultdict(list)
    for idx in range(num_layers):
        engine_group_idx = (
            per_layer_engine_group_idx[idx]
            if per_layer_engine_group_idx is not None
            else 0
        )
        # Skip layers explicitly excluded from grouping (e.g. cross-layer
        # KV-sharing layers, whose KV lives in their target owner's blocks).
        if engine_group_idx == EXCLUDED_ENGINE_GROUP:
            continue
        layer_format = engine_kv_formats[idx]
        mla = is_mla(layer_format)
        kv_size = 1 if mla else 2
        nh = 1 if mla else get_num_heads(kv_caches, layer_format, idx)
        hs = get_head_size(kv_caches, layer_format, idx)
        dt = get_dtype(kv_caches, layer_format, idx)
        bs = get_block_size(kv_caches, layer_format, idx)

        identity = LayerGroupIdentity(
            kv_size=kv_size,
            num_heads=nh,
            head_size=hs,
            block_size=bs,
            engine_group_idx=engine_group_idx,
            dtype=dt,
            engine_kv_format=layer_format,
        )
        groups_dict[identity].append(idx)
    return sorted(groups_dict.items(), key=lambda kv: kv[1][0])


@dataclass
class KernelGroupInfo:
    """A single transfer-kernel dispatch unit: a set of KV layers that can
    ride one kernel launch with one ``PageBufferShapeDesc``.

    Membership is decided by :class:`KVLayerGroupsManager` according to
    :data:`LayerGroupIdentity`; every layer referenced by
    ``layer_indices`` shares the same
    ``(kv_size, num_heads, head_size, block_size, engine_group_idx,
    dtype, engine_kv_format)`` signature.
    Consumers use ``layer_indices`` to pull the matching device pointers
    out of ``kv_caches`` (via
    :func:`~lmcache.v1.gpu_connector.utils.get_group_data_ptrs`) and
    feed them to the kernel alongside ``shape_desc``.

    ``dtype`` is carried alongside ``shape_desc`` because
    ``PageBufferShapeDesc.element_size`` is a byte width, which cannot
    distinguish dtypes that share a byte count (e.g. bfloat16 and
    float16 are both 2 bytes). Kernel template instantiation keys on the
    torch dtype, not the byte width, so we keep it explicit.

    Treat instances as immutable after construction; callers may hold
    references for the lifetime of the manager.
    """

    layer_indices: list[int]
    """0-based layer indices belonging to this group, in the order the
    kernel should iterate them. Fed to ``get_group_data_ptrs`` to build
    the per-group pointer array."""
    shape_desc: "lmc_ops.PageBufferShapeDesc"
    """Kernel-facing shape descriptor shared by every layer in the group.
    All eight fields (``kv_size, nl, nb, bs, nh, hs, element_size,
    block_stride_elems``) are stamped once at construction."""
    dtype: torch.dtype
    """Torch dtype of the KV cache tensors for this group. Used for
    kernel template instantiation; see class docstring for why we keep
    this alongside ``shape_desc.element_size``."""
    engine_kv_format: "lmc_ops.EngineKVFormat | None" = None
    """Per-group Engine KV format, read via
    ``BaseCacheContext.get_engine_kv_format`` so mixed-format models dispatch each
    group with its own. ``None`` only for bench bookkeeping groups from
    :func:`parse_kvcache_shape_spec`, which have no detected format and never
    transfer; detection-built groups always set it."""
    tokens_per_block: int = 0
    """Logical engine tokens covered by one paged chunk (one engine block
    ID) of this group, as declared by the engine's KV cache spec at
    initialization time (carried in ``EngineGroupInfo.tokens_per_block``).
    ``0`` means the engine did not report it; the group is then treated as
    uncompressed (``compress_ratio == 1``)."""
    engine_group_idx: int = 0
    """Engine group index (paged-block address space). 0 for non-hybrid."""
    sw_size_tokens: int = -1
    """Sliding window size in logical tokens for this group's layers.
    ``-1`` means the layers are not sliding-window attention."""

    def __repr__(self) -> str:
        if not self.layer_indices:
            indices_repr = "[]"
        else:
            indices_repr = f"{self.layer_indices[0]}-{self.layer_indices[-1]}"
        sd = self.shape_desc
        return (
            f"KernelGroupInfo(layers={len(self.layer_indices)}, "
            f"indices={indices_repr}, "
            f"shape_desc=(kv={sd.kv_size}, nl={sd.nl}, nb={sd.nb}, "
            f"bs={sd.bs}, nh={sd.nh}, hs={sd.hs}, "
            f"element_size={sd.element_size}, "
            f"block_stride_elems={sd.block_stride_elems}), "
            f"dtype={self.dtype}, "
            f"tokens_per_block={self.tokens_per_block}, "
            f"slots_per_block={self.slots_per_block}, "
            f"engine_group_idx={self.engine_group_idx}, "
            f"sw_size_tokens={self.sw_size_tokens})"
        )

    @property
    def num_layers(self) -> int:
        """Number of layers in this group."""
        return len(self.layer_indices)

    @property
    def hidden_dim_size(self) -> int:
        """Hidden dimension size (``num_heads * head_size``)."""
        return self.shape_desc.nh * self.shape_desc.hs

    @property
    def slots_per_block(self) -> int:
        """Physical slots in one paged chunk of this group, detected from
        the registered KV tensors at registration time (the batch-size
        dimension, ``shape_desc.bs``)."""
        return self.shape_desc.bs

    def calculate_slots(self, num_tokens: int) -> int:
        """Calculate the number of slots for the specified number of
        tokens. Assuming the number of tokens are already aligned.
        """
        return num_tokens * self.slots_per_block // self.tokens_per_block


KVLayerGroupInfo = KernelGroupInfo  # Alias for compatibility


@dataclass
class ObjectGroupInfo:
    """Metadata for an 'object group'.

    An object group contains one or more kernel groups whose
    KV caches will be stored in the same memory object.

    This will be useful for dealing with sliding window or mamba
    KV caches that needs a different prefix matching logic from
    the full attention KV caches.
    """

    kernel_group_indices: list[int]
    """Indices of the kernel groups belonging to this object group, in the
    order they should be laid out in memory."""

    sw_size_chunks: int = -1
    """Cross-chunk sliding window size in LMCache chunks shared by every
    kernel group in this object group. ``-1`` means the kernel groups are
    not sliding-window attention."""


class KVLayerGroupsManager:
    """Partition a model's KV layers into transfer-kernel dispatch units.

    At construction time, every layer in ``kv_caches`` is bucketed by its
    :data:`LayerGroupIdentity` (``(kv_size, num_heads, head_size,
    block_size, engine_group_idx, dtype, engine_kv_format)``). Each bucket
    becomes one
    :class:`KernelGroupInfo` holding the layer indices, a shared
    :class:`PageBufferShapeDesc`, and the group's torch dtype.

    Downstream consumers (``VLLMPagedMemGPUConnectorV3``,
    ``GPUCacheContext``, the multiprocess server) iterate
    ``self._kernel_groups`` and issue one transfer-kernel launch per
    group. The manager itself is a pure metadata object — it does not
    own any GPU buffers or perform any transfers.

    Layout parsing is delegated entirely to
    :mod:`lmcache.v1.gpu_connector.utils`; this class only drives the
    grouping and look-up.
    """

    def __init__(
        self,
        kv_caches: "DiscoverableKVCache",
        engine_kv_formats: "Sequence[lmc_ops.EngineKVFormat]",
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        lmcache_tokens_per_chunk: int = 256,
        separate_object_groups: bool = True,
    ) -> None:
        """Partition the layers into kernel groups for this set of KV caches.

        Group order is deterministic across runs.

        Args:
            kv_caches: KV cache structure accepted by
                :func:`normalize_and_discover_per_layer_formats`.
            engine_kv_formats: One Engine KV format per layer (its length is the
                layer count), from
                :func:`normalize_and_discover_per_layer_formats`. A model that
                mixes formats across engine groups supplies the differing
                per-layer formats so each group is shaped with its own;
                homogeneous models repeat one shared format.
            engine_group_infos: Engine KV cache group metadata, one info per
                kernel group in kernel-group order, or empty.
            lmcache_logical_chunk_size: Tokens per LMCache chunk
            separate_object_groups: When True (default), split kernel groups
                into one object group per sliding-window size; when False, all
                kernel groups share a single full-attention object group.
        """
        # Import here to break a circular import via
        # lmcache.v1.gpu_connector.__init__ → metadata → kv_layer_groups.
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_num_blocks,
            make_page_buffer_shape_desc,
            resolve_block_stride_and_log_layout,
        )
        from lmcache.v1.multiprocess.group_view import get_engine_group_indices

        self._kernel_groups: list[KernelGroupInfo] = []
        self._object_groups: list[ObjectGroupInfo] = []

        num_layers = len(engine_kv_formats)
        if num_layers == 0:
            logger.debug("No KV caches available, skipping KV layer groups building")
            return

        per_layer_engine_group_idx = get_engine_group_indices(
            engine_group_infos, num_layers
        )
        groups_by_identity = group_layers_by_identity(
            kv_caches, engine_kv_formats, per_layer_engine_group_idx
        )

        # Engine group infos are produced by the same group_layers_by_identity
        # bucketing on the engine side, so they correspond 1:1, in order, to
        # the kernel groups built below.
        if engine_group_infos and len(engine_group_infos) != len(groups_by_identity):
            raise ValueError(
                f"Got {len(engine_group_infos)} engine group infos for "
                f"{len(groups_by_identity)} kernel groups; expecting one "
                "engine group info per kernel group"
            )

        # Emit groups in order of their first-appearing layer so that group
        # indices remain deterministic across runs.
        for group_idx, (identity, indices) in enumerate(groups_by_identity):
            bs = identity.block_size
            engine_group_idx = identity.engine_group_idx
            dt = identity.dtype
            # Format is part of the identity, so every layer in the group shares
            # it -- this is the whole group's format, by construction.
            group_format = identity.engine_kv_format
            # Block count is per engine group (each is its own block-id space), so
            # read it from this group's own tensor rather than a context-wide value.
            group_num_blocks = get_num_blocks([kv_caches[indices[0]]], group_format)
            block_stride_elems = resolve_block_stride_and_log_layout(
                kv_caches,
                group_format,
                layer_idx=indices[0],
                group_idx=group_idx,
            )
            shape_desc = make_page_buffer_shape_desc(
                kv_caches,
                group_format,
                layer_idx=indices[0],
                num_layers_in_group=len(indices),
                num_blocks=group_num_blocks,
                block_size=bs,
                block_stride_elems=block_stride_elems,
            )

            info = engine_group_infos[group_idx] if engine_group_infos else None
            if info is not None and tuple(indices) != tuple(info.layer_indices):
                raise ValueError(
                    f"group {group_idx}: engine group info covers layers "
                    f"{info.layer_indices}, but the kernel group covers "
                    f"layers {indices}"
                )

            # tokens_per_block comes from the engine's KV cache spec; when
            # absent, fall back to the physical slot count so the group is
            # treated as non-compressed (compress_ratio == 1).
            tokens_per_block = (
                info.tokens_per_block
                if info is not None and info.tokens_per_block > 0
                else bs
            )
            sw_size_tokens = info.sw_size_tokens if info is not None else -1

            self._validate_block_chunk_size_config(
                group_idx,
                slots_per_block=bs,
                tokens_per_block=tokens_per_block,
                lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
                sw_size_tokens=sw_size_tokens,
            )

            self._kernel_groups.append(
                KernelGroupInfo(
                    layer_indices=indices,
                    shape_desc=shape_desc,
                    dtype=dt,
                    engine_kv_format=group_format,
                    tokens_per_block=tokens_per_block,
                    engine_group_idx=engine_group_idx,
                    sw_size_tokens=sw_size_tokens,
                )
            )

        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._separate_object_groups = separate_object_groups

        # When True, sliding-window groups store/transfer FULL per-chunk KV
        self._full_sw_kv = False

        logger.info(
            "KV layer groups: ---\n%s\n---",
            "\n".join(repr(g) for g in self._kernel_groups),
        )

        # Detect the object groups
        self._object_groups = self._detect_object_groups(engine_group_infos)

    @property
    def kernel_groups(self) -> list[KernelGroupInfo]:
        """List of :class:`KernelGroupInfo`, one per kernel group."""
        return self._kernel_groups

    @property
    def num_blocks(self) -> int:
        """Paged block count, shared across kernel groups (one block-id space).

        Read from the first kernel group's ``shape_desc.nb``, which was computed
        from that group's own tensor and format -- not a guessed representative.
        Returns ``0`` when there are no kernel groups.
        """
        if not self._kernel_groups:
            return 0
        return self._kernel_groups[0].shape_desc.nb

    @property
    @lmcache_deprecate("`kv_layer_groups` is an outdated alias for `kernel_groups`")
    def kv_layer_groups(self) -> list[KernelGroupInfo]:
        """List of :class:`KernelGroupInfo`, one per kernel group."""
        return self._kernel_groups

    @property
    def num_kernel_groups(self) -> int:
        """Number of :class:`KernelGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self._kernel_groups)

    @property
    def object_groups(self) -> list[ObjectGroupInfo]:
        """List of :class:`ObjectGroupInfo`, one per object group."""
        return self._object_groups

    @property
    def num_object_groups(self) -> int:
        """Number of :class:`ObjectGroupInfo` entries."""
        return len(self._object_groups)

    @property
    @lmcache_deprecate("`num_groups` is an outdated alias for `num_kernel_groups`")
    def num_groups(self) -> int:
        """Number of :class:`KernelGroupInfo` entries.

        Zero if ``kv_caches`` had no layers at construction time.
        """
        return len(self._kernel_groups)

    def get_shape_desc(self, kernel_group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Return the :class:`PageBufferShapeDesc` for *kernel_group_idx*.

        Args:
            kernel_group_idx: 0-based kernel group index.

        Raises:
            IndexError: If *kernel_group_idx* is out of range.
        """
        return self._kernel_groups[kernel_group_idx].shape_desc

    @lmcache_deprecate("It does not have hybrid model support")
    def get_slots_per_chunk(self, kernel_group_idx: int) -> int:
        """Return the per-chunk slot count for *kernel_group_idx*.

        Args:
            kernel_group_idx: 0-based kernel group index.

        Note:
            This is a deprecated function because it does not have
            hybrid model support
        """
        group = self._kernel_groups[kernel_group_idx]
        return group.calculate_slots(self._lmcache_tokens_per_chunk)

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        """Return the per-chunk *transfer* slot count for *kernel_group_idx*.

        For sub-chunk sliding window groups, the transfer slots is smaller
        than the physical slots in a chunk.

        Args:
            kernel_group_idx: 0-based kernel group index.
        """
        group = self._kernel_groups[kernel_group_idx]
        sw_size = self.get_subchunk_sw_size_tokens(kernel_group_idx)
        return group.calculate_slots(sw_size)

    def enable_full_sw_kv(self) -> None:
        """Store/transfer the FULL per-chunk KV for sliding-window groups.

        Keeps every block so a chunk stays valid when reused at any position
        (default windowing keeps only each chunk's last sub-chunk window).
        """
        self._full_sw_kv = True

    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        """Return the sub-chunk sliding window size of a given kernel group.
        The size is measured in the number of tokens.

        This is for the models like DSV4 where the sliding window size is
        smaller than the tokens in a single lmcache chunk.

        Args:
            kernel_group_idx: 0-based kernel group index.

        Returns:
            The sub-chunk sliding window size. Will be the same as the
            chunk size for non-slding-window models or big-sliding-
            window models.
        """
        sw_size_tokens = self._kernel_groups[kernel_group_idx].sw_size_tokens
        # full_sw_kv: report the full chunk as the window so store/transfer keep
        # every block (chunk stays valid for reuse at any position).
        if self._full_sw_kv:
            return self._lmcache_tokens_per_chunk
        if sw_size_tokens == -1 or sw_size_tokens >= self._lmcache_tokens_per_chunk:
            return self._lmcache_tokens_per_chunk
        return sw_size_tokens

    def get_attn_desc(self) -> AttnWindowDesc:
        """Return the cross-chunk attention windows of all object groups.

        Returns:
            An :class:`AttnWindowDesc` with one entry per object group, in
            object-group order; the entry is ``-1`` for a non-sliding-window
            group.

        Note:
            With object-group separation disabled (the default), the result
            has a single full-attention entry.
        """
        if self._full_sw_kv:
            # full_sw_kv: every group reports full attention, no cross-chunk
            # window skipping (mirrors get_subchunk_sw_size_tokens).
            return AttnWindowDesc(num_chunks_in_sw=[-1] * len(self._object_groups))
        return AttnWindowDesc(
            num_chunks_in_sw=[
                w if w >= 1 else -1
                for w in (g.sw_size_chunks for g in self._object_groups)
            ]
        )

    def calculate_num_blocks(self, kernel_group_idx: int, num_tokens: int) -> int:
        """Calculate the number of blocks for a given number of tokens in a
        specified kernel group.

        Args:
            kernel_group_idx: 0-based index of the kernel group.
            num_tokens: The total number of tokens to be processed for the group.

        Returns:
            The number of blocks.

        Raises:
            IndexError: If *kernel_group_idx* is out of range.
        """
        group = self._kernel_groups[kernel_group_idx]
        # Physical slots for num_tokens, derived from the per-block geometry
        # (slots_per_block / tokens_per_block) rather than a compress ratio.
        num_physical_slots = (
            num_tokens * group.slots_per_block // group.tokens_per_block
        )
        return num_physical_slots // group.shape_desc.bs

    ### Helper methods
    def _detect_object_groups(
        self, engine_group_infos: "Sequence[EngineGroupInfo]"
    ) -> list[ObjectGroupInfo]:
        """Bucket kernel groups into object groups.

        Puts all kernel groups into a single object group when object-group
        separation is disabled (the default). Otherwise groups the kernel groups
        by sliding-window size measured in number of chunks.

        Args:
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Returns:
            One :class:`ObjectGroupInfo` per object group.
        """
        if not self._separate_object_groups:
            return [
                ObjectGroupInfo(
                    kernel_group_indices=list(range(len(self._kernel_groups)))
                )
            ]

        chunk_size = self._lmcache_tokens_per_chunk
        groups_by_sw_size: dict[int, list[int]] = defaultdict(list)
        for kernel_group_idx, group in enumerate(self._kernel_groups):
            if group.sw_size_tokens == -1:
                sw_size_chunks = -1
            else:
                sw_size_chunks = (group.sw_size_tokens + chunk_size - 1) // chunk_size
            groups_by_sw_size[sw_size_chunks].append(kernel_group_idx)
        return [
            ObjectGroupInfo(
                kernel_group_indices=kernel_group_indices,
                sw_size_chunks=sw_size_chunks,
            )
            for sw_size_chunks, kernel_group_indices in sorted(
                groups_by_sw_size.items(), key=lambda kv: kv[1][0]
            )
        ]

    @staticmethod
    def _validate_block_chunk_size_config(
        group_idx: int,
        slots_per_block: int,
        tokens_per_block: int,
        lmcache_tokens_per_chunk: int,
        sw_size_tokens: int = -1,
    ) -> None:
        """Validate the chunk size configuration against the slot and
        tokens block detected from the serving engine.

        Raises:
            ValueError: If one of the following conditions is met:
                - ``tokens_per_block`` is not a whole multiple of
                  ``slots_per_block``
                - ``lmcache_tokens_per_chunk`` is not a whole multiple of
                  ``tokens_per_block``
                - a sub-chunk sliding window is not a whole multiple of
                  ``tokens_per_block``
        """
        if tokens_per_block % slots_per_block != 0:
            raise ValueError(
                f"group {group_idx}: tokens_per_block {tokens_per_block} "
                f"must be a multiple of slots_per_block {slots_per_block}"
            )
        if lmcache_tokens_per_chunk % tokens_per_block != 0:
            raise ValueError(
                f"group {group_idx}: lmcache_tokens_per_chunk "
                f"{lmcache_tokens_per_chunk} must be a multiple of "
                f"tokens_per_block {tokens_per_block}"
            )
        if (
            0 < sw_size_tokens < lmcache_tokens_per_chunk
            and sw_size_tokens % tokens_per_block != 0
        ):
            raise ValueError(
                f"group {group_idx}: sub-chunk sliding window size "
                f"{sw_size_tokens} must be a multiple of tokens_per_block "
                f"{tokens_per_block}"
            )
        if slots_per_block != tokens_per_block:
            logger.info(
                "group %d: compressed (tokens_per_block=%d, slots_per_block=%d)",
                group_idx,
                tokens_per_block,
                slots_per_block,
            )


# ------------------------------------------------------------------ #
#  CLI shape-spec parser                                               #
# ------------------------------------------------------------------ #


def parse_kvcache_shape_spec(
    spec_str: str,
) -> list[KernelGroupInfo]:
    """Parse a ``--kvcache-shape-spec`` string into layer groups.

    **Grammar** (EBNF-ish)::

        spec        := group { ";" group }
        group       := "(" shape ")" ":" dtype ":" layer_count
        shape       := kv_size "," NB "," BS "," NH "," HS
        dtype       := "float16" | "float32" | "bfloat16" | "uint8"
        layer_count := positive integer

    **Field semantics** (names aligned with ``EngineKVFormat``; see
    :func:`lmcache.v1.gpu_connector.utils.get_engine_kv_shape_description`):

    * ``kv_size`` -- leading dim (``2`` for standard K/V, ``1`` for MLA).
    * ``NB`` -- ``num_blocks``: paged-KV block count.
    * ``BS`` -- ``block_size``: tokens per paged-KV block.
    * ``NH`` -- ``num_heads``: attention heads per layer.
    * ``HS`` -- ``head_size``: per-head hidden dim.
    * ``dtype`` -- element dtype (case-insensitive). ``uint8`` is used
      by FP8-quantized layouts.
    * ``layer_count`` -- number of consecutive layers sharing this
      group's geometry. Groups are concatenated in declaration order;
      ``layer_indices`` are assigned sequentially starting from 0.

    When consumed by the ``lmcache bench server`` CLI, ``NB``/``BS``
    from the spec take precedence over ``--num-blocks`` / ``--block-size``
    CLI flags when set to a positive value.

    **Examples**::

        # Single homogeneous group: 32 layers of standard K/V
        (2,1024,16,8,128):float16:32

        # Heterogeneous model: 30 dense layers + 2 MLA-ish layers
        (2,1024,16,8,128):float16:30;(1,1024,16,4,64):bfloat16:2

        # FP8-quantized KV cache
        (2,1024,16,8,128):uint8:32

    See also :func:`format_kvcache_shape_spec` for the inverse -- it
    turns a parsed group list back into a human-readable spec string
    (handy for CLI echo-back / debug logging).

    Returns:
        A list of :class:`KernelGroupInfo`, one per group.

    Raises:
        ValueError: Malformed spec, unknown dtype, or a shape with a
            wrong number of dimensions.
    """
    if not spec_str:
        raise ValueError("KV shape specification cannot be empty")

    groups: list[KernelGroupInfo] = []
    layer_offset = 0

    for group_spec in spec_str.split(";"):
        group_spec = group_spec.strip()
        if not group_spec:
            continue

        if not (group_spec.startswith("(") and "):" in group_spec):
            raise ValueError("Invalid group spec format: %s" % group_spec)

        shape_end = group_spec.find(")")
        shape_str = group_spec[1:shape_end]

        remaining = group_spec[shape_end + 2 :]  # Skip "):"
        parts = remaining.split(":")
        if len(parts) != 2:
            raise ValueError("Invalid group spec format: %s" % group_spec)

        dtype_str = parts[0].strip()
        layer_count_str = parts[1].strip()

        dtype_key = dtype_str.lower()
        if dtype_key not in DTYPE_MAP:
            raise ValueError(
                "Unrecognized dtype '%s' in group spec: %s. "
                "Supported: %s" % (dtype_str, group_spec, list(DTYPE_MAP.keys()))
            )
        try:
            shape = tuple(int(p.strip()) for p in shape_str.split(","))
            layer_count = int(layer_count_str)
        except ValueError as exc:
            raise ValueError("Invalid number in group spec: %s" % group_spec) from exc
        dtype = DTYPE_MAP[dtype_key]

        if len(shape) != 5:
            raise ValueError(
                "Shape must be a 5-tuple (kv_size,nb,bs,nh,hs): %s" % group_spec
            )
        kv_size, nb, bs, nh, hs = shape
        shape_desc = lmc_ops.PageBufferShapeDesc()
        shape_desc.kv_size = kv_size
        shape_desc.nl = layer_count
        shape_desc.nb = nb
        shape_desc.bs = bs
        shape_desc.nh = nh
        shape_desc.hs = hs
        shape_desc.element_size = dtype.itemsize
        set_shape_desc_dtype(shape_desc, dtype)

        indices = list(range(layer_offset, layer_offset + layer_count))
        groups.append(
            KernelGroupInfo(
                layer_indices=indices,
                shape_desc=shape_desc,
                dtype=dtype,
            )
        )
        layer_offset += layer_count

    if not groups:
        raise ValueError("No valid layer groups found in spec")

    return groups


def format_kvcache_shape_spec(groups: list[KernelGroupInfo]) -> str:
    """Format layer groups back into a ``--kvcache-shape-spec`` string.

    This is the inverse of :func:`parse_kvcache_shape_spec`; the
    result is round-trip safe (i.e. ``parse(format(x)) == x`` for any
    ``x`` that ``parse`` would produce).

    The returned string is also human-readable and is used by the
    ``lmcache bench server`` CLI to echo the resolved KV cache
    geometry at startup, so operators can verify that their spec was
    interpreted as intended.

    Example::

        >>> groups = parse_kvcache_shape_spec(
        ...     "(2,1024,16,8,128):float16:30;"
        ...     "(1,1024,16,4,64):bfloat16:2"
        ... )
        >>> format_kvcache_shape_spec(groups)
        '(2,1024,16,8,128):float16:30;(1,1024,16,4,64):bfloat16:2'

    Args:
        groups: Layer groups as returned by
            :func:`parse_kvcache_shape_spec`.

    Raises:
        ValueError: If *groups* is empty or contains an unsupported
            dtype (one that is not present in :data:`DTYPE_MAP`).
    """
    if not groups:
        raise ValueError("Cannot format an empty layer group list")

    # Invert DTYPE_MAP once: torch.dtype -> canonical string name.
    dtype_names = {v: k for k, v in DTYPE_MAP.items()}

    parts: list[str] = []
    for g in groups:
        sd = g.shape_desc
        try:
            dtype_str = dtype_names[g.dtype]
        except KeyError as exc:
            raise ValueError("dtype %s is not present in DTYPE_MAP" % g.dtype) from exc
        parts.append(
            "(%d,%d,%d,%d,%d):%s:%d"
            % (sd.kv_size, sd.nb, sd.bs, sd.nh, sd.hs, dtype_str, sd.nl)
        )
    return ";".join(parts)
