# SPDX-License-Identifier: Apache-2.0
"""Centralized edits to vLLM kv cache specs.

This is needed to mask out attention-specific details while making sure that
LMCache can still store / load KV cache correctly.

Currently there are three edits, one per :class:`KVCacheGroupEdit` subclass
below. All are for Mamba-hybrid models; the registry is only consulted when
``kv_cache_config.has_mamba_layers``. :func:`validate_kv_cache_groups`
additionally rejects, at startup, group specs the transfer path cannot serve
correctly yet (see its docstring).

Reference design: vLLM PR #42828 ("[KVConnector][DSV4] HMA support for
Mooncake store connector") solves the same problem class for an external KV
store. Check it again when working on any of the following:

- Extending :func:`validate_kv_cache_groups` (its connector validates and
  rejects unsupported specs up front, with one aggregated error).
- Per-group store/load masks (SWA / Mamba tail-only transfer, the
  "sliding-window load-plan trimming" deferred in the hybrid design doc):
  its ``MooncakeStoreCoordinator.store_mask`` / ``load_mask`` derive masks
  from vLLM's own per-spec managers. Requires per-group objects in LMCache
  (``ObjectKey.object_group_id``, LMCache #3608).
- Hit-length computation for hybrid models: its ``ExternalCachedBlockPool``
  duck-types vLLM's ``BlockPool`` over a ``(group_id, hash)`` existence set
  and reuses ``KVCacheSpecRegistry`` manager classes, so promised hits are
  consumable by construction. NOTE: LMCache's lookup doubles as prefetch, so
  vLLM's lookup-first-then-trim flow does not map directly -- this needs its
  own design before borrowing.
- Eagle + HMA: see its ``apply_eagle`` notes (the eagle last-block prune must
  be applied exactly once between hit-length and mask computation).
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from typing import TypeAlias

# Third Party
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints

logger = init_logger(__name__)

# One registered cache value: a paged KV tensor, or [conv_state, ssm_state]
# for Mamba layers.
RegisteredKVCache: TypeAlias = torch.Tensor | list[torch.Tensor]

# Synthetic head count for a reinterpreted page. The page is opaque bytes, so
# one "head" holding the whole per-(K/V) slab is enough; head_size is derived
# to fill the page.
_SYNTHETIC_NUM_HEADS = 1

# Standard-paged (non-MLA) attention kinds eligible for the sub-paged edit.
_SUBPAGEABLE_ATTENTION_KINDS = frozenset(
    {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.SLIDING_WINDOW,
        KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
    }
)


def _declares_slot_compression(spec: KVCacheSpec) -> bool:
    """Return whether a spec declares slot compression (must not be edited).

    Covers ``MLAAttentionSpec.compress_ratio > 1`` (DeepSeek-V4 slot packing)
    and ``TQFullAttentionSpec.tq_slot_size > 0`` (TurboQuant slots); such
    groups belong to the compression path in ``lmcache.v1.kv_layer_groups``.
    """
    return (
        getattr(spec, "compress_ratio", 1) > 1 or getattr(spec, "tq_slot_size", 0) > 0
    )


def _leaf_specs(spec: KVCacheSpec) -> list[KVCacheSpec]:
    """Return a group spec's leaf specs, unwrapping ``UniformTypeKVCacheSpecs``."""
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict):
        return list(inner.values())
    return [spec]


def validate_kv_cache_groups(kv_cache_config: KVCacheConfig | None) -> None:
    """Reject KV cache group specs the transfer path cannot serve correctly.

    Rejected, with one aggregated error listing every offending group:

    - ``CrossAttentionSpec`` (encoder-decoder caches).
    - Mamba groups with ``mamba_cache_mode != "align"``: other modes keep no
      reusable per-block state snapshots.

    Specs declaring slot compression (``compress_ratio > 1`` /
    ``tq_slot_size > 0``, e.g. DeepSeek-V4) are NOT rejected: they are served
    by the compression path in ``lmcache.v1.kv_layer_groups`` and merely
    skipped by the edits here (see ``_declares_slot_compression``).

    Args:
        kv_cache_config: vLLM ``KVCacheConfig``; ``None`` skips validation
            (callers without the config validate again at registration).

    Raises:
        ValueError: Listing every unsupported group and why.
    """
    if kv_cache_config is None:
        return
    unsupported: list[str] = []
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        for spec in _leaf_specs(group.kv_cache_spec):
            kind = get_kv_cache_spec_kind(spec)
            if kind == KVCacheSpecKind.CROSS_ATTENTION:
                unsupported.append(f"group {group_idx}: CrossAttentionSpec")
            elif (
                kind == KVCacheSpecKind.MAMBA
                and getattr(spec, "mamba_cache_mode", "none") != "align"
            ):
                unsupported.append(
                    f"group {group_idx}: MambaSpec with mamba_cache_mode="
                    f"'{getattr(spec, 'mamba_cache_mode', 'none')}' "
                    f"(only 'align' keeps reusable state snapshots)"
                )
    if unsupported:
        raise ValueError(
            "LMCache cannot serve this model's KV cache groups: "
            + "; ".join(unsupported)
            + ". See lmcache/integration/vllm/kv_cache_group_edits.py."
        )


def _synthetic_attention_shape(elems_per_page: int, block_size: int) -> tuple[int, int]:
    """Factor a page's element count into the synthetic attention layout.

    Args:
        elems_per_page: Total elements in one page (one logical block).
        block_size: Logical block size (tokens per page).

    Returns:
        ``(num_heads, head_size)`` such that
        ``2 * block_size * num_heads * head_size == elems_per_page``.

    Raises:
        ValueError: If the page size does not factor into the target shape.
    """
    denom = 2 * block_size * _SYNTHETIC_NUM_HEADS
    if elems_per_page % denom != 0:
        raise ValueError(
            f"page ({elems_per_page} elems) does not factor into "
            f"(2, block_size={block_size}, num_heads={_SYNTHETIC_NUM_HEADS}, head_size)"
        )
    return _SYNTHETIC_NUM_HEADS, elems_per_page // denom


class KVCacheGroupEdit(ABC):
    """One structural edit rule for a KV cache group's registered cache.

    ``matches`` must be side-effect free and decide purely from the vLLM spec
    and the registered cache value; ``apply`` must return a view over the same
    storage (never a copy). ``name`` labels the rule in logs.
    """

    name: str

    @abstractmethod
    def matches(self, spec: KVCacheSpec, kv_cache: RegisteredKVCache) -> bool:
        """Return whether this rule applies to the layer's registered cache.

        Args:
            spec: The layer's vLLM KV cache spec (from its group).
            kv_cache: The layer's registered cache value -- a tensor, or a
                list of tensors for Mamba layers.
        """

    @abstractmethod
    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        layout_hints: LayoutHints,
    ) -> torch.Tensor:
        """Return the edited view for a layer this rule matched.

        Args:
            spec: The layer's vLLM KV cache spec (from its group).
            kv_cache: The layer's registered cache value.
            layout_hints: The layout hints from vLLM (see :class:`LayoutHints`).

        Raises:
            ValueError: If the cache's layout violates the rule's invariants.
        """


class _MambaPageViewEdit(KVCacheGroupEdit):
    """Convert a Mamba page to its equivalent: a sliding-window-style layer.

    A Mamba layer registers ``[conv_state, ssm_state]`` -- two tensors with
    different shapes and dtypes sharing one padded page per block
    (``conv | ssm | pad``) -- which LMCache's attention-shaped transfer path
    cannot represent. Each page is one recurrent state snapshot, equivalent
    for caching purposes to one block of a sliding-window attention layer with
    window == block_size: only the last matched block is ever consumed. The
    edit reinterprets the page buffer as one
    ``[#blocks, 2, block_size, 1, head_size]`` tensor in the conv state's
    dtype, with ``head_size`` derived to fill the page exactly.

    The view's dims are addressing metadata only; the bytes are opaque
    (conv | ssm | pad, not K/V), so content-aware processing does not apply.
    """

    name = "mamba-page-view"

    def matches(self, spec: KVCacheSpec, kv_cache: RegisteredKVCache) -> bool:
        return get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA and isinstance(
            kv_cache, list
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
    ) -> torch.Tensor:
        # vLLM lays out one padded page per block as (conv | ssm | pad), and
        # the conv state is the view that starts at the page base: its leading
        # dim is the block count and its per-block stride is one full page.
        # Re-striding it therefore covers the whole page, ssm and pad included.
        if not isinstance(kv_cache, list) or not kv_cache:
            raise ValueError(
                f"expected a Mamba [conv_state, ssm_state] tensor list, "
                f"got {type(kv_cache).__name__}"
            )
        conv_state = kv_cache[0]
        if conv_state.storage_offset() != 0:
            raise ValueError(
                f"Mamba conv state must view the page base, got "
                f"storage_offset={conv_state.storage_offset()}"
            )
        if conv_state.stride(0) * conv_state.element_size() != spec.page_size_bytes:
            raise ValueError(
                f"Mamba conv state per-block stride "
                f"({conv_state.stride(0) * conv_state.element_size()} bytes) "
                f"does not equal the page size ({spec.page_size_bytes} bytes)"
            )
        num_blocks = conv_state.shape[0]
        elems_per_page = spec.page_size_bytes // conv_state.element_size()
        num_heads, head_size = _synthetic_attention_shape(
            elems_per_page, spec.block_size
        )
        flat = conv_state.as_strided((num_blocks, elems_per_page), (elems_per_page, 1))
        return flat.reshape(num_blocks, 2, spec.block_size, num_heads, head_size)


class _SubpagedAttentionViewEdit(KVCacheGroupEdit):
    """Re-view a kernel-paged attention tensor as logical-block pages.

    For a Mamba-hybrid model vLLM inflates the attention block size to align
    with the Mamba page (e.g. 544 for Qwen3.5-0.8B), and that size is used for
    all prefix-caching logic at the scheduler. But at the worker the attention
    kernel has to run at block size 32 for numerical stability (vLLM #27753,
    working around the NaN-propagation issue
    Dao-AILab/flash-attention#1974), so the registered tensor is paged as

        ``[#blocks, 2, 32, #heads, head_size]``

    which makes LMCache detect block size 32, mistake the group for a
    DeepSeek-compression layer (``block size < scheduler block size``), and
    corrupt the store/retrieve. Fix: re-view the tensor at the scheduler
    block size (one logical block = its 17 contiguous kernel pages),

        ``[#blocks / 17, 2, 544, 1, head_size']``

    Cost: before this fix ``kv_caches[:, 0]`` is just the K tensor; after, it
    interleaves K and V at kernel-page granularity. The bytes round-trip
    correctly (store and retrieve share the mapping), but the dims are no
    longer semantic, so content-aware processing does not apply.
    """

    name = "subpaged-attention-view"

    def matches(self, spec: KVCacheSpec, kv_cache: RegisteredKVCache) -> bool:
        return (
            # Standard-paged attention only; MLA layouts and declared slot
            # compression (DeepSeek) belong to other transfer paths.
            get_kv_cache_spec_kind(spec) in _SUBPAGEABLE_ATTENTION_KINDS
            and not _declares_slot_compression(spec)
            # (num_blocks, 2, block_size, num_heads, head_size) layout whose
            # block dim disagrees with the scheduler block-id unit -- the
            # backend re-paged the tensor at its kernel block size.
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 5
            and kv_cache.shape[2] != spec.block_size
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
    ) -> torch.Tensor:
        """Re-view ``kv_cache`` at logical-block granularity.

        The tensor is kernel-paged as ``(num_kernel_pages, 2,
        kernel_block_size, num_kv_heads, head_size)``; the result is
        ``(num_logical_blocks, 2, spec.block_size, num_heads, head_size)``
        over the same storage.

        Raises:
            ValueError: If the layout is not the expected kernel-paged shape,
                the sizes do not divide evenly, or the kernel pages of one
                logical block do not tile its page bytes exactly (which would
                indicate an undeclared packed layout that must not be edited).
        """
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.shape[1] != 2:
            got = (
                tuple(kv_cache.shape)
                if isinstance(kv_cache, torch.Tensor)
                else type(kv_cache).__name__
            )
            raise ValueError(
                f"expected a (num_blocks, 2, block_size, num_heads, head_size) "
                f"attention KV tensor, got {got}"
            )
        logical_block_size = spec.block_size
        kernel_block_size = kv_cache.shape[2]
        if logical_block_size % kernel_block_size != 0:
            raise ValueError(
                f"logical block size {logical_block_size} is not a multiple of "
                f"kernel block size {kernel_block_size}"
            )
        ratio = logical_block_size // kernel_block_size

        num_kernel_pages = kv_cache.shape[0]
        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel block ratio {ratio}"
            )
        kernel_page_bytes = kv_cache.shape[1:].numel() * kv_cache.element_size()
        if kernel_page_bytes * ratio != spec.page_size_bytes:
            raise ValueError(
                f"{ratio} kernel pages ({kernel_page_bytes * ratio} bytes) do "
                f"not tile the logical page ({spec.page_size_bytes} bytes)"
            )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "kernel-paged attention KV tensor must be contiguous to "
                "re-view as logical pages"
            )

        num_blocks = num_kernel_pages // ratio
        elems_per_page = spec.page_size_bytes // kv_cache.element_size()
        num_heads, head_size = _synthetic_attention_shape(
            elems_per_page, logical_block_size
        )
        return kv_cache.view(num_blocks, 2, logical_block_size, num_heads, head_size)


######################
# vLLM 0.26.0 or later
######################


class _MambaUnifiedViewEdit(KVCacheGroupEdit):
    """Re-view mamba's unified state as a single attention tensor

    This is for vLLM >= 0.26.0, where the unified KV cache layout is implemented.

    In this case, Mamba's KV layer will be a single tensor with the shape of:
    - [num_blocks, 1, 1, context_size]
    Where the context size equals to vllm_block_size * ``head_size''


    What we do here is to convert the shape to
    - [num_blocks, 1, vllm_block_size, head_size]
    """

    name = "mamba-unified-view"

    def matches(self, spec: KVCacheSpec, kv_cache: RegisteredKVCache) -> bool:
        """
        Matches the mamba kv cache with unified layout.
        """
        return (
            get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 4
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        layout_hints: LayoutHints,
    ) -> torch.Tensor:
        """
        Convert [num_blocks, 1, 1, context_size] to
        [num_blocks, 1, vllm_block_size, head_size] for HND layout, or
        [num_blocks, vllm_block_size, 1, head_size] for NHD layout.
        """
        assert isinstance(kv_cache, torch.Tensor), (
            "single-layer KV cache must be a torch.Tensor"
        )
        kv_layout = layout_hints.get("kv_layout", "none")
        if kv_layout == "NHD":
            return kv_cache.view(kv_cache.shape[0], spec.block_size, 1, -1)
        elif kv_layout == "HND":
            return kv_cache.view(kv_cache.shape[0], 1, spec.block_size, -1)
        else:
            raise ValueError(
                f"Unsupported kv_layout: {kv_layout}. Only NHD and HND are supported."
            )


class _SubpagedMLAAttentionViewEdit(KVCacheGroupEdit):
    """Re-view a kernel-paged attention tensor as logical-block pages.

    For vLLM 0.26 or later

    Example on Kimi K3 , where 768 is the block size
    - Input: [N * 12, 64, 576]
    - Output: [N, 768, 576] (where 768 = 12 * 64)
    """

    name = "subpaged-mla-attention-view"

    def matches(self, spec: KVCacheSpec, kv_cache: RegisteredKVCache) -> bool:
        return (
            get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MLA_ATTENTION
            and not _declares_slot_compression(spec)
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 3
            and kv_cache.shape[1] != spec.block_size
        )

    def apply(
        self,
        spec: KVCacheSpec,
        kv_cache: RegisteredKVCache,
        _layout_hints: LayoutHints,
    ) -> torch.Tensor:
        """Re-view ``kv_cache`` at logical-block granularity.

        The tensor is kernel-paged as ``(num_kernel_pages, 2,
        kernel_block_size, num_kv_heads, head_size)``; the result is
        ``(num_logical_blocks, 2, spec.block_size, num_heads, head_size)``
        over the same storage.

        Raises:
            ValueError: If the layout is not the expected kernel-paged shape,
                the sizes do not divide evenly, or the kernel pages of one
                logical block do not tile its page bytes exactly (which would
                indicate an undeclared packed layout that must not be edited).
        """
        assert isinstance(kv_cache, torch.Tensor)
        logical_block_size = spec.block_size
        kernel_block_size = kv_cache.shape[1]
        if logical_block_size % kernel_block_size != 0:
            raise ValueError(
                f"logical block size {logical_block_size} is not a multiple of "
                f"kernel block size {kernel_block_size}"
            )
        ratio = logical_block_size // kernel_block_size

        num_kernel_pages = kv_cache.shape[0]
        if num_kernel_pages % ratio != 0:
            raise ValueError(
                f"kernel page count {num_kernel_pages} is not a multiple of "
                f"the logical/kernel block ratio {ratio}"
            )
        kernel_page_bytes = kv_cache.shape[1:].numel() * kv_cache.element_size()
        if kernel_page_bytes * ratio != spec.page_size_bytes:
            raise ValueError(
                f"{ratio} kernel pages ({kernel_page_bytes * ratio} bytes) do "
                f"not tile the logical page ({spec.page_size_bytes} bytes)"
            )
        if not kv_cache.is_contiguous():
            raise ValueError(
                "kernel-paged attention KV tensor must be contiguous to "
                "re-view as logical pages"
            )

        return kv_cache.view(num_kernel_pages // ratio, logical_block_size, -1)


# Rule registry, in match priority order.
_EDITS: tuple[KVCacheGroupEdit, ...] = (
    _MambaUnifiedViewEdit(),
    _MambaPageViewEdit(),
    _SubpagedMLAAttentionViewEdit(),
    _SubpagedAttentionViewEdit(),
)


def apply_kv_cache_group_edits(
    kv_cache_config: KVCacheConfig | None,
    kv_caches: Mapping[str, RegisteredKVCache],
    layout_hints: LayoutHints,
) -> dict[str, RegisteredKVCache]:
    """Apply all KV cache group metadata edits for LMCache registration.

    Each layer is checked against the ``_EDITS`` rules (first match wins) and
    re-viewed by the matching rule; layers matching no rule pass through
    unchanged. ``None`` configs and configs without Mamba groups are returned
    as-is (as a dict): all current rules only apply to Mamba-hybrid models.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (read for per-group specs).
        kv_caches: Registered tensors keyed by layer name. Mamba entries are
            ``[conv_state, ssm_state]`` lists; others are single tensors.
        layout_hints: layout hints from vLLM (see :class:`LayoutHints`).

    Returns:
        A new ``dict`` with edited layers re-viewed, others untouched.

    Raises:
        ValueError: If the groups fail :func:`validate_kv_cache_groups`, or a
            matched layer's cache layout violates its rule's invariants (see
            each rule's ``apply``).
    """
    # Backstop for connectors initialized without a kv_cache_config.
    validate_kv_cache_groups(kv_cache_config)
    if kv_cache_config is None or not kv_cache_config.has_mamba_layers:
        return dict(kv_caches)

    edited = dict(kv_caches)
    counts: Counter[str] = Counter()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            for edit in _EDITS:
                if edit.matches(spec, kv_caches[name]):
                    edited[name] = edit.apply(spec, kv_caches[name], layout_hints)
                    counts[edit.name] += 1
                    break
    logger.info(
        "KV cache group edits applied: %s",
        dict(counts) if counts else "none",
    )
    return edited
