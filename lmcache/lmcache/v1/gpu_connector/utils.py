# SPDX-License-Identifier: Apache-2.0
# Format-dispatched geometry now lives in the ``kv_format`` package: each
# ``EngineKVFormat`` has a :class:`KVFormatSpec` (geometry accessors) and
# detection is split per engine in ``kv_format.detection``. The public
# functions below are a thin, backwards-compatible facade that delegates
# to ``get_spec`` / ``get_spec_class`` / ``detect_format`` so existing
# callers keep working unchanged.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, lmcache_deprecate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.kv_format import (
    concrete_shape,
    describe_shape,
    detect_format,
    extract_kv_cache_shapes,
    get_spec,
    get_spec_class,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
from lmcache.v1.platform.ops_types import set_shape_desc_dtype

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface

# First Party
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def assert_contiguous(tensor: torch.Tensor) -> None:
    """Assert that *tensor* has a contiguous physical layout with zero offset.

    LMCache transfer kernels assume logical and physical views match for
    coalesced memory accesses. Used at boundaries where we receive a
    tensor we can't or shouldn't permute (e.g. raw CUDA-IPC reconstruction
    in :class:`~lmcache.v1.platform.cuda.ipc_wrapper.RawCudaIPCWrapper`).

    Raises:
        ValueError: If *tensor* has a nonzero storage offset, or is
            non-contiguous.
    """
    if tensor.storage_offset() != 0:
        raise ValueError(f"expected storage_offset 0, got {tensor.storage_offset()}")
    if not tensor.is_contiguous():
        raise ValueError("tensor is not contiguous")


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    """
    Check if the GPU Connector needs to create an intermediate
    buffer on the GPU
    """
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def assert_layerwise_gpu_connector(gpu_connector: "GPUConnectorInterface"):
    """
    Assert that a GPU Connector is a layerwise connector.
    """
    # Import at runtime to avoid circular dependency
    # First Party
    from lmcache.v1.gpu_connector import gpu_connectors, xpu_connectors

    valid_connectors = (
        gpu_connectors.VLLMPagedMemLayerwiseGPUConnector,
        gpu_connectors.VLLMBufferLayerwiseGPUConnector,
        gpu_connectors.SGLangLayerwiseGPUConnector,
        xpu_connectors.VLLMPagedMemLayerwiseXPUConnector,
        xpu_connectors.VLLMBufferLayerwiseXPUConnector,
        xpu_connectors.SGLangLayerwiseXPUConnector,
    )

    assert isinstance(gpu_connector, valid_connectors)


def is_mla(engine_kv_format: "lmc_ops.EngineKVFormat") -> bool:
    """Return ``True`` for a Multi-head Latent Attention (MLA) layout."""
    return lmc_ops.is_mla(engine_kv_format)


def get_engine_kv_shape_description(engine_kv_format: "lmc_ops.EngineKVFormat") -> str:
    """Return a human-readable symbolic shape legend for the Engine KV format.

    Uses short names matching the ``EngineKVFormat`` enum convention:
    NB=num_blocks, NL=num_layers, BS=block_size, NH=num_heads,
    HS=head_size, PBS=page_buffer_size (NB*BS).
    """
    try:
        return describe_shape(engine_kv_format)
    except KeyError:
        return f"Unknown ({engine_kv_format})"


def get_attention_backend(engine_kv_format: "lmc_ops.EngineKVFormat") -> str:
    """Return a representative attention-backend label for the format.

    Diagnostic only. A format may be produced by several (engine,
    attention-backend) combinations; the spec lists them in
    ``attention_backends`` and this returns the first (canonical) one.
    """
    try:
        backends = get_spec_class(engine_kv_format).attention_backends
    except ValueError:
        return f"Unknown ({engine_kv_format})"
    return backends[0] if backends else f"Unknown ({engine_kv_format})"


def get_concrete_engine_kv_shape(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> str:
    """Return the shape with actual numeric values substituted.

    For example, instead of ``NL x [2, NB, BS, NH, HS]``
    this returns ``80 x [2, 2048, 128, 8, 128]``.
    """
    try:
        return get_spec(kv_caches, engine_kv_format).concrete_shape_str()
    except ValueError:
        return f"Unknown ({engine_kv_format})"


def get_concrete_engine_kv_shape_from_shape_desc(
    shape_desc: "lmc_ops.PageBufferShapeDesc",
    engine_kv_format: "lmc_ops.EngineKVFormat",
) -> str:
    """Return the concrete shape for a single kernel group's ``shape_desc``.

    Like :func:`get_concrete_engine_kv_shape`, but the numeric values are
    read from a per-group :class:`PageBufferShapeDesc` rather than from the
    whole ``kv_caches`` structure. This makes the result *group-accurate*:
    ``shape_desc.nl`` is the layer count of the group (not the model total),
    so each kernel group of a hybrid model reports its own shape.

    For example, instead of ``NL x [2, NB, BS, NH, HS]`` this returns
    ``80 x [2, 2048, 128, 8, 128]``.

    Args:
        shape_desc: The kernel group's shape descriptor. Sizes are read
            from its ``nl``/``nb``/``bs``/``nh``/``hs`` fields; fused
            page-buffer (``NBBS``) formats use ``nb * bs``.
        engine_kv_format: The format whose member name is the shape template.

    Returns:
        The shape string with numeric values substituted, or
        ``"Unknown (<format>)"`` for an unrecognised format.
    """
    sizes = {
        "NB": shape_desc.nb,
        "NL": shape_desc.nl,
        "BS": shape_desc.bs,
        "NH": shape_desc.nh,
        "HS": shape_desc.hs,
        "PBS": shape_desc.nb * shape_desc.bs,
    }
    try:
        return concrete_shape(engine_kv_format, lambda label: sizes[label])
    except KeyError:
        return f"Unknown ({engine_kv_format})"


def legible_print_engine_kv_format(engine_kv_format: "lmc_ops.EngineKVFormat"):
    """
    Print the Engine KV Format in a legible way
    """
    shape = get_engine_kv_shape_description(engine_kv_format)
    backend = get_attention_backend(engine_kv_format)
    if shape.startswith("Unknown"):
        logger.warning(f"Unknown Engine KV Format: {engine_kv_format}")
    else:
        logger.info("Engine KV Format: %s", shape)
        logger.info("Currently used by:\n  - %s", backend)


def normalize_kv_and_discover_format(
    kv_caches: DiscoverableKVCache,
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> tuple["lmc_ops.EngineKVFormat", DiscoverableKVCache]:
    """Normalize ``kv_caches`` into canonical form and discover its Engine KV format.

    Thin wrapper over
    :func:`lmcache.v1.gpu_connector.kv_format.detect_format`; see that
    function for the full contract.

    Args:
        kv_caches: The KV cache tensors (possibly nested lists of tensors).
        serving_engine: Which serving engine produced the caches.
        layout_hints: See :class:`LayoutHints`.

    Returns:
        ``(engine_kv_format, normalized_kv_caches)``. Callers must use the
        returned tensor structure for subsequent operations -- it shares
        storage with the input but may be a permuted view.
    """
    return detect_format(kv_caches, serving_engine, layout_hints)


def normalize_and_discover_per_layer_formats(
    kv_caches: "DiscoverableKVCache",
    layer_index_groups: "Sequence[Sequence[int]]",
    serving_engine: EngineType,
    layout_hints: "LayoutHints | None" = None,
) -> "tuple[DiscoverableKVCache, list[lmc_ops.EngineKVFormat]]":
    """Normalize the KV caches and return one Engine KV format per layer.

    Reports each layer's own format, so models whose layers do not all share one
    format -- e.g. a K+V main cache (``kv_size=2``) alongside a key-only MLA index
    cache (``kv_size=1``) -- get a correct per-layer format rather than a single
    model-wide one.

    Args:
        kv_caches: The registered KV caches: a per-layer list, or a single fused
            tensor for cross-layer formats.
        layer_index_groups: Layer indices of each engine group (one inner
            sequence per group). Empty means a single non-hybrid group.
        serving_engine: Which serving engine produced the caches.
        layout_hints: See :class:`LayoutHints`.

    Returns:
        ``(normalized_kv_caches, engine_kv_formats)``: the canonical KV cache
        structure and one format per layer (length equals the layer count),
        ready for :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`.
    """

    # Detect the whole structure once. A format that isn't a per-layer list (a
    # cross-layer tensor, or a K/V-split) is single-format -- return it whole.
    extracted_shapes = extract_kv_cache_shapes(kv_caches)
    if len(extracted_shapes) == 1:
        whole_format, whole_normalized = detect_format(
            kv_caches, serving_engine, layout_hints
        )
        if not lmc_ops.is_layer_list(whole_format):
            return whole_normalized, [whole_format] * get_num_layers(
                whole_normalized, whole_format
            )

    # Per-layer list: re-detect per engine group, split by tensor shape so a group
    # that mixes layouts gets the right format per layer.
    groups = layer_index_groups or [range(len(kv_caches))]
    detected: dict[int, tuple[DiscoverableKVCache, "lmc_ops.EngineKVFormat"]] = {}
    for indices in groups:
        layers_by_shape: dict[Hashable, list[int]] = {}
        for i in indices:
            shape = getattr(kv_caches[i], "shape", None)
            key = tuple(shape) if shape is not None else None
            layers_by_shape.setdefault(key, []).append(i)
        for same_shape_indices in layers_by_shape.values():
            fmt, normalized = detect_format(
                [kv_caches[i] for i in same_shape_indices],
                serving_engine,
                layout_hints,
            )
            for sub_idx, layer_idx in enumerate(same_shape_indices):
                detected[layer_idx] = (normalized[sub_idx], fmt)

    # A layer in no group (cross-layer KV sharing) keeps its own tensor and is
    # skipped downstream; give it any detected format so every layer has one.
    fallback_format = next(fmt for _, fmt in detected.values())
    normalized_per_layer = [
        detected[i][0] if i in detected else kv_caches[i] for i in range(len(kv_caches))
    ]
    engine_kv_formats = [
        detected[i][1] if i in detected else fallback_format
        for i in range(len(kv_caches))
    ]
    return normalized_per_layer, engine_kv_formats


def get_num_layers(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> int:
    """Return the number of layers from ``kv_caches``."""
    return get_spec(kv_caches, engine_kv_format).num_layers()


def get_num_blocks(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> int:
    """Return the number of blocks from ``kv_caches``.

    Raises:
        ValueError: For NBBS-fused formats with no separate block axis.
    """
    return get_spec(kv_caches, engine_kv_format).num_blocks()


def get_block_size(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the block size (tokens per block) for layer ``layer_idx``.

    ``layer_idx`` is honoured only for per-layer formats where BS may
    differ across layers (e.g. mixed-compression MLA pools). For
    cross-layer formats BS is shared across layers and ``layer_idx``
    is ignored.

    Raises:
        ValueError: For NBBS-fused formats with no separate block axis.
    """
    return get_spec(kv_caches, engine_kv_format).block_size(layer_idx)


@lmcache_deprecate(
    "page_buffer_size is only used by the legacy non-MP (in-process) connectors; "
    "the MP transfer path reads geometry from a per-group PageBufferShapeDesc instead"
)
def get_page_buffer_size(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> int:
    """Return the page buffer size (num_blocks * block_size) from ``kv_caches``."""
    return get_spec(kv_caches, engine_kv_format).page_buffer_size()


def get_kv_size(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> int:
    """Return the K/V axis size (2 for split K/V, 1 for fused)."""
    return get_spec(kv_caches, engine_kv_format).kv_size()


def get_num_heads(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the number of heads for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, engine_kv_format).num_heads(layer_idx)


def get_hidden_dim_size(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the hidden dimension for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, engine_kv_format).hidden_dim(layer_idx)


def get_head_size(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int = 0,
) -> int:
    """Return the head size for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, engine_kv_format).head_size(layer_idx)


def get_tokens_per_layer(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> int:
    """Return the number of tokens per layer (num_blocks * block_size)."""
    return get_spec(kv_caches, engine_kv_format).tokens_per_layer()


def get_elements_per_layer(
    kv_caches: DiscoverableKVCache, engine_kv_format: "lmc_ops.EngineKVFormat"
) -> int:
    """Return the number of elements per layer (both K and V for non-MLA)."""
    return get_spec(kv_caches, engine_kv_format).elements_per_layer()


def get_dtype(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int = 0,
) -> torch.dtype:
    """Return the dtype for a layer (defaults to layer 0)."""
    return get_spec(kv_caches, engine_kv_format).dtype(layer_idx)


def get_group_data_ptrs(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_indices: list[int],
) -> list[int]:
    """Return device pointers for a group of layers in kernel-expected order.

    See :meth:`KVFormatSpec.data_ptrs` for the per-format pointer-array
    shape (per-layer list, SGLang K-then-V, or single cross-layer base).

    Args:
        kv_caches: Full kv_caches structure.
        engine_kv_format: Format returned by :func:`normalize_kv_and_discover_format`.
        layer_indices: 0-based layer indices in the group, in kernel order.

    Returns:
        Device pointers (int), in kernel-expected order.

    Raises:
        ValueError: If *engine_kv_format* is not recognized.
    """
    return get_spec(kv_caches, engine_kv_format).data_ptrs(layer_indices)


def assert_is_vllm_flash_attn_or_flash_infer(
    engine_kv_format: "lmc_ops.EngineKVFormat",
):
    """
    Ensure that we have an Engine KV Cache Format
    that is either vLLM's flash attention or flash infer.
    """
    assert engine_kv_format in (
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        # Blocks-first fused K/V (HND and NHD): per-layer non-MLA layouts that
        # share this transfer path even though they are not literally flash-*.
        lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_CS,
        lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_CS,
    )


def assert_is_vllm_mla_or_flash_attn_or_flash_infer(
    engine_kv_format: "lmc_ops.EngineKVFormat",
) -> None:
    """
    Ensure that we have an Engine KV Cache Format that is either
    vLLM's MLA, flash attention, or flash infer.

    Accepted formats:
        - ``NL_X_TWO_NB_BS_NH_HS`` (flash attention, NHD)
        - ``NL_X_NB_TWO_BS_NH_HS`` (flash infer, NHD)
        - ``NL_X_TWO_NB_NH_BS_HS`` (flash attention, HND)
        - ``NL_X_NB_TWO_NH_BS_HS`` (flash infer, HND)
        - ``NL_X_NB_BS_HS`` (MLA)

    Raises:
        AssertionError: If *engine_kv_format* is not one of the accepted formats.
    """
    assert engine_kv_format in (
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS,
    )


def get_device(kv_caches: DiscoverableKVCache) -> torch.device:
    """Return the device of the KV cache tensors.

    Descends into any list nesting until a tensor is found; assumes all
    tensors in *kv_caches* live on the same device (true for every
    current :class:`EngineKVFormat`).
    """
    probe: DiscoverableKVCache = kv_caches
    while isinstance(probe, list):
        probe = probe[0]
    return probe.device


# Formats whose per-layer tensor dim-0 is the *block* axis AND for
# which we currently support dim-0 padding (e.g. DeepSeek V4
# compressor / indexer caches sharing a KV pool with larger attn
# groups). Today only the MLA layout (``NL_X_NB_BS_HS``, kv_size==1)
# is exercised by real mixed-compression workloads.
#
# ``NL_X_NB_TWO_BS_NH_HS`` *could* in principle also be the block
# axis on dim-0, but no real serving engine emits a padded layout of
# that format yet, and supporting it would require: (a) deciding
# (without a ground-truth example) which axis carries the padding
# -- NB boundary vs K<->V offset -- and (b) a coordinated change in
# ``attempt_permute_to_contiguous_view`` to let interior-dim padding
# through for that one format. Rather than ship an unverifiable code
# path, we keep ``NL_X_NB_TWO_BS_NH_HS`` out of this set, which means
# any padded tensor of that format will fail loudly via the
# non-block-axis dim-0-padding check below. Revisit and add a
# properly-tested branch when a concrete use case lands.
_BLOCK_AXIS_FORMATS: frozenset = frozenset(
    {
        lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
        lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS,
    }
)


def resolve_block_stride_and_log_layout(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int,
    group_idx: int,
) -> Optional[int]:
    """Resolve the per-block stride for a KV layer group and log its layout.

    Single entry point for :class:`KVLayerGroupsManager` to obtain the
    ``block_stride_elems`` value for :class:`PageBufferShapeDesc` and emit
    a one-shot layout audit line. All ``EngineKVFormat``-aware reasoning is
    kept here so callers never touch a "representative KV cache" tensor.

    * Block-axis formats (:data:`_BLOCK_AXIS_FORMATS`): ``stride(0)`` is
      the per-block step and is returned as-is. A value larger than the
      tight stride indicates dim-0 padding (e.g. DeepSeek V4 compressor
      caches sharing a KV pool with larger attn groups).
    * Other formats: dim-0 is not the block axis, so ``None`` is
      returned and ``shape_desc`` falls back to the tight stride. Any
      dim-0 padding in such formats is rejected with ``ValueError``
      since downstream kernels cannot honour it.

    Args:
        kv_caches: Full KV cache structure (already normalised).
        engine_kv_format: Format of ``kv_caches``.
        layer_idx: 0-based layer index used as the layout probe.
        group_idx: 0-based group index, used only for logging.

    Returns:
        ``stride(0)`` for block-axis formats; ``None`` otherwise.

    Raises:
        ValueError: Non-block-axis format carries dim-0 padding.
    """

    def _pick_layout_probe_tensor() -> torch.Tensor:
        # Layout probe only (shape/stride/storage_offset/dtype); not
        # the K/V slice fed to the transfer kernel.
        # - Cross-layer formats: ``kv_caches`` is the single backing
        #   tensor packing all layers along dim-1; indexing dim-0
        #   (= NB) would yield a per-block slice, so return the whole
        #   tensor (its ``stride(0)`` is the authoritative per-NB step).
        # - SGL MHA: outer list is K/V (length 2), inner list is
        #   per-layer; K & V share shape/stride by construction.
        # - Other formats: ``kv_caches`` is already a per-layer list.
        if lmc_ops.is_cross_layer(engine_kv_format):
            if not isinstance(kv_caches, torch.Tensor):
                raise TypeError(
                    "Cross-layer EngineKVFormat expects a single backing "
                    f"torch.Tensor, got {type(kv_caches).__name__}."
                )
            return kv_caches
        if lmc_ops.is_kv_list(engine_kv_format):
            return kv_caches[0][layer_idx]  # type: ignore[index,return-value]
        return kv_caches[layer_idx]  # type: ignore[index,return-value]

    rep = _pick_layout_probe_tensor()

    block_stride_elems: Optional[int]
    if engine_kv_format in _BLOCK_AXIS_FORMATS and rep.ndim > 0:
        block_stride_elems = int(rep.stride(0))
    else:
        # Non-block-axis format: detect forbidden dim-0 padding.
        if rep.ndim >= 2:
            tight_dim0 = 1
            for d in range(1, rep.ndim):
                tight_dim0 *= int(rep.shape[d])
            padding = int(rep.stride(0)) - tight_dim0
            if padding > 0:
                raise ValueError(
                    "resolve_block_stride_and_log_layout: group's probe "
                    f"tensor has dim-0 padding ({padding} elements per "
                    f"block) but engine_kv_format={engine_kv_format!r} is not "
                    "a supported dim-0-padded format (only "
                    "NL_X_NB_BS_HS is); downstream transfer kernels "
                    "cannot honour this padding and would read/write "
                    "wrong bytes. "
                    f"layer_idx={layer_idx}, shape={tuple(rep.shape)}, "
                    f"stride={tuple(rep.stride())}, "
                    f"tight_stride0={tight_dim0}, "
                    f"storage_offset={int(rep.storage_offset())}, "
                    f"dtype={rep.dtype}."
                )
        block_stride_elems = None

    # Best-effort layout audit log; the log line itself must not raise.
    shape = tuple(rep.shape)
    stride = tuple(rep.stride())
    try:
        inner = 1
        for s in shape[1:]:
            inner *= int(s)
        padding_per_block = stride[0] - inner if stride else 0
    except Exception:
        padding_per_block = -1
    try:
        storage_nbytes = rep.untyped_storage().nbytes()
    except Exception:
        storage_nbytes = -1
    logger.info(
        "Group %d first-layer tensor: layer_idx=%d shape=%s "
        "stride=%s is_contiguous=%s dtype=%s device=%s "
        "storage_offset=%d numel=%d storage_nbytes=%d "
        "padding_per_block=%d",
        group_idx,
        layer_idx,
        shape,
        stride,
        rep.is_contiguous(),
        rep.dtype,
        rep.device,
        rep.storage_offset(),
        rep.numel(),
        storage_nbytes,
        padding_per_block,
    )

    return block_stride_elems


def make_page_buffer_shape_desc(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
    layer_idx: int,
    num_layers_in_group: int,
    num_blocks: int,
    block_size: int,
    block_stride_elems: Optional[int] = None,
) -> "lmc_ops.PageBufferShapeDesc":
    """Build a :class:`PageBufferShapeDesc` from a representative layer.

    Args:
        kv_caches: Full kv_caches structure.
        engine_kv_format: Format returned by :func:`normalize_kv_and_discover_format`.
        layer_idx: 0-based index of the representative layer.
        num_layers_in_group: Number of layers in the group (``nl``).
        num_blocks: Number of paged blocks (``nb``).
        block_size: Tokens per block (``bs``).
        block_stride_elems: Physical per-block stride in *elements*
            (= ``tensor.stride(0)`` of the representative layer). Pass
            the real value whenever the group's KV pool may be
            dim-0-padded (e.g. DeepSeek V4 compressor/indexer caches
            sharing a row width with a larger group in the same pool);
            otherwise downstream transfer kernels will skip into
            padding and corrupt data. Leave as ``None`` for unpadded
            pools -- the kernel's ``per_block_stride()`` fallback
            (block_stride_elems <= 0) will reconstruct the tight
            stride from ``kv_size`` and ``scalars_per_block`` itself,
            so we don't duplicate that arithmetic on the Python side.

    Returns:
        A populated ``PageBufferShapeDesc``.
    """
    desc = lmc_ops.PageBufferShapeDesc()
    desc.kv_size = get_kv_size(kv_caches, engine_kv_format)
    desc.nl = num_layers_in_group
    desc.nb = num_blocks
    desc.bs = block_size
    desc.nh = (
        1
        if is_mla(engine_kv_format)
        else get_num_heads(kv_caches, engine_kv_format, layer_idx)
    )
    desc.hs = get_head_size(kv_caches, engine_kv_format, layer_idx)
    dtype = get_dtype(kv_caches, engine_kv_format, layer_idx)
    desc.element_size = dtype.itemsize
    # The C++ PageBufferShapeDesc has no ``dtype`` field, but the pure-Python
    # CPU fallback does -- and needs it to disambiguate float16 vs bfloat16
    # (both have itemsize 2, so element_size alone is not enough). Best-effort.
    set_shape_desc_dtype(desc, dtype)

    resolved_stride = int(block_stride_elems) if block_stride_elems else 0
    desc.block_stride_elems = resolved_stride
    return desc


def _split_token2d_kv(token2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Accepts either:
      - [2, T, D]
      - [T, 2, D]
    Returns:
      - k_tok: [T, D]
      - v_tok: [T, D]
    """
    if token2d.dim() != 3:
        raise ValueError(f"Expected token2d dim=3, got {token2d.shape}")
    if token2d.shape[0] == 2:  # [2, T, D]
        return token2d[0], token2d[1]
    if token2d.shape[1] == 2:  # [T, 2, D]
        return token2d[:, 0, :], token2d[:, 1, :]
    raise ValueError(f"Unrecognized token2d layout: {token2d.shape}")


def _get_head_size_view(
    kv_cache_layer: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    *,
    use_mla: bool,
    engine_kv_format: Optional["lmc_ops.EngineKVFormat"] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns flattened views for index_copy/index_select.

    If engine_kv_format is provided, use it to interpret tensor layout explicitly.
    If not provided, fall back to current structural behavior:
      - MLA: expects Tensor [P, B, HS]
      - Non-MLA: expects either
          * Tensor [2, P, B, NH, HS]  OR
          * (k, v) tuple each [P, B, NH, HS]
        (and also supports [P, 2, B, NH, HS] as a safe extension)
    """
    # -------------------------
    # MLA
    # -------------------------
    if use_mla:
        if not isinstance(kv_cache_layer, torch.Tensor):
            raise ValueError("MLA expects kv_cache_layer as Tensor")
        if kv_cache_layer.dim() != 3:
            raise ValueError(f"MLA expects 3D [P,B,HS], got {kv_cache_layer.shape}")
        p, b, hs = kv_cache_layer.shape
        return kv_cache_layer.view(p * b, hs)

    # -------------------------
    # non-MLA (K/V)
    # -------------------------
    # If already provided (k, v) in canonical per-layer form, no format needed.
    if not isinstance(kv_cache_layer, torch.Tensor):
        k, v = kv_cache_layer
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(f"Expected (k,v) 4D [P,B,NH,HS], got {k.shape}, {v.shape}")
        p, b, nh, hs = k.shape
        if v.shape != (p, b, nh, hs):
            raise ValueError(f"k/v shape mismatch: {k.shape} vs {v.shape}")
        return k.view(p * b, nh * hs), v.view(p * b, nh * hs)

    t = kv_cache_layer
    if t.dim() != 5:
        raise ValueError(f"Expected 5D tensor for non-MLA, got {t.shape}")

    # If we have the format enum, decode explicitly.
    if engine_kv_format is not None:
        if engine_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
            # per-layer: [2, NB, BS, NH, HS]
            if t.shape[0] != 2:
                raise ValueError(
                    f"{engine_kv_format} expects [2,NB,BS,NH,HS], got {t.shape}"
                )
            k, v = t[0], t[1]  # [NB,BS,NH,HS]

        elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
            # per-layer: [NB, 2, BS, NH, HS]
            if t.shape[1] != 2:
                raise ValueError(
                    f"{engine_kv_format} expects [NB,2,BS,NH,HS], got {t.shape}"
                )
            k, v = t[:, 0], t[:, 1]  # [NB,BS,NH,HS]

        else:
            # Other formats are either MLA-only or require upstream normalization.
            raise NotImplementedError(
                f"engine_kv_format={engine_kv_format} not supported in non-MLA "
                "path here. Normalize to (k,v) tuple [NB,BS,NH,HS] per-layer "
                "before calling."
            )

    else:
        # No enum available: Assumed [2,P,B,H,D] (or [2,NB,BS,NH,HS] per-layer).
        # Also accept [P,2,B,H,D] (or [NB,2,BS,NH,HS]) to be more robust.
        if t.shape[0] == 2:
            k, v = t[0], t[1]
        elif t.shape[1] == 2:
            k, v = t[:, 0], t[:, 1]
        else:
            raise ValueError(
                f"engine_kv_format is None and tensor does not look like stacked KV. "
                f"Expected axis0==2 or axis1==2, got {t.shape}"
            )

    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"Expected k/v 4D [NB,BS,NH,HS], got {k.shape}, {v.shape}")

    nb, bs, nh, hs = k.shape
    if v.shape != (nb, bs, nh, hs):
        raise ValueError(f"k/v shape mismatch after decode: {k.shape} vs {v.shape}")

    return k.view(nb * bs, nh * hs), v.view(nb * bs, nh * hs)
