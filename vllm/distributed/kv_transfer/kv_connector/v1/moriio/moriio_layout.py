# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Mapping
from typing import NamedTuple

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)


class LayerTransferGeometry(NamedTuple):
    num_blocks: int
    block_size: int
    block_len: int
    slot_size_bytes: int
    block_stride: int
    local_kv_stride: int | None
    remote_kv_stride: int | None
    transfers_per_block: int
    regions_per_block: int
    split_kv_regions: bool


class MambaTransferGeometry(NamedTuple):
    """Geometry of a hybrid (mamba/KDA) layer's recurrent state.

    A KDA layer's cache is a ``(conv_state, ssm_state)`` tuple; each is a
    slot-strided view whose slot stride (``stride(0)``) is larger than the
    per-slot element count, so a slot's bytes live at
    ``slot * slot_stride * element_size`` and span ``slot_bytes``. The two
    tensors are registered as two separate MoRIIO regions per layer.
    """

    num_slots: int
    conv_slot_stride: int
    ssm_slot_stride: int
    conv_slot_bytes: int
    ssm_slot_bytes: int
    conv_element_size: int
    ssm_element_size: int
    conv_region_len: int
    ssm_region_len: int


def get_mamba_transfer_geometry(
    layer_name: str,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> MambaTransferGeometry:
    """Derive the slot-strided geometry of a KDA layer's conv + ssm state.

    ``conv``/``ssm`` are the two tensors of the layer's ``(conv, ssm)`` cache
    tuple. Slot byte offset is ``slot * slot_stride * element_size`` and each
    slot spans ``subtensor[0].numel() * element_size`` bytes; the whole tensor
    is registered as one region (``*_region_len`` bytes).
    """
    return MambaTransferGeometry(
        num_slots=conv.shape[0],
        conv_slot_stride=conv.stride(0),
        ssm_slot_stride=ssm.stride(0),
        conv_slot_bytes=conv[0].numel() * conv.element_size(),
        ssm_slot_bytes=ssm[0].numel() * ssm.element_size(),
        conv_element_size=conv.element_size(),
        ssm_element_size=ssm.element_size(),
        # Registered region must span the full slot-strided extent: stride(0)
        # may exceed the per-slot element count, so numel() would under-size the
        # region and truncate the highest-slot transfers. shape[0]*stride(0)
        # covers every byte a ``slot * slot_stride * elem`` offset can address.
        # Cap at the bytes remaining from each view's storage offset so a
        # shared conv+ssm page buffer (dev19253 packed layout: ssm sits at
        # a non-zero intra-buffer offset) is not registered past the buffer
        # end; for separate buffers (offset 0, full storage) it is the same
        # slot-strided extent as before.
        conv_region_len=min(
            conv.shape[0] * conv.stride(0) * conv.element_size(),
            conv.untyped_storage().nbytes()
            - conv.storage_offset() * conv.element_size(),
        ),
        ssm_region_len=min(
            ssm.shape[0] * ssm.stride(0) * ssm.element_size(),
            ssm.untyped_storage().nbytes()
            - ssm.storage_offset() * ssm.element_size(),
        ),
    )


def build_layer_to_spec(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    layer_to_spec: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            layer_to_spec.update(
                {
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in group.layer_names
                }
            )
        else:
            layer_to_spec.update(
                {layer_name: group_spec for layer_name in group.layer_names}
            )
    return layer_to_spec


def is_mla_cache_layer(
    layer_to_spec: Mapping[str, KVCacheSpec], layer_name: str
) -> bool:
    try:
        spec = layer_to_spec[layer_name]
    except KeyError as e:
        raise ValueError(f"Missing KV cache spec for layer {layer_name}") from e
    return isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))


def _content_packed_dim(spec: AttentionSpec) -> int:
    head_size_v = getattr(spec, "head_size_v", spec.head_size)
    return spec.head_size + head_size_v


def _spec_dim_matches(value: int, expected: int | None) -> bool:
    return expected is None or value == expected


def _kernel_layout_matches(
    spec: KVCacheSpec, kernel_block_size: int, num_kv_heads: int, head_dim: int
) -> bool:
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size != 0:
        return False
    return _spec_dim_matches(
        num_kv_heads, getattr(spec, "num_kv_heads", None)
    ) and _spec_dim_matches(head_dim, getattr(spec, "head_size", None))


def _select_kernel_block_layout(
    layer_name: str, shape: torch.Size, spec: KVCacheSpec
) -> tuple[int, int, int]:
    axis2_matches = _kernel_layout_matches(spec, shape[2], shape[3], shape[4])
    axis3_matches = _kernel_layout_matches(spec, shape[3], shape[2], shape[4])

    if axis2_matches and axis3_matches and shape[2] != shape[3]:
        raise ValueError(
            f"Ambiguous MoRIIO kernel-block K/V cache shape for layer "
            f"{layer_name}: {tuple(shape)}"
        )
    if axis2_matches:
        return shape[2], shape[3], shape[4]
    if axis3_matches:
        return shape[3], shape[2], shape[4]

    raise ValueError(
        f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
        f"{tuple(shape)} does not contain block size {spec.block_size}"
    )


def get_layer_transfer_geometry(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    remote_num_blocks: int | None = None,
) -> LayerTransferGeometry | MambaTransferGeometry:
    spec = layer_to_spec[layer_name]
    if isinstance(spec, MambaSpec):
        # Hybrid/KDA layer: cache is a ``(conv, ssm)`` tuple, which has no
        # ``.shape``; handle it before touching the attention-only shape logic.
        conv, ssm = kda_conv_ssm(kv_cache, spec)
        return get_mamba_transfer_geometry(layer_name, conv, ssm, layer_to_spec)
    shape = kv_cache.shape
    stride = kv_cache.stride()
    element_size = kv_cache.element_size()
    is_mla_cache = is_mla_cache_layer(layer_to_spec, layer_name)

    if is_mla_cache and len(shape) == 3:
        num_blocks, block_size, latent_dim = shape
        slot_size_bytes = latent_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    if not is_mla_cache and len(shape) == 5 and shape[0] == 2:
        _, num_blocks = shape[:2]
        kernel_blocks_per_block = 1
        if shape[2] == spec.block_size:
            block_size, num_kv_heads, head_dim = shape[2:]
        elif shape[3] == spec.block_size:
            num_kv_heads, block_size, head_dim = shape[2:]
        else:
            kernel_num_blocks = num_blocks
            kernel_block_size, num_kv_heads, head_dim = _select_kernel_block_layout(
                layer_name, shape, spec
            )
            kernel_blocks_per_block = spec.block_size // kernel_block_size
            if kernel_num_blocks % kernel_blocks_per_block != 0:
                raise ValueError(
                    f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
                    f"{tuple(shape)} has {kernel_num_blocks} kernel blocks, "
                    f"not divisible by {kernel_blocks_per_block}"
                )
            num_blocks = kernel_num_blocks // kernel_blocks_per_block
            block_size = spec.block_size
        slot_size_bytes = num_kv_heads * head_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[1] * kernel_blocks_per_block,
            local_kv_stride=stride[0],
            remote_kv_stride=(
                stride[1] * kernel_blocks_per_block * (remote_num_blocks or num_blocks)
            ),
            transfers_per_block=2,
            regions_per_block=1,
            split_kv_regions=True,
        )

    if not is_mla_cache and len(shape) == 5 and shape[1] == 2:
        num_blocks = shape[0]
        if shape[2] == spec.block_size:
            block_size, num_kv_heads, head_dim = shape[2:]
            slot_size_bytes = num_kv_heads * head_dim * element_size
            block_len = block_size * slot_size_bytes
            return LayerTransferGeometry(
                num_blocks=num_blocks,
                block_size=block_size,
                block_len=block_len,
                slot_size_bytes=slot_size_bytes,
                block_stride=stride[0],
                local_kv_stride=stride[1],
                remote_kv_stride=stride[1],
                transfers_per_block=2,
                regions_per_block=2,
                split_kv_regions=False,
            )
        elif shape[3] == spec.block_size:
            num_kv_heads, block_size, head_dim = shape[2:]
        else:
            kernel_num_blocks = num_blocks
            kernel_block_size, _, _ = _select_kernel_block_layout(
                layer_name, shape, spec
            )
            kernel_blocks_per_block = spec.block_size // kernel_block_size
            if kernel_num_blocks % kernel_blocks_per_block != 0:
                raise ValueError(
                    f"Unsupported MoRIIO K/V cache shape for layer {layer_name}: "
                    f"{tuple(shape)} has {kernel_num_blocks} kernel blocks, "
                    f"not divisible by {kernel_blocks_per_block}"
                )
            num_blocks = kernel_num_blocks // kernel_blocks_per_block
            block_size = spec.block_size
            block_stride = stride[0] * kernel_blocks_per_block
            block_len = block_stride * element_size
            slot_size_bytes = block_len // block_size
            return LayerTransferGeometry(
                num_blocks=num_blocks,
                block_size=block_size,
                block_len=block_len,
                slot_size_bytes=slot_size_bytes,
                block_stride=block_stride,
                local_kv_stride=None,
                remote_kv_stride=None,
                transfers_per_block=1,
                regions_per_block=1,
                split_kv_regions=False,
            )
        slot_size_bytes = num_kv_heads * head_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=stride[1],
            remote_kv_stride=stride[1],
            transfers_per_block=2,
            regions_per_block=2,
            split_kv_regions=False,
        )

    if (
        not is_mla_cache
        and isinstance(spec, AttentionSpec)
        and len(shape) == 4
        and shape[1] == spec.num_kv_heads
        and shape[2] == spec.block_size
        and shape[3] == _content_packed_dim(spec)
    ):
        num_blocks, num_kv_heads, block_size, packed_dim = shape
        slot_size_bytes = num_kv_heads * packed_dim * element_size
        block_len = block_size * slot_size_bytes
        return LayerTransferGeometry(
            num_blocks=num_blocks,
            block_size=block_size,
            block_len=block_len,
            slot_size_bytes=slot_size_bytes,
            block_stride=stride[0],
            local_kv_stride=None,
            remote_kv_stride=None,
            transfers_per_block=1,
            regions_per_block=1,
            split_kv_regions=False,
        )

    cache_kind = "MLA" if is_mla_cache else "K/V"
    raise ValueError(
        f"Unsupported MoRIIO {cache_kind} cache shape for layer "
        f"{layer_name}: {tuple(shape)}"
    )


def kda_conv_ssm(
    kv_cache: "torch.Tensor | tuple | list",
    spec: "KVCacheSpec | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Return (conv_state, ssm_state) for a KDA/Mamba layer, version-robustly.

    Two vLLM kv-cache layouts are supported:
      * legacy k3-release (e.g. dev19033): ``kv_cache`` is a ``(conv, ssm)``
        tuple/list of separate slot-strided tensors -> returned as-is (``spec``
        unused, so callers on this path may pass ``None``).
      * dev19253+: ``kv_cache`` is a single ``[num_blocks, 1, 1, page_bytes]``
        int8 page view holding both states packed per block. Slice the conv
        and ssm byte sub-ranges and reinterpret per ``spec.shapes``/``dtypes``,
        mirroring ``MambaBase.bind_kv_cache`` so the returned views are
        byte-identical to what the model reads/writes: slot stride == page
        bytes, per-slot span == that state's bytes, conv/ssm sharing one
        backing buffer (handled by the connector's single-region + ssm
        intra-buffer offset path). Byte-exact for homogeneous P/D attn-TP.

    NOTE: the packed page interleaves conv+ssm per block, so the hetero-TP
    conv sub-projection split cannot be expressed by slicing this layout
    alone; hetero-TP GDN remains a documented follow-up (as with the legacy
    layout's whole-item transfer).
    """
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) != 2:
            raise ValueError(
                "Expected a 2-tuple (conv, ssm) KDA kv-cache, got "
                f"len={len(kv_cache)}"
            )
        return kv_cache[0], kv_cache[1]
    if spec is None:
        raise ValueError("kda_conv_ssm needs the MambaSpec to unpack a packed page tensor")
    from math import prod

    from vllm.utils.torch_utils import get_dtype_size

    shapes = list(spec.shapes)
    dtypes = list(spec.dtypes)
    if len(dtypes) == 1 and len(shapes) > 1:
        dtypes = dtypes * len(shapes)
    pages = kv_cache.reshape(kv_cache.shape[0], -1)  # [num_blocks, page_bytes]
    states: list[torch.Tensor] = []
    offset = 0
    for shp, dt in zip(shapes, dtypes):
        nbytes = prod(shp) * get_dtype_size(dt)
        state = pages[:, offset : offset + nbytes].view(dt)
        states.append(state.view(-1, *shp))
        offset += nbytes
    if len(states) < 2:
        raise ValueError(f"KDA page unpack expected >=2 states, got {len(states)}")
    return states[0], states[1]


def iter_layer_registration_regions(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
) -> list[tuple[torch.Tensor, int]]:
    spec = layer_to_spec[layer_name]
    if isinstance(spec, MambaSpec):
        # KDA layer: register the conv and ssm tensors as two whole-tensor
        # regions (the conv 3-subprojection split is expressed as transfer
        # offset triples, not extra registrations).
        conv, ssm = kda_conv_ssm(kv_cache, spec)
        geom = get_mamba_transfer_geometry(layer_name, conv, ssm, layer_to_spec)
        return [(conv, geom.conv_region_len), (ssm, geom.ssm_region_len)]
    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len
    if geometry.split_kv_regions:
        return [(cache, region_len) for cache in kv_cache]
    return [(kv_cache, region_len)]


def merge_contiguous_offsets(
    offsets_local: list[int],
    offsets_remote: list[int],
    sizes: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if not offsets_local:
        return [], [], []
    if not (len(offsets_local) == len(offsets_remote) == len(sizes)):
        raise ValueError("Input list lengths mismatch")

    rows = sorted(zip(offsets_local, offsets_remote, sizes), key=lambda row: row[0])
    merged: list[list[int]] = []
    for local, remote, size in rows:
        if (
            merged
            and local == merged[-1][0] + merged[-1][2]
            and remote == merged[-1][1] + merged[-1][2]
        ):
            merged[-1][2] += size
        else:
            merged.append([local, remote, size])

    return (
        [row[0] for row in merged],
        [row[1] for row in merged],
        [row[2] for row in merged],
    )


def compute_block_transfer_offsets(
    layer_name: str,
    kv_cache: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    local_block_ids: list[int],
    remote_block_ids: list[int],
    remote_num_blocks: int,
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
) -> tuple[list[int], list[int], list[int]]:
    # A shorter (or empty) local list is the READ-mode "drop the transfer, just
    # free the prefill blocks" case (full-prefix-hit / aborted-before-scheduled):
    # decode pulls fewer blocks than the prefill holds. The zip loop below pairs
    # local[i]<->remote[i] and sizes by len(local), so a short local transfers
    # only what decode allocated and an empty local is a no-op. A longer local
    # list is a genuine bug and still fails loudly.
    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )
    geometry = get_layer_transfer_geometry(
        layer_name, kv_cache, layer_to_spec, remote_num_blocks
    )
    element_size = kv_cache.element_size()
    transfer_size_byte = geometry.block_len
    per_block = geometry.transfers_per_block
    total = len(local_block_ids) * per_block
    offset_local = [0] * total
    offset_remote = [0] * total
    sizes = [transfer_size_byte] * total

    w = 0
    for lb, rb in zip(local_block_ids, remote_block_ids):
        offset_local[w] = element_size * (lb * geometry.block_stride)
        offset_remote[w] = element_size * (rb * geometry.block_stride)
        w += 1
        if per_block == 2:
            assert geometry.local_kv_stride is not None
            assert geometry.remote_kv_stride is not None
            offset_local[w] = element_size * (
                geometry.local_kv_stride + lb * geometry.block_stride
            )
            offset_remote[w] = element_size * (
                geometry.remote_kv_stride + rb * geometry.block_stride
            )
            w += 1

    return merge_fn(offset_local, offset_remote, sizes)


class MambaOffsetTemplate(NamedTuple):
    """Slot-independent conv+ssm offset decomposition for a KDA layer.

    Homogeneous-TP geometry (conv/ssm slot strides, conv sub-projection
    offsets, ssm per-slot size) is identical across requests and across the
    homogeneous GDN layers, so it is computed once and reused; only the
    per-request slot bases vary (see ``apply_mamba_offset_template``).
    """

    conv_slot_stride_bytes: int
    ssm_slot_stride_bytes: int
    ssm_read_bytes: int
    ssm_remote_extra: int
    conv_subprojs: tuple[tuple[int, int, int], ...]


def build_mamba_offset_template(
    layer_name: str,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    split_info: MambaConvSplitInfo,
    tp_ratio: int,
    tp_rank: int,
    world_size: int,
) -> MambaOffsetTemplate:
    """Precompute the slot-independent conv+ssm offset decomposition.

    Only homogeneous TP (``tp_ratio == 1``) is supported; heterogeneous TP is
    gated with ``NotImplementedError`` (see the design doc). The returned
    template is a pure function of the layer geometry and ``split_info``, so it
    can be cached and reused across requests without recomputation.
    """
    if tp_ratio != 1:
        # Heterogeneous-TP recurrent-state transfer needs the remote page's
        # slot stride (which differs from the local page) and, for
        # P_TP > D_TP, a multi-rank gather of conv/ssm slices. Both are
        # follow-ups (see docs/design/moriio_kda_transfer.md). The mamba path
        # bypasses the attention KV-head validator, so fail loudly here
        # instead of silently transferring wrong bytes.
        raise NotImplementedError(
            "heterogeneous-TP mamba/KDA transfer is not supported yet "
            f"(tp_ratio={tp_ratio}); use equal prefill/decode TP for "
            "hybrid models"
        )
    # Past the gate above this is always the homogeneous path (the reviewer's
    # "is tp_ratio always 1 here?"): make that invariant explicit.
    assert tp_ratio == 1, (
        f"homogeneous KDA offset path requires tp_ratio == 1, got {tp_ratio}"
    )
    geom = get_mamba_transfer_geometry(layer_name, conv, ssm, layer_to_spec)
    local_offset = tp_rank % max(tp_ratio, 1)
    conv_local_offsets = split_info.local_conv_offsets
    conv_remote_offsets = split_info.remote_conv_offsets(local_offset, tp_ratio)
    ssm_read_bytes = geom.ssm_slot_bytes
    conv_subprojs = tuple(
        (loff, roff, rsz)
        for (loff, _lsz), (roff, rsz) in zip(conv_local_offsets, conv_remote_offsets)
    )
    return MambaOffsetTemplate(
        conv_slot_stride_bytes=geom.conv_slot_stride * geom.conv_element_size,
        ssm_slot_stride_bytes=geom.ssm_slot_stride * geom.ssm_element_size,
        ssm_read_bytes=ssm_read_bytes,
        ssm_remote_extra=local_offset * ssm_read_bytes,
        conv_subprojs=conv_subprojs,
    )


def apply_mamba_offset_template(
    template: MambaOffsetTemplate,
    local_slots: list[int],
    remote_slots: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Apply per-request slot bases to a cached ``MambaOffsetTemplate``.

    Produces byte-identical output to computing the offsets from scratch:
    ``compute_mamba_conv_ssm_offsets`` is defined in terms of this function, so
    the cached fast path and the fresh path share one arithmetic definition.
    Conv sub-projection entries (all slots) come first, followed by one ssm
    entry per slot.
    """
    if len(local_slots) > len(remote_slots):
        raise ValueError(
            "local_slots longer than remote_slots: "
            f"{len(local_slots)} > {len(remote_slots)}"
        )
    local_offs: list[int] = []
    remote_offs: list[int] = []
    sizes: list[int] = []

    # Conv sub-projections (region 0), all slots first.
    conv_slot_stride_bytes = template.conv_slot_stride_bytes
    for ls, rs in zip(local_slots, remote_slots):
        lbase = ls * conv_slot_stride_bytes
        rbase = rs * conv_slot_stride_bytes
        for loff, roff, rsz in template.conv_subprojs:
            local_offs.append(lbase + loff)
            remote_offs.append(rbase + roff)
            sizes.append(rsz)

    # SSM temporal state (region 1), one entry per slot.
    ssm_slot_stride_bytes = template.ssm_slot_stride_bytes
    ssm_read_bytes = template.ssm_read_bytes
    ssm_remote_extra = template.ssm_remote_extra
    for ls, rs in zip(local_slots, remote_slots):
        local_offs.append(ls * ssm_slot_stride_bytes)
        remote_offs.append(rs * ssm_slot_stride_bytes + ssm_remote_extra)
        sizes.append(ssm_read_bytes)

    return local_offs, remote_offs, sizes


def compute_mamba_conv_ssm_offsets(
    layer_name: str,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    layer_to_spec: Mapping[str, KVCacheSpec],
    local_slots: list[int],
    remote_slots: list[int],
    split_info: MambaConvSplitInfo,
    tp_ratio: int,
    tp_rank: int,
    world_size: int,
) -> tuple[list[int], list[int], list[int]]:
    """Compute conv + ssm byte offsets for a KDA layer's state transfer.

    Returns ``(local_offsets, remote_offsets, sizes)`` where the conv
    sub-projection entries come first (``len(local_slots) *
    len(split_info.local_conv_offsets)`` of them) followed by one ssm entry
    per slot. Conv and ssm live in two separate registered regions, so the
    caller must route the conv slice to the conv session and the ssm slice to
    the ssm session (``compute_mamba_conv_split_count`` gives the boundary).

    Offsets are region-relative byte offsets. Each slot's base is
    ``slot * slot_stride * element_size`` (slot-strided view); conv
    sub-projections are placed at ``slot_base + local_conv_offsets`` locally
    and ``slot_base + remote_conv_offsets(local_offset, tp_ratio)`` remotely,
    while the ssm state uses the whole per-slot block (heads TP-sharded via
    ``local_offset`` for heterogeneous TP).
    """
    if len(local_slots) > len(remote_slots):
        raise ValueError(
            "local_slots longer than remote_slots: "
            f"{len(local_slots)} > {len(remote_slots)}"
        )
    template = build_mamba_offset_template(
        layer_name,
        conv,
        ssm,
        layer_to_spec,
        split_info,
        tp_ratio,
        tp_rank,
        world_size,
    )
    return apply_mamba_offset_template(template, local_slots, remote_slots)


def compute_mamba_conv_split_count(
    local_slots: list[int],
    split_info: MambaConvSplitInfo,
) -> int:
    """Number of leading conv entries in ``compute_mamba_conv_ssm_offsets``.

    Entries ``[:count]`` are conv sub-projections (conv region/session);
    entries ``[count:]`` are the per-slot ssm state (ssm region/session).
    """
    return len(local_slots) * len(split_info.local_conv_offsets)
