# SPDX-License-Identifier: Apache-2.0
"""Optional native MUSA KV-transfer adapter for LMCache.

The adapter is deliberately fail-closed. It never makes ``musa_aiter`` a
required dependency and returns ``False`` whenever native dispatch is not
available, so callers can keep using the torch implementation as the fallback.
"""

# Standard
from importlib import import_module
from typing import Any
import os

# Third Party
import torch

ENV_MUSA_NATIVE_KV_TRANSFER = "LMCACHE_MUSA_NATIVE_KV_TRANSFER"
NATIVE_LMCACHE_KV_TRANSFER_ABI_VERSION = 1

_REQUIRED_NATIVE_SYMBOLS = (
    "native_lmcache_kv_transfer_abi_version",
    "lmcache_kv_paged_to_buffer",
    "lmcache_kv_buffer_to_paged",
    "lmcache_mla_paged_to_buffer",
    "lmcache_mla_buffer_to_paged",
)

_SUPPORTED_BLOCK_TRANSFER_FORMATS = {
    "NL_X_TWO_NB_BS_NH_HS",
    "NL_X_NB_BS_HS",
}
_MLA_BLOCK_TRANSFER_FORMATS = {"NL_X_NB_BS_HS"}
_TRANSFER_DIRECTION_H2D = 0
_TRANSFER_DIRECTION_D2H = 1


def is_native_musa_kv_transfer_enabled() -> bool:
    """Return whether LMCache should try optional native MUSA KV transfer."""
    return os.environ.get(ENV_MUSA_NATIVE_KV_TRANSFER, "").lower() in {
        "1",
        "true",
        "yes",
    }


def load_native_musa_module() -> Any | None:
    """Import optional ``musa_aiter``, returning ``None`` when unavailable."""
    try:
        return import_module("musa_aiter")
    except Exception:
        return None


def check_native_abi(module: Any) -> bool:
    """Return whether ``module`` exposes the Stage2 LMCache KV-transfer ABI."""
    for name in _REQUIRED_NATIVE_SYMBOLS:
        if not callable(getattr(module, name, None)):
            return False
    try:
        version = int(module.native_lmcache_kv_transfer_abi_version())
    except Exception:
        return False
    return version == NATIVE_LMCACHE_KV_TRANSFER_ABI_VERSION


def _is_musa_contiguous_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor can be passed directly to MUSA native kernels."""
    return tensor.device.type == "musa" and tensor.is_contiguous()


def _native_tensors_ready(
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> bool:
    """Return whether native MUSA dispatch can consume these tensors directly."""
    return (
        _is_musa_contiguous_tensor(memory_tensor)
        and _is_musa_contiguous_tensor(slot_mapping)
        and all(_is_musa_contiguous_tensor(kvcache) for kvcache in kvcaches)
    )


def try_native_multi_layer_block_kv_transfer(
    *,
    paged_layers: Any,
    object_tensors: list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    direction: Any,
    shape_desc: Any,
    lmcache_chunk_size: int,
    engine_kv_format: Any,
    skip_prefix_n_blocks: int,
) -> bool:
    """Try native MUSA acceleration for block-based paged-KV transfer.

    This is the optional native fast path used by
    ``lmcache.v1.platform.musa.ops`` behind
    ``lmc_ops.multi_layer_block_kv_transfer``. It hides the MUSA-specific
    eligibility checks, slot mapping, and CPU staging from the generic
    multiprocess transfer context.

    Args:
        paged_layers: Normalized per-layer paged KV tensors.
        object_tensors: LMCache chunk tensors.
        block_ids: Engine block IDs in object order.
        direction: Transfer direction enum or integer.
        shape_desc: Page buffer shape descriptor.
        lmcache_chunk_size: Tokens per LMCache object.
        engine_kv_format: Engine KV layout enum.
        skip_prefix_n_blocks: Whole blocks to skip before H2D scatter.

    Returns:
        ``True`` when native dispatch completed and the caller should skip the
        torch fallback. ``False`` when the transfer is unsupported, disabled,
        unavailable, or rejected by the native module.
    """
    if not is_native_musa_kv_transfer_enabled():
        return False
    if not object_tensors or lmcache_chunk_size <= 0:
        return False
    if not _is_supported_musa_block_transfer_format(engine_kv_format):
        return False
    if not _is_musa_block_transfer_candidate(paged_layers):
        return False

    is_d2h = int(direction) == _TRANSFER_DIRECTION_D2H
    is_h2d = int(direction) == _TRANSFER_DIRECTION_H2D
    if not (is_d2h or is_h2d):
        return False
    if is_d2h and skip_prefix_n_blocks != 0:
        return False

    block_size = int(getattr(shape_desc, "bs", 0))
    if block_size <= 0 or lmcache_chunk_size % block_size != 0:
        return False
    blocks_per_object = lmcache_chunk_size // block_size
    block_id_list = _to_block_id_list(block_ids)

    layer_tensors = _as_tensor_list(paged_layers)
    if layer_tensors is None:
        return False
    use_mla = _is_mla_block_transfer_format(engine_kv_format)
    native_dims = _native_transfer_dims(layer_tensors, shape_desc, use_mla)
    if native_dims is None:
        return False
    num_heads, head_size = native_dims

    if is_d2h:
        return _try_native_block_transfer_from_gpu(
            layer_tensors=layer_tensors,
            object_tensors=object_tensors,
            block_ids=block_id_list,
            blocks_per_object=blocks_per_object,
            block_size=block_size,
            use_mla=use_mla,
            num_heads=num_heads,
            head_size=head_size,
        )
    return _try_native_block_transfer_to_gpu(
        layer_tensors=layer_tensors,
        object_tensors=object_tensors,
        block_ids=block_id_list,
        blocks_per_object=blocks_per_object,
        block_size=block_size,
        use_mla=use_mla,
        num_heads=num_heads,
        head_size=head_size,
        skip_prefix_n_blocks=skip_prefix_n_blocks,
    )


def try_native_to_gpu(
    *,
    use_mla: bool,
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
    skip_prefix_n_tokens: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Try native contiguous-buffer-to-paged-KV scatter.

    Args:
        use_mla: Whether the active layout is MLA.
        memory_tensor: LMCache contiguous memory object tensor.
        kvcaches: Engine paged KV tensors.
        slot_mapping: Full engine slot mapping tensor.
        start: Inclusive token start for this transfer.
        end: Exclusive token end for this transfer.
        skip_prefix_n_tokens: Prefix tokens already cached by the engine.
        block_size: Engine paged KV block size.
        num_heads: Number of KV heads for non-MLA layouts.
        head_size: KV head size or MLA hidden size.

    Returns:
        ``True`` when native dispatch completed and the caller should skip the
        torch fallback. ``False`` when native dispatch is disabled, unavailable,
        ABI-incompatible, when tensors are not contiguous MUSA tensors, or when
        the native module rejects the transfer.
    """
    module = _native_module_if_ready()
    if module is None:
        return False

    transfer_start = start + skip_prefix_n_tokens
    if transfer_start >= end:
        return True
    if not _native_tensors_ready(memory_tensor, kvcaches, slot_mapping):
        return False

    slot_slice = slot_mapping[transfer_start:end]
    try:
        if use_mla:
            return bool(
                module.lmcache_mla_buffer_to_paged(
                    memory_tensor,
                    kvcaches,
                    slot_slice,
                    skip_prefix_n_tokens,
                    block_size,
                    head_size,
                )
            )
        return bool(
            module.lmcache_kv_buffer_to_paged(
                memory_tensor,
                kvcaches,
                slot_slice,
                skip_prefix_n_tokens,
                block_size,
                num_heads,
                head_size,
            )
        )
    except Exception:
        return False


def try_native_from_gpu(
    *,
    use_mla: bool,
    memory_tensor: torch.Tensor,
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    start: int,
    end: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> bool:
    """Try native paged-KV-to-contiguous-buffer gather.

    Args:
        use_mla: Whether the active layout is MLA.
        memory_tensor: LMCache contiguous memory object tensor to populate.
        kvcaches: Engine paged KV tensors.
        slot_mapping: Full engine slot mapping tensor.
        start: Inclusive token start for this transfer.
        end: Exclusive token end for this transfer.
        block_size: Engine paged KV block size.
        num_heads: Number of KV heads for non-MLA layouts.
        head_size: KV head size or MLA hidden size.

    Returns:
        ``True`` when native dispatch completed and the caller should skip the
        torch fallback. ``False`` when native dispatch is unavailable, when
        tensors are not contiguous MUSA tensors, or when native dispatch fails.
    """
    module = _native_module_if_ready()
    if module is None:
        return False
    if start >= end:
        return True
    if not _native_tensors_ready(memory_tensor, kvcaches, slot_mapping):
        return False

    slot_slice = slot_mapping[start:end]
    try:
        if use_mla:
            return bool(
                module.lmcache_mla_paged_to_buffer(
                    kvcaches,
                    memory_tensor,
                    slot_slice,
                    block_size,
                    head_size,
                )
            )
        return bool(
            module.lmcache_kv_paged_to_buffer(
                kvcaches,
                memory_tensor,
                slot_slice,
                block_size,
                num_heads,
                head_size,
            )
        )
    except Exception:
        return False


def _native_module_if_ready() -> Any | None:
    """Return a usable native module when opt-in and ABI-compatible."""
    if not is_native_musa_kv_transfer_enabled():
        return None
    module = load_native_musa_module()
    if module is None or not check_native_abi(module):
        return None
    return module


def _engine_kv_format_name(engine_kv_format: Any) -> str:
    """Return a stable enum member name for Python and pybind enum values."""
    name = getattr(engine_kv_format, "name", None)
    if isinstance(name, str):
        return name
    return str(engine_kv_format).rsplit(".", maxsplit=1)[-1]


def _is_supported_musa_block_transfer_format(engine_kv_format: Any) -> bool:
    """Return whether the native MUSA block path supports ``engine_kv_format``."""
    return _engine_kv_format_name(engine_kv_format) in _SUPPORTED_BLOCK_TRANSFER_FORMATS


def _is_mla_block_transfer_format(engine_kv_format: Any) -> bool:
    """Return whether ``engine_kv_format`` is a native-supported MLA layout."""
    return _engine_kv_format_name(engine_kv_format) in _MLA_BLOCK_TRANSFER_FORMATS


def _as_tensor_list(value: Any) -> list[torch.Tensor] | None:
    """Return ``value`` as a flat tensor list, or ``None`` when unsupported."""
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(tensor, torch.Tensor) for tensor in value):
        return None
    return value


def _is_musa_block_transfer_candidate(paged_layers: Any) -> bool:
    """Return whether normalized paged layers can use native MUSA kernels."""
    layer_tensors = _as_tensor_list(paged_layers)
    return layer_tensors is not None and all(
        _is_musa_contiguous_tensor(tensor) for tensor in layer_tensors
    )


def _to_block_id_list(block_ids: torch.Tensor | list[int]) -> list[int]:
    """Convert block IDs from tensor/list form into Python integers."""
    if isinstance(block_ids, torch.Tensor):
        return [int(x) for x in block_ids.to(dtype=torch.int64).cpu().tolist()]
    return [int(x) for x in block_ids]


def _native_transfer_dims(
    layer_tensors: list[torch.Tensor],
    shape_desc: Any,
    use_mla: bool,
) -> tuple[int, int] | None:
    """Return ``(num_heads, head_size)`` for native MUSA transfer."""
    if use_mla:
        head_size = int(getattr(shape_desc, "hs", 0)) or int(layer_tensors[0].shape[-1])
        return 1, head_size
    num_heads = int(getattr(shape_desc, "nh", 0))
    head_size = int(getattr(shape_desc, "hs", 0))
    if num_heads <= 0 or head_size <= 0:
        sample = layer_tensors[0][0]
        if sample.ndim < 4:
            return None
        num_heads = int(sample.shape[-2])
        head_size = int(sample.shape[-1])
    return num_heads, head_size


def _block_ids_to_slot_mapping(
    block_ids: list[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand paged block IDs into token slot IDs on ``device``."""
    block_ids_tensor = torch.tensor(block_ids, dtype=torch.long, device=device)
    token_offsets = torch.arange(block_size, dtype=torch.long, device=device)
    return (block_ids_tensor.unsqueeze(1) * block_size + token_offsets).flatten()


def _object_block_ids(
    block_ids: list[int],
    object_idx: int,
    blocks_per_object: int,
) -> list[int]:
    """Return block IDs belonging to one LMCache object."""
    start = object_idx * blocks_per_object
    end = min(start + blocks_per_object, len(block_ids))
    return block_ids[start:end]


def _try_native_block_transfer_from_gpu(
    *,
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: list[int],
    blocks_per_object: int,
    block_size: int,
    use_mla: bool,
    num_heads: int,
    head_size: int,
) -> bool:
    """Try native paged-KV-to-object gather for each LMCache object."""
    device = layer_tensors[0].device
    for object_idx, object_tensor in enumerate(object_tensors):
        chunk_block_ids = _object_block_ids(block_ids, object_idx, blocks_per_object)
        if not chunk_block_ids:
            continue
        slot_mapping = _block_ids_to_slot_mapping(chunk_block_ids, block_size, device)
        memory_tensor = (
            object_tensor
            if _is_musa_contiguous_tensor(object_tensor)
            else torch.empty_like(object_tensor, device=device)
        )
        if not try_native_from_gpu(
            use_mla=use_mla,
            memory_tensor=memory_tensor,
            kvcaches=layer_tensors,
            slot_mapping=slot_mapping,
            start=0,
            end=int(slot_mapping.numel()),
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ):
            return False
        if memory_tensor is not object_tensor:
            object_tensor.copy_(
                memory_tensor.to(object_tensor.device), non_blocking=True
            )
    return True


def _try_native_block_transfer_to_gpu(
    *,
    layer_tensors: list[torch.Tensor],
    object_tensors: list[torch.Tensor],
    block_ids: list[int],
    blocks_per_object: int,
    block_size: int,
    use_mla: bool,
    num_heads: int,
    head_size: int,
    skip_prefix_n_blocks: int,
) -> bool:
    """Try native object-to-paged-KV scatter for each LMCache object."""
    device = layer_tensors[0].device
    for object_idx, object_tensor in enumerate(object_tensors):
        chunk_block_ids = _object_block_ids(block_ids, object_idx, blocks_per_object)
        if not chunk_block_ids:
            continue
        object_start_block = object_idx * blocks_per_object
        chunk_skip_blocks = max(0, skip_prefix_n_blocks - object_start_block)
        if chunk_skip_blocks >= len(chunk_block_ids):
            continue

        memory_tensor = object_tensor.to(device, non_blocking=True)
        if memory_tensor is object_tensor:
            memory_tensor = object_tensor.clone()
        elif not memory_tensor.is_contiguous():
            memory_tensor = memory_tensor.contiguous()
        slot_mapping = _block_ids_to_slot_mapping(chunk_block_ids, block_size, device)
        if not try_native_to_gpu(
            use_mla=use_mla,
            memory_tensor=memory_tensor,
            kvcaches=layer_tensors,
            slot_mapping=slot_mapping,
            start=0,
            end=int(slot_mapping.numel()),
            skip_prefix_n_tokens=chunk_skip_blocks * block_size,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ):
            return False
    return True
