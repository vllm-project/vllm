# SPDX-License-Identifier: Apache-2.0
"""Contiguous-view recovery for raw engine KV caches.

Metadata-only (zero-copy) preprocessing that makes a tensor's ``.shape``
reflect its physical layout. Engine- and format-agnostic: ``detect_format``
runs it before detection, and CUDA-IPC callers use it standalone.
"""

# mypy: disable-error-code="union-attr"
# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache

logger = init_logger(__name__)


def _get_expected_shape(stride: tuple[int, ...], num_elems: int) -> tuple[int, ...]:
    """Return the expected shape for a stride-sorted tensor view.

    The returned shape is the tightest possible (contiguous) layout
    that preserves the original storage size and stride ordering. For
    example, a tensor with ``stride=(16, 4, 1)`` has an expected shape
    of ``(4, 4, 4)``, which is the smallest shape that can be laid out
    in memory with those strides.

    Args:
        stride: The stride of the tensor.
        num_elems: The total number of elements in the tensor.

    Returns:
        The expected shape for a contiguous layout.
    """
    assert len(stride) > 0, "Stride must have at least one dimension"
    expected_shape = [num_elems // stride[0]]
    for i in range(0, len(stride) - 1):
        expected_shape.append(stride[i] // stride[i + 1])
    return tuple(dim for dim in expected_shape)


def attempt_permute_to_contiguous_view(
    kv_caches: DiscoverableKVCache,
) -> DiscoverableKVCache:
    """Return a contiguous view of *kv_caches*, metadata-only (no copy).

    For a tensor leaf: reorders the dims by stride magnitude so shape
    lines up with a contiguous layout. For a list: recurses into each
    element. Tensor leaves alias the input's storage; list nodes are
    freshly allocated but hold the same tensor objects (or their
    permuted views).

    Recovers the vLLM HND case: tensors allocated physically as
    ``[2, NB, NH, BS, HS]`` but exposed logically as
    ``[2, NB, BS, NH, HS]`` via a dim permute. Sorting dims by stride
    undoes the permute without touching storage.

    For tensors that remain non-contiguous even after dim-permute
    recovery (e.g. vLLM unified KV pool views where dim-0 has an
    inflated periodic stride because every block slot is padded to a
    model-wide maximum), this function returns the tensor unchanged.
    Rationale: :class:`CudaIPCWrapper` transports ``(shape, stride,
    storage_offset)`` verbatim and the receiver rebuilds the view via
    ``torch.Tensor.set_(storage, offset, shape, stride)``, which
    supports arbitrary strided views (including periodic-dim-0 and
    ``as_strided``-produced layouts) and yields a bit-identical view.
    Downstream consumers that rely on ``shape`` alone to infer the
    physical layout must therefore also consult ``stride``.

    We deliberately never fall back to ``.contiguous()`` (which would
    allocate and copy), so the caller's zero-copy invariant is
    preserved.
    """
    if isinstance(kv_caches, torch.Tensor):
        strides = kv_caches.stride()
        perm = sorted(range(kv_caches.ndim), key=lambda i: strides[i], reverse=True)
        result = kv_caches.permute(perm)
        if result.is_contiguous():
            return result.view(_get_expected_shape(result.stride(), result.numel()))
        padding_per_block = _validate_dim0_padded_layout(result)
        logger.debug(
            "attempt_permute_to_contiguous_view: accepting dim-0-padded "
            "view; downstream kernels must honour block_stride_elems. "
            "shape=%s, stride=%s, padding_per_block_elems=%d, "
            "storage_nbytes=%s, dtype=%s",
            tuple(result.shape),
            tuple(result.stride()),
            padding_per_block,
            int(result.untyped_storage().nbytes()),
            result.dtype,
        )
        return result
    return [attempt_permute_to_contiguous_view(sub) for sub in kv_caches]


def _validate_dim0_padded_layout(tensor: torch.Tensor) -> int:
    """Validate that *tensor* matches the dim-0-padding-only strided layout.

    Mainly used for DeepSeek V4 integration, where compressor / indexer
    KV groups share a pool with larger attn groups and end up with
    per-block dim-0 padding. The downstream KV transfer kernels only
    honour this single non-contiguous shape (via
    :class:`PageBufferShapeDesc.block_stride_elems`); any other strided
    view would cause wrong reads/writes and is rejected here.

    The accepted layout requires:

    * ``stride[-1] == 1`` and ``stride[-2] == shape[-1]`` -- each block
      row is internally tightly packed.
    * Every interior dim ``i`` satisfies
      ``stride[i] == prod(shape[i+1:])`` -- only dim-0 may carry
      padding, with ``stride[0] >= prod(shape[1:])``.

    Callers must pass the stride-sorted permuted view (not the original
    tensor): for tensors that are both permuted and dim-0-padded, the
    original's unsorted inner strides would falsely trip the tight-
    packing check. ``permute`` shares storage and preserves
    ``storage_offset``/``numel``/storage bytes, so those checks are
    equivalent on either view.

    Returns:
        ``padding_per_block_elems`` (= ``stride[0] - prod(shape[1:])``).

    Raises:
        ValueError: *tensor* violates any of the invariants above.
    """
    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    ndim = tensor.ndim
    storage_offset = int(tensor.storage_offset())

    def _fail(reason: str) -> None:
        raise ValueError(
            "attempt_permute_to_contiguous_view: tensor is non-contiguous "
            f"and not a supported (dim-0 padding only) layout -- {reason}. "
            f"shape={shape}, stride={stride}, "
            f"storage_offset={storage_offset}, numel={int(tensor.numel())}, "
            f"storage_nbytes={int(tensor.untyped_storage().nbytes())}, "
            f"dtype={tensor.dtype}. "
            "Downstream KV transfer kernels only understand dim-0 "
            "block-row padding; other strided views would produce "
            "wrong reads/writes and are rejected."
        )

    if ndim < 2:
        _fail("ndim < 2")
    if stride[-1] != 1:
        _fail("stride[-1] != 1 (inner dim not contiguous)")
    if stride[-2] != shape[-1]:
        _fail("stride[-2] != shape[-1] (last-two dims not tightly packed)")
    inner_tight = 1
    for i in range(ndim - 1, 0, -1):
        if i < ndim - 1 and stride[i] != inner_tight:
            _fail(
                f"dim {i} stride {stride[i]} != tight {inner_tight} "
                "(interior-dim padding is not supported)"
            )
        inner_tight *= shape[i]
    if stride[0] < inner_tight:
        _fail(
            f"dim-0 stride {stride[0]} < prod(shape[1:])={inner_tight} "
            "(overlapping blocks)"
        )
    return stride[0] - inner_tight
