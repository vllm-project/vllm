# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side KV checksum computation.

Each worker computes its part of a block's checksum: the slices (layer, KV
head) it holds, each weighed by its global layer and head (see
``checksum.py``). Parts from all TP ranks and PP stages add up to the block's
checksum. Drafter (speculative decoding) layers are not checksummed.

The results stay on the device until the model runner copies them to host
memory with the step's other outputs, on its output copy stream (see
``PendingKVChecksums`` and ``vllm/v1/worker/gpu/async_utils.py``).
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    MAX_KV_HEADS_PER_LAYER,
    KVChecksumBlocks,
    KVChecksumGroup,
    KVChecksumRecord,
    KVChecksumScheduled,
    ReplicaKey,
    kv_checksum_enabled,
)
from vllm.distributed.kv_transfer.kv_checksum.kernels import layers_checksum
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import PIN_MEMORY, get_dtype_size
from vllm.v1.core.kv_cache_planning import layer_tp_replicas
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorTransferResults,
    )

logger = init_logger(__name__)

# The leading states of a Mamba layer's page that are checksummed: the states
# NIXL transfers (conv and ssm; see derive_mamba_conv_split and
# _build_mamba_local), or the only state of a layer with one.
_MAMBA_CHECKSUMMED_STATES = 2

# Upper bound on the bytes of KV gathered per kernel launch, which bounds the
# transient device memory of a checksum pass for long prompts.
_MAX_GATHER_BYTES = 64 * 1024 * 1024


class _LayerView:
    """One layer's KV cache as integer pieces, for weighted sums.

    ``pieces`` is ``[num_blocks, kernel_blocks, heads, tokens, pieces]``. A
    piece ``j`` of slice ``(layer, head)`` weighs ``m * (2 * j + 1)``, with
    ``m = 2 * (layer * H + head) + 1`` for the global head (see
    ``checksum.py``). ``weights`` are those of a block holding the request's
    first tokens; a block starting at token ``f`` adds ``f`` times
    ``offset_weights`` to them. Pieces narrower than 8 bytes count as
    unsigned.

    On CUDA with Triton the sums come from ``_FusedLayers``, for all layers
    of a layout in one launch; otherwise from PyTorch ops on gathered blocks.
    """

    def __init__(
        self, pieces: torch.Tensor, first_slice: int, replica_key: ReplicaKey
    ) -> None:
        """Set up the weighted sums of this rank's slices.

        Args:
            pieces: The layer's cache viewed as integer pieces.
            first_slice: ``layer * H + head`` of this rank's first head.
            replica_key: The replicas and replica index of this rank's heads.

        """
        self.pieces = pieces
        self.first_slice = first_slice
        self.replica_key = replica_key
        self.use_triton = pieces.is_cuda and HAS_TRITON
        if self.use_triton:
            return
        _, kernel_blocks, heads, tokens, num_pieces = pieces.shape
        tokens_per_block = kernel_blocks * tokens
        device = pieces.device
        # Token index within the logical block, as [kernel_block, 1, token, 1].
        self.token_index = torch.arange(
            tokens_per_block, device=device, dtype=torch.int64
        ).view(kernel_blocks, 1, tokens, 1)
        head = torch.arange(heads, device=device, dtype=torch.int64).view(
            1, heads, 1, 1
        )
        piece = torch.arange(num_pieces, device=device, dtype=torch.int64).view(
            1, 1, 1, num_pieces
        )
        slice_weight = 2 * (first_slice + head) + 1
        position = self.token_index * num_pieces + piece
        self.weights = slice_weight * (2 * position + 1)
        self.offset_weights = 2 * num_pieces * slice_weight
        piece_bits = 8 * pieces.element_size()
        self.piece_mask = (1 << piece_bits) - 1 if piece_bits < 64 else None
        # Gathered rows are widened to int64.
        self.rows_per_gather = max(
            1, _MAX_GATHER_BYTES // (tokens_per_block * heads * num_pieces * 8)
        )

    def accumulate(
        self,
        block_ids: torch.Tensor,
        first_tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Add this layer's weighted sum of each block row into ``out``.

        Args:
            block_ids: ``[n]`` logical block ids on the device.
            first_tokens: ``[n]`` position in the request of each block's first
                token.
            valid_tokens: ``[n]`` tokens to include from the start of each block.
            out: ``[n]`` int64 accumulator, wrapping around (mod 2**64).

        """
        assert not self.use_triton
        dims = (1, 2, 3, 4)
        for start in range(0, block_ids.shape[0], self.rows_per_gather):
            end = start + self.rows_per_gather
            rows = self.pieces.index_select(0, block_ids[start:end])
            valid = valid_tokens[start:end].view(-1, 1, 1, 1, 1)
            rows.masked_fill_(self.token_index >= valid, 0)
            rows = rows.to(torch.int64)
            if self.piece_mask is not None:
                rows &= self.piece_mask
            out[start:end] += (rows * self.weights).sum(dim=dims) + first_tokens[
                start:end
            ] * (rows * self.offset_weights).sum(dim=dims)


class _FusedLayers:
    """Layers of a group whose caches share one layout (shape, strides and
    dtype), each adding into its column of the group's checksums: in one
    Triton launch on CUDA, else layer by layer."""

    def __init__(self, views: list[_LayerView], columns: list[int]) -> None:
        self.views = views
        self.columns = columns
        self.use_triton = views[0].use_triton
        if self.use_triton:
            self.layers = torch.tensor(
                [
                    [view.pieces.data_ptr(), view.first_slice, column]
                    for view, column in zip(views, columns)
                ],
                dtype=torch.int64,
                device=views[0].pieces.device,
            )

    def accumulate(
        self,
        block_ids: torch.Tensor,
        first_tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Like ``_LayerView.accumulate``, into the ``[n, num_columns]``
        ``out``."""
        if self.use_triton:
            layers_checksum(
                self.views[0].pieces,
                self.layers,
                block_ids,
                first_tokens,
                valid_tokens,
                out,
            )
            return
        for view, column in zip(self.views, self.columns):
            view.accumulate(block_ids, first_tokens, valid_tokens, out[:, column])

    def warm_up(self, num_columns: int) -> None:
        """Compile the Triton kernel now rather than in a serving step."""
        if self.use_triton:
            device = self.views[0].pieces.device
            zeros = torch.zeros(1, dtype=torch.int64, device=device)
            out = torch.zeros(1, num_columns, dtype=torch.int64, device=device)
            self.accumulate(zeros, zeros, zeros, out)


@dataclass
class _GroupPlan:
    """How a worker checksums a group's layers: one launch per cache layout,
    adding into one column per replica key."""

    replica_keys: list[ReplicaKey]
    launches: list[_FusedLayers]

    @classmethod
    def create(cls, views: list[_LayerView]) -> "_GroupPlan | None":
        if not views:
            return None
        replica_keys = list(dict.fromkeys(view.replica_key for view in views))
        layouts: dict[tuple, list[_LayerView]] = defaultdict(list)
        for view in views:
            pieces = view.pieces
            layouts[
                (pieces.dtype, pieces.device, pieces.shape, pieces.stride())
            ].append(view)
        return cls(
            replica_keys,
            [
                _FusedLayers(
                    same_layout,
                    [replica_keys.index(view.replica_key) for view in same_layout],
                )
                for same_layout in layouts.values()
            ],
        )

    @property
    def device(self) -> torch.device:
        return self.launches[0].views[0].pieces.device


def _layer_spec(group_spec: KVCacheSpec, layer_name: str) -> KVCacheSpec:
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return group_spec.kv_cache_specs[layer_name]
    return group_spec


def _group_fingerprint(group: KVCacheGroupSpec, tp_size: int) -> int:
    """The sum of a hash of each of the group's layers on this worker."""
    total = 0
    for layer_name in group.layer_names:
        spec = _layer_spec(group.kv_cache_spec, layer_name)
        identity: tuple = (layer_name, type(spec).__name__)
        if isinstance(spec, MambaSpec) and spec.state_splits is not None:
            identity += (
                _global_state_shapes(spec, tp_size),
                [str(d) for d in spec.dtypes[: len(spec.state_splits)]],
            )
        elif isinstance(spec, MambaSpec):
            states = _MAMBA_CHECKSUMMED_STATES
            identity += (spec.shapes[:states], [str(d) for d in spec.dtypes[:states]])
        else:
            identity += (str(getattr(spec, "dtype", None)),)
        digest = hashlib.sha256(repr(identity).encode()).digest()
        total += int.from_bytes(digest[:8], "little")
    return total % 2**64


class _UnsupportedGroupError(Exception):
    """A KV cache group whose layers cannot be checksummed."""


def _tp_replicas(
    spec: AttentionSpec, tp_size: int, uniform_kv_heads: int | None
) -> int:
    """Consecutive TP ranks holding the same KV heads of a layer.

    Specs that declare ``max_tp_shards`` say how they shard, and MLA-type
    caches that do not are held whole by every rank, as NIXL transfers them.
    Other attention layers shard their KV heads across TP and, once TP exceeds them,
    replicate one head on consecutive ranks. More than one head per rank thus
    means no replicas, while one head per rank is ambiguous and is resolved
    with the model's KV head count when every layer shares it.
    """
    if spec.max_tp_shards is not None:
        return layer_tp_replicas(spec, tp_size, dcp_size=1)
    if isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
        return tp_size
    if spec.num_kv_heads > 1 or tp_size == 1:
        return 1
    if uniform_kv_heads is None or tp_size % uniform_kv_heads:
        raise _UnsupportedGroupError(
            "its KV head replication across TP is unknown (one head per rank, "
            "and the model's layers do not share one KV head count)"
        )
    return tp_size // uniform_kv_heads


def _layer_module(layer_name: str) -> tuple[int, str]:
    """The model-wide layer number in a layer name, the same on every rank,
    and the name of its layer module (up to the number)."""
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        number = extract_layer_index(layer_name)
    except AssertionError as e:
        raise _UnsupportedGroupError(
            f"layer {layer_name} has no single layer number"
        ) from e
    parts = layer_name.split(".")
    end = next(i for i, part in enumerate(parts) if part.isdigit()) + 1
    return number, ".".join(parts[:end])


def _attention_pieces(
    spec: KVCacheSpec,
    cache: torch.Tensor,
    num_blocks: int,
    tp_size: int,
    uniform_kv_heads: int | None,
) -> tuple[torch.Tensor, int, int]:
    """An attention layer's 8-byte piece view, TP replicas and heads per rank."""
    if not isinstance(spec, AttentionSpec) or not spec.has_layer_views:
        raise _UnsupportedGroupError(
            f"{type(spec).__name__} has no per-token KV blocks"
        )
    if spec.tokens_per_state != 1:
        raise _UnsupportedGroupError(
            f"{type(spec).__name__} stores {spec.tokens_per_state} tokens per state"
        )
    if spec.num_head_slots is not None:
        raise _UnsupportedGroupError("its KV heads are packed into head slots")
    pieces = _as_pieces(cache, num_blocks)
    if pieces.shape[1] * pieces.shape[3] != spec.block_size:
        raise _UnsupportedGroupError(
            f"the kernel blocks of its cache of shape {tuple(cache.shape)} do not "
            f"tile its {spec.block_size}-token blocks"
        )
    return pieces, _tp_replicas(spec, tp_size, uniform_kv_heads), spec.num_kv_heads


def _mamba_pieces(
    spec: MambaSpec, cache: torch.Tensor, num_blocks: int, tp_size: int
) -> tuple[torch.Tensor, int, int]:
    """Like ``_attention_pieces``, for a Mamba layer: each block's leading
    ``_MAMBA_CHECKSUMMED_STATES`` states are one slice of one head and token.
    """
    covered = sum(
        prod(shape) * get_dtype_size(dtype)
        for shape, dtype in zip(
            spec.shapes[:_MAMBA_CHECKSUMMED_STATES],
            spec.dtypes[:_MAMBA_CHECKSUMMED_STATES],
        )
    )
    if cache.dim() != 4 or cache.shape[0] != num_blocks or cache.shape[1:3] != (1, 1):
        raise _UnsupportedGroupError(
            f"its state cache of shape {tuple(cache.shape)} is not one page per block"
        )
    try:
        pages = cache.view(torch.uint8)[:, 0, 0, :covered].view(torch.int64)
    except RuntimeError as e:
        raise _UnsupportedGroupError(
            "its states or pages are not whole 8-byte pieces"
        ) from e
    replicas = tp_size if spec.tp_replicated else 1
    return pages[:, None, None, None, :], replicas, 1


def _global_state_shapes(spec: MambaSpec, tp_size: int) -> list[tuple]:
    """Per split state: its global segment sizes and its other dimensions."""
    assert spec.state_splits is not None
    shapes = []
    for shape, split in zip(spec.shapes, spec.state_splits):
        segments = tuple(size * tp_size // r for size, r in split.segments)
        others = shape[: split.axis] + shape[split.axis + 1 :]
        shapes.append((segments, others))
    return shapes


def _mamba_rows(
    spec: MambaSpec, cache: torch.Tensor, num_blocks: int, number: int, tp_size: int
) -> list["_LayerCache"]:
    """A Mamba layer's TP-split states, each global row (the TP-split
    dimension, see ``MambaStateSplit``) one slice of its elements, read as
    unsigned 2- or 4-byte pieces. Slices are numbered across the layer's
    states in their global order, so a rank holds those of its shard of each
    segment whatever the TP size.
    """
    assert spec.state_splits is not None
    if cache.dim() != 4 or cache.shape[0] != num_blocks or cache.shape[1:3] != (1, 1):
        raise _UnsupportedGroupError(
            f"its state cache of shape {tuple(cache.shape)} is not one page per block"
        )
    pages = cache.view(torch.uint8)[:, 0, 0]
    caches: list[_LayerCache] = []
    offset = global_rows = 0
    for shape, dtype, split in zip(spec.shapes, spec.dtypes, spec.state_splits):
        size = get_dtype_size(dtype)
        if size not in (2, 4):
            raise _UnsupportedGroupError(f"its {dtype} states are not 2 or 4 bytes")
        nbytes = prod(shape) * size
        if offset + nbytes > pages.shape[1]:
            raise _UnsupportedGroupError("its state cache is smaller than its states")
        state = pages[:, offset : offset + nbytes]
        offset += nbytes
        try:
            # [blocks, rows, pieces]: the TP-split dimension, then the rest.
            elements = state.view(torch.int16 if size == 2 else torch.int32)
            rows = elements.unflatten(1, shape).movedim(split.axis + 1, 1)
            if rows.dim() > 3:
                rows = rows.view(num_blocks, rows.shape[1], -1)
        except RuntimeError as e:
            raise _UnsupportedGroupError("its states cannot be viewed by rows") from e
        if sum(local for local, _ in split.segments) != rows.shape[1]:
            raise _UnsupportedGroupError("its state splits do not match its states")
        local_rows = 0
        for local, replicas in split.segments:
            caches.append(
                _LayerCache(
                    number,
                    rows[:, None, local_rows : local_rows + local, None],
                    replicas,
                    heads=local,
                    first_head=global_rows,
                )
            )
            local_rows += local
            global_rows += local * tp_size // replicas
    return caches


def _as_pieces(cache: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """A ``[blocks, kernel_blocks, heads, tokens, pieces]`` 8-byte piece view."""
    if cache.dim() != 4 or cache.shape[0] % num_blocks:
        raise _UnsupportedGroupError(
            f"its cache of shape {tuple(cache.shape)} does not hold {num_blocks} blocks"
        )
    try:
        pieces = cache.view(torch.uint8).view(torch.int64)
    except RuntimeError as e:
        raise _UnsupportedGroupError(
            "its KV cells or strides are not whole 8-byte pieces"
        ) from e
    # Kernel blocks subdivide manager blocks contiguously (group_kernel_blocks).
    return pieces.unflatten(0, (num_blocks, -1))


@dataclass
class _LayerCache:
    """One distinct KV cache of a group, or segment of a Mamba state, under the
    lowest layer number using it. Rank ``r`` holds shard ``r // replicas``:
    the ``heads`` slices from ``first_head + shard * heads`` in the layer.
    """

    number: int
    pieces: torch.Tensor
    replicas: int
    heads: int
    first_head: int = 0


def _build_layer_views(
    group: KVCacheGroupSpec,
    kv_caches: dict[str, torch.Tensor],
    num_blocks: int,
    tp_rank: int,
    tp_size: int,
    uniform_kv_heads: int | None,
) -> list[_LayerView]:
    """The views of a group's layers, each KV cache counted once.

    Raises:
        _UnsupportedGroupError: If the group cannot be checksummed.

    """
    if group.host_resident:
        raise _UnsupportedGroupError("its KV is host-resident")
    # KV-sharing layers alias their target's cache: count each cache once,
    # under the lowest (layer number, name) that uses it.
    owners: dict[tuple[int, tuple[int, ...]], tuple[int, str]] = {}
    entries: dict[tuple[int, tuple[int, ...]], list[_LayerCache]] = {}
    for layer_name in group.layer_names:
        spec = _layer_spec(group.kv_cache_spec, layer_name)
        cache = kv_caches.get(layer_name)
        if cache is None:
            raise _UnsupportedGroupError(f"layer {layer_name} has no KV cache")
        if isinstance(spec, MambaSpec) and isinstance(
            group.kv_cache_spec, UniformTypeKVCacheSpecs
        ):
            # The scheduler picks a Mamba group's state block by its group spec.
            raise _UnsupportedGroupError("its Mamba layers have different specs")
        number, _ = _layer_module(layer_name)
        identity = (cache.data_ptr(), cache.stride())
        if identity in owners:
            owners[identity] = min(owners[identity], (number, layer_name))
            continue
        owners[identity] = (number, layer_name)
        if isinstance(spec, MambaSpec) and spec.state_splits is not None:
            entries[identity] = _mamba_rows(spec, cache, num_blocks, number, tp_size)
        elif isinstance(spec, MambaSpec):
            pieces, replicas, heads = _mamba_pieces(spec, cache, num_blocks, tp_size)
            entries[identity] = [_LayerCache(number, pieces, replicas, heads)]
        else:
            pieces, replicas, heads = _attention_pieces(
                spec, cache, num_blocks, tp_size, uniform_kv_heads
            )
            entries[identity] = [_LayerCache(number, pieces, replicas, heads)]

    # The caches of one layer module (e.g. an attention's KV and its indexer's,
    # always on one PP stage) take consecutive slices, in layer name order.
    layers: list[_LayerCache] = []
    modules: dict[int, str] = {}
    next_head: dict[int, int] = defaultdict(int)
    for identity, (number, name) in sorted(owners.items(), key=lambda item: item[1]):
        module = _layer_module(name)[1]
        if modules.setdefault(number, module) != module:
            raise _UnsupportedGroupError(
                f"layers {modules[number]} and {module} share a layer number"
            )
        for layer in entries[identity]:
            layer.number = number
            layer.first_head += next_head[number]
        next_head[number] = max(
            layer.first_head + layer.heads * (tp_size // layer.replicas)
            for layer in entries[identity]
        )
        layers.extend(entries[identity])
    if any(heads > MAX_KV_HEADS_PER_LAYER for heads in next_head.values()):
        raise _UnsupportedGroupError(
            f"a layer has more than {MAX_KV_HEADS_PER_LAYER} KV heads"
        )
    views = []
    for layer in layers:
        shard, replica = divmod(tp_rank, layer.replicas)
        views.append(
            _LayerView(
                layer.pieces,
                first_slice=layer.number * MAX_KV_HEADS_PER_LAYER
                + layer.first_head
                + shard * layer.heads,
                replica_key=(layer.replicas, replica),
            )
        )
    return views


@dataclass
class _GroupBatch:
    """The rows of one KV cache group to checksum in one pass."""

    segments: list[tuple[str, int, int]] = field(default_factory=list)
    """``(req_id, start_position, num_rows)`` per request, in row order."""
    block_ids: list[int] = field(default_factory=list)
    num_tokens: list[int] = field(default_factory=list)
    """The request's number of tokens, per segment."""

    def rows(self, block_size: int) -> np.ndarray:
        """The ``[block_ids, first_tokens, valid_tokens]`` of the rows."""
        starts = np.array([start for _, start, _ in self.segments], dtype=np.int64)
        lengths = np.array([n for _, _, n in self.segments], dtype=np.int64)
        first_rows = np.cumsum(lengths) - lengths
        positions = np.arange(len(self.block_ids), dtype=np.int64)
        positions += np.repeat(starts - first_rows, lengths)
        first_tokens = positions * block_size
        num_tokens = np.repeat(np.array(self.num_tokens, dtype=np.int64), lengths)
        valid_tokens = np.clip(num_tokens - first_tokens, 0, block_size)
        block_ids = np.array(self.block_ids, dtype=np.int64)
        return np.stack([block_ids, first_tokens, valid_tokens])


class PendingKVChecksums:
    """A worker's KV checksums of one step, until they reach host memory.

    ``post_forward`` queues the checksum kernels on the main stream. The model
    runner then calls ``enqueue_async_cpu_copy`` on its output copy stream, ordered
    after them, and reads the record it returns once that copy is done.
    """

    def __init__(
        self,
        pp_rank: int,
        tp_rank: int,
        skipped_groups: frozenset[int],
        fingerprints: dict[int, int],
        received: dict[str, bytes],
    ):
        self._rank = (pp_rank, tp_rank)
        self._skipped_groups = skipped_groups
        self._fingerprints = fingerprints
        self._received = received
        self._req_ids: list[str] = []
        # (group id, segments, replica keys, [rows, len(replica keys)] int64
        # values) per checksum pass of a group: the loads and the sends of one
        # step are separate passes over the same groups.
        self._groups: list[
            tuple[int, list[tuple[str, int, int]], list[ReplicaKey], torch.Tensor]
        ] = []

    def __bool__(self) -> bool:
        return bool(self._req_ids or self._received)

    def add_requests(self, req_ids: Iterable[str]) -> None:
        self._req_ids.extend(req_ids)

    def add_group(
        self,
        group_id: int,
        segments: list[tuple[str, int, int]],
        replica_keys: list[ReplicaKey],
        values: torch.Tensor,
    ) -> None:
        self._groups.append((group_id, segments, replica_keys, values))

    def enqueue_async_cpu_copy(self) -> KVChecksumRecord:
        """Enqueue the asynchronous D2H copy; call on the output copy stream,
        after it waits for the main stream."""
        checksums: dict[str, dict[int, KVChecksumGroup]] = {
            req_id: {} for req_id in self._req_ids
        }
        for group_id, segments, replica_keys, device_values in self._groups:
            values = device_values.to("cpu", non_blocking=True).numpy()
            values = values.view(np.uint64)
            row = 0
            for req_id, start, num_rows in segments:
                checksums[req_id][group_id] = KVChecksumGroup(
                    start, replica_keys, values[row : row + num_rows]
                )
                row += num_rows
        return KVChecksumRecord(
            *self._rank,
            checksums,
            self._skipped_groups,
            self._fingerprints,
            self._received,
        )


class KVChecksumWorker:
    """Computes the KV checksums the scheduler asks a worker for.

    Requests to send are checksummed in the step that finishes their KV.
    Requests to receive are checksummed in the step whose transfer results
    report their load finished, before any later step can write their blocks.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, torch.Tensor],
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        uniform_kv_heads: int | None,
        draft_layer_names: frozenset[str] = frozenset(),
    ):
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        # Per group: loads check every replica of this worker's layers, while
        # a producer only needs the first replica of each layer, as any replica
        # describes the same KV and consumers check each of theirs against it.
        self._recv_plans: dict[int, _GroupPlan] = {}
        self._send_plans: dict[int, _GroupPlan] = {}
        self._block_sizes: dict[int, int] = {}
        skipped_groups = set()
        self._fingerprints: dict[int, int] = {}
        for group_id in kv_cache_config.transfer_group_ids:
            group = kv_cache_config.kv_cache_groups[group_id]
            # The drafter's KV is not checksummed: it only affects which draft
            # tokens are accepted, not the output.
            group = replace(
                group,
                layer_names=[
                    name for name in group.layer_names if name not in draft_layer_names
                ],
            )
            self._fingerprints[group_id] = _group_fingerprint(group, tp_size)
            try:
                views = _build_layer_views(
                    group,
                    kv_caches,
                    kv_cache_config.num_blocks,
                    tp_rank,
                    tp_size,
                    uniform_kv_heads,
                )
            except _UnsupportedGroupError as e:
                logger.warning(
                    "KV cache group %d is not covered by KV checksums: %s.",
                    group_id,
                    e,
                )
                skipped_groups.add(group_id)
                continue
            self._block_sizes[group_id] = group.kv_cache_spec.block_size
            if recv_plan := _GroupPlan.create(views):
                self._recv_plans[group_id] = recv_plan
            if send_plan := _GroupPlan.create(
                [view for view in views if view.replica_key[1] == 0]
            ):
                self._send_plans[group_id] = send_plan

        for plan in (*self._recv_plans.values(), *self._send_plans.values()):
            for launch in plan.launches:
                launch.warm_up(len(plan.replica_keys))
        self._skipped_groups = frozenset(skipped_groups)
        self._reqs_to_send: dict[str, KVChecksumBlocks] = {}
        self._reqs_to_recv: dict[str, KVChecksumBlocks] = {}

    def bind(self, scheduled: KVChecksumScheduled | None) -> None:
        """Take the requests the scheduler set for checksumming this step."""
        self._reqs_to_send = scheduled.reqs_to_send if scheduled is not None else {}
        if scheduled is not None:
            self._reqs_to_recv.update(scheduled.reqs_to_recv)

    def post_forward(
        self,
        transfer_results: "KVConnectorTransferResults",
        finished_req_ids: set[str],
    ) -> PendingKVChecksums | None:
        """Queue the checksums of this step's sends and finished loads.

        Args:
            transfer_results: The connector's transfer results of this step:
                the loads it finished (failed ones are not checked) and the
                producer checksums it received, which the first worker passes
                on to the scheduler.
            finished_req_ids: Requests that ended, e.g. aborted while loading.

        Returns:
            The step's checksums, still on the device, or None if there is
            nothing to report.

        """
        for req_id in finished_req_ids:
            self._reqs_to_recv.pop(req_id, None)
        loaded = {
            req_id: blocks
            for req_id in transfer_results.finished_recving
            if (blocks := self._reqs_to_recv.pop(req_id, None)) is not None
            and req_id not in transfer_results.failed_recving
        }

        pending = PendingKVChecksums(
            self.pp_rank,
            self.tp_rank,
            self._skipped_groups,
            self._fingerprints,
            # Every worker's connector receives the producer's checksums, so
            # the first worker passes them on for all.
            transfer_results.kv_checksums if self.pp_rank == self.tp_rank == 0 else {},
        )
        self._compute(loaded, pending, self._recv_plans)
        self._compute(self._reqs_to_send, pending, self._send_plans)
        self._reqs_to_send = {}
        return pending or None

    def _compute(
        self,
        reqs: dict[str, KVChecksumBlocks],
        output: PendingKVChecksums,
        plans: dict[int, _GroupPlan],
    ) -> None:
        """Checksum the blocks of ``reqs`` into ``output``, as ``plans`` say."""
        output.add_requests(reqs)
        batches: dict[int, _GroupBatch] = defaultdict(_GroupBatch)
        for req_id, blocks in reqs.items():
            for group_id, (start, block_ids) in blocks.groups.items():
                if group_id not in plans or not block_ids:
                    continue
                batch = batches[group_id]
                batch.segments.append((req_id, start, len(block_ids)))
                batch.block_ids.extend(block_ids)
                batch.num_tokens.append(blocks.num_tokens)
        if not batches:
            return
        groups = [
            (group_id, batch, plans[group_id]) for group_id, batch in batches.items()
        ]
        device = groups[0][2].device

        # One copy for every group. Each group starts at an even column, so
        # the slices are 16-byte aligned like those the kernel was warmed up on.
        columns = []
        width = 0
        for _, batch, _ in groups:
            columns.append(width)
            width += len(batch.block_ids) + len(batch.block_ids) % 2
        host_tensor = torch.zeros((3, width), dtype=torch.int64, pin_memory=PIN_MEMORY)
        host = host_tensor.numpy()
        for (group_id, batch, _), column in zip(groups, columns):
            rows = batch.rows(self._block_sizes[group_id])
            host[:, column : column + rows.shape[1]] = rows
        device_rows = host_tensor.to(device, non_blocking=True)

        for (group_id, batch, plan), column in zip(groups, columns):
            num_rows = len(batch.block_ids)
            block_ids, first_tokens, valid_tokens = device_rows[
                :, column : column + num_rows
            ]
            values = torch.zeros(
                (num_rows, len(plan.replica_keys)),
                dtype=torch.int64,
                device=device,
            )
            for launch in plan.launches:
                launch.accumulate(block_ids, first_tokens, valid_tokens, values)
            output.add_group(group_id, batch.segments, plan.replica_keys, values)


def create_kv_checksum_worker(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
) -> KVChecksumWorker | None:
    """This worker's KV checksum worker, or None if KV checksums are off."""
    if not kv_checksum_enabled(vllm_config):
        return None
    return KVChecksumWorker(
        kv_cache_config,
        kv_caches,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        pp_rank=get_pp_group().rank_in_group,
        uniform_kv_heads=_uniform_kv_heads(vllm_config),
        draft_layer_names=_draft_layer_names(vllm_config),
    )


def _draft_layer_names(vllm_config: VllmConfig) -> frozenset[str]:
    """The drafter's KV cache layers, marked when the drafter was loaded."""
    layers = get_layers_from_vllm_config(
        vllm_config,
        AttentionLayerBase,  # type: ignore[type-abstract]
    )
    return frozenset(name for name, layer in layers.items() if layer.is_draft_layer)


def _uniform_kv_heads(vllm_config: VllmConfig) -> int | None:
    """The model's KV head count if every attention layer has it, else None.

    Only needed to tell sharded from replicated heads at one head per rank.
    """
    model_config = vllm_config.model_config
    total = model_config.get_total_num_kv_heads()
    overrides = model_config.model_arch_config.per_layer_overrides or []
    global_heads = getattr(
        model_config.hf_text_config, "num_global_key_value_heads", None
    )
    if global_heads not in (None, total) or any(
        layer.get("total_num_kv_heads", total) != total for layer in overrides
    ):
        return None
    return total
