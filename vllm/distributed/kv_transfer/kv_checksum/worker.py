# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side KV checksum computation.

Each worker computes its part of a block's checksum: the slices (layer, KV
head) it holds, each weighed by its global layer and head (see
``checksum.py``). Parts from all TP ranks and PP stages add up to the block's
checksum.

Results are copied to pinned host memory on a side stream ordered after the
work that produces them; readers wait only for that copy's event, never for
the main stream.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    MAX_KV_HEADS_PER_LAYER,
    KVChecksumBlocks,
    KVChecksumGroup,
    KVChecksumRecord,
    KVChecksumScheduled,
    ReplicaKey,
    kv_checksum_enabled,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY, current_stream
from vllm.v1.core.kv_cache_planning import layer_tp_replicas
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)

# Upper bound on the bytes of KV gathered at once, which bounds the
# transient device memory of a checksum pass for long prompts.
_MAX_GATHER_BYTES = 64 * 1024 * 1024


class _LayerView:
    """One layer's KV cache as 8-byte pieces, for weighted sums.

    ``pieces`` is ``[num_blocks, kernel_blocks, heads, tokens, pieces]``. A
    piece ``j`` of slice ``(layer, head)`` weighs ``m * (2 * j + 1)``, with
    ``m = 2 * (layer * H + head) + 1`` for the global head (see
    ``checksum.py``). ``weights`` are those of a block holding the request's
    first tokens; a block starting at token ``f`` adds ``f`` times
    ``offset_weights`` to them.
    """

    def __init__(
        self, pieces: torch.Tensor, first_slice: int, replica_key: ReplicaKey
    ) -> None:
        """Set up the weighted sums of this rank's slices.

        Args:
            pieces: The layer's cache viewed as 8-byte pieces.
            first_slice: ``layer * H + head`` of this rank's first head.
            replica_key: The replicas and replica index of this rank's heads.

        """
        self.pieces = pieces
        self.replica_key = replica_key
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
        dims = (1, 2, 3, 4)
        for start in range(0, block_ids.shape[0], self.rows_per_gather):
            end = start + self.rows_per_gather
            rows = self.pieces.index_select(0, block_ids[start:end])
            valid = valid_tokens[start:end].view(-1, 1, 1, 1, 1)
            rows.masked_fill_(self.token_index >= valid, 0)
            out[start:end] += (rows * self.weights).sum(dim=dims) + first_tokens[
                start:end
            ] * (rows * self.offset_weights).sum(dim=dims)


def _layer_spec(group_spec: KVCacheSpec, layer_name: str) -> KVCacheSpec:
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return group_spec.kv_cache_specs[layer_name]
    return group_spec


def _group_fingerprint(group: KVCacheGroupSpec) -> int:
    """The sum of a hash of each of the group's layers on this worker."""
    total = 0
    for layer_name in group.layer_names:
        spec = _layer_spec(group.kv_cache_spec, layer_name)
        identity = (layer_name, type(spec).__name__, str(getattr(spec, "dtype", None)))
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
    """One distinct KV cache of a group, under the lowest layer number using
    it. Rank ``r`` holds shard ``r // replicas``:
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
        number, _ = _layer_module(layer_name)
        identity = (cache.data_ptr(), cache.stride())
        if identity in owners:
            owners[identity] = min(owners[identity], (number, layer_name))
            continue
        owners[identity] = (number, layer_name)
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


class KVChecksumOutput:
    """KV checksums of one step.

    A worker's output holds its device results and their in-flight copy to
    host memory. ``finalize`` waits for the copy and turns them into a record,
    and ``aggregate`` merges the records of all workers.
    """

    def __init__(
        self,
        pp_rank: int = 0,
        tp_rank: int = 0,
        skipped_groups: frozenset[int] = frozenset(),
        fingerprints: dict[int, int] | None = None,
    ):
        self._rank = (pp_rank, tp_rank)
        self._skipped_groups = skipped_groups
        self._fingerprints = fingerprints or {}
        self._req_ids: list[str] = []
        self._received: dict[str, bytes] = {}
        # group id -> (segments, replica keys, [rows, len(replica keys)] int64
        # values). The device values stay referenced until the copy is done.
        self._groups: dict[
            int, tuple[list[tuple[str, int, int]], list[ReplicaKey], torch.Tensor]
        ] = {}
        self._host: dict[int, torch.Tensor] = {}
        self._copy_event: torch.cuda.Event | None = None
        self._records: list[KVChecksumRecord] = []

    def __bool__(self) -> bool:
        return bool(self._req_ids or self._received or self._records)

    def add_requests(self, req_ids: Iterable[str]) -> None:
        self._req_ids.extend(req_ids)

    def add_received(self, received: dict[str, bytes]) -> None:
        self._received.update(received)

    def add_group(
        self,
        group_id: int,
        segments: list[tuple[str, int, int]],
        replica_keys: list[ReplicaKey],
        values: torch.Tensor,
    ) -> None:
        self._groups[group_id] = (segments, replica_keys, values)

    def start_cpu_copy(self, copy_stream: torch.cuda.Stream | None) -> None:
        """Queue the copy to host memory after the work queued so far.

        Args:
            copy_stream: Side stream for the copy. None (devices without CUDA
                streams) copies synchronously, which blocks the host.

        """
        if not self._groups:
            return
        if copy_stream is None:
            self._host = {g: v.cpu() for g, (_, _, v) in self._groups.items()}
            return
        main_stream = current_stream()
        copy_stream.wait_stream(main_stream)
        torch.cuda.set_stream(copy_stream)
        try:
            self._host = {
                g: v.to("cpu", non_blocking=True)
                for g, (_, _, v) in self._groups.items()
            }
        finally:
            torch.cuda.set_stream(main_stream)
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self._copy_event = torch.cuda.Event(blocking=True)
        self._copy_event.record(copy_stream)

    def finalize(self) -> list[KVChecksumRecord]:
        """Wait for the copy, if any, and return the records."""
        if not self._req_ids and not self._received:
            return self._records
        if self._groups and not self._host:
            self.start_cpu_copy(None)
        if self._copy_event is not None:
            self._copy_event.synchronize()
        checksums: dict[str, dict[int, KVChecksumGroup]] = {
            req_id: {} for req_id in self._req_ids
        }
        for group_id, (segments, replica_keys, _) in self._groups.items():
            values = self._host[group_id].numpy().view(np.uint64)
            row = 0
            for req_id, start, num_rows in segments:
                checksums[req_id][group_id] = KVChecksumGroup(
                    start, replica_keys, values[row : row + num_rows]
                )
                row += num_rows
        self._records.append(
            KVChecksumRecord(
                *self._rank,
                checksums,
                self._skipped_groups,
                self._fingerprints,
                self._received,
            )
        )
        self._req_ids.clear()
        self._received = {}
        self._groups.clear()
        self._host.clear()
        self._copy_event = None
        return self._records

    def aggregate(self, other: "KVChecksumOutput") -> "KVChecksumOutput":
        self._records = self.finalize() + other.finalize()
        return self

    def __getstate__(self) -> dict:
        # Device tensors and CUDA events cannot cross processes.
        self.finalize()
        return self.__dict__


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
    ):
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self._layer_views: dict[int, list[_LayerView]] = {}
        self._block_sizes: dict[int, int] = {}
        skipped_groups = set()
        self._fingerprints: dict[int, int] = {}
        for group_id in kv_cache_config.transfer_group_ids:
            group = kv_cache_config.kv_cache_groups[group_id]
            self._fingerprints[group_id] = _group_fingerprint(group)
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
            if views:
                self._layer_views[group_id] = views
                self._block_sizes[group_id] = group.kv_cache_spec.block_size

        device = next(
            (views[0].pieces.device for views in self._layer_views.values()), None
        )
        self._copy_stream = (
            torch.cuda.Stream(device)
            if device is not None and device.type == "cuda"
            else None
        )
        self._skipped_groups = frozenset(skipped_groups)
        self._reqs_to_send: dict[str, KVChecksumBlocks] = {}
        self._reqs_to_recv: dict[str, KVChecksumBlocks] = {}
        self._deferred_output: KVChecksumOutput | None = None

    def bind(self, scheduled: KVChecksumScheduled | None) -> None:
        """Take the requests the scheduler set for checksumming this step."""
        self._reqs_to_send = scheduled.reqs_to_send if scheduled is not None else {}
        if scheduled is not None:
            self._reqs_to_recv.update(scheduled.reqs_to_recv)

    def post_forward(
        self,
        finished_recving: set[str] | None,
        failed_recving: set[str],
        finished_req_ids: set[str],
        received: dict[str, bytes] | None = None,
        defer_sends: bool = False,
    ) -> KVChecksumOutput | None:
        """Checksum this step's sends and finished loads, and start copying the
        results to host.

        Args:
            finished_recving: Requests whose load this worker finished.
            failed_recving: Requests whose load failed; nothing to check.
            finished_req_ids: Requests that ended, e.g. aborted while loading.
            received: Producer checksums the connector received with finished
                loads, passed on to the scheduler.
            defer_sends: Leave the requests to send, and the copy, to
                ``finalize_sends``, for a drafter that writes this step's KV
                after this call.

        Returns:
            The step's checksums, or None if there is nothing to report.

        """
        for req_id in finished_req_ids:
            self._reqs_to_recv.pop(req_id, None)
        loaded = {
            req_id: blocks
            for req_id in finished_recving or ()
            if (blocks := self._reqs_to_recv.pop(req_id, None)) is not None
            and req_id not in failed_recving
        }

        output = KVChecksumOutput(
            self.pp_rank, self.tp_rank, self._skipped_groups, self._fingerprints
        )
        output.add_received(received or {})
        self._compute(loaded, output, replica_zero_only=False)
        if defer_sends and self._reqs_to_send:
            self._deferred_output = output
            return output
        self._compute_sends(output)
        return output or None

    def finalize_sends(self) -> None:
        """Checksum the requests to send that ``post_forward`` deferred."""
        if (output := self._deferred_output) is not None:
            self._deferred_output = None
            self._compute_sends(output)

    def _compute_sends(self, output: KVChecksumOutput) -> None:
        self._compute(self._reqs_to_send, output, replica_zero_only=True)
        self._reqs_to_send = {}
        output.start_cpu_copy(self._copy_stream)

    def _compute(
        self,
        reqs: dict[str, KVChecksumBlocks],
        output: KVChecksumOutput,
        replica_zero_only: bool,
    ) -> None:
        """Checksum the blocks of ``reqs`` into ``output``.

        A producer only needs the first replica of each layer: any replica of
        it describes the same KV, and consumers check each of theirs against it.
        """
        output.add_requests(reqs)
        batches: dict[int, _GroupBatch] = defaultdict(_GroupBatch)
        for req_id, blocks in reqs.items():
            for group_id, (start, block_ids) in blocks.groups.items():
                if group_id not in self._layer_views or not block_ids:
                    continue
                batch = batches[group_id]
                batch.segments.append((req_id, start, len(block_ids)))
                batch.block_ids.extend(block_ids)
                batch.num_tokens.append(blocks.num_tokens)

        device = None
        groups = []
        for group_id, batch in batches.items():
            layers = [
                layer
                for layer in self._layer_views[group_id]
                if not replica_zero_only or layer.replica_key[1] == 0
            ]
            if layers:
                device = layers[0].pieces.device
                groups.append((group_id, batch, layers))
        if device is None:
            return

        # One copy for every group.
        columns = []
        width = 0
        for _, batch, _ in groups:
            columns.append(width)
            width += len(batch.block_ids)
        host = np.zeros((3, width), dtype=np.int64)
        for (group_id, batch, _), column in zip(groups, columns):
            rows = batch.rows(self._block_sizes[group_id])
            host[:, column : column + rows.shape[1]] = rows
        device_rows = _to_device(host, device)

        for (group_id, batch, layers), column in zip(groups, columns):
            num_rows = len(batch.block_ids)
            block_ids, first_tokens, valid_tokens = device_rows[
                :, column : column + num_rows
            ]
            replica_keys = list(dict.fromkeys(layer.replica_key for layer in layers))
            values = torch.zeros(
                (num_rows, len(replica_keys)),
                dtype=torch.int64,
                device=device,
            )
            for layer in layers:
                layer.accumulate(
                    block_ids,
                    first_tokens,
                    valid_tokens,
                    values[:, replica_keys.index(layer.replica_key)],
                )
            output.add_group(group_id, batch.segments, replica_keys, values)


def _to_device(values: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy ``values`` to ``device`` without blocking the host."""
    host = torch.from_numpy(values)
    if device.type == "cpu":
        return host
    if PIN_MEMORY:
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


_KV_CHECKSUM_WORKER: KVChecksumWorker | None = None


def init_kv_checksum_worker(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """Create this worker's KV checksum worker if KV checksums are enabled."""
    global _KV_CHECKSUM_WORKER
    _KV_CHECKSUM_WORKER = None
    if not kv_checksum_enabled(vllm_config):
        return
    _KV_CHECKSUM_WORKER = KVChecksumWorker(
        kv_cache_config,
        kv_caches,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        pp_rank=get_pp_group().rank_in_group,
        uniform_kv_heads=_uniform_kv_heads(vllm_config),
    )


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
    speculative_config = vllm_config.speculative_config
    draft = speculative_config.draft_model_config if speculative_config else None
    if draft is not None and draft.get_total_num_kv_heads() != total:
        return None
    return total


def get_kv_checksum_worker() -> KVChecksumWorker | None:
    return _KV_CHECKSUM_WORKER
