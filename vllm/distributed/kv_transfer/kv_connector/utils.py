# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache helper for store."""

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from vllm.config import (
    VllmConfig,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

EngineId = str
# block ids as returned by the hybrid KV cache manager. list[list[int]] are allow
# mutability and are for connector internal use only.
BlockIds = tuple[list[int], ...] | list[list[int]]

# kv_transfer_params key under which a prefill instance reports how far the
# prompt's M-RoPE positions trail its token indices. A decode instance that gets
# the expanded prompt without its multimodal inputs (e.g. one started with
# --language-model-only) cannot derive this, and would otherwise place every
# token it computes as if the prompt were plain text.
MROPE_POSITION_DELTA_KEY = "mrope_position_delta"


def get_transferred_mrope_position_delta(
    vllm_config: VllmConfig, sampling_params: "SamplingParams | None"
) -> int | None:
    """The M-RoPE position delta the prefill instance reported for a request
    whose KV this instance receives, or None if there is none to apply."""
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or not kv_transfer_config.is_kv_consumer:
        return None
    if sampling_params is None or not sampling_params.extra_args:
        return None
    kv_transfer_params = sampling_params.extra_args.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict):
        return None
    delta = kv_transfer_params.get(MROPE_POSITION_DELTA_KEY)
    if delta is None:
        return None
    # Transports that carry kv_transfer_params as a protobuf Struct turn every
    # number into a float.
    if isinstance(delta, float) and delta.is_integer():
        delta = int(delta)
    if isinstance(delta, bool) or not isinstance(delta, int):
        logger.warning_once(
            "Ignoring non-integer %s in kv_transfer_params: %r",
            MROPE_POSITION_DELTA_KEY,
            delta,
        )
        return None
    return delta


def get_max_offload_tokens(kv_transfer_params: dict[str, Any] | None) -> int | None:
    """Parse the request's offload cap, preserving zero and rejecting booleans."""
    raw = kv_transfer_params.get("max_offload_tokens") if kv_transfer_params else None
    if type(raw) is int and raw >= 0:
        return raw
    if raw is not None:
        logger.warning(
            "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
        )
    return None


def clip_ssm_state_blocks(blocks: list[int], num_spec_blocks: int) -> list[int]:
    """Drop speculative scratch slots and keep only the running-state slot."""
    if not blocks:
        return blocks
    # Keep at least the running-state slot, even before scratch allocation.
    if num_scratch := min(num_spec_blocks, len(blocks) - 1):
        blocks = blocks[:-num_scratch]
    return blocks[-1:]


def get_kv_connector_cache_layout(vllm_config: VllmConfig | None = None):
    # NOTE (NickLucche) When running disaggregated PD with NIXL, LBHNC layout is
    # used for faster transfer.
    if vllm_config is None:
        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    kv_config = vllm_config.kv_transfer_config
    if kv_config is not None:
        connector_cls = KVConnectorFactory.get_connector_class(kv_config)
        required_kvcache_layout = connector_cls.get_required_kvcache_layout(vllm_config)
        if required_kvcache_layout is not None:
            return required_kvcache_layout
    return None


def kv_connector_may_load_foreign_kv(vllm_config: VllmConfig) -> bool:
    """Whether the configured KV connector may load bytes that vLLM's own
    cache writer did not produce for this KV cache layout.

    Attention backends whose fast paths assume no foreign bytes (e.g. NaNs in
    unwritten slots) reach the KV cache keep them only when this is False.
    """
    kv_config = vllm_config.kv_transfer_config
    if kv_config is None:
        return False
    connector_cls = KVConnectorFactory.get_connector_class(kv_config)
    return not connector_cls.only_restores_canonical_kv(kv_config)


class KVOutputAggregator:
    """Utility class to aggregate the output of all workers into a single
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, expected_finished_count: int):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        self._recv_remaining_count = dict[str, int]()
        self._send_remaining_count = dict[str, int]()
        self._failed_recving_pending = set[str]()
        self._expected_finished_count = expected_finished_count

        self._salvaged_outputs: list[KVConnectorOutput] = []

    @classmethod
    def from_connector(cls, connector: "KVConnectorBase", world_size: int):
        return cls(connector.get_finished_count() or world_size)

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        if not outputs[output_rank]:
            return None

        # Aggregate kv_connector_output from all workers

        def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
        ) -> None:
            for req_id in req_ids or ():
                remaining_count = remaining_count_dict.get(
                    req_id, self._expected_finished_count
                )
                remaining_count_dict[req_id] = remaining_count - 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_kv_connector_stats = None
        aggregated_kv_connector_worker_meta = None
        combined_kv_cache_events = None
        invalid_block_ids = set[int]()
        mrope_position_deltas: dict[str, int] = {}

        salvaged, self._salvaged_outputs = self._salvaged_outputs, []

        kv_outputs: list[KVConnectorOutput | None] = []
        for o in outputs:
            assert o is not None
            kv_outputs.append(o.kv_connector_output)

        # Salvaged outputs from failed workers join this round so their
        # transfer progress isn't lost.
        for kv_output in chain(salvaged, kv_outputs):
            if not kv_output:
                continue
            # Allow the worker to dynamically update the expected number of
            # finished sending/recving for new requests.
            if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
            ):
                logger.debug(
                    "Expected finished requests updated from %d to %d",
                    self._expected_finished_count,
                    kv_output.expected_finished_count,
                )
                self._expected_finished_count = kv_output.expected_finished_count

            update_finished_set(
                kv_output.finished_sending, self._send_remaining_count, finished_sending
            )
            update_finished_set(
                kv_output.finished_recving, self._recv_remaining_count, finished_recving
            )

            # Aggregate kv_connector_stats from all workers.
            if aggregated_kv_connector_stats is None:
                # Use the first worker's kv_connector_stats as accumulator.
                aggregated_kv_connector_stats = kv_output.kv_connector_stats
            elif kv_connector_stats := kv_output.kv_connector_stats:
                assert isinstance(
                    aggregated_kv_connector_stats, type(kv_connector_stats)
                )
                aggregated_kv_connector_stats = aggregated_kv_connector_stats.aggregate(
                    kv_connector_stats
                )

            # Aggregate kv_connector_worker_meta from all workers.
            if aggregated_kv_connector_worker_meta is None:
                # Use the first worker's kv_connector_worker_meta as accumulator.
                aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
            elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
                aggregated_kv_connector_worker_meta = (
                    aggregated_kv_connector_worker_meta.aggregate(
                        kv_connector_worker_meta
                    )
                )

            # Combine kv_cache_events from all workers.
            if combined_kv_cache_events is None:
                # Use the first worker's kv_cache events as start event list.
                combined_kv_cache_events = kv_output.kv_cache_events
            elif kv_cache_events := kv_output.kv_cache_events:
                assert isinstance(
                    combined_kv_cache_events,
                    type(kv_cache_events),
                )
                worker_kv_cache_events = kv_cache_events.get_all_events()
                combined_kv_cache_events.add_events(worker_kv_cache_events)
                combined_kv_cache_events.increment_workers(1)

            invalid_block_ids |= kv_output.invalid_block_ids
            self._failed_recving_pending |= kv_output.failed_recving
            # Every rank computes the same deltas for the requests it adds.
            if kv_output.mrope_position_deltas:
                mrope_position_deltas.update(kv_output.mrope_position_deltas)

        failed_recving = self._failed_recving_pending & finished_recving
        self._failed_recving_pending -= failed_recving

        # select output of the worker specified by output_rank
        output = outputs[output_rank]

        assert output is not None
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            kv_connector_stats=aggregated_kv_connector_stats or None,
            kv_cache_events=combined_kv_cache_events or None,
            kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
            invalid_block_ids=invalid_block_ids,
            failed_recving=failed_recving,
            mrope_position_deltas=mrope_position_deltas or None,
            expected_finished_count=self._expected_finished_count,
        )

        return output

    def ft_defer_kv_connector_outputs(
        self, kv_connector_outputs: list[KVConnectorOutput | None]
    ) -> None:
        """Stash KV connector outputs from failed workers for replay at the
        next aggregate().

        kv_cache_events are dropped: replaying them could break the live
        workers' event quorum, and a missed event only costs a cache miss.
        """
        for o in kv_connector_outputs:
            if o is not None:
                o.kv_cache_events = None
        self._salvaged_outputs.extend(o for o in kv_connector_outputs if o)


def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: torch.device | str,
    dst_device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _to(block_ids: list[int], device: torch.device | str) -> torch.Tensor:
        device = torch.device(device) if isinstance(device, str) else device
        if device.type == "cpu":
            return torch.tensor(block_ids, dtype=torch.int64, device=device)
        return async_tensor_h2d(block_ids, dtype=torch.int64, device=device)

    return _to(src_block_ids, src_device), _to(dst_block_ids, dst_device)


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None:
    """Copy kv blocks between different buffers."""
    if (
        not src_kv_caches
        or not dst_kv_caches
        or not src_block_ids
        or not dst_block_ids
        or len(src_block_ids) != len(dst_block_ids)
    ):
        return

    src_device = next(iter(src_kv_caches.values())).device
    dst_device = next(iter(dst_kv_caches.values())).device

    src_indices, dst_indices = _make_src_and_dst_indices(
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        src_device=src_device,
        dst_device=dst_device,
    )

    if direction == "h2d":
        copy_fn = current_platform.insert_blocks_to_device
    else:
        copy_fn = current_platform.swap_out_blocks_to_host
    for layer_name in src_kv_caches:
        src_tensor = src_kv_caches[layer_name]
        dst_tensor = dst_kv_caches[layer_name]
        copy_fn(src_tensor, dst_tensor, src_indices, dst_indices)


def kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio):
    """Transforms the layout of received KV cache blocks to the local block_size.
    (Only works for local blocksize > remote blocksize)

    Example:
    local blocksize = 16 tokens, remote blocksize = 4 tokens
    local block[0] = remote block[0, 1, 2, 3]
    remote is |h0-b0|h1-b0|h2-b0|h3-b0|h0-b1|h1-b1|h2-b1|h3-b1|...
    local is  |h0-b0..................|h1-b0..................|...
    permute is to:
    1. view => view remote as n_blocks * remote_shape(H,remoteN,D)
    2. permute => (H, nblocks, remoteN, D)
    3. flatten => (H, localN, D)

    """
    blocks_to_update = cache.index_select(0, indices)
    n_kv_heads, block_size, head_size = blocks_to_update.shape[1:]
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    permuted_blocks = (
        blocks_to_update.reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 2, 1, 3, 4)
        .flatten(2, 3)
    )
    cache.index_copy_(0, indices, permuted_blocks)


def kv_postprocess_blksize_and_layout_on_receive(cache, indices, block_size_ratio):
    """Transforms the layout of received KV cache to the local block_size and LBHNC.
    (Only works for local blocksize >= remote blocksize)

    prefill is LBHNC, smaller block_size
    decode(local) is LBNHC, larger block_size
    """
    blocks_to_update = cache.index_select(0, indices)

    n_kv_heads, block_size, head_size = blocks_to_update.shape[1:]
    remote_block_size = block_size // block_size_ratio
    n_blocks = block_size_ratio

    # View the received bytes in memory order as the remote head-major blocks.
    permuted_blocks = (
        blocks_to_update.transpose(1, 2)
        .reshape(-1, n_blocks, n_kv_heads, remote_block_size, head_size)
        .permute(0, 2, 1, 3, 4)
        .flatten(2, 3)
    )
    cache.index_copy_(0, indices, permuted_blocks)


def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...] | None, bool]]:
    """Yields:
    (req_id, block_id_groups, is_full_table)

    `is_full_table` is True when `block_id_groups` is the request's whole
    block table (new and resumed requests, which are both in
    `scheduled_new_reqs`) rather than this step's additions.
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, True

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    for req_id, block_id_groups in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
        yield req_id, block_id_groups, False


def get_current_attn_backends(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> list[type[AttentionBackend]]:
    """Get all distinct attention backends for the given layers.

    Args:
        vllm_config: The current vLLM configuration.
        layer_names: Optional list of layer names to scope the lookup.
            When None, all attention layers are considered.

    Returns:
        Deduplicated list of attention backend classes.

    """
    layer_type = cast(type[Any], AttentionLayerBase)
    layers = get_layers_from_vllm_config(vllm_config, layer_type, layer_names)
    if layers:
        seen: dict[str, type[AttentionBackend]] = {}
        for layer in layers.values():
            backend = layer.get_attn_backend()
            seen[backend.full_cls_name()] = backend
        return list(seen.values())

    # Fallback for tests, when static_forward_context is empty.
    logger.debug(
        "No layers found in the vLLM config. Falling back to default attention backend."
    )
    from vllm.v1.attention.selector import get_attn_backend

    with set_current_vllm_config(vllm_config):
        return [
            get_attn_backend(
                head_size=vllm_config.model_config.get_head_size(),
                dtype=vllm_config.model_config.dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                use_mla=vllm_config.model_config.use_mla,
            )
        ]


def get_current_attn_backends_and_specs(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    fallback: list[type[AttentionBackend]],
) -> tuple[
    list[type[AttentionBackend]], list[AttentionSpec | None], list[list[str]] | None
]:
    """Distinct (backend, spec, query heads) groups of the transfer layers, with
    their layer names, else `fallback` with no layer names.

    The layer names let kernel block selection match the model runner's, which
    picks per attention group. Attention caches in a block-outermost layout are
    skipped: blocks there are never split group-wide
    (`prepare_kernel_block_sizes`), and a cache that needs smaller kernel blocks
    views them inside each block, so its kernel pages never split transferred
    blocks. Compressed caches are skipped too (`check_compressed_kernel_split`
    guards them).
    """
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    groups: dict[
        tuple[type[AttentionBackend], AttentionSpec | None, int], list[str]
    ] = {}
    has_layers = False
    layer_type = cast(type[Any], AttentionLayerBase)
    for group in kv_cache_config.transfer_groups:
        layers = get_layers_from_vllm_config(vllm_config, layer_type, group.layer_names)
        for name, layer in layers.items():
            has_layers = True
            group_spec = group.kv_cache_specs[name]
            spec: AttentionSpec | None = None
            if isinstance(group_spec, AttentionSpec):
                if group_spec.tokens_per_state != 1 or layout.is_block_outermost:
                    continue
                spec = group_spec
            key = (layer.get_attn_backend(), spec, getattr(layer, "num_heads", 0))
            groups.setdefault(key, []).append(name)
    if not has_layers:
        return fallback, [None] * len(fallback), None
    return (
        [backend for backend, _, _ in groups],
        [spec for _, spec, _ in groups],
        list(groups.values()),
    )


def check_compressed_kernel_split(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, kernel_block_size: int
) -> None:
    """A layer-outer layout cannot split compressed pages into kernel blocks."""
    if vllm_config.cache_config.get_resolved_kv_cache_layout().is_block_outermost:
        return
    for group in kv_cache_config.transfer_groups:
        for spec in group.kv_cache_specs.values():
            if (
                isinstance(spec, AttentionSpec)
                and spec.tokens_per_state > 1
                and spec.block_size % kernel_block_size == 0
                and spec.block_size != kernel_block_size
            ):
                raise NotImplementedError(
                    "Compressed KV caches cannot be split into kernel blocks "
                    "with a layer-outermost KV cache layout."
                )


def get_current_attn_backend(
    vllm_config: VllmConfig, layer_names: list[str] | None = None
) -> type[AttentionBackend]:
    """Get the first attention backend for the given layers."""
    return get_current_attn_backends(vllm_config, layer_names)[0]


# The offset, shape and strides of a byte view in its block.
SerializedMetaTensor = tuple[int, tuple[int, ...], tuple[int, ...]]
# One layer's views, by component.
SerializedLayerMetaTensors = dict[str, SerializedMetaTensor]


def byte_view(cache: torch.Tensor) -> torch.Tensor:
    """`cache` as bytes on the meta device, from its first element."""
    size = cache.element_size()
    assert cache.stride(-1) == 1
    return torch.empty(0, dtype=torch.int8, device="meta").as_strided(
        (*cache.shape[:-1], cache.shape[-1] * size),
        (*(stride * size for stride in cache.stride()[:-1]), 1),
    )


# Dims of a `KVCacheSpec.slice_for_tp_transfer` view: its blocks, what TP
# shards (a rank holds a run of it, or a copy of all of it) and its states.
_VIEW_DIM_B, _VIEW_DIM_TP, _VIEW_DIM_N = 0, 1, 2


def view_block_size_ratio(
    views: dict[str, torch.Tensor], other_views: dict[str, torch.Tensor]
) -> int:
    """How many blocks of the other side a block of ours holds, from the
    states of both sides' `KVCacheSpec.slice_for_tp_transfer` views of a
    layer."""
    for name, view in views.items():
        if name in other_views:
            num_states = other_views[name].shape[_VIEW_DIM_N]
            if view.shape[_VIEW_DIM_N] % num_states:
                raise NotImplementedError(
                    f"A block of {name!r} holds {view.shape[_VIEW_DIM_N]} states "
                    f"here, which is not a multiple of the other side's "
                    f"{num_states}."
                )
            return view.shape[_VIEW_DIM_N] // num_states
    return 1


def serialize_layer_meta_tensors(
    views: dict[str, torch.Tensor],
) -> SerializedLayerMetaTensors:
    return {
        name: (view.storage_offset(), tuple(view.shape), tuple(view.stride()))
        for name, view in views.items()
    }


def deserialize_layer_meta_tensors(
    serialized: SerializedLayerMetaTensors,
) -> dict[str, torch.Tensor]:
    """The serialized byte views as tensors on the meta device."""
    block = torch.empty(0, dtype=torch.int8, device="meta")
    return {
        name: block.as_strided(shape, strides, offset)
        for name, (offset, shape, strides) in serialized.items()
    }


def pair_shards(
    views: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
    other_views: dict[str, torch.Tensor],
    other_tp_size: int,
    *,
    receiving: bool,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Pair a TP rank's `KVCacheSpec.slice_for_tp_transfer` views with those of
    the ranks of another TP size: other rank -> `(shard, other shard)` pairs.

    Both sides cut what TP shards of a view into shards of one length, the
    longest that fits both, so equal TP sizes leave the views whole. Side by
    side in rank order, the shards of a side are the whole layer:

        TP 2   | rank 0          | rank 1          |   two shards to a rank
        TP 4   | rank 0 | rank 1 | rank 2 | rank 3 |   one shard to a rank

    A receiving shard takes the sending shard at the same position. A side
    with more shards holds copies (heads replicated across ranks, Mamba2
    groups or a state every rank holds whole), one shard to a rank: a receiver
    takes one copy, and a sender gives to every copy.
    """
    pairs: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    for name, view in views.items():
        if name not in other_views:
            raise NotImplementedError(
                f"The other rank has no {name!r}: the two do not shard this "
                "layer the same way."
            )
        other_view = other_views[name]
        length = math.gcd(view.shape[_VIEW_DIM_TP], other_view.shape[_VIEW_DIM_TP])
        shards = view.split(length, dim=_VIEW_DIM_TP)
        other_shards = other_view.split(length, dim=_VIEW_DIM_TP)
        num_shards = tp_size * len(shards)
        other_num_shards = other_tp_size * len(other_shards)
        copies = shards if num_shards > other_num_shards else other_shards
        if shards[0].shape != other_shards[0].shape or (
            num_shards != other_num_shards and len(copies) != 1
        ):
            raise NotImplementedError(
                f"{name!r} is {num_shards} shards of {tuple(shards[0].shape)} here "
                f"but {other_num_shards} of {tuple(other_shards[0].shape)} on the "
                "other side: the two do not shard this layer the same way."
            )
        # The position of every shard of each side: [rank, shard of the rank].
        ours = np.arange(num_shards).reshape(tp_size, len(shards))
        theirs = np.arange(other_num_shards).reshape(other_tp_size, len(other_shards))
        # Count the receiver's shards the way the sender counts its own.
        if receiving:
            ours = ours * other_num_shards // num_shards
        else:
            theirs = theirs * num_shards // other_num_shards
        for other_rank in range(other_tp_size):
            _, mine, others = np.intersect1d(
                ours[tp_rank], theirs[other_rank], return_indices=True
            )
            for i, j in zip(mine, others):
                pairs[other_rank].append((shards[i], other_shards[j]))
    return pairs


def expand_block_ids(block_ids: Sequence[int] | np.ndarray, num: int) -> np.ndarray:
    """The ids of the `num` smaller blocks each block spans, e.g. `[1, 3]`
    at 2 to a block is `[2, 3, 6, 7]`."""
    ids = np.asarray(block_ids, dtype=np.int64)
    return (ids[:, None] * num + np.arange(num)).reshape(-1)


def transfer_block_positions(
    num_transfer_blocks: int,
    transfer_blocks_per_block: int,
    *,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    num_computed_blocks: int = 0,
) -> np.ndarray:
    """Where in a request the transfer blocks of a rank sit, in the order of
    its block ids.

    A DCP rank holds every `dcp_size`-th block of a request from its
    `dcp_rank`-th, every block without DCP; its ids start after the
    `num_computed_blocks` of them already in its prefix cache. A block is
    `transfer_blocks_per_block` transfer blocks. With DCP 2 and two transfer
    blocks to a block, rank 1 holds:

        block of the request    |  0  |  1  |  2  |  3  |  4  |  5  |
        block of rank 1               |  0  |     |  1  |     |  2  |
        position                      | 2 3 |     | 6 7 |     |10 11|
    """
    num_blocks = cdiv(num_transfer_blocks, transfer_blocks_per_block)
    block = dcp_rank + (num_computed_blocks + np.arange(num_blocks)) * dcp_size
    return expand_block_ids(block, transfer_blocks_per_block)[:num_transfer_blocks]


def contiguous_byte_ranges(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[int, int, int]]:
    """Split pairs of same-shape views into byte ranges contiguous in both.

    `pairs` are the views `pair_shards` pairs up from each side's
    `KVCacheSpec.slice_for_tp_transfer`, which may differ in physical layout.
    Returns `(src_offset, dst_offset, num_bytes)` per range, with offsets in
    bytes from the start of each view's storage.
    """
    ranges = []
    for src, dst in pairs:
        if src.shape != dst.shape or src.element_size() != dst.element_size():
            raise ValueError(
                f"Cannot pair {tuple(src.shape)} x {src.element_size()}B with "
                f"{tuple(dst.shape)} x {dst.element_size()}B."
            )
        if src.numel() == 0:
            continue
        # Coalesce the pair: the fewest dims, outermost first, that address
        # the same bytes of both views.
        dims: list[tuple[int, int, int]] = []
        for s, d, size in sorted(
            zip(src.stride(), dst.stride(), src.shape), reverse=True
        ):
            if size == 1:
                continue
            if dims and dims[-1][:2] == (s * size, d * size):
                size *= dims.pop()[2]
            dims.append((s, d, size))
        contiguous = dims.pop()[2] if dims and dims[-1][:2] == (1, 1) else 1

        src_offsets = torch.tensor(
            [src.storage_offset()], dtype=torch.int64, device="cpu"
        )
        dst_offsets = torch.tensor(
            [dst.storage_offset()], dtype=torch.int64, device="cpu"
        )
        for s, d, size in dims:
            idx = torch.arange(size, dtype=torch.int64, device="cpu")
            src_offsets = (src_offsets[:, None] + idx * s).flatten()
            dst_offsets = (dst_offsets[:, None] + idx * d).flatten()
        elem = src.element_size()
        ranges += [
            (s * elem, d * elem, contiguous * elem)
            for s, d in zip(src_offsets.tolist(), dst_offsets.tolist())
        ]
    return ranges


def coalesce_byte_ranges(ranges: Iterable[tuple]) -> list[tuple]:
    """Merge each `(offset, other_offset, num_bytes, *rest)` range into the
    one before it when it starts where that one ends on both sides. The
    merged range keeps the `rest` of its first."""
    merged: list[tuple] = []
    for src, dst, size, *rest in ranges:
        if merged:
            last_src, last_dst, last_size, *last_rest = merged[-1]
            if (last_src + last_size, last_dst + last_size) == (src, dst):
                merged[-1] = (last_src, last_dst, last_size + size, *last_rest)
                continue
        merged.append((src, dst, size, *rest))
    return merged


# ---- Per-engine transfer info ----


@dataclass(frozen=True, slots=True)
class EngineTransferInfo:
    """Common per-remote-engine transfer state, computed at handshake.

    Stored per `(engine_id, pp_rank)` inside `TransferTopology._engines`.
    """

    remote_tp_size: int

    remote_block_len: int
    """Block length (bytes)"""

    remote_block_size: int
    """Tokens per block."""

    remote_physical_blocks_per_logical: int
    """Physical blocks per logical block."""

    remote_pp_rank: int = 0
    """Remote producer PP rank for this engine."""

    start_layer: int = 0
    """Global index of the first layer owned by this PP rank."""

    end_layer: int = 0
    """Exclusive global index after the last layer owned by this PP rank."""

    remote_dcp_size: int = 1
    """Remote decode context parallel size."""


# ---- Transfer topology ----


@dataclass
class TransferTopology:
    """Single source of truth for local TP identity and per-engine remote info."""

    tp_rank: int
    tp_size: int
    block_size: int
    engine_id: EngineId
    is_mla: bool
    is_mamba: bool
    total_num_kv_heads: int
    attn_backends: list[type[AttentionBackend]]
    dcp_size: int = 1
    tensor_shape: torch.Size | None = None

    def __post_init__(self):
        self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)

        self._engines: dict[tuple[EngineId, int], EngineTransferInfo] = {}

    # ============================================================
    # Engine registration
    # ============================================================

    def register_remote_engine(
        self,
        remote_engine_id: EngineId,
        info: EngineTransferInfo,
    ) -> EngineTransferInfo:
        """Register a remote engine, unifying worker dicts state.

        The caller (worker) is responsible for computing the info via
        the transfer policy.  This method only stores and deduplicates.
        """
        assert remote_engine_id != self.engine_id, (
            f"Cannot register local engine {self.engine_id} as remote. "
            f"Local identity is set via __init__ params."
        )
        engine_key = (remote_engine_id, info.remote_pp_rank)
        if engine_key in self._engines:
            return self._engines[engine_key]
        self._engines[engine_key] = info
        return info

    def get_engine_info(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> EngineTransferInfo:
        return self._engines[(remote_engine_id, remote_pp_rank)]

    def unregister_remote_engine(self, remote_engine_id: EngineId) -> None:
        # Remove all pp_rank entries for the remote engine.
        for key in [k for k in self._engines if k[0] == remote_engine_id]:
            del self._engines[key]

    @property
    def dcp_rank(self) -> int:
        """This rank's DCP coverage rank.

        with `dcp_size in (1, tp_size)` enforced at the connector boundary, a
        rank's DCP identity is always exactly `tp_rank % dcp_size`.
        """
        return self.tp_rank % self.dcp_size

    # ============================================================
    # Common methods
    # ============================================================

    def tp_ratio(self, remote_tp_size: int) -> int:
        """Calculate the tensor parallel ratio between local and remote TP.

        Positive when local_tp >= remote_tp (local workers read from the
        same remote worker in groups of size `tp_ratio`).  Negative when
        remote_tp > local_tp (ratio is flipped).
        """
        if self.tp_size >= remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size
        assert remote_tp_size % self.tp_size == 0, (
            f"Remote tensor parallel size {remote_tp_size} is not divisible "
            f"by local tensor parallel size {self.tp_size}."
        )
        return -(remote_tp_size // self.tp_size)

    def block_size_ratio(self, remote_block_size: int) -> int:
        """Calculate the block size ratio between local and remote."""
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    def is_kv_replicated(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> bool:
        """Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        """
        return (
            self._engines[(remote_engine_id, remote_pp_rank)].remote_tp_size
            > self.total_num_kv_heads
        )

    def replicates_kv_cache(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id, remote_pp_rank)

    @property
    def local_replicates_kv_cache(self) -> bool:
        """Whether the local engine's KV cache is replicated."""
        return self.is_mla or self.tp_size > self.total_num_kv_heads

    def handshake_target_ranks(
        self, remote_tp_size: int, remote_dcp_size: int = 1
    ) -> list[int]:
        """The remote TP ranks this rank handshakes with: those holding some
        of what it holds. Pure math on the TP and DCP sizes of the two
        sides; the remote engine need not be registered yet.

        DCP: a DCP rank holds every `dcp_size`-th block of a request from
        its `dcp_rank`-th, so two ranks hold blocks in common when their
        DCP ranks agree modulo `shared`, the greatest common divisor of the
        DCP sizes.

        TP: the ranks of a DCP group hold the same shards of a layer, so
        along what TP shards the remote counts as its DCP groups (as its
        ranks, without DCP). Our ranks holding blocks in common with a
        remote rank are every `shared`-th of ours, and each talks to the
        groups at its place among those, so that every remote rank hears
        from one. With two such ranks and four remote groups:

            every `shared`-th of ours  | 0                 | 1                 |
            remote DCP groups          | group 0 | group 1 | group 2 | group 3 |
        """
        if remote_tp_size % remote_dcp_size:
            raise NotImplementedError(
                f"A remote of tensor parallel size {remote_tp_size} cannot have "
                f"DCP size {remote_dcp_size}."
            )
        num_dcp_groups = remote_tp_size // remote_dcp_size
        shared = math.gcd(self.dcp_size, remote_dcp_size)
        place, num_places = self.tp_rank // shared, self.tp_size // shared
        assert num_places % num_dcp_groups == 0 or num_dcp_groups % num_places == 0, (
            f"Local tensor parallel size {self.tp_size} and remote tensor "
            f"parallel size {remote_tp_size} are not divisible by one another."
        )
        dcp_groups = range(
            place * num_dcp_groups // num_places,
            cdiv((place + 1) * num_dcp_groups, num_places),
        )
        return [
            rank
            for dcp_group in dcp_groups
            for rank in range(
                dcp_group * remote_dcp_size, (dcp_group + 1) * remote_dcp_size
            )
            if self.shares_blocks(rank, remote_dcp_size)
        ]

    def shares_blocks(self, remote_rank: int, remote_dcp_size: int) -> bool:
        """Whether a remote rank holds blocks of a request that this rank
        holds too."""
        shared = math.gcd(self.dcp_size, remote_dcp_size)
        return (remote_rank % remote_dcp_size - self.dcp_rank) % shared == 0

    def dcp_consumer_count(self, remote_tp_size: int, remote_dcp_size: int) -> int:
        """How many local ranks (in aggregate) read from a given remote rank.

        Used by the producer side to know how many reader notifications to
        wait for before freeing a request's blocks.
        """
        remote_rank = self.handshake_target_ranks(remote_tp_size, remote_dcp_size)[0]
        return sum(
            remote_rank
            in replace(self, tp_rank=tp_rank).handshake_target_ranks(
                remote_tp_size, remote_dcp_size
            )
            for tp_rank in range(self.tp_size)
        )

    def target_remote_ranks(
        self, remote_engine_id: EngineId, remote_pp_rank: int = 0
    ) -> list[int]:
        """Get the remote TP rank(s) that the current local TP rank will
        read from.  When remote tp_size > local tp_size, reads from
        multiple remote ranks.
        """
        info = self._engines[(remote_engine_id, remote_pp_rank)]
        return self.handshake_target_ranks(info.remote_tp_size)

    def describe(self, remote_engine_id: EngineId, remote_pp_rank: int = 0) -> str:
        """One-line summary of transfer config for logging."""
        info = self._engines[(remote_engine_id, remote_pp_rank)]
        return (
            f"TransferTopology("
            f"tp_ratio={self.tp_ratio(info.remote_tp_size)}, "
            f"num_kv_heads={self.total_num_kv_heads if not self.is_mla else 1}, "
            f"local_tp={self.tp_size}, "
            f"remote_tp={info.remote_tp_size}, "
            f"remote_pp={remote_pp_rank}, "
            f"local_dcp={self.dcp_size}, "
            f"remote_dcp={info.remote_dcp_size}, "
            f"local_rank={self.tp_rank}, "
            f"remote_block_len={info.remote_block_len})"
        )


class CircuitBreaker:
    """Opens for `cooldown_s` after `failure_threshold` consecutive
    failures; callers skip requests while it is open. After the cooldown,
    `allow()` admits a single probe and holds others for `probe_window_s`:
    a failed probe reopens the breaker, a success closes it, and a probe that
    never reports back lets the next request probe once the window passes."""

    def __init__(
        self,
        name: str,
        cooldown_s: float,
        failure_threshold: int,
        probe_window_s: float,
    ):
        assert cooldown_s > 0 and failure_threshold > 0 and probe_window_s > 0
        self.name = name
        self.cooldown_s = cooldown_s
        self.failure_threshold = failure_threshold
        self.probe_window_s = probe_window_s
        self._failures = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def allow(self) -> bool:
        now = time.monotonic()
        if now < self._open_until:
            return False
        if self._failures >= self.failure_threshold:
            self._open_until = now + self.probe_window_s
        return True

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.cooldown_s
            logger.warning(
                "%s circuit breaker opened after %d consecutive failures; "
                "skipping requests for %s seconds",
                self.name,
                self._failures,
                self.cooldown_s,
            )

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0
