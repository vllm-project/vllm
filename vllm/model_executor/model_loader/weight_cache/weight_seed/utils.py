# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for weight shard seeding (manifest format, fingerprint, launcher logic)."""

import socket
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from vllm.config import replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorKind,
    WeightCacheKey,
    json_to_dataclass,
    parse_seed_address,
    recv_msg,
    send_msg,
    verify_socket_owner,
)
from vllm.model_executor.model_loader.weight_cache.transport import (
    SeedConfig,
    SeedTransport,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_CONNECT_TIMEOUT_S = 5.0


class WeightsSeedPayload(TypedDict):
    """What a source daemon's ``weights`` resolve sends to a peer loader."""

    cache_config: dict[str, Any]
    """``dataclass_to_json(WeightCacheKey)`` of the source's shard."""
    model: str
    manifest: dict[str, Any]
    """Storage manifest, as :func:`build_manifest` produces."""
    aliases: dict[str, str]
    attrs: dict[str, bool]
    seed: dict[str, Any]
    """Mover-specific transfer seed; ``seed["backend"]`` picks the mover."""


def build_manifest(
    tensors: Mapping[str, torch.Tensor], kinds: Mapping[str, TensorKind]
) -> tuple[dict[str, Any], list[torch.UntypedStorage]]:
    """Describe ``tensors`` as views over their unique storages."""
    storages: list[torch.UntypedStorage] = []
    index_by_storage: dict[tuple[torch.device, int], int] = {}
    views: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        index = index_by_storage.setdefault(
            (tensor.device, storage.data_ptr()), len(storages)
        )
        if index == len(storages):
            storages.append(storage)
        views[name] = {
            "storage": index,
            "offset": tensor.storage_offset(),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "is_param": kinds[name] == "param",
        }
    manifest = {
        "storages": [
            {"nbytes": storage.nbytes(), "device": storage.device.type}
            for storage in storages
        ],
        "tensors": views,
    }
    return manifest, storages


def manifest_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"Weight-cache seed manifest uses unsupported dtype {dtype_name!r}"
        )
    return dtype


def manifest_nbytes(manifest: Mapping[str, Any]) -> int:
    return sum(storage["nbytes"] for storage in manifest["storages"])


def _view_extent(view: Mapping[str, Any], itemsize: int) -> int:
    """Bytes from the start of its storage that a strided view reaches."""
    if 0 in view["shape"]:
        return 0
    last = view["offset"] + sum(
        (size - 1) * stride
        for size, stride in zip(view["shape"], view["stride"], strict=True)
    )
    return (last + 1) * itemsize


def rebuild_tensors(
    manifest: Mapping[str, Any], buffers: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Recreate every manifest tensor on the mirror's copies; bounds-checked first."""
    tensors: dict[str, torch.Tensor] = {}
    for name, view in manifest["tensors"].items():
        index = view["storage"]
        if not isinstance(index, int) or not 0 <= index < len(buffers):
            raise RuntimeError(f"Seed manifest entry {name!r} names no storage")
        buffer = buffers[index]
        dtype = manifest_dtype(view["dtype"])
        if _view_extent(view, dtype.itemsize) > buffer.nbytes:
            raise RuntimeError(f"Seed manifest entry {name!r} overruns its storage")
        tensors[name] = torch.empty(0, dtype=dtype, device=buffer.device).set_(
            buffer.untyped_storage(), view["offset"], view["shape"], view["stride"]
        )
    return tensors


def allocate_storage(storage: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """A fresh byte buffer for one manifest storage, on the source's device type."""
    target = torch.device("cpu") if storage["device"] == "cpu" else device
    return torch.empty(storage["nbytes"], dtype=torch.uint8, device=target)


def storage_bytes(storage: torch.UntypedStorage) -> torch.Tensor:
    """A flat byte tensor covering the whole storage."""
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def seed_mismatches(
    local: WeightCacheKey,
    source: WeightCacheKey,
    model: str,
    source_model: str | None,
) -> list[str]:
    """Fields on which a seed source's shard differs (model-path fallback)."""
    mismatched = local.mismatched_fields(source)
    if "checkpoint" in mismatched and local.checkpoint == model == source_model:
        mismatched.remove("checkpoint")
    return mismatched


def split_addresses(value: str | None) -> list[str]:
    return [address.strip() for address in (value or "").split(",") if address.strip()]


def seed_address(
    value: str | None,
    local_rank: int,
    global_rank: int,
    port_offset: int = 0,
    world_size: int | None = None,
) -> str | tuple[str, int] | None:
    """Resolve one daemon's seed source address.

    A comma-separated list of Unix socket paths names one per local GPU
    (same-host peer_ipc copies). One or more ``host:base_port`` entries name
    a remote replica's nodes in node order, with one entry per source node:
    global rank ``g`` dials entry ``g // (world_size // N)`` at
    ``port + port_offset + g``.
    """
    parsed = [parse_seed_address(address) for address in split_addresses(value)]
    if not parsed:
        return None
    socket_paths = [address for address in parsed if isinstance(address, str)]
    remote_hosts = [address for address in parsed if isinstance(address, tuple)]
    if socket_paths and remote_hosts:
        raise ValueError(
            "--weight-cache-seed cannot mix Unix socket paths with host:port "
            "entries; a same-host mirror takes socket paths only, and a "
            "remote source takes one host:base_port per source node"
        )
    if socket_paths:
        if len(socket_paths) == 1:
            return socket_paths[0]
        if local_rank >= len(socket_paths):
            raise ValueError(
                f"--weight-cache-seed has {len(socket_paths)} addresses but "
                f"local GPU {local_rank} needs one; pass one address per "
                "local GPU"
            )
        return socket_paths[local_rank]
    if len(remote_hosts) > 1:
        if world_size is None or world_size % len(remote_hosts) != 0:
            raise ValueError(
                "Multi-host --weight-cache-seed passes one host:base_port "
                f"per source node, so its count ({len(remote_hosts)}) must "
                f"divide the world size ({world_size})"
            )
        ranks_per_node = world_size // len(remote_hosts)
        host, port = remote_hosts[global_rank // ranks_per_node]
    else:
        host, port = remote_hosts[0]
    return host, port + port_offset + global_rank


def _request_topology(
    address: tuple[str, int], token: str, *, hint: str = ""
) -> dict[str, Any] | None:
    """One ``topology`` request; None only when the source predates the kind."""
    transport = SeedTransport(seed_config=SeedConfig(source=address, token=token))
    response = transport.request({"kind": "topology"})
    host, port = address
    if response is None:
        raise RuntimeError(
            f"Cannot reach the seed source at {host}:{port}: is the source "
            "replica still serving, and was it launched with "
            f"--seed-listen-host/--seed-listen-port?{hint}"
        )
    status = response.get("status")
    if status == "ok":
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Seed source at {host}:{port} returned a malformed topology payload"
            )
        return payload
    message = str(response.get("message", response))
    if "Unauthorized" in message:
        raise RuntimeError(
            f"Seed source at {host}:{port} rejected the topology request "
            f"({message}); check that --seed-token matches the source "
            "replica's token"
        )
    if status == "error" and "unsupported seed kind" in message:
        return None
    raise RuntimeError(
        f"Seed source at {host}:{port} refused the topology request "
        f"(status {status}): {message}"
    )


def _parse_topology_ranks(
    payload: dict[str, Any],
    world_size: int,
    *,
    is_draft: bool,
    address: tuple[str, int],
) -> dict[int, tuple[str, int]]:
    """Validate one topology payload into a ``{rank: (host, port)}`` map."""
    host, port = address
    label = "draft " if is_draft else ""
    source_world_size = payload.get("world_size")
    if source_world_size != world_size:
        raise RuntimeError(
            f"Seed source at {host}:{port} reports a {label}world size of "
            f"{source_world_size}, but this replica's world size is "
            f"{world_size}; the source and mirror replicas must use the "
            "same DP x PP x TP layout"
        )
    if payload.get("is_draft") is not is_draft:
        raise RuntimeError(
            f"Seed source at {host}:{port} reported is_draft="
            f"{payload.get('is_draft')} where is_draft={is_draft} was "
            "expected; the source and mirror replicas must run the same "
            "speculative draft configuration"
        )
    ranks = payload.get("ranks")
    if not isinstance(ranks, dict):
        raise RuntimeError(
            f"Seed source at {host}:{port} returned a topology payload "
            "without a ranks map"
        )
    resolved: dict[int, tuple[str, int]] = {}
    for key, entry in ranks.items():
        if (
            not isinstance(entry, list | tuple)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], int)
        ):
            raise RuntimeError(
                f"Seed source at {host}:{port} returned a malformed "
                f"topology entry for rank {key!r}: {entry!r}"
            )
        try:
            rank = int(key)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Seed source at {host}:{port} returned a topology entry "
                f"with a non-integer rank {key!r}"
            ) from None
        resolved[rank] = (entry[0], entry[1])
    missing = [rank for rank in range(world_size) if rank not in resolved]
    if missing:
        raise RuntimeError(
            f"Seed source at {host}:{port} reported no address for "
            f"{label}rank(s) {missing}; the source replica looks partially "
            "started, retry once every source daemon is serving"
        )
    return resolved


def discover_seed_topology(
    address: tuple[str, int],
    token: str,
    world_size: int,
    need_draft: bool,
) -> dict[int, dict[int, tuple[str, int]]] | None:
    """Discover a source replica's per-rank daemon addresses.

    Queries the ``topology`` seedable of the source at ``address`` (the
    base ``host:base_port`` of a single-address ``--weight-cache-seed``)
    and returns ``{group_index: {rank: (host, port)}}``: group 0 is the
    target group, group 1 the speculative draft group when ``need_draft``.
    Returns None when the source predates the ``topology`` kind, so the
    caller can fall back to the single-host assumption of
    :func:`seed_address`.

    Raises:
        RuntimeError: If the source is unreachable, rejects the token, or
            reports a topology inconsistent with this replica.

    """
    payload = _request_topology(address, token)
    if payload is None:
        logger.info(
            "Seed source at %s:%d predates topology discovery; assuming "
            "every source rank lives at that host:base_port",
            *address,
        )
        return None
    maps: dict[int, dict[int, tuple[str, int]]] = {
        0: _parse_topology_ranks(payload, world_size, is_draft=False, address=address)
    }
    if need_draft:
        rank0_host, rank0_port = maps[0][0]
        draft_address = (rank0_host, rank0_port + world_size)
        draft_payload = _request_topology(draft_address, token)
        if draft_payload is None:
            raise RuntimeError(
                "This mirror runs a speculative draft daemon group, but the "
                f"seed source has no draft listener at {draft_address[0]}:"
                f"{draft_address[1]}; the source replica predates topology "
                "discovery or was launched without draft daemons"
            )
        maps[1] = _parse_topology_ranks(
            draft_payload, world_size, is_draft=True, address=draft_address
        )
    logger.info(
        "Discovered seed source topology: %d rank(s)%s",
        world_size,
        ", plus a draft daemon group" if need_draft else "",
    )
    return maps


def preflight_seed_sources(
    maps: dict[int, dict[int, tuple[str, int]]],
    token: str,
    world_size: int,
) -> None:
    """Probe every resolved source daemon before any mirror daemon spawns.

    Sends one ``topology`` request to each address in ``maps`` (as returned
    by :func:`discover_seed_topology`) so a missing or misconfigured source
    daemon fails the launch up front instead of mid-seed.

    Raises:
        RuntimeError: If any source daemon is unreachable, rejects the
            token, predates the ``topology`` kind, or reports a world size
            or role inconsistent with this replica.

    """
    for group_index, ranks in maps.items():
        expect_draft = group_index == 1
        for rank, address in sorted(ranks.items()):
            hint = f" (preflighting group {group_index} rank {rank})"
            payload = _request_topology(address, token, hint=hint)
            host, port = address
            if payload is None:
                raise RuntimeError(
                    f"Seed source rank {rank} (group {group_index}) at "
                    f"{host}:{port} predates topology discovery; every "
                    "source daemon must run a topology-capable version "
                    "when the mirror discovers the source topology"
                )
            source_world_size = payload.get("world_size")
            if source_world_size != world_size:
                raise RuntimeError(
                    f"Seed source rank {rank} (group {group_index}) at "
                    f"{host}:{port} reports a world size of "
                    f"{source_world_size}, but this replica's world size "
                    f"is {world_size}; the source and mirror replicas must "
                    "use the same DP x PP x TP layout"
                )
            if payload.get("is_draft") is not expect_draft:
                raise RuntimeError(
                    f"Seed source rank {rank} (group {group_index}) at "
                    f"{host}:{port} reported is_draft="
                    f"{payload.get('is_draft')} where is_draft="
                    f"{expect_draft} was expected; the source and mirror "
                    "replicas must run the same speculative draft "
                    "configuration"
                )
    total = sum(len(ranks) for ranks in maps.values())
    logger.info(
        "Preflighted %d source daemon(s) across %d group(s)",
        total,
        len(maps),
    )


def peer_cache_config(config: "VllmConfig", *, is_draft: bool) -> "VllmConfig":
    """Switch one daemon's config to the ``peer_cache`` load format."""
    extra = dict(config.load_config.model_loader_extra_config or {})
    extra["is_draft"] = is_draft
    return replace(
        config,
        load_config=replace(
            config.load_config,
            load_format="peer_cache",
            model_loader_extra_config=extra,
        ),
    )


def mirror_device_indices(seed_value: str, num_local: int) -> list[int]:
    """A same-host mirror's GPU indices: the free ones past the source's."""
    source_uuids = set()
    for address in split_addresses(seed_value):
        # The launcher only calls this for the all-Unix-socket form.
        path = parse_seed_address(address)
        assert isinstance(path, str)
        verify_socket_owner(path, strict_perms=False)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_CONNECT_TIMEOUT_S)
            sock.connect(path)
            send_msg(sock, {"cmd": "get_memory"})
            response = recv_msg(sock)
        gpu_uuid = response.get("gpu_uuid") if isinstance(response, dict) else None
        if not isinstance(gpu_uuid, str):
            raise ValueError(
                f"Source daemon at {path} did not report its GPU UUID; "
                "is it an up-to-date weight cache daemon?"
            )
        source_uuids.add(gpu_uuid)
    device_count = torch.accelerator.device_count()
    occupied = {
        index
        for index in range(device_count)
        if current_platform.get_device_uuid(index) in source_uuids
    }
    if len(occupied) != len(source_uuids):
        raise ValueError(
            "A --weight-cache-seed source socket belongs to a GPU that is "
            "not visible here; a same-host mirror must see every GPU"
        )
    free = [index for index in range(device_count) if index not in occupied]
    if len(free) < num_local:
        raise ValueError(
            f"Only {len(free)} GPU(s) are free of the source replica, but "
            f"{num_local} local daemon(s) each need one"
        )
    return free


def validate_weights_payload(
    payload: Any, expected_key: WeightCacheKey, model: str
) -> WeightsSeedPayload:
    """Structural checks plus the shard fingerprint against the expected key."""
    if not isinstance(payload, dict):
        raise ValueError("Seed daemon returned a malformed payload")
    source_model = payload.get("model")
    if source_model is not None and not isinstance(source_model, str):
        raise ValueError("Seed daemon returned a malformed model name")
    manifest = payload.get("manifest")
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("storages"), list)
        or not isinstance(manifest.get("tensors"), dict)
    ):
        raise ValueError("Seed daemon returned a malformed manifest")
    seed = payload.get("seed")
    if not isinstance(seed, dict) or not isinstance(seed.get("backend"), str):
        raise ValueError("Seed daemon returned a malformed transfer seed")
    for key in ("aliases", "attrs"):
        if payload.get(key) is not None and not isinstance(payload[key], dict):
            raise ValueError(f"Seed daemon returned malformed {key}")
    source_key = json_to_dataclass(WeightCacheKey, payload.get("cache_config"))
    mismatched = seed_mismatches(expected_key, source_key, model, source_model)
    if mismatched:
        hint = (
            f"; no safetensors under {model} to fingerprint here, so the "
            "mirror needs the source's --model path or the checkpoint files"
            if mismatched == ["checkpoint"] and expected_key.checkpoint == model
            else ""
        )
        raise ValueError(
            f"Seed daemon cache differs on fields {mismatched}; refusing "
            f"to mirror a different weight shard{hint}"
        )
    return cast(WeightsSeedPayload, payload)
