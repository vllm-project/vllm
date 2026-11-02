# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper functions for weight shard seeding: the storage manifest wire
format, the fingerprint comparison, and the launcher's address/placement
logic, shared by the source side (:mod:`.seed`), the movers, the peer
loader (:mod:`.loader`), and ``vllm preload``."""

import socket
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from vllm.config import replace
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorKind,
    WeightCacheKey,
    json_to_dataclass,
    parse_seed_address,
    recv_msg,
    send_msg,
    verify_socket_owner,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

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
    """Describe ``tensors`` as views over their unique storages.

    Returns:
        The JSON-safe manifest a mirror rebuilds the tensors from, and the
        storages it indexes, in order, for the mover to publish.

    """
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
    """Recreate every manifest tensor on the mirror's copies of the storages.

    ``set_`` silently grows a storage too small for the view, which would
    move the buffer away from the other views on it, so a manifest from a
    remote peer is checked against the buffers first.
    """
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
    """Fields on which a seed source's shard differs from this daemon's.

    A mirror on a host without the checkpoint's safetensors cannot hash
    them, so its fingerprint falls back to the model path. It then matches
    the source by that path, as an engine without the files matches its
    daemon, rather than refusing every source that could hash its files.
    """
    mismatched = local.mismatched_fields(source)
    if "checkpoint" in mismatched and local.checkpoint == model == source_model:
        mismatched.remove("checkpoint")
    return mismatched


def split_addresses(value: str | None) -> list[str]:
    return [address.strip() for address in (value or "").split(",") if address.strip()]


def seed_address(
    value: str | None, local_rank: int, global_rank: int, port_offset: int = 0
) -> str | tuple[str, int] | None:
    """Resolve one daemon's seed source address.

    One address per local GPU, in ``local_rank`` order, names each source's
    Unix socket, which is the same-host form. A single ``host:base_port``
    instead makes every rank dial ``base_port + port_offset + its global
    rank``, matching how ``--seed-listen-port`` lays a source's listeners
    out, so one flag covers a whole cross-host replica and both daemon
    groups.
    """
    addresses = split_addresses(value)
    if not addresses:
        return None
    if len(addresses) > 1:
        if local_rank >= len(addresses):
            raise ValueError(
                f"--weight-cache-seed has {len(addresses)} addresses but local "
                f"GPU {local_rank} needs one; pass one address per local GPU"
            )
        address = parse_seed_address(addresses[local_rank])
        if not isinstance(address, str):
            raise ValueError(
                "Per-GPU --weight-cache-seed addresses must be Unix socket "
                "paths; a remote source takes a single host:base_port"
            )
        return address
    address = parse_seed_address(addresses[0])
    if isinstance(address, str):
        return address
    host, port = address
    return host, port + port_offset + global_rank


def peer_cache_config(config: "VllmConfig", *, is_draft: bool) -> "VllmConfig":
    """Switch one daemon's config to loading from a source replica.

    The daemon stays a normal weight cache daemon; only its loader changes
    to ``peer_cache``, which pulls the post-processed weights from the
    source daemon instead of reading the checkpoint. The source address and
    token ride in the daemon's own ``SeedConfig`` (used by the warmup-cache
    adoption too), and the daemon injects its D2D endpoint into the loader.
    """
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
    """Local GPU indices for a same-host mirror's daemons, by free index.

    A same-host mirror keeps every GPU visible (a CUDA IPC handle names its
    device by index) and places its ranks on the GPUs the source replica
    does not use. The source's GPUs are identified by UUID through its
    sockets, so the operator does not place the mirror by hand, and an
    arbitrary source placement (not just a leading block) works.
    """
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
