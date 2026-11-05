# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Publish an immutable Hugging Face checkpoint through vLLM's native NCCL WTE.

This module intentionally has no trainer, Ray, VIME, Megatron, or Transformers
dependency.  It reads safetensors in checkpoint-name order, stages one tensor at
a time on the publisher GPU, and delegates the complete transfer transaction to
vLLM's stateful trainer-side engine.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import cached_property
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import regex as re
import requests
import torch
from safetensors import safe_open
from update_timing import summarize_update_timing

from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ParamMeta,
    WeightSource,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    packed_nccl_broadcast_producer,
)
from vllm.utils.network_utils import get_ip, get_open_port

_SAFETENSORS_DTYPE_TO_TORCH = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "U16": (torch.uint16, 2),
    "I32": (torch.int32, 4),
    "U32": (torch.uint32, 4),
    "I64": (torch.int64, 8),
    "U64": (torch.uint64, 8),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "F8_E8M0": (torch.float8_e8m0fnu, 1),
}

ExpertTensorOrder = Literal["natural", "lexical"]


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    file: str
    shape: list[int]
    safetensors_dtype: str
    nbytes: int
    # Absolute byte offsets in the safetensors file. These are internal source
    # metadata used only by the direct file-backed expert transfer path.
    file_data_offsets: tuple[int, int] | None = None


@dataclass(frozen=True)
class WeightTransferBucket:
    """One Vime-style metadata RPC and NCCL broadcast batch."""

    phase: str
    metadata: tuple[ParamMeta, ...]
    total_bytes: int


@dataclass(frozen=True)
class CheckpointManifest:
    model: str
    revision: str
    checkpoint_path: str
    tensors: tuple[TensorMetadata, ...]
    tensor_count: int
    total_bytes: int
    manifest_sha256: str
    expert_tensor_order: ExpertTensorOrder = "natural"

    @classmethod
    def load(
        cls,
        *,
        model: str,
        revision: str,
        checkpoint_path: str | None = None,
        expert_tensor_order: ExpertTensorOrder = "natural",
    ) -> CheckpointManifest:
        root = _resolve_checkpoint(model, revision, checkpoint_path)
        tensor_entries = _checkpoint_tensor_entries(root, expert_tensor_order)
        # Tensors a trainer does not sync (e.g. frozen embedding tables too
        # large for one packed chunk) stay at their cold-start values.
        exclude = os.environ.get("VERIFIER_EXCLUDE_REGEX")
        if exclude:
            tensor_entries = [
                (name, file_name)
                for name, file_name in tensor_entries
                if not re.search(exclude, name)
            ]
        tensors: list[TensorMetadata] = []
        with ExitStack() as stack:
            opened_files: dict[str, Any] = {}
            file_layouts: dict[str, dict[str, tuple[int, int]]] = {}
            for name, file_name in tensor_entries:
                if file_name not in opened_files:
                    opened_files[file_name] = stack.enter_context(
                        safe_open(
                            root / file_name,
                            framework="pt",
                            device="cpu",
                        )
                    )
                    file_layouts[file_name] = _read_safetensors_layout(root / file_name)
                tensor_slice = opened_files[file_name].get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                try:
                    torch_dtype, itemsize = _SAFETENSORS_DTYPE_TO_TORCH[dtype_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported safetensors dtype {dtype_name!r} for {name}"
                    ) from exc
                shape = list(tensor_slice.get_shape())
                numel = 1
                for dimension in shape:
                    numel *= dimension
                tensors.append(
                    TensorMetadata(
                        name=name,
                        file=file_name,
                        shape=shape,
                        safetensors_dtype=dtype_name,
                        nbytes=numel * itemsize,
                        file_data_offsets=file_layouts[file_name].get(name),
                    )
                )

        manifest_sha256 = _manifest_sha256(tensors)
        return cls(
            model=model,
            revision=revision,
            checkpoint_path=str(root),
            tensors=tuple(tensors),
            tensor_count=len(tensors),
            total_bytes=sum(tensor.nbytes for tensor in tensors),
            manifest_sha256=manifest_sha256,
            expert_tensor_order=expert_tensor_order,
        )

    def iter_cuda_tensors(
        self, device: torch.device
    ) -> Iterator[tuple[str, torch.Tensor]]:
        return _CheckpointTensorIterator(self, device)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "revision": self.revision,
            "checkpoint_path": self.checkpoint_path,
            "tensor_count": self.tensor_count,
            "total_bytes": self.total_bytes,
            "manifest_sha256": self.manifest_sha256,
            "expert_tensor_order": self.expert_tensor_order,
        }


class SafetensorsCheckpointSource(WeightSource):
    """Re-iterable, bounded-host-memory source for one immutable checkpoint."""

    def __init__(self, manifest: CheckpointManifest, device: torch.device) -> None:
        self.manifest = manifest
        self.device = device

    # vLLM's WeightSource requires a cached_property: engines read it repeatedly
    @cached_property
    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(
                name=tensor.name,
                dtype=_SAFETENSORS_DTYPE_TO_TORCH[tensor.safetensors_dtype][0],
                shape=tuple(tensor.shape),
            )
            for tensor in self.manifest.tensors
        ]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        return self.manifest.iter_cuda_tensors(self.device)

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self)


class _CheckpointTensorIterator(Iterator[tuple[str, torch.Tensor]]):
    """Stream checkpoint tensors while allowing bounded mmap lifetimes."""

    def __init__(self, manifest: CheckpointManifest, device: torch.device) -> None:
        self._root = Path(manifest.checkpoint_path)
        self._metadata = tuple(manifest.tensors)
        self._metadata_index = 0
        self._device = device
        self._stack = ExitStack()
        self._opened_files: dict[str, Any] = {}
        self._mapped_files: dict[str, tuple[Any, mmap.mmap]] = {}
        self._closed = False
        self._timing_enabled = os.environ.get("K3_WEIGHT_UPDATE_SOURCE_TIMING") == "1"
        self._timing_tensor_count = 0
        self._timing_total_bytes = 0
        self._timing_get_tensor_seconds = 0.0
        self._timing_to_device_seconds = 0.0
        self._timing_cpu_pack_seconds = 0.0
        self._timing_h2d_copy_count = 0
        self._timing_h2d_modes: set[str] = set()
        self._timing_files: set[str] = set()

    def __iter__(self) -> _CheckpointTensorIterator:
        return self

    def __next__(self) -> tuple[str, torch.Tensor]:
        if self._closed:
            raise StopIteration
        if self._metadata_index >= len(self._metadata):
            self.close_handles()
            raise StopIteration
        metadata = self._metadata[self._metadata_index]
        self._metadata_index += 1

        opened_file = self._opened_files.get(metadata.file)
        if opened_file is None:
            opened_file = self._stack.enter_context(
                safe_open(
                    self._root / metadata.file,
                    framework="pt",
                    device="cpu",
                )
            )
            self._opened_files[metadata.file] = opened_file
        if self._timing_enabled:
            get_started = time.perf_counter()
            cpu_tensor = opened_file.get_tensor(metadata.name)
            get_seconds = time.perf_counter() - get_started
            self._timing_tensor_count += 1
            self._timing_total_bytes += metadata.nbytes
            self._timing_get_tensor_seconds += get_seconds
            self._timing_files.add(metadata.file)
        else:
            cpu_tensor = opened_file.get_tensor(metadata.name)

        if self._device.type == "cpu":
            return metadata.name, cpu_tensor

        to_device_started = time.perf_counter()
        cuda_tensor = cpu_tensor.to(device=self._device)
        if self._timing_enabled:
            self._timing_to_device_seconds += time.perf_counter() - to_device_started
            self._timing_h2d_copy_count += 1
            self._timing_h2d_modes.add("per-tensor")
        del cpu_tensor
        return metadata.name, cuda_tensor

    def materialize_contiguous_expert(
        self,
        bucket: WeightTransferBucket,
        device: torch.device,
    ) -> list[tuple[str, torch.Tensor]] | None:
        """Map a physically contiguous expert layer and copy it once."""
        if device.type != "cuda" or bucket.phase != "expert":
            return None
        count = len(bucket.metadata)
        current = self._metadata[self._metadata_index : self._metadata_index + count]
        if len(current) != count or [item.name for item in current] != [
            item.name for item in bucket.metadata
        ]:
            raise ValueError("Direct expert source order does not match manifest")
        if not current or any(item.file_data_offsets is None for item in current):
            return None
        file_name = current[0].file
        if any(item.file != file_name for item in current):
            return None
        offsets = [item.file_data_offsets for item in current]
        assert all(offset is not None for offset in offsets)
        concrete_offsets = [
            (offset[0], offset[1]) for offset in offsets if offset is not None
        ]
        if any(
            concrete_offsets[index][1] != concrete_offsets[index + 1][0]
            for index in range(len(concrete_offsets) - 1)
        ):
            return None

        self._metadata_index += count
        _, mapped = self._mapped_file(file_name)
        first_offset = concrete_offsets[0][0]
        last_offset = concrete_offsets[-1][1]
        span = last_offset - first_offset
        cpu_span = torch.frombuffer(
            mapped,
            dtype=torch.uint8,
            count=span,
            offset=first_offset,
        )
        to_device_started = time.perf_counter()
        device_span = cpu_span.to(device=device)
        to_device_seconds = time.perf_counter() - to_device_started

        byte_views = torch.split(
            device_span,
            [end - start for start, end in concrete_offsets],
        )
        named_tensors: list[tuple[str, torch.Tensor]] = []
        for item, view in zip(current, byte_views):
            dtype = _SAFETENSORS_DTYPE_TO_TORCH[item.safetensors_dtype][0]
            named_tensors.append((item.name, view.view(dtype).view(item.shape)))
        self._record_direct_timing(
            file_name=file_name,
            tensor_count=count,
            total_bytes=span,
            to_device_seconds=to_device_seconds,
        )
        return named_tensors

    def _mapped_file(self, file_name: str) -> tuple[Any, mmap.mmap]:
        existing = self._mapped_files.get(file_name)
        if existing is not None:
            return existing
        file_handle = (self._root / file_name).open("rb")
        mapped = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_COPY)
        self._mapped_files[file_name] = (file_handle, mapped)
        return file_handle, mapped

    def _record_direct_timing(
        self,
        *,
        file_name: str,
        tensor_count: int,
        total_bytes: int,
        to_device_seconds: float,
    ) -> None:
        if not self._timing_enabled:
            return
        self._timing_tensor_count += tensor_count
        self._timing_total_bytes += total_bytes
        self._timing_to_device_seconds += to_device_seconds
        self._timing_h2d_copy_count += 1
        self._timing_h2d_modes.add("direct-file-expert")
        self._timing_files.add(file_name)

    def record_transfer_timing(
        self,
        *,
        to_device_seconds: float,
        copy_count: int,
        mode: str,
        cpu_pack_seconds: float = 0.0,
    ) -> None:
        """Record bucket-level transfer work performed outside the iterator."""
        if not self._timing_enabled:
            return
        self._timing_to_device_seconds += to_device_seconds
        self._timing_cpu_pack_seconds += cpu_pack_seconds
        self._timing_h2d_copy_count += copy_count
        self._timing_h2d_modes.add(mode)

    def drain_timing(self) -> dict[str, Any]:
        """Return and reset source timing accumulated since the last drain."""
        if not self._timing_enabled:
            return {}
        result = {
            "source_tensor_count": self._timing_tensor_count,
            "source_total_bytes": self._timing_total_bytes,
            "source_get_tensor_seconds": self._timing_get_tensor_seconds,
            "source_to_device_seconds": self._timing_to_device_seconds,
            "source_cpu_pack_seconds": self._timing_cpu_pack_seconds,
            "source_h2d_copy_count": self._timing_h2d_copy_count,
            "source_h2d_modes": sorted(self._timing_h2d_modes),
            "source_files": sorted(self._timing_files),
        }
        self._timing_tensor_count = 0
        self._timing_total_bytes = 0
        self._timing_get_tensor_seconds = 0.0
        self._timing_to_device_seconds = 0.0
        self._timing_cpu_pack_seconds = 0.0
        self._timing_h2d_copy_count = 0
        self._timing_h2d_modes.clear()
        self._timing_files.clear()
        return result

    def close_handles(self) -> None:
        """Release file-backed mappings without resetting iteration state."""
        if self._closed:
            return
        self._stack.close()
        self._stack = ExitStack()
        self._opened_files.clear()
        for file_handle, mapped in self._mapped_files.values():
            mapped.close()
            file_handle.close()
        self._mapped_files.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.close_handles()
        self._closed = True


def _manifest_sha256(tensors: list[TensorMetadata] | tuple[TensorMetadata, ...]) -> str:
    canonical = json.dumps(
        [
            {
                "name": tensor.name,
                "file": tensor.file,
                "shape": tensor.shape,
                "safetensors_dtype": tensor.safetensors_dtype,
                "nbytes": tensor.nbytes,
            }
            for tensor in tensors
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _read_safetensors_layout(
    path: Path,
) -> dict[str, tuple[int, int]]:
    """Read absolute payload offsets without materializing tensor data."""
    with path.open("rb") as file_handle:
        header_length_bytes = file_handle.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header = json.loads(file_handle.read(header_length))
    data_start = 8 + header_length
    return {
        name: (
            data_start + int(entry["data_offsets"][0]),
            data_start + int(entry["data_offsets"][1]),
        )
        for name, entry in header.items()
        if name != "__metadata__"
    }


UpdateTarget = Literal["main", "draft"]


class _TargetWeightSyncClient(HTTPVLLMWeightSyncClient):
    """Route an update transaction to the main or speculative draft model."""

    def __init__(self, base_url: str, *, timeout: float) -> None:
        super().__init__(base_url, timeout=timeout)
        self.target: UpdateTarget = "main"

    def set_target(self, target: UpdateTarget) -> None:
        if target not in {"main", "draft"}:
            raise ValueError(f"update target must be 'main' or 'draft', got {target!r}")
        self.target = target

    def start_weight_update(self) -> None:
        endpoint = (
            "start_draft_weight_update"
            if self.target == "draft"
            else "start_weight_update"
        )
        self._post(endpoint)


def _fail_fast_trainer_init(engine_cls: Any, *args: Any, **kwargs: Any) -> Any:
    """``trainer_init`` that exits instead of hanging when the trainer endpoint fails.

    The rendezvous joins the worker-init RPC thread before re-raising, and the
    workers wait in NCCL init for a trainer that is gone, so a trainer-side error
    would otherwise hang the publisher with the error never printed.
    """
    open_endpoint = engine_cls._open_trainer_endpoint

    def _open(self: Any, init_info: Any) -> None:
        try:
            open_endpoint(self, init_info)
        except BaseException:
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(3)

    engine_cls._open_trainer_endpoint = _open
    try:
        return engine_cls.trainer_init(*args, **kwargs)
    finally:
        engine_cls._open_trainer_endpoint = open_endpoint


class NcclCheckpointPublisher:
    """Own one Vime-compatible trainer-side NCCL group and ordered updates."""

    def __init__(
        self,
        *,
        base_url: str,
        manifest: CheckpointManifest,
        device: str,
        update_bucket_size_bytes: int = 512 * 1024**2,
        direct_file_expert_h2d: bool = False,
        timeout_seconds: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.manifest = manifest
        self.device = torch.device(device)
        self.update_bucket_size_bytes = update_bucket_size_bytes
        self.direct_file_expert_h2d = direct_file_expert_h2d
        self.timeout_seconds = timeout_seconds
        self.group = None
        self.client: _TargetWeightSyncClient | None = None
        self.update_version = 0
        self.weight_epoch = 0

    def initialize(self) -> dict[str, Any]:
        if self.group is not None:
            raise RuntimeError("NCCL publisher is already initialized")
        torch.accelerator.set_device_index(self.device)
        inference_world_size = self._get("/get_world_size")["world_size"]
        init_info = NCCLTrainerInitInfo(
            master_address=get_ip(),
            master_port=get_open_port(),
            world_size=inference_world_size + 1,
            rank=0,
            # VERIFIER_UNPACKED=1: one broadcast per tensor (the packed
            # consumer cannot view tensors at unaligned offsets)
            packed=os.environ.get("VERIFIER_UNPACKED") != "1",
        )
        self.client = _TargetWeightSyncClient(
            self.base_url,
            timeout=self.timeout_seconds,
        )
        trainer_engine = _fail_fast_trainer_init(
            NCCLTrainerWeightTransferEngine,
            init_info,
            client=self.client,
            source=SafetensorsCheckpointSource(self.manifest, self.device),
        )
        self.group = trainer_engine.model_update_group
        return {
            "backend": "nccl",
            "inference_world_size": inference_world_size,
            "trainer_rank": 0,
            "worker_rank_offset": 1,
            "rendezvous": {
                "master_address": init_info.master_address,
                "master_port": init_info.master_port,
                "world_size": init_info.world_size,
            },
            "trainer_transport": "packed_nccl_broadcast_producer",
            "transport_mode": "packed",
        }

    def publish_update(
        self,
        *,
        target: UpdateTarget = "main",
        manifest: CheckpointManifest | None = None,
    ) -> dict[str, Any]:
        """Transfer the current source and return transfer provenance.

        Lifecycle operations such as pause, sleep, cache invalidation, and
        resume belong to the caller. This method only performs the weight
        transfer transaction itself.
        """
        if self.group is None or self.client is None:
            raise RuntimeError("initialize() must complete before publish")
        if target == "draft" and self.weight_epoch == 0:
            raise RuntimeError("draft update requires a completed main update")
        transfer_manifest = manifest or self.manifest
        source_device = (
            torch.device("cpu") if self.direct_file_expert_h2d else self.device
        )
        source = SafetensorsCheckpointSource(
            transfer_manifest,
            source_device,
        )
        weight_epoch = self.weight_epoch + 1 if target == "main" else self.weight_epoch
        self.client.set_target(target)
        with torch.inference_mode():
            bucket_results, timing = self._send_weights(source)
        torch.accelerator.synchronize(self.device)
        self.update_version += 1
        self.weight_epoch = weight_epoch
        return {
            "update_version": self.update_version,
            "weight_epoch": weight_epoch,
            "target": target,
            "send_weights_completed": True,
            "update_bucket_size_bytes": self.update_bucket_size_bytes,
            "direct_file_expert_h2d": self.direct_file_expert_h2d,
            "transport_mode": "packed",
            "timing": timing,
            "buckets": bucket_results,
            **transfer_manifest.summary(),
        }

    def _send_weights(
        self,
        source: SafetensorsCheckpointSource,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Send one update using the required non-expert/expert order."""
        if self.group is None or self.client is None:
            raise RuntimeError("NCCL publisher has no initialized trainer group")
        group = self.group
        transfer_started = time.perf_counter()

        metadata_started = time.perf_counter()
        source_metadata = source.metadata
        metadata_seconds = time.perf_counter() - metadata_started
        expected_names = [item.name for item in source_metadata]
        duplicate_names = _duplicate_names(expected_names)
        if duplicate_names:
            raise ValueError(
                "Weight source metadata contains duplicate names: "
                f"{duplicate_names[:20]}"
            )
        bucket_build_started = time.perf_counter()
        buckets = _build_vime_transfer_buckets(
            source_metadata,
            self.update_bucket_size_bytes,
        )
        bucket_build_seconds = time.perf_counter() - bucket_build_started
        transfer_names = [item.name for bucket in buckets for item in bucket.metadata]
        if transfer_names != expected_names:
            raise ValueError(
                "Weight source metadata is not in Vime transfer order; expected "
                "non-experts first followed by complete expert-layer groups"
            )
        expected_total_bytes = sum(_param_nbytes(item) for item in source_metadata)
        source_iter = iter(source)
        bucket_results: list[dict[str, Any]] = []
        sent_names: list[str] = []
        sent_total_bytes = 0
        start_rpc_started = time.perf_counter()
        self.client.start_weight_update()
        start_rpc_seconds = time.perf_counter() - start_rpc_started
        update_failed = False
        try:
            for index, bucket in enumerate(buckets):
                bucket_started = time.perf_counter()
                named_tensors, materialize_seconds = _materialize_bucket_for_device(
                    source_iter,
                    bucket,
                    self.device,
                    direct_file_expert_h2d=self.direct_file_expert_h2d,
                )
                drain_timing = getattr(source_iter, "drain_timing", None)
                source_timing = drain_timing() if drain_timing is not None else {}
                update_info = NCCLWeightTransferUpdateInfo(
                    names=[name for name, _ in named_tensors],
                    dtype_names=[
                        str(tensor.dtype).removeprefix("torch.")
                        for _, tensor in named_tensors
                    ],
                    shapes=[list(tensor.shape) for _, tensor in named_tensors],
                )
                rpc_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.client.update_weights,
                        asdict(update_info),
                    )
                    if future.done():
                        future.result()
                    send_started = time.perf_counter()
                    if os.environ.get("VERIFIER_UNPACKED") == "1":
                        for _, tensor in named_tensors:
                            group.broadcast(
                                tensor.contiguous(),
                                src=0,
                                stream=torch.cuda.current_stream(),
                            )
                    else:
                        packed_nccl_broadcast_producer(
                            iterator=iter(named_tensors),
                            group=group,
                            src=0,
                            post_iter_func=lambda item: item[1],
                        )
                    # The RPC below only returns once the server has received
                    # this bucket, so waiting for the send here costs nothing.
                    torch.accelerator.synchronize(self.device)
                    sent = time.perf_counter()
                    future.result()
                    returned = time.perf_counter()
                rpc_seconds = returned - rpc_started
                bucket_seconds = time.perf_counter() - bucket_started

                bucket_results.append(
                    {
                        "index": index,
                        "phase": bucket.phase,
                        "tensor_count": len(named_tensors),
                        "total_bytes": bucket.total_bytes,
                        "materialize_seconds": materialize_seconds,
                        "rpc_and_nccl_seconds": rpc_seconds,
                        "nccl_send_seconds": sent - send_started,
                        "server_wait_seconds": returned - sent,
                        "total_seconds": bucket_seconds,
                        "throughput_gib_per_second": (
                            bucket.total_bytes / 1024**3 / bucket_seconds
                            if bucket_seconds > 0
                            else 0.0
                        ),
                        **source_timing,
                    }
                )
                sent_names.extend(name for name, _ in named_tensors)
                sent_total_bytes += bucket.total_bytes
                del named_tensors
                close_handles = getattr(source_iter, "close_handles", None)
                if close_handles is not None:
                    close_handles()

            try:
                extra_name, extra_tensor = next(source_iter)
            except StopIteration:
                pass
            else:
                del extra_tensor
                raise ValueError(
                    "Weight source emitted a tensor not declared in metadata: "
                    f"{extra_name!r}"
                )
            if sent_names != expected_names:
                raise ValueError(
                    "Weight transfer manifest mismatch before finish: "
                    f"expected={len(expected_names)} sent={len(sent_names)}"
                )
            if sent_total_bytes != expected_total_bytes:
                raise ValueError(
                    "Weight transfer byte count mismatch before finish: "
                    f"expected={expected_total_bytes} sent={sent_total_bytes}"
                )
        except BaseException:
            # vLLM resets its active-update flag when a receiver-side load
            # fails. Do not issue a second finish RPC, which would mask the
            # original loader error with a spurious unmatched-finish error.
            update_failed = True
            raise
        finally:
            close_iterator = getattr(source_iter, "close", None)
            if close_iterator is not None:
                close_iterator()
            finish_started = time.perf_counter()
            if not update_failed:
                self.client.finish_weight_update()
            finish_seconds = time.perf_counter() - finish_started
        total_seconds = time.perf_counter() - transfer_started
        timing = summarize_update_timing(
            bucket_results,
            start_rpc_seconds=start_rpc_seconds,
            finish_rpc_seconds=finish_seconds,
            total_seconds=total_seconds,
        )
        timing["metadata_seconds"] = metadata_seconds
        timing["bucket_build_seconds"] = bucket_build_seconds
        return bucket_results, timing

    def shutdown(self) -> None:
        self.group = None
        self.client = None

    def _get(self, path: str) -> Any:
        response = requests.get(f"{self.base_url}{path}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()


def _resolve_checkpoint(
    model: str,
    revision: str,
    checkpoint_path: str | None,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).resolve()
    elif Path(model).exists():
        path = Path(model).resolve()
    else:
        raise FileNotFoundError(
            "Refusing an implicit Hugging Face download. Pass the already-"
            "materialized local checkpoint with --checkpoint-path, or use a "
            f"local model path; got model={model!r}, revision={revision!r}."
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    return path


def _checkpoint_tensor_entries(
    root: Path,
    expert_tensor_order: ExpertTensorOrder = "natural",
) -> list[tuple[str, str]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return sorted(
            weight_map.items(),
            key=lambda item: _vime_transfer_name_key(item[0], expert_tensor_order),
        )

    single_file = root / "model.safetensors"
    if single_file.is_file():
        with safe_open(single_file, framework="pt", device="cpu") as f:
            return [
                (name, single_file.name)
                for name in sorted(
                    f.keys(),
                    key=lambda name: _vime_transfer_name_key(name, expert_tensor_order),
                )
            ]

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint found under {root}")
    result: list[tuple[str, str]] = []
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            result.extend((name, path.name) for name in f)
    return sorted(
        result,
        key=lambda item: _vime_transfer_name_key(item[0], expert_tensor_order),
    )


def _natural_name_key(name: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", name)
    )


def _expert_parent(name: str) -> str | None:
    if ".experts." not in name:
        return None
    return name.split(".experts.", 1)[0]


def _vime_transfer_name_key(
    name: str,
    expert_tensor_order: ExpertTensorOrder = "natural",
) -> tuple[
    int,
    tuple[tuple[int, int | str], ...],
    tuple[tuple[int, int | str], ...],
]:
    """Keep expert layers grouped while selecting their within-layer order."""
    parent = _expert_parent(name)
    if parent is None:
        return (0, _natural_name_key(name), ())
    suffix = name.split(".experts.", 1)[1]
    if expert_tensor_order == "natural":
        suffix_key = _natural_name_key(suffix)
    elif expert_tensor_order == "lexical":
        suffix_key = ((1, suffix),)
    else:
        raise ValueError(f"unknown expert tensor order: {expert_tensor_order!r}")
    return (1, _natural_name_key(parent), suffix_key)


def _param_nbytes(metadata: ParamMeta) -> int:
    numel = 1
    for dimension in metadata.shape:
        numel *= dimension
    return numel * metadata.dtype.itemsize


def _duplicate_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates, key=_natural_name_key)


def _build_vime_transfer_buckets(
    metadata: list[ParamMeta],
    bucket_size_bytes: int,
) -> list[WeightTransferBucket]:
    """Reproduce Vime's non-expert buckets and complete expert-layer buckets."""
    if bucket_size_bytes <= 0:
        raise ValueError("update bucket size must be positive")

    buckets: list[WeightTransferBucket] = []
    non_expert = [item for item in metadata if _expert_parent(item.name) is None]
    current: list[ParamMeta] = []
    current_bytes = 0
    for item in non_expert:
        item_bytes = _param_nbytes(item)
        if current and current_bytes + item_bytes > bucket_size_bytes:
            buckets.append(
                WeightTransferBucket(
                    phase="non_expert",
                    metadata=tuple(current),
                    total_bytes=current_bytes,
                )
            )
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
    if current:
        buckets.append(
            WeightTransferBucket(
                phase="non_expert",
                metadata=tuple(current),
                total_bytes=current_bytes,
            )
        )

    expert_groups: dict[str, list[ParamMeta]] = {}
    for item in metadata:
        parent = _expert_parent(item.name)
        if parent is not None:
            expert_groups.setdefault(parent, []).append(item)

    current = []
    current_bytes = 0
    for parent, items in expert_groups.items():
        group_bytes = sum(_param_nbytes(item) for item in items)
        if current and current_bytes + group_bytes > bucket_size_bytes:
            buckets.append(
                WeightTransferBucket(
                    phase="expert",
                    metadata=tuple(current),
                    total_bytes=current_bytes,
                )
            )
            current = []
            current_bytes = 0
        current.extend(items)
        current_bytes += group_bytes
    if current:
        buckets.append(
            WeightTransferBucket(
                phase="expert",
                metadata=tuple(current),
                total_bytes=current_bytes,
            )
        )
    return buckets


def _materialize_bucket(
    source_iter: Iterator[tuple[str, torch.Tensor]],
    bucket: WeightTransferBucket,
) -> list[tuple[str, torch.Tensor]]:
    named_tensors: list[tuple[str, torch.Tensor]] = []
    for expected in bucket.metadata:
        try:
            name, tensor = next(source_iter)
        except StopIteration as exc:
            raise ValueError(
                f"Weight source ended before expected tensor {expected.name!r}"
            ) from exc
        if name != expected.name:
            raise ValueError(
                "Weight source order mismatch: "
                f"expected {expected.name!r}, got {name!r}"
            )
        if tensor.dtype != expected.dtype or tuple(tensor.shape) != tuple(
            expected.shape
        ):
            raise ValueError(
                f"Weight source schema mismatch for {name}: "
                f"expected={(expected.dtype, tuple(expected.shape))}, "
                f"got={(tensor.dtype, tuple(tensor.shape))}"
            )
        named_tensors.append((name, tensor))
    return named_tensors


def _materialize_bucket_for_device(
    source_iter: Iterator[tuple[str, torch.Tensor]],
    bucket: WeightTransferBucket,
    device: torch.device,
    *,
    direct_file_expert_h2d: bool = False,
) -> tuple[list[tuple[str, torch.Tensor]], float]:
    """Materialize one bucket on a worker thread with an explicit CUDA device."""
    if device.type == "cuda":
        torch.accelerator.set_device_index(device)
    started = time.perf_counter()
    if direct_file_expert_h2d and bucket.phase == "expert":
        direct_materialize = getattr(
            source_iter,
            "materialize_contiguous_expert",
            None,
        )
        if direct_materialize is not None:
            named_tensors = direct_materialize(bucket, device)
            if named_tensors is not None:
                return named_tensors, time.perf_counter() - started
    named_tensors = _materialize_bucket(source_iter, bucket)
    if named_tensors and named_tensors[0][1].device.type == "cpu":
        to_device_started = time.perf_counter()
        named_tensors = [
            (name, tensor.to(device=device)) for name, tensor in named_tensors
        ]
        _record_transfer_timing(
            source_iter,
            to_device_seconds=time.perf_counter() - to_device_started,
            copy_count=len(named_tensors),
            mode="per-tensor",
        )
    return named_tensors, time.perf_counter() - started


def _record_transfer_timing(
    source_iter: Iterator[tuple[str, torch.Tensor]],
    *,
    to_device_seconds: float,
    copy_count: int,
    mode: str,
    cpu_pack_seconds: float = 0.0,
) -> None:
    record = getattr(source_iter, "record_transfer_timing", None)
    if record is not None:
        record(
            to_device_seconds=to_device_seconds,
            copy_count=copy_count,
            mode=mode,
            cpu_pack_seconds=cpu_pack_seconds,
        )
