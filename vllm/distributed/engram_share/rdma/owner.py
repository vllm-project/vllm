# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single pinned-Host owner for the real DSV4.1 Engram layers 1 and 14.

Startup is intentionally explicit: this program allocates ~188.83 GiB pinned
Host memory and registers two row264 native MRs. It is not run by
syntax/mock tests.
"""

import argparse
import base64
import hashlib
import json
import logging
import mmap
import os
import re
import signal
import struct
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from vllm.distributed.engram_share.rdma.runtime import ControlError, OwnerServer

LAYERS = (1, 14)
WEIGHT_DTYPE = "F8_E4M3"
SCALE_DTYPE = "F8_E8M0"
WEIGHT_COLS = 256
SCALE_COLS = 8
MAX_HEADER_BYTES = 512 * 1024 * 1024
MAX_COPY_CHUNK = 64 * 1024 * 1024
LOG = logging.getLogger("real_checkpoint_owner")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _safetensors_header(path):
    """Read only the bounded safetensors header, returning parsed header/hash."""
    with open(path, "rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"short safetensors header: {path}")
        (length,) = struct.unpack("<Q", prefix)
        if not 2 <= length <= MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length {length}: {path}")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header, _sha256(prefix + raw), 8 + length


def inspect_checkpoint(root):
    """Validate config/index/selected tensor headers without reading tensor data."""
    root = Path(root).resolve(strict=True)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config_bytes, index_bytes = config_path.read_bytes(), index_path.read_bytes()
    config, index = json.loads(config_bytes), json.loads(index_bytes)
    text_config = config.get("text_config", config)
    rows = tuple(text_config.get("engram_num_embeddings", ()))
    layers = tuple(text_config.get("engram_layer_ids", ()))
    if layers != LAYERS or len(rows) != len(LAYERS):
        raise ValueError(f"unexpected Engram config layers/rows: {layers}, {rows}")
    if int(text_config.get("engram_head_dim", WEIGHT_COLS)) != WEIGHT_COLS:
        raise ValueError("Engram head dimension is not 256")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index has no weight_map")

    specs, shard_headers = [], {}
    for layer, row_count in zip(LAYERS, rows):
        for kind, dtype, columns in (
            ("weight", WEIGHT_DTYPE, WEIGHT_COLS),
            ("scale", SCALE_DTYPE, SCALE_COLS),
        ):
            name = f"layers.{layer}.engram.embed.{kind}"
            shard_name = weight_map.get(name)
            if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
                raise ValueError(f"missing or unsafe index entry for {name}")
            shard_path = root / shard_name
            shard_path.resolve(strict=True).relative_to(root)
            if shard_name not in shard_headers:
                header, header_hash, data_start = _safetensors_header(shard_path)
                shard_headers[shard_name] = {
                    "path": str(shard_path),
                    "header": header,
                    "header_sha256": header_hash,
                    "data_start": data_start,
                    "file_size": shard_path.stat().st_size,
                }
            shard = shard_headers[shard_name]
            tensor = shard["header"].get(name)
            shape = [int(row_count), columns]
            if not isinstance(tensor, dict) or tensor.get("dtype") != dtype:
                raise ValueError(f"{name}: expected dtype {dtype}, got {tensor}")
            if tensor.get("shape") != shape:
                raise ValueError(
                    f"{name}: expected shape {shape}, got {tensor.get('shape')}"
                )
            offsets = tensor.get("data_offsets")
            size = int(row_count) * columns
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or offsets[1] - offsets[0] != size
                or offsets[0] < 0
                or shard["data_start"] + offsets[1] > shard["file_size"]
            ):
                raise ValueError(f"{name}: invalid data_offsets {offsets}")
            specs.append(
                {
                    "name": name,
                    "layer": layer,
                    "kind": kind,
                    "dtype": dtype,
                    "shape": shape,
                    "rows": int(row_count),
                    "columns": columns,
                    "nbytes": size,
                    "data_offsets": offsets,
                    "shard": shard_name,
                }
            )

    header_records = {
        name: {"header_sha256": item["header_sha256"], "file_size": item["file_size"]}
        for name, item in sorted(shard_headers.items())
    }
    identity_material = {
        "config_sha256": _sha256(config_bytes),
        "index_sha256": _sha256(index_bytes),
        "selected_shard_headers": header_records,
        "tensors": [
            {k: spec[k] for k in ("name", "dtype", "shape", "data_offsets", "shard")}
            for spec in specs
        ],
    }
    identity = _sha256(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
    )
    return {
        "root": root,
        "rows": rows,
        "config": config,
        "index": index,
        "shards": shard_headers,
        "specs": specs,
        "identity_material": identity_material,
        "checkpoint_identity": "sha256:" + identity,
    }


def _copy_and_sample_row264(manifest, destination, chunk_bytes):
    """Pack weight[256] + scale[8] into each row using bounded strided copies."""
    import numpy as np

    if not 0 < chunk_bytes <= MAX_COPY_CHUNK:
        raise ValueError(f"copy chunk must be in (0, {MAX_COPY_CHUNK}]")
    by_layer = {}
    for spec in manifest["specs"]:
        by_layer.setdefault(spec["layer"], {})[spec["kind"]] = spec
    offset = 0
    tables, samples = [], []
    chunk_rows = max(1, chunk_bytes // (WEIGHT_COLS + SCALE_COLS))
    for layer in LAYERS:
        parts = by_layer.get(layer, {})
        if set(parts) != {"weight", "scale"}:
            raise ValueError(f"layer {layer} needs checkpoint weight and scale specs")
        weight, scale = parts["weight"], parts["scale"]
        rows = weight["rows"]
        if (
            scale["rows"] != rows
            or weight["columns"] != WEIGHT_COLS
            or scale["columns"] != SCALE_COLS
        ):
            raise ValueError(f"layer {layer} cannot be packed as row264")
        table_bytes = rows * (WEIGHT_COLS + SCALE_COLS)
        row_ids = sorted({0, 1, rows // 2, rows - 1})
        digests = {"weight": hashlib.sha256(), "scale": hashlib.sha256()}
        with ExitStack() as stack:
            mappings = {}
            for part in (weight, scale):
                shard = manifest["shards"][part["shard"]]
                if part["shard"] not in mappings:
                    stream = stack.enter_context(open(shard["path"], "rb"))
                    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                    stack.callback(mapping.close)
                    mappings[part["shard"]] = (mapping, shard["data_start"])
            weight_map, weight_data_start = mappings[weight["shard"]]
            scale_map, scale_data_start = mappings[scale["shard"]]
            weight_start = weight_data_start + weight["data_offsets"][0]
            scale_start = scale_data_start + scale["data_offsets"][0]
            for row0 in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - row0)
                packed_offset = offset + row0 * (WEIGHT_COLS + SCALE_COLS)
                source_weight = np.ndarray(
                    (count, WEIGHT_COLS),
                    dtype=np.uint8,
                    buffer=weight_map,
                    offset=weight_start + row0 * WEIGHT_COLS,
                )
                target_weight = np.ndarray(
                    (count, WEIGHT_COLS),
                    dtype=np.uint8,
                    buffer=destination,
                    offset=packed_offset,
                    strides=(WEIGHT_COLS + SCALE_COLS, 1),
                )
                source_scale = np.ndarray(
                    (count, SCALE_COLS),
                    dtype=np.uint8,
                    buffer=scale_map,
                    offset=scale_start + row0 * SCALE_COLS,
                )
                target_scale = np.ndarray(
                    (count, SCALE_COLS),
                    dtype=np.uint8,
                    buffer=destination,
                    offset=packed_offset + WEIGHT_COLS,
                    strides=(WEIGHT_COLS + SCALE_COLS, 1),
                )
                np.copyto(target_weight, source_weight, casting="no")
                np.copyto(target_scale, source_scale, casting="no")
                del source_weight, target_weight, source_scale, target_scale

            for row in row_ids:
                packed_row = offset + row * (WEIGHT_COLS + SCALE_COLS)
                expected_weight = weight_map[
                    weight_start + row * WEIGHT_COLS : weight_start
                    + (row + 1) * WEIGHT_COLS
                ]
                expected_scale = scale_map[
                    scale_start + row * SCALE_COLS : scale_start
                    + (row + 1) * SCALE_COLS
                ]
                copied_weight = bytes(
                    destination[packed_row : packed_row + WEIGHT_COLS]
                )
                copied_scale = bytes(
                    destination[
                        packed_row + WEIGHT_COLS : packed_row + WEIGHT_COLS + SCALE_COLS
                    ]
                )
                if copied_weight != expected_weight or copied_scale != expected_scale:
                    raise RuntimeError(
                        "row264 sampled bytes differ from checkpoint, "
                        f"layer={layer} row={row}"
                    )
                digests["weight"].update(row.to_bytes(8, "little") + expected_weight)
                digests["scale"].update(row.to_bytes(8, "little") + expected_scale)

        tables.append(
            {
                "name": f"layers.{layer}.engram.embed.row264",
                "layer": layer,
                "kind": "row264",
                "dtype": "uint8",
                "shape": [rows, 264],
                "rows": rows,
                "columns": 264,
                "nbytes": table_bytes,
                "offset": offset,
                "component_layout": {
                    "weight": {
                        "offset": 0,
                        "bytes_per_row": WEIGHT_COLS,
                        "dtype": WEIGHT_DTYPE,
                    },
                    "scale": {
                        "offset": WEIGHT_COLS,
                        "bytes_per_row": SCALE_COLS,
                        "dtype": SCALE_DTYPE,
                    },
                },
            }
        )
        samples.append(
            {
                "layer": layer,
                "rows": row_ids,
                "weight_sample_bytes_sha256": digests["weight"].hexdigest(),
                "scale_sample_bytes_sha256": digests["scale"].hexdigest(),
                "exact_match": True,
            }
        )
        offset += table_bytes
    if offset != len(destination):
        raise RuntimeError(
            f"row264 backing size mismatch: packed={offset} "
            f"allocated={len(destination)}"
        )
    return tables, samples


def build_owner_metadata(
    manifest,
    tables,
    addresses,
    lengths,
    owner_gpu,
    rdma_port,
    native_md,
    storage_layout="row264",
):
    """Build the lease payload consumed by ReaderRuntime, in MR order."""
    if storage_layout != "row264":
        raise ValueError(f"unsupported storage layout: {storage_layout}")
    expected_order = [(1, "row264"), (14, "row264")]
    if (
        len(tables) != len(expected_order)
        or len(addresses) != len(expected_order)
        or len(lengths) != len(expected_order)
    ):
        raise ValueError(
            f"{storage_layout} requires {len(expected_order)} ordered table MRs"
        )
    actual_order = [(table["layer"], table["kind"]) for table in tables]
    if actual_order != expected_order:
        raise ValueError(f"unexpected table MR order: {actual_order}")
    if any(
        int(table["nbytes"]) != int(length) for table, length in zip(tables, lengths)
    ):
        raise ValueError("table specs do not match registered MR lengths")
    if any(
        table.get("columns") != 264 or table.get("shape") != [table.get("rows"), 264]
        for table in tables
    ):
        raise ValueError("row264 table specs must be [rows, 264]")
    identity = manifest["identity_material"]
    return {
        "checkpoint_id": manifest["checkpoint_identity"],
        "config_sha256": identity["config_sha256"],
        "index_sha256": identity["index_sha256"],
        "storage_layout": storage_layout,
        "table_specs": tables,
        "selected_shard_headers": identity["selected_shard_headers"],
        "addresses": list(addresses),
        "lengths": list(lengths),
        "owner_gpu": owner_gpu,
        "rdma_port": rdma_port,
        "native_md": base64.b64encode(native_md).decode("ascii"),
    }


class CountingOwnerServer(OwnerServer):
    """Owner control server with accepted lifecycle-operation counters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters = {"attach_ok": 0, "release_ok": 0, "drain_ok": 0}
        self._counter_lock = threading.Lock()

    def _dispatch(self, request):
        result = super()._dispatch(request)
        field = {
            "attach": "attach_ok",
            "release": "release_ok",
            "drain": "drain_ok",
        }.get(request.get("op"))
        if field:
            with self._counter_lock:
                self.counters[field] += 1
            LOG.info(
                "control %s leases=%d counters=%s",
                field,
                self.lease_count,
                self.counter_snapshot(),
            )
        return result

    def counter_snapshot(self):
        with self._counter_lock:
            return dict(self.counters)


class RealCheckpointOwner:
    """Own pinned checkpoint bytes, registered tables and the lease server."""

    def __init__(self, args):
        self.args = args
        self.backing = None
        self.numpy_view = None
        self.table_views = []
        self.context = None
        self.control = None
        self.manifest = None
        self.host_mapping = None
        self.host_pointer = None

    def start(self):
        import torch

        from vllm.distributed.engram_share.rdma.runtime import Context

        self.manifest = inspect_checkpoint(self.args.checkpoint)
        storage_layout = getattr(self.args, "storage_layout", "row264")
        if storage_layout != "row264":
            raise ValueError(f"unsupported storage layout: {storage_layout}")
        pinned_bytes = sum(spec["nbytes"] for spec in self.manifest["specs"])
        # The configured GiB value is displayed to two decimals. Permit at most
        # half of the final displayed decimal as rounding, not extra backing.
        budget = int((self.args.max_pinned_gib + 0.005) * (1024**3))
        if pinned_bytes > budget:
            raise MemoryError(f"pinned backing {pinned_bytes} exceeds budget {budget}")
        if not self.args.ready_file.parent.exists():
            raise FileNotFoundError(
                f"ready directory does not exist: {self.args.ready_file.parent}"
            )
        if self.args.ready_file.exists() or self.args.stop_file.exists():
            raise FileExistsError(
                "ready/stop file already exists; refusing stale lifecycle files"
            )
        if self.args.copy_chunk_mib < 1 or self.args.copy_chunk_mib > 64:
            raise ValueError("copy chunk must be between 1 and 64 MiB")

        LOG.info(
            "allocating one pinned Host backing: %d bytes (%.2f GiB), budget %.2f GiB",
            pinned_bytes,
            pinned_bytes / 2**30,
            self.args.max_pinned_gib,
        )
        # CUDA-register an exact-sized anonymous mapping: PyTorch 2.11's
        # pinned allocator lacks the threshold control and rounds this size.
        self.host_mapping = mmap.mmap(-1, pinned_bytes)
        self.backing = torch.frombuffer(self.host_mapping, dtype=torch.uint8)
        self.host_pointer = self.backing.data_ptr()
        status = torch.cuda.cudart().cudaHostRegister(
            self.host_pointer, pinned_bytes, 0
        )
        if (
            status != torch.cuda.cudart().cudaError.success
            or not self.backing.is_pinned()
        ):
            raise RuntimeError(f"cudaHostRegister failed: {status}")
        pointer = self.backing.data_ptr()
        actual_mapping = None
        for region in re.split(
            r"(?=^[0-9a-f]+-[0-9a-f]+ )",
            Path("/proc/self/smaps").read_text(),
            flags=re.M,
        ):
            if not region:
                continue
            lo, hi = (int(x, 16) for x in region.split()[0].split("-"))
            if lo <= pointer < hi:
                actual_mapping = {
                    "mapping_bytes": hi - lo,
                    "resident_bytes": int(
                        re.search(r"^Rss:\s+(\d+)", region, re.M).group(1)
                    )
                    * 1024,
                }
                break
        if actual_mapping is None or actual_mapping["mapping_bytes"] > budget:
            raise MemoryError(
                f"actual pinned mapping exceeds budget or is unknown: {actual_mapping}"
            )
        LOG.info("actual pinned allocation: %s", actual_mapping)
        self.numpy_view = memoryview(self.backing.numpy())
        tables, samples = _copy_and_sample_row264(
            self.manifest, self.numpy_view, self.args.copy_chunk_mib * 1024 * 1024
        )
        self.numpy_view.release()
        self.numpy_view = None
        self.table_views = [
            self.backing.narrow(0, table["offset"], table["nbytes"]) for table in tables
        ]
        # No copy or mutation of the backing is performed after this point.
        self.context = Context(
            self.args.library, self.args.owner_gpu, self.args.rdma_port
        )
        self.context.register(self.table_views, host=True)
        if len(self.context.addresses) != len(tables) or len(
            self.context.lengths
        ) != len(tables):
            raise RuntimeError(f"expected {len(tables)} native table registrations")
        checkpoint_id = self.manifest["checkpoint_identity"]
        metadata = build_owner_metadata(
            self.manifest,
            tables,
            self.context.addresses,
            self.context.lengths,
            self.args.owner_gpu,
            self.args.rdma_port,
            self.context.metadata(),
            storage_layout=storage_layout,
        )
        self.control = CountingOwnerServer(
            metadata, host=self.args.control_host, port=self.args.control_port
        ).start()
        self.ready = {
            "state": "ready",
            "host": self.args.advertise_host,
            "owner_pid": os.getpid(),
            "owner_start_ticks": Path("/proc/self/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
            "owner_source_sha256": _sha256(Path(__file__).read_bytes()),
            "library_sha256": _sha256(self.args.library.read_bytes()),
            "copy_completed_monotonic_ns": time.monotonic_ns(),
            "port": self.control.address[1],
            "control_port": self.control.address[1],
            "checkpoint_id": checkpoint_id,
            "rdma_port": self.args.rdma_port,
            "storage_layout": storage_layout,
            "generation": self.control.generation,
            "checkpoint_identity": checkpoint_id,
            "identity_scope": (
                "config/index/selected safetensors headers and tensor descriptors; "
                "not full weight-file hashes"
            ),
            "pinned_backing_bytes": pinned_bytes,
            "pinned_backing_gib": pinned_bytes / 2**30,
            "actual_pinned_allocation": actual_mapping,
            "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", ""),
            "copy_chunk_bytes": self.args.copy_chunk_mib * 1024 * 1024,
            "sample_validation": samples,
            "metadata": metadata,
            "counters": self.control.counter_snapshot(),
        }
        self._write_ready()
        LOG.info(
            "owner ready checkpoint=%s layout=%s leases=0 MRs=%d backing_bytes=%d",
            checkpoint_id,
            storage_layout,
            len(tables),
            pinned_bytes,
        )

    def _write_ready(self):
        path = self.args.ready_file
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(self.ready, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    def _request_stop(self):
        LOG.info(
            "stop requested; beginning drain with leases=%d counters=%s",
            self.control.lease_count,
            self.control.counter_snapshot(),
        )
        last_count = None
        while True:
            try:
                self.control.close()
                break
            except Exception as exc:
                # OwnerServer.close sets draining before refusing live leases.
                if not isinstance(exc, ControlError):
                    raise
                current = self.control.lease_count
                if current != last_count:
                    LOG.info(
                        "draining leases=%d counters=%s",
                        current,
                        self.control.counter_snapshot(),
                    )
                    last_count = current
                time.sleep(0.25)
        LOG.info(
            "drain complete; deregistering native MRs, counters=%s",
            self.control.counter_snapshot(),
        )
        while True:
            try:
                self.context.close()
                break
            except Exception:
                LOG.exception(
                    "native deregistration failed; retaining pinned backing "
                    "and retrying"
                )
                time.sleep(1.0)
        # Context.close deregisters before its buffer references and our views/backing
        # are dropped. Never reverse this order.
        self.table_views.clear()
        if self.numpy_view is not None:
            self.numpy_view.release()
            self.numpy_view = None
        import torch

        status = torch.cuda.cudart().cudaHostUnregister(self.host_pointer)
        if status != torch.cuda.cudart().cudaError.success:
            raise RuntimeError(
                f"cudaHostUnregister failed; retaining backing: {status}"
            )
        self.backing = None
        self.host_mapping.close()
        self.host_mapping = None
        self.ready.update(
            state="closed",
            counters=self.control.counter_snapshot(),
            native_deregistered=True,
            remaining_leases=self.control.lease_count,
        )
        self._write_ready()
        LOG.info(
            "owner closed; native deregistered, backing released, leases=0 counters=%s",
            self.control.counter_snapshot(),
        )

    def serve(self):
        stop_requested = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
        signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
        while not stop_requested.is_set() and not self.args.stop_file.exists():
            time.sleep(0.2)
        self._request_stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--owner-gpu", type=int, default=0)
    parser.add_argument("--rdma-port", type=int, required=True)
    parser.add_argument(
        "--control-host",
        default="127.0.0.1",
        help="bind host; use an externally reachable interface for cross-node readers",
    )
    parser.add_argument(
        "--advertise-host",
        default="127.0.0.1",
        help="host readers use to reach this owner",
    )
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument(
        "--max-pinned-gib",
        type=float,
        default=188.83,
        help="pinned byte budget; permits <=0.005 GiB display-rounding tolerance",
    )
    parser.add_argument("--copy-chunk-mib", type=int, default=8)
    parser.add_argument("--storage-layout", choices=("row264",), default="row264")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if args.max_pinned_gib <= 0 or args.owner_gpu < 0:
        parser.error("max-pinned-gib must be positive and owner-gpu nonnegative")
    owner = RealCheckpointOwner(args)
    owner.start()
    owner.serve()


if __name__ == "__main__":
    main()
