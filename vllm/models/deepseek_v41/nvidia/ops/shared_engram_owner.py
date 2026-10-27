# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish immutable Engram tables to independent CUDA processes.

Run with ``python -m vllm.models.deepseek_v41.nvidia.ops.shared_engram_owner``.
The control socket only exchanges metadata and leases; lookup rows never pass
through it. Every participant needs access to the same NVIDIA IMEX channel.
"""

import argparse
import atexit
import hashlib
import json
import logging
import os
import secrets
import signal
import socket
import socketserver
import threading
from collections import Counter
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm.models.deepseek_v41.common.engram import EngramLayout
from vllm.models.deepseek_v41.nvidia.ops.shared_engram_vmm import (
    TableHandles,
    TableMapping,
    create_table,
    gpu_host_numa_nodes,
    import_table,
)

ROWS_PER_READ = 1 << 18
MAX_CONTROL_LINE = 4096
logger = logging.getLogger(__name__)
_attachments: list[tuple[socket.socket, TableMapping, str]] = []
_attachments_lock = threading.Lock()


def range_chunks(table: TableHandles):
    return (
        (offset, min(table.chunk_bytes, table.padded_bytes - offset))
        for offset in range(0, table.padded_bytes, table.chunk_bytes)
    )


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_tables(model_dir: Path) -> tuple[dict, dict]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    index = json.loads(index_path.read_text())["weight_map"]
    layers = tuple(text_config["engram_layer_ids"])
    rows = tuple(text_config["engram_num_embeddings"])
    dim = int(text_config["engram_head_dim"])
    if len(layers) != len(rows) or dim <= 0 or dim % 32:
        raise ValueError("invalid Engram checkpoint layout")
    layout = EngramLayout(SimpleNamespace(**text_config))
    tables = {}
    for layer_index, (layer, num_rows) in enumerate(zip(layers, rows)):
        keys = (
            f"layers.{layer}.engram.embed.weight",
            f"layers.{layer}.engram.embed.scale",
        )
        if num_rows <= 0 or any(key not in index for key in keys):
            raise ValueError(f"missing Engram checkpoint table for layer {layer}")
        tables[str(layer)] = {
            "rows": int(num_rows),
            "dim": dim,
            "scale_width": dim // 32,
            "head_sizes": tuple(
                size for order in layout.primes[layer_index] for size in order
            ),
            "keys": keys,
            "files": tuple(index[key] for key in keys),
        }
    return {
        "config_sha256": _digest_file(config_path),
        "index_sha256": _digest_file(index_path),
        "format": "FP8_E4M3FN_E8M0_BLOCK32",
    }, tables


def _fill_tensor(
    destination: torch.Tensor,
    model_dir: Path,
    filename: str,
    key: str,
    rows: int,
    width: int,
    expected_dtype: torch.dtype,
) -> str:
    digest = hashlib.sha256()
    with safe_open(model_dir / filename, framework="pt", device="cpu") as source:
        lazy = source.get_slice(key)
        shape = lazy.get_shape()
        if len(shape) != 2 or shape != [rows, width]:
            raise ValueError(f"wrong checkpoint shape for {key}: {shape}")
        for start in range(0, rows, ROWS_PER_READ):
            chunk = lazy[start : min(start + ROWS_PER_READ, rows)]
            if chunk.dtype != expected_dtype:
                raise ValueError(f"wrong checkpoint dtype for {key}: {chunk.dtype}")
            raw = chunk.contiguous().view(torch.uint8).flatten()
            byte_start = start * width
            destination[byte_start : byte_start + raw.numel()].copy_(raw)
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def publish_checkpoint(model_dir: Path, device: int) -> tuple[dict, list[TableMapping]]:
    """Load full table bytes once and publish after both tensors are complete."""
    identity, tables = _checkpoint_tables(model_dir)
    mappings: list[TableMapping] = []
    published = {}
    numa_nodes = gpu_host_numa_nodes()
    try:
        for layer, spec in tables.items():
            rows, dim, scale_width = (
                spec["rows"],
                spec["dim"],
                spec["scale_width"],
            )
            weight_bytes = rows * dim
            mapping = create_table(
                rows * (dim + scale_width), device, numa_nodes=numa_nodes
            )
            mappings.append(mapping)
            assert mapping.cpu is not None
            weights = mapping.cpu[:weight_bytes]
            scales = mapping.cpu[weight_bytes:]
            weight_key, scale_key = spec["keys"]
            weight_file, scale_file = spec["files"]
            weight_hash = _fill_tensor(
                weights,
                model_dir,
                weight_file,
                weight_key,
                rows,
                dim,
                torch.float8_e4m3fn,
            )
            scale_hash = _fill_tensor(
                scales,
                model_dir,
                scale_file,
                scale_key,
                rows,
                scale_width,
                torch.float8_e8m0fnu,
            )
            published[layer] = {
                "rows": rows,
                "dim": dim,
                "scale_width": scale_width,
                "head_sizes": spec["head_sizes"],
                "weight_sha256": weight_hash,
                "scale_sha256": scale_hash,
                "table": mapping.table.to_wire(),
            }
        torch.accelerator.synchronize(device)
        return {
            "schema": 1,
            "generation": secrets.token_hex(16),
            **identity,
            "tables": published,
        }, mappings
    except BaseException:
        for mapping in reversed(mappings):
            mapping.close()
        raise


class _OwnerState:
    def __init__(self, manifest: dict, token: str) -> None:
        self.manifest = manifest
        self.token = token
        self.active: set[str] = set()
        self.uncertain: set[str] = set()
        self.draining = False
        self.condition = threading.Condition()

    def attach(self, token: str) -> str:
        with self.condition:
            if self.draining or not secrets.compare_digest(token, self.token):
                raise PermissionError("shared Engram attachment rejected")
            lease = secrets.token_hex(16)
            self.active.add(lease)
            return lease

    def detach(self, lease: str) -> None:
        with self.condition:
            self.active.remove(lease)
            self.condition.notify_all()

    def lose_control(self, lease: str) -> None:
        with self.condition:
            self.active.discard(lease)
            self.uncertain.add(lease)
            self.condition.notify_all()


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: _OwnerState = self.server.owner_state  # type: ignore[attr-defined]
        lease = None
        try:
            self.request.settimeout(30)
            request = json.loads(self.rfile.readline(MAX_CONTROL_LINE))
            if request.get("op") != "attach":
                raise ValueError("expected attach")
            lease = state.attach(str(request.get("token", "")))
            reply = {"ok": True, "lease": lease, "manifest": state.manifest}
            self.wfile.write(json.dumps(reply).encode() + b"\n")
            self.wfile.flush()
            self.request.settimeout(None)
            line = self.rfile.readline(MAX_CONTROL_LINE)
            if not line:
                raise ConnectionError("consumer control socket closed")
            request = json.loads(line)
            if request != {"op": "detach", "lease": lease}:
                raise ValueError("consumer did not detach its lease")
            state.detach(lease)
            lease = None
            self.wfile.write(b'{"ok":true}\n')
            self.wfile.flush()
        except Exception as exc:
            if lease is not None:
                state.lose_control(lease)
            try:
                self.wfile.write(
                    json.dumps({"ok": False, "error": str(exc)}).encode() + b"\n"
                )
                self.wfile.flush()
            except OSError:
                pass


class _OwnerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: _OwnerState) -> None:
        super().__init__(address, _RequestHandler)
        self.owner_state = state


def run_owner(model_dir: Path, address: tuple[str, int], token_file: Path, device: int):
    token = token_file.read_text().strip()
    if len(token) < 32 or token_file.stat().st_mode & 0o077:
        raise ValueError("token file must hold at least 32 private characters")
    torch.accelerator.set_device_index(device)
    manifest, mappings = publish_checkpoint(model_dir, device)
    state = _OwnerState(manifest, token)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _sig, _frame: stop.set())
    try:
        server = _OwnerServer(address, state)
    except BaseException:
        for mapping in reversed(mappings):
            mapping.close()
        raise
    with server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        node_bytes: Counter[int] = Counter()
        for mapping in mappings:
            for node, (_, length) in zip(
                mapping.table.placement,
                range_chunks(mapping.table),
            ):
                node_bytes[node] += length
        print(
            "ENGRAM_OWNER_BACKING "
            + json.dumps(
                {
                    "table_bytes": sum(
                        mapping.table.size_bytes for mapping in mappings
                    ),
                    "padded_bytes": sum(
                        mapping.table.padded_bytes for mapping in mappings
                    ),
                    "node_bytes": dict(node_bytes),
                }
            ),
            flush=True,
        )
        bound_host = server.server_address[0]
        host = bound_host.decode() if isinstance(bound_host, bytes) else bound_host
        print(
            f"ENGRAM_OWNER_READY {host}:{server.server_address[1]} "
            f"generation={manifest['generation']}",
            flush=True,
        )
        stop.wait()
        with state.condition:
            state.draining = True
        server.shutdown()
        thread.join()
        with state.condition:
            while state.active or state.uncertain:
                state.condition.wait()
    for mapping in reversed(mappings):
        mapping.close()


def attach_table(
    endpoint: str,
    token_file: Path,
    layer: int,
    rows: int,
    dim: int,
    head_sizes: tuple[int, ...],
    device: int,
    expected_generation: str | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[socket.socket, TableMapping, dict]:
    """Attach one layer and retain the control socket as its owner lease."""
    host, port = endpoint.rsplit(":", 1)
    connection = socket.create_connection((host, int(port)), timeout=30)
    lease = None
    try:
        token = token_file.read_text().strip()
        request = {"op": "attach", "token": token}
        connection.sendall(json.dumps(request).encode() + b"\n")
        response = b""
        while not response.endswith(b"\n"):
            part = connection.recv(1 << 20)
            if not part:
                raise ConnectionError("Engram owner closed during attach")
            response += part
            if len(response) > 1 << 20:
                raise ValueError("Engram owner manifest is too large")
        reply = json.loads(response)
        if not reply.get("ok"):
            raise RuntimeError(
                f"Engram owner rejected attachment: {reply.get('error')}"
            )
        lease = reply["lease"]
        manifest = reply["manifest"]
        if (
            expected_generation is not None
            and manifest["generation"] != expected_generation
        ):
            raise ValueError("unexpected shared Engram generation")
        if checkpoint_dir is not None:
            if not checkpoint_dir.is_dir():
                raise ValueError("shared Engram needs a local checkpoint directory")
            expected_hashes = {
                "config_sha256": _digest_file(checkpoint_dir / "config.json"),
                "index_sha256": _digest_file(
                    checkpoint_dir / "model.safetensors.index.json"
                ),
            }
            if any(
                manifest.get(key) != value for key, value in expected_hashes.items()
            ):
                raise ValueError("shared Engram checkpoint metadata mismatch")
        spec = manifest["tables"][str(layer)]
        if (
            manifest["schema"] != 1
            or manifest["format"] != "FP8_E4M3FN_E8M0_BLOCK32"
            or spec["rows"] != rows
            or spec["dim"] != dim
            or spec["scale_width"] != dim // 32
            or tuple(spec["head_sizes"]) != head_sizes
        ):
            raise ValueError("Engram owner generation has incompatible table layout")
        table = TableHandles.from_wire(spec["table"])
        if table.size_bytes != rows * (dim + dim // 32):
            raise ValueError("Engram owner table byte size mismatch")
        mapping = import_table(table, device)
        return connection, mapping, {"lease": lease, **manifest}
    except BaseException:
        if lease is not None:
            with suppress(OSError):
                connection.sendall(
                    json.dumps({"op": "detach", "lease": lease}).encode() + b"\n"
                )
        connection.close()
        raise


def detach_table(connection: socket.socket, mapping: TableMapping, lease: str) -> None:
    """Quiesce local work before releasing the owner's lease."""
    mapping.close()
    try:
        connection.sendall(
            json.dumps({"op": "detach", "lease": lease}).encode() + b"\n"
        )
    finally:
        connection.close()


def retain_attachment(
    connection: socket.socket, mapping: TableMapping, lease: str
) -> None:
    """Keep mappings until worker shutdown has quiesced its graphs."""
    with _attachments_lock:
        _attachments.append((connection, mapping, lease))


def close_attached_tables() -> None:
    """Release all tables after the model runner stops replaying graphs."""
    with _attachments_lock:
        for attachment in reversed(_attachments[:]):
            try:
                detach_table(*attachment)
            except Exception:
                logger.exception("Cannot safely detach an external Engram table")
            else:
                _attachments.remove(attachment)


atexit.register(close_attached_tables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Own immutable shared Engram tables")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be a local checkpoint directory")
    if args.listen_port < 1 or args.listen_port > 65535:
        parser.error("--listen-port must be between 1 and 65535")
    os.umask(0o077)
    run_owner(
        args.model_dir,
        (args.listen_host, args.listen_port),
        args.token_file,
        args.device,
    )


if __name__ == "__main__":
    main()
