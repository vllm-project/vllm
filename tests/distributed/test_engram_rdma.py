# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import socket
import struct
import tempfile
import unittest
from pathlib import Path

import pytest

from vllm.distributed.engram_share.provider import get_provider
from vllm.distributed.engram_share.rdma import provider as rdma_provider
from vllm.distributed.engram_share.rdma.owner import (
    MAX_COPY_CHUNK,
    _copy_and_sample_row264,
    build_owner_metadata,
    inspect_checkpoint,
)
from vllm.distributed.engram_share.rdma.provider import (
    ReaderRuntime,
    verify_checkpoint_samples,
)
from vllm.distributed.engram_share.rdma.runtime import (
    ControlError,
    OwnerClient,
    OwnerServer,
)


def test_external_provider_is_initialized_only_by_the_loader(tmp_path, monkeypatch):
    """Loading the interface does not open a remote reader or acquire a lease."""
    (tmp_path / "test_engram_provider.py").write_text(
        "reader = None\n"
        "def get_reader(checkpoint_path=None):\n"
        "    global reader\n"
        "    if reader is None:\n"
        "        if checkpoint_path is None: raise RuntimeError('not initialized')\n"
        "        reader = checkpoint_path\n"
        "    return reader\n"
        "def close_reader():\n"
        "    global reader\n"
        "    reader = None\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    provider = get_provider("test_engram_provider")
    with pytest.raises(RuntimeError, match="not initialized"):
        provider.get_reader()
    assert provider.get_reader("checkpoint") == "checkpoint"
    assert provider.get_reader() == "checkpoint"
    provider.close_reader()
    with pytest.raises(RuntimeError, match="not initialized"):
        provider.get_reader()


@pytest.mark.parametrize("retirement_fails", [False, True])
def test_reader_setup_failure_retires_or_quarantines_runtime(
    tmp_path, monkeypatch, retirement_fails
):
    """A CUDA event failure must not release unsafe registered resources."""
    runtimes = []

    class Runtime:
        storage_layout = "row264"

        def __init__(self, checkpoint_path):
            self.closed = False
            runtimes.append(self)

        def close(self):
            if retirement_fails:
                raise RuntimeError("retirement failed")
            self.closed = True

    def fail_event():
        raise RuntimeError("event creation failed")

    quarantined = []
    monkeypatch.setattr(rdma_provider, "verify_checkpoint_samples", lambda path: None)
    monkeypatch.setattr(rdma_provider, "ReaderRuntime", Runtime)
    monkeypatch.setattr(rdma_provider.torch.cuda, "Event", fail_event)
    monkeypatch.setattr(rdma_provider, "_quarantined", quarantined)
    with pytest.raises(RuntimeError, match="event creation failed"):
        rdma_provider.VllmRemoteReader(tmp_path)
    assert runtimes[0].closed is not retirement_fails
    assert quarantined == ([runtimes[0]] if retirement_fails else [])


class OwnerControlTests(unittest.TestCase):
    def setUp(self):
        self.owner = OwnerServer(
            {"checkpoint_id": "ckpt-sha256:abc", "table": {"rows": 7}}
        ).start()
        host, port = self.owner.address
        self.client = OwnerClient(host, port, "ckpt-sha256:abc", self.owner.generation)
        self.leases = []

    def attach(self, client=None):
        lease = (client or self.client).attach()
        self.leases.append(lease)
        return lease

    def tearDown(self):
        for lease in self.leases:
            if not lease._released:
                lease.release()
        self.owner.close()

    def test_independent_clients_and_reconnect_keep_generation(self):
        host, port = self.owner.address
        other = OwnerClient(host, port, "ckpt-sha256:abc", self.owner.generation)
        first, second = self.attach(), self.attach(other)
        self.assertNotEqual(first.token, second.token)
        self.assertEqual(first.metadata["table"]["rows"], 7)
        generation = self.owner.generation
        first.release()
        reconnected = self.attach(
            OwnerClient(host, port, "ckpt-sha256:abc", generation)
        )
        self.assertEqual(reconnected.client.generation, generation)
        second.release()
        reconnected.release()

    def test_close_enters_drain_and_refuses_live_lease(self):
        lease = self.attach()
        with self.assertRaisesRegex(ControlError, r"lease\(s\) remain"):
            self.owner.close()
        self.assertTrue(self.owner.draining)
        with self.assertRaisesRegex(ControlError, "draining"):
            self.client.attach()
        lease.release()
        self.owner.close()

    def test_explicit_drain_rejects_new_attach(self):
        lease = self.attach()
        result = self.client.begin_drain()
        self.assertEqual(result["leases"], 1)
        with self.assertRaisesRegex(ControlError, "draining"):
            self.client.attach()
        lease.release()

    def test_status_reports_live_leases_and_drain_without_attaching(self):
        self.assertEqual(self.client.status()["leases"], 0)
        lease = self.attach()
        self.assertEqual(self.client.status()["leases"], 1)
        self.client.begin_drain()
        self.assertTrue(self.client.status()["draining"])
        lease.release()
        self.assertEqual(self.client.status()["leases"], 0)

    def test_stale_generation_and_lease_rejected(self):
        host, port = self.owner.address
        stale = OwnerClient(host, port, "ckpt-sha256:abc", "old-generation")
        with self.assertRaisesRegex(ControlError, "generation"):
            stale.attach()
        lease = self.attach()
        lease.release()
        with self.assertRaisesRegex(ControlError, "unknown or already"):
            self.client._request("release", lease=lease.token)

    def test_checkpoint_mismatch_and_metadata_is_immutable(self):
        host, port = self.owner.address
        wrong = OwnerClient(host, port, "other-checkpoint", self.owner.generation)
        with self.assertRaisesRegex(ControlError, "checkpoint"):
            wrong.attach()
        with self.assertRaises(TypeError):
            self.owner.metadata["checkpoint_id"] = "changed"

    def test_malformed_and_oversize_frames_rejected(self):
        for payload in (b"{bad json}\n", b"{}" + b" " * (64 * 1024) + b"\n"):
            with socket.create_connection(self.owner.address, timeout=2) as sock:
                sock.settimeout(2)
                sock.sendall(payload)
                response = json.loads(sock.makefile("rb").readline())
            self.assertFalse(response["ok"])
            self.assertIn("invalid request", response["error"])


def write_safetensors(path, tensors, payload):
    header = json.dumps(tensors, separators=(",", ":")).encode()
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class Row264OwnerCpuTests(unittest.TestCase):
    def make_checkpoint(self, root):
        config = {
            "text_config": {
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [2, 3],
                "engram_head_dim": 256,
            }
        }
        (root / "config.json").write_text(json.dumps(config))
        tensors, weight_map = {}, {}
        payloads = {}
        for layer, rows, shard in (
            (1, 2, "model-00047-of-00048.safetensors"),
            (14, 3, "model-00048-of-00048.safetensors"),
        ):
            weight_name = f"layers.{layer}.engram.embed.weight"
            scale_name = f"layers.{layer}.engram.embed.scale"
            weight = bytes((layer + i) % 256 for i in range(rows * 256))
            scale = bytes((layer + i + 31) % 256 for i in range(rows * 8))
            tensors[weight_name] = {
                "dtype": "F8_E4M3",
                "shape": [rows, 256],
                "data_offsets": [0, len(weight)],
            }
            tensors[scale_name] = {
                "dtype": "F8_E8M0",
                "shape": [rows, 8],
                "data_offsets": [len(weight), len(weight) + len(scale)],
            }
            weight_map[weight_name] = weight_map[scale_name] = shard
            payloads[shard] = (weight, scale)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        for shard, (weight, scale) in payloads.items():
            write_safetensors(
                root / shard,
                {k: v for k, v in tensors.items() if weight_map[k] == shard},
                weight + scale,
            )
        return payloads

    def test_row264_matches_checkpoint_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = self.make_checkpoint(root)
            manifest = inspect_checkpoint(root)
            total = sum(spec["nbytes"] for spec in manifest["specs"])
            packed_buffer = bytearray(total)
            packed_tables, packed_samples = _copy_and_sample_row264(
                manifest, memoryview(packed_buffer), chunk_bytes=2 * 264
            )

            self.assertEqual(
                [(x["layer"], x["kind"]) for x in packed_tables],
                [(1, "row264"), (14, "row264")],
            )
            self.assertTrue(all(x["exact_match"] for x in packed_samples))
            self.assertEqual(
                manifest["identity_material"],
                inspect_checkpoint(root)["identity_material"],
            )

            offset = 0
            for layer, rows, shard in (
                (1, 2, "model-00047-of-00048.safetensors"),
                (14, 3, "model-00048-of-00048.safetensors"),
            ):
                weight, scale = payloads[shard]
                for row in range(rows):
                    expected = (
                        weight[row * 256 : (row + 1) * 256]
                        + scale[row * 8 : (row + 1) * 8]
                    )
                    self.assertEqual(
                        bytes(
                            packed_buffer[offset + row * 264 : offset + (row + 1) * 264]
                        ),
                        expected,
                    )
                offset += rows * 264

            row_md = build_owner_metadata(
                manifest,
                packed_tables,
                [21, 22],
                [x["nbytes"] for x in packed_tables],
                0,
                55000,
                b"native-md",
                storage_layout="row264",
            )
            self.assertEqual(row_md["storage_layout"], "row264")
            self.assertEqual(manifest["checkpoint_identity"], row_md["checkpoint_id"])
            self.assertEqual(len(row_md["addresses"]), 2)
            self.assertEqual(
                [
                    (x["layer"], x["kind"], x["rows"], x["columns"])
                    for x in row_md["table_specs"]
                ],
                [(1, "row264", 2, 264), (14, "row264", 3, 264)],
            )
            self.assertEqual(
                row_md["table_specs"][0]["component_layout"]["scale"]["offset"], 256
            )

    def test_row264_copy_enforces_bounded_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_checkpoint(root)
            manifest = inspect_checkpoint(root)
            backing = bytearray(sum(spec["nbytes"] for spec in manifest["specs"]))
            with self.assertRaisesRegex(ValueError, "copy chunk"):
                _copy_and_sample_row264(
                    manifest, memoryview(backing), MAX_COPY_CHUNK + 1
                )

    def test_row264_metadata_rejects_wrong_mr_count_or_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_checkpoint(root)
            manifest = inspect_checkpoint(root)
            tables = [
                {
                    "layer": 1,
                    "kind": "row264",
                    "nbytes": 528,
                    "rows": 2,
                    "columns": 264,
                    "shape": [2, 264],
                },
                {
                    "layer": 14,
                    "kind": "row264",
                    "nbytes": 792,
                    "rows": 3,
                    "columns": 264,
                    "shape": [3, 264],
                },
            ]
            with self.assertRaisesRegex(ValueError, "ordered table MRs"):
                build_owner_metadata(
                    manifest,
                    tables,
                    [1, 2, 3],
                    [528, 792, 0],
                    0,
                    55000,
                    b"md",
                    storage_layout="row264",
                )
            with self.assertRaisesRegex(ValueError, "MR order"):
                build_owner_metadata(
                    manifest,
                    list(reversed(tables)),
                    [1, 2],
                    [792, 528],
                    0,
                    55000,
                    b"md",
                    storage_layout="row264",
                )


def test_reader_setup_failure_releases_owner_lease(tmp_path, monkeypatch):
    """A failed checkpoint check must not block an otherwise safe owner drain."""
    metadata = {"checkpoint_id": "fixture", "storage_layout": "row264"}
    with OwnerServer(metadata).start() as owner:
        host, port = owner.address
        ready = tmp_path / "ready.json"
        ready.write_text(
            json.dumps(
                {
                    "host": host,
                    "port": port,
                    "checkpoint_id": "fixture",
                    "generation": owner.generation,
                }
            )
        )
        monkeypatch.setenv("ENGRAM_SHARED_READY", str(ready))
        monkeypatch.setenv("ENGRAM_SHARED_MODEL", str(tmp_path / "missing"))
        with pytest.raises(FileNotFoundError):
            ReaderRuntime(tmp_path)
        assert owner.lease_count == 0


def test_engine_checkpoint_must_match_owner_directory(tmp_path, monkeypatch):
    """Identical headers and tensor shapes cannot make two checkpoints equal."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    fixture = Row264OwnerCpuTests()
    fixture.make_checkpoint(a)
    fixture.make_checkpoint(b)
    manifest = inspect_checkpoint(a)
    shard = b / "model-00047-of-00048.safetensors"
    content = bytearray(shard.read_bytes())
    content[-1] ^= 1
    shard.write_bytes(content)
    assert (
        inspect_checkpoint(b)["checkpoint_identity"] == manifest["checkpoint_identity"]
    )
    total = sum(spec["nbytes"] for spec in manifest["specs"])
    _, samples = _copy_and_sample_row264(manifest, memoryview(bytearray(total)), 528)
    ready = tmp_path / "ready.json"
    ready.write_text(
        json.dumps(
            {
                "checkpoint_id": manifest["checkpoint_identity"],
                "sample_validation": samples,
            }
        )
    )
    monkeypatch.setenv("ENGRAM_SHARED_READY", str(ready))
    monkeypatch.setenv("ENGRAM_SHARED_MODEL", str(a))
    verify_checkpoint_samples(a)
    with pytest.raises(ValueError, match="engine checkpoint differs"):
        verify_checkpoint_samples(b)
