# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import socket
import unittest

from vllm.distributed.engram_rdma.control import ControlError, OwnerClient, OwnerServer


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


if __name__ == "__main__":
    unittest.main()
