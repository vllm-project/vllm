# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact-source patch: fence retired reads; use original native cleanup routine."""

import ast
import hashlib
import sysconfig
from pathlib import Path

HASHES = {
    "base_worker.py": (
        "156cf9f8a44b139c6a328491299e2fc00d80b6560b586ac4d10a858f3df9dd9b"
    ),
    "pull_worker.py": (
        "ba16c48417f3648ef3441cb60fea5f01efcb3ee5eaac8b4afa57a19871cc5ed5"
    ),
}


def transform(name, text):
    if hashlib.sha256(text.encode()).hexdigest() != HASHES[name]:
        raise RuntimeError("Source does not match the approved v0.26.0: " + name)
    if name == "base_worker.py":
        old = "            if now - last_active > self._engine_ttl:\n"
        new = (
            "            from peer_lab import busy_reason, is_fenced\n"
            "            if is_fenced(self, eid) or busy_reason(self):\n"
            "                continue\n" + old
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
        old = (
            "        for engine_id, hb_info in metadata.heartbeat_by_engine.items():\n"
        )
        new = old + (
            "            from peer_lab import is_fenced\n"
            "            if is_fenced(self, engine_id):\n"
            "                continue\n"
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
        old = (
            "        with self._handshake_lock:\n"
            "            if engine_id in self._remote_agents:\n"
        )
        new = (
            "        with self._handshake_lock:\n"
            "            from peer_lab import is_fenced\n"
            "            if is_fenced(self, engine_id):\n"
            '                raise RuntimeError("Retired peer handshake rejected")\n'
            "            if engine_id in self._remote_agents:\n"
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
        old = "                        remote_agents, clock_offset = f.result()\n"
        new = old + (
            "                        if is_fenced(self, eid):\n"
            "                            raise RuntimeError("
            '"Retired peer handshake callback rejected")\n'
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
    else:
        old = "            self._recving_metadata[req_id] = meta\n"
        new = old + (
            "            from peer_lab import is_fenced\n"
            "            if is_fenced(self, remote_engine_id):\n"
            '                logger.warning("Rejected retired P generation %s", '
            "remote_engine_id)\n"
            "                raise RuntimeError("
            '"Peer-lab admission fence was bypassed")\n'
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
        old = "    def _read_blocks(\n"
        new = (
            "        # Notification-only cache hits never enter _recving_transfers.\n"
            "        # All rank notifications above have been attempted; no data\n"
            "        # transfer exists whose get_finished() would remove this entry.\n"
            "        if not local_block_ids and "
            "req_id not in self._recving_transfers:\n"
            "            self._recving_metadata.pop(req_id, None)\n\n" + old
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
        old = "        engine_id = meta.remote.engine_id\n"
        new = old + (
            "        from peer_lab import is_fenced\n"
            "        if is_fenced(self, engine_id):\n"
            '            raise RuntimeError("Peer-lab admission fence was bypassed")\n'
        )
        assert text.count(old) == 1
        text = text.replace(old, new)
    ast.parse(text)
    return text


def main():
    root = (
        Path(sysconfig.get_paths()["purelib"])
        / "vllm/distributed/kv_transfer/kv_connector/v1/nixl"
    )
    # Validate every file before changing either source file in the new layer.
    patched = {name: transform(name, (root / name).read_text()) for name in HASHES}
    for name, text in patched.items():
        (root / name).write_text(text)
        print(name, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
