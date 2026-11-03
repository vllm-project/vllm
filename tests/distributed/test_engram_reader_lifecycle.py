# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.distributed.engram_rdma.control import OwnerServer
from vllm.distributed.engram_rdma.reader import ReaderRuntime


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
            ReaderRuntime()
        assert owner.lease_count == 0
