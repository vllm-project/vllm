# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ownership and manifest checks for externally supplied Engram tables."""

import threading

import pytest
import torch
from safetensors.torch import save_file

from vllm.models.deepseek_v41.nvidia.ops import shared_engram_owner as owner
from vllm.models.deepseek_v41.nvidia.ops.shared_engram_owner import (
    _OwnerServer,
    _OwnerState,
)
from vllm.models.deepseek_v41.nvidia.ops.shared_engram_vmm import TableHandles


def test_disconnected_consumer_prevents_owner_drain():
    state = _OwnerState({"generation": "generation-1"}, "private-token")
    lease = state.attach("private-token")
    state.lose_control(lease)
    assert not state.active
    assert state.uncertain == {lease}
    state.draining = True
    with pytest.raises(PermissionError, match="rejected"):
        state.attach("private-token")


def test_fabric_manifest_rejects_missing_or_malformed_chunks():
    table = TableHandles(100, 2 << 30, 0, (bytes(64),))
    assert TableHandles.from_wire(table.to_wire()) == table
    for bad_handles in ([], ["aa"], ["00" * 64, "00" * 64]):
        bad = {**table.to_wire(), "fabric_handles": bad_handles}
        with pytest.raises(ValueError, match="extent"):
            TableHandles.from_wire(bad)


def test_worker_shutdown_releases_leases_after_mapping_quiescence():
    events = []

    class Mapping:
        def close(self):
            events.append("quiesce")

    class Connection:
        def sendall(self, data):
            assert b'"op": "detach"' in data
            events.append("detach")

        def close(self):
            events.append("close")

    owner.retain_attachment(Connection(), Mapping(), "lease-1")
    owner.close_attached_tables()
    assert events == ["quiesce", "detach", "close"]
    assert not owner._attachments


def test_checkpoint_metadata_mismatch_detaches_without_import(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("private-token")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    state = _OwnerState(
        {
            "generation": "test-generation",
            "config_sha256": "wrong",
            "index_sha256": "wrong",
            "tables": {},
        },
        "private-token",
    )
    with _OwnerServer(("127.0.0.1", 0), state) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f"127.0.0.1:{server.server_address[1]}"
            with pytest.raises(ValueError, match="checkpoint metadata mismatch"):
                owner.attach_table(
                    endpoint,
                    token_file,
                    1,
                    3,
                    64,
                    (3,),
                    0,
                    expected_generation="test-generation",
                    checkpoint_dir=tmp_path,
                )
            with state.condition:
                assert state.condition.wait_for(lambda: not state.active, timeout=2)
                assert not state.uncertain
        finally:
            server.shutdown()
            thread.join()


def test_owner_rejects_wrong_table_encoding_before_publication(tmp_path):
    name = "layers.1.engram.embed.scale"
    filename = tmp_path / "mixed.safetensors"
    save_file({name: torch.full((3, 2), 127, dtype=torch.uint8)}, filename)
    with pytest.raises(ValueError, match="wrong checkpoint dtype"):
        owner._fill_tensor(
            torch.empty(6, dtype=torch.uint8),
            tmp_path,
            filename.name,
            name,
            3,
            2,
            torch.float8_e8m0fnu,
        )
