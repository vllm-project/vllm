# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ownership and manifest checks for externally supplied Engram tables."""

import hashlib
import threading

import pytest
import torch
from safetensors.torch import save_file

from vllm.distributed.engram_share.vmm import mapping as vmm
from vllm.distributed.engram_share.vmm import owner
from vllm.distributed.engram_share.vmm.mapping import TableHandles
from vllm.distributed.engram_share.vmm.owner import (
    _OwnerServer,
    _OwnerState,
)
from vllm.models.deepseek_v41.common.engram import external_table_views


def test_disconnected_consumer_prevents_owner_drain(caplog):
    state = _OwnerState({"generation": "generation-1"}, "private-token")
    client = {"peer": "192.0.2.7", "hostname": "worker-7", "pid": 42}
    lease = state.attach("private-token", client)
    state.lose_control(lease)
    assert not state.active
    assert state.uncertain == {lease}
    assert state.drain_status()["clients"][lease] == client
    assert "retaining backing" in caplog.text
    state.draining = True
    with pytest.raises(PermissionError, match="rejected"):
        state.attach("private-token")


def test_normal_detach_removes_client_from_drain_status(capsys):
    state = _OwnerState({"generation": "generation-1"}, "private-token")
    lease = state.attach("private-token", {"peer": "127.0.0.1", "pid": 4})
    state.detach(lease)
    output = capsys.readouterr().out
    assert "ENGRAM_OWNER_ATTACH" in output
    assert "ENGRAM_OWNER_DETACH" in output
    assert state.drain_status() == {
        "generation": "generation-1",
        "active": [],
        "uncertain": [],
        "clients": {},
    }


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


def test_failed_import_cleanup_keeps_owner_lease(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text("private-token")
    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    config.write_text("{}")
    index.write_text("{}")
    table = TableHandles(198, 256, 0, (bytes(64),), 256)
    manifest = {
        "generation": "generation-1",
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "schema": 1,
        "format": "FP8_E4M3FN_E8M0_BLOCK32",
        "tables": {
            "1": {
                "rows": 3,
                "dim": 64,
                "scale_width": 2,
                "head_sizes": [3],
                "table": table.to_wire(),
            }
        },
    }
    state = _OwnerState(manifest, "private-token")

    def failed_import(_table, _device):
        raise vmm.IncompleteTableMapCleanup("mapping may remain")

    monkeypatch.setattr(owner, "import_table", failed_import)
    with _OwnerServer(("127.0.0.1", 0), state) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            endpoint = f"127.0.0.1:{server.server_address[1]}"
            with pytest.raises(vmm.IncompleteTableMapCleanup):
                owner.attach_table(
                    endpoint,
                    token_file,
                    1,
                    3,
                    64,
                    (3,),
                    0,
                    expected_generation="generation-1",
                    checkpoint_dir=tmp_path,
                )
            with state.condition:
                assert state.condition.wait_for(
                    lambda: bool(state.uncertain), timeout=2
                )
                assert not state.active
        finally:
            server.shutdown()
            thread.join()


def test_failed_vmm_map_cleanup_reports_uncertain_mapping(monkeypatch):
    calls = []

    def driver_call(fn, *_args):
        name = fn.__name__
        calls.append(name)
        if name == "cuMemAddressReserve":
            return 4096
        if name == "cuMemMap" and calls.count(name) == 2:
            raise RuntimeError("second map failed")
        if name in ("cuMemUnmap", "cuMemAddressFree"):
            raise RuntimeError(f"{name} failed")
        return None

    monkeypatch.setattr(vmm, "_call", driver_call)
    table = TableHandles(128, 128, 0, (bytes(64), bytes(64)), 64)
    with pytest.raises(vmm.IncompleteTableMapCleanup, match="retain the owner lease"):
        vmm._map(table, [1, 2], 0, cpu_access=False)
    assert calls.count("cuMemMap") == 2
    assert calls.count("cuMemUnmap") == 1


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


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_external_table_views_follow_tp_head_ranges(tp_size):
    head_sizes = (3, 5, 7, 11)
    rows, dim, scale_width = sum(head_sizes), 64, 2
    weight = torch.arange(rows, dtype=torch.uint8)[:, None].expand(rows, dim)
    scales = (torch.arange(rows, dtype=torch.uint8) + 100)[:, None].expand(
        rows, scale_width
    )
    packed = torch.cat((weight.flatten(), scales.flatten()))
    heads_per_rank = (len(head_sizes) + tp_size - 1) // tp_size
    for rank in range(tp_size):
        head_start = rank * heads_per_rank
        head_end = min(head_start + heads_per_rank, len(head_sizes))
        row_start = sum(head_sizes[:head_start])
        row_end = sum(head_sizes[:head_end])
        selected_weight, selected_scales = external_table_views(
            packed, rows, dim, 32, row_start, row_end - row_start
        )
        torch.testing.assert_close(
            selected_weight.view(torch.uint8), weight[row_start:row_end]
        )
        torch.testing.assert_close(selected_scales, scales[row_start:row_end])
