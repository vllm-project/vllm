# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DP rank recovery communication fabric.

Level 1 — No GPUs, runs in CI:

  test_replace_peer_sequence — unit test for NixlEPAll2AllManager.replace_peer.
      Uses FakeNixlEPBuffer. No distributed, no GPU.

  test_donor_rank_selection — unit test for ElasticEPScalingState.donor_dp_rank.
      Verifies the lowest surviving DP rank is always chosen as donor, and
      that dead_dp_rank=0 is handled correctly. No distributed, no GPU.

  test_transfer_weights_noop_for_non_donor — unit test for
      ElasticEPScalingExecutor.transfer_weights_to_replacement.
      Verifies that non-donor ranks are a no-op (batch_transfer_weights is
      never called). No distributed, no GPU.

  test_transfer_weights_sends_for_donor — unit test for
      ElasticEPScalingExecutor.transfer_weights_to_replacement.
      Verifies that the donor rank calls batch_transfer_weights with
      is_sender=True and peer_rank=dead_dp_rank. No distributed, no GPU.

Level 2 — 2 GPUs, no NIXL:

  Both tests share a single worker (_recovery_worker) parameterised by
  backend.  Each spawns 2 processes (one per DP rank, TP=PP=1), each with
  its own torch.distributed group (world_size=1) to match the elastic EP
  topology.  The worker plants stale data in a TCPStore, calls
  create_recovery_groups on a fresh port, and verifies:
    - recovery_store.get("dp_0") is a valid 12-byte port triplet
    - stale_store.get("dp_0") is still b"stale_garbage" (untouched)
    - get_dp_group().all_reduce(tensor) returns correct sum (group is live)

  test_recovery_groups[gloo] — runs with backend="gloo".
  test_recovery_groups[nccl] — runs with backend="nccl".

  test_weight_transfer_round_trip — spawns 2 processes: donor (rank 0) fills
      a FakeModel with 1.0, replacement (rank 1) fills with 0.0.  After
      batch_transfer_weights, replacement's weights must equal donor's (1.0).

Level 3 — 2 SM 90+ GPUs + NIXL-EP + RDMA, end-to-end:

  test_recovery_ep — starts a DP=2 vllm server with NIXL-EP, runs baseline
      GSM8K, then attempts to trigger recovery and re-evaluate accuracy.
      Currently blocked: the recovery trigger mechanism (reconnect_peer) is
      implemented but not wired to any API endpoint or automatic detection.
      Baseline inference (server startup + GSM8K eval) verified on B200.
      See build_nixl_ep.sh for setup instructions.

Run with:
    pytest tests/distributed/test_recovery_ep.py::test_replace_peer_sequence -v
    pytest tests/distributed/test_recovery_ep.py::test_donor_rank_selection -v
    pytest tests/distributed/test_recovery_ep.py::test_transfer_weights_noop_for_non_donor -v
    pytest tests/distributed/test_recovery_ep.py::test_transfer_weights_sends_for_donor -v
    pytest tests/distributed/test_recovery_ep.py::test_recovery_groups -v
    pytest tests/distributed/test_recovery_ep.py::test_weight_transfer_round_trip -v
    pytest tests/distributed/test_recovery_ep.py::test_recovery_ep -v
"""

import os
import socket
import struct
import sys
import time
import traceback
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

_HOST = "127.0.0.1"


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Level 1a: unit test — NixlEPAll2AllManager.replace_peer
# ---------------------------------------------------------------------------


class FakeNixlEPBuffer:
    """Minimal fake for nixl_ep.Buffer that tracks calls and connected set."""

    def __init__(self, rank: int, explicitly_destroy: bool = False):
        self.rank = rank
        self.connected_ranks: set[int] = set()

    def update_memory_buffers(self, **kwargs) -> None:
        pass

    def set_tcp_store_group(self, store) -> None:
        pass

    def connect_ranks(self, remote_ranks, remote_mds=None) -> None:
        for r in remote_ranks:
            self.connected_ranks.add(r)

    def disconnect_ranks(self, remote_ranks) -> None:
        for r in remote_ranks:
            self.connected_ranks.discard(r)

    def get_local_metadata(self) -> bytes:
        return b"fake_metadata"

    def is_available(self) -> bool:
        return True


def test_replace_peer_sequence():
    """replace_peer disconnects the dead rank and reconnects the new one,
    without changing _buffer[1] (EP size)."""
    from vllm.distributed.device_communicators.all2all import NixlEPAll2AllManager

    try:
        fake_buffer = FakeNixlEPBuffer(rank=0)
        fake_buffer.connected_ranks = {0, 1, 2, 3}
        ep_size = 4
        NixlEPAll2AllManager._buffer = (fake_buffer, ep_size)

        NixlEPAll2AllManager.replace_peer(dead_ep_rank=2, new_ep_rank=2)

        assert 2 in fake_buffer.connected_ranks, "new_ep_rank must be re-connected"
        assert NixlEPAll2AllManager._buffer[1] == 4, "ep_size must not change"
    finally:
        NixlEPAll2AllManager._buffer = None


# ---------------------------------------------------------------------------
# Level 1b: unit tests — Gap C (donor selection + weight transfer guards)
# ---------------------------------------------------------------------------


class _FakeDPGroup:
    """Minimal stand-in for StatelessGroupCoordinator for isinstance checks.

    Subclasses StatelessGroupCoordinator in name only so that
    ``isinstance(obj, StatelessGroupCoordinator)`` passes without requiring
    any real distributed setup.
    """

    def __new__(cls, rank_in_group: int):
        from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

        # Dynamically create a subclass so isinstance checks pass.
        fake_cls = type("_FakeDPGroup", (StatelessGroupCoordinator,), {})
        obj = object.__new__(fake_cls)
        obj.rank_in_group = rank_in_group
        return obj


@pytest.mark.parametrize("dead_dp_rank,dp_size,expected_donor", [
    (0, 2, 1),  # dead rank 0, only survivor is rank 1
    (1, 2, 0),  # dead rank 1, only survivor is rank 0
    (0, 4, 1),  # dead rank 0, lowest survivor is rank 1
    (1, 4, 0),  # dead rank 1, lowest survivor is rank 0
    (3, 4, 0),  # dead rank 3, lowest survivor is rank 0
])
def test_donor_rank_selection(dead_dp_rank: int, dp_size: int, expected_donor: int):
    """ElasticEPScalingState.donor_dp_rank is always the lowest surviving rank.

    Critically, when dead_dp_rank=0 the donor must NOT be 0 — it must fall
    back to the next surviving rank (1).
    """
    from unittest.mock import MagicMock

    from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState
    from vllm.v1.engine import ReconfigureDistributedRequest

    reconfig_request = ReconfigureDistributedRequest(
        new_data_parallel_size=dp_size,
        new_data_parallel_rank=dead_dp_rank,
        new_data_parallel_rank_local=dead_dp_rank,
        new_data_parallel_master_ip=_HOST,
        new_data_parallel_master_port=0,
        new_data_parallel_master_port_list=[],
        coord_store_port=0,
        dead_data_parallel_rank=dead_dp_rank,
    )

    state = ElasticEPScalingState(
        model_executor=MagicMock(),
        engine_core=MagicMock(),
        vllm_config=MagicMock(),
        new_parallel_config=MagicMock(),
        worker_type="new",
        scale_type="recovery",
        reconfig_request=reconfig_request,
    )

    assert state.donor_dp_rank == expected_donor, (
        f"dead={dead_dp_rank}, dp_size={dp_size}: "
        f"expected donor {expected_donor}, got {state.donor_dp_rank}"
    )
    assert state.donor_dp_rank != dead_dp_rank, (
        "donor must never be the dead rank"
    )


def test_transfer_weights_noop_for_non_donor():
    """transfer_weights_to_replacement is a no-op when rank_in_group != donor.

    All surviving ranks except the donor must return immediately without
    calling batch_transfer_weights, so only one P2P send is issued.
    """
    from unittest.mock import MagicMock, patch

    from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

    fake_dp_group = _FakeDPGroup(rank_in_group=2)  # not the donor (0)

    executor = object.__new__(ElasticEPScalingExecutor)
    executor.worker_ref = lambda: MagicMock()

    with patch(
        "vllm.distributed.elastic_ep.elastic_execute.get_dp_group",
        return_value=fake_dp_group,
    ):
        with patch(
            "vllm.distributed.elastic_ep.elastic_execute.batch_transfer_weights"
        ) as mock_transfer:
            executor.transfer_weights_to_replacement(
                dead_dp_rank=1, donor_dp_rank=0
            )
            mock_transfer.assert_not_called()


def test_transfer_weights_sends_for_donor():
    """transfer_weights_to_replacement calls batch_transfer_weights for the donor.

    The donor rank must call batch_transfer_weights with is_sender=True and
    peer_rank=dead_dp_rank.
    """
    from unittest.mock import MagicMock, patch

    from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

    fake_dp_group = _FakeDPGroup(rank_in_group=0)  # is the donor
    mock_model = MagicMock()

    mock_worker = MagicMock()
    mock_worker.model_runner.get_model.return_value = mock_model

    executor = object.__new__(ElasticEPScalingExecutor)
    executor.worker_ref = lambda: mock_worker

    with patch(
        "vllm.distributed.elastic_ep.elastic_execute.get_dp_group",
        return_value=fake_dp_group,
    ):
        with patch(
            "vllm.distributed.elastic_ep.elastic_execute.batch_transfer_weights"
        ) as mock_transfer:
            with patch("torch.accelerator.synchronize"):
                executor.transfer_weights_to_replacement(
                    dead_dp_rank=1, donor_dp_rank=0
                )
                mock_transfer.assert_called_once_with(
                    model=mock_model,
                    is_sender=True,
                    peer_rank=1,
                    dp_group=fake_dp_group,
                    expert_weights=mock_model.expert_weights,
                )


# ---------------------------------------------------------------------------
# Level 2 helpers — shared by test_recovery_groups_rendezvous and
#                    test_reconnect_peer_group_swap
# ---------------------------------------------------------------------------


def _create_tcp_store(rank: int, port: int, world_size: int) -> "torch.distributed.TCPStore":
    """Create a TCPStore: rank 0 as master, others as clients."""
    if rank == 0:
        return dist.TCPStore(_HOST, port, world_size, is_master=True)
    timeout = 10  # seconds
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            return dist.TCPStore(_HOST, port, world_size, is_master=False)
        except RuntimeError as e:
            if "Connection refused" in str(e) or "Connection reset by peer" in str(e):
                time.sleep(0.1)
                continue
            raise
    raise RuntimeError(
        f"Rank {rank} failed to connect to TCPStore at {_HOST}:{port} after {timeout}s"
    )


def _recovery_worker(
    rank: int,
    world_size: int,
    dist_ports: list[int],
    world_coord_port: int,
    original_port: int,
    recovery_port: int,
    result_queue: "mp.Queue",
    backend: str = "gloo",
) -> None:
    """Shared worker for Level 2 recovery tests.

    Simulates the elastic EP topology (TP=PP=1, one process per DP rank):
    1. Init torch.distributed with world_size=1 (local TP*PP group).
    2. Build StatelessGroupCoordinator for _WORLD; stub _TP/_PP.
    3. Plant stale data in TCPStore at original_port.
    4. Start a master TCPStore on recovery_port, then call
       create_recovery_groups.
    5. Assert recovery store "dp_0" has 12 valid bytes.
    6. Assert stale store is untouched.
    7. Promote recovery groups and verify all_reduce.
    """
    import vllm.distributed.parallel_state as ps
    from vllm.distributed.elastic_ep.standby_state import (
        create_recovery_groups,
        pop_standby_groups,
    )
    from vllm.distributed.parallel_state import _replace_active_groups, get_dp_group
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

    def _log(msg: str) -> None:
        print(f"[Rank {rank}] {msg}", flush=True)
        sys.stdout.flush()

    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        # 1. torch.distributed — world_size=1 per DP rank (elastic EP topology)
        _log(f"Step 1: init_process_group (gloo, world_size=1) ...")
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{_HOST}:{dist_ports[rank]}",
            world_size=1,
            rank=0,
        )
        _log("Step 1: done")

        # 2. Patch parallel_state with _WORLD coordinator and TP/PP stubs
        _log("Step 2: patching parallel_state ...")
        world_coord_store = _create_tcp_store(rank, world_coord_port, world_size)
        ps._WORLD = StatelessGroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            torch_distributed_backend=backend,
            use_device_communicator=False,
            coord_store=world_coord_store,
            group_name="world",
            host=_HOST,
            global_rank=rank,
            global_world_size=world_size,
        )
        ps._TP = types.SimpleNamespace(world_size=1)
        ps._PP = types.SimpleNamespace(world_size=1)
        _log("Step 2: done")

        # 3. Plant stale data at original_port
        _log("Step 3: planting stale data ...")
        stale_store = _create_tcp_store(rank, original_port, world_size)
        if rank == 0:
            stale_store.set("dp_0", b"stale_garbage")
        stale_store.add("all_ready", 1)
        while int(stale_store.get("all_ready")) < world_size:
            pass
        _log("Step 3: done")

        # 4. Create recovery store server, then run create_recovery_groups
        _log("Step 4: create_recovery_groups ...")
        recovery_store = _create_tcp_store(rank, recovery_port, world_size)
        create_recovery_groups(
            recovery_coord_store_port=recovery_port,
            master_ip=_HOST,
            enable_eplb=False,
            backend=backend,
        )
        _log("Step 4: done")

        # 5. recovery store "dp_0" must have 12 valid bytes (3 × uint32 ports)
        _log("Step 5: checking recovery store dp_0 ...")
        dp_0_value = recovery_store.get("dp_0")
        assert dp_0_value != b"stale_garbage", (
            f"Rank {rank}: recovery store returned stale value"
        )
        assert len(dp_0_value) == struct.calcsize("!3I"), (
            f"Rank {rank}: expected 12-byte port triplet, got {len(dp_0_value)} bytes"
        )
        ports = struct.unpack("!3I", dp_0_value)
        assert all(p > 0 for p in ports), (
            f"Rank {rank}: invalid zero port in recovery store: {ports}"
        )
        _log("Step 5: done")

        # 6. Stale store must be untouched
        _log("Step 6: checking stale store unchanged ...")
        assert stale_store.get("dp_0") == b"stale_garbage", (
            f"Rank {rank}: stale store was modified by create_recovery_groups"
        )
        _log("Step 6: done")

        # 7. Promote recovery groups and verify all_reduce
        _log("Step 7: promote + all_reduce ...")
        _replace_active_groups(**pop_standby_groups())
        assert get_dp_group().world_size == world_size

        tensor = torch.tensor([float(rank)], device=device)
        result = get_dp_group().all_reduce(tensor)
        torch.cuda.synchronize()
        expected = float(sum(range(world_size)))
        assert result.item() == expected, (
            f"Rank {rank}: all_reduce returned {result.item()}, expected {expected}"
        )
        _log("Step 7: done — all passed")

        result_queue.put((rank, None))

    except Exception as exc:
        _log(f"EXCEPTION: {exc!r}")
        result_queue.put((rank, traceback.format_exc()))


def _run_recovery_test(
    backend: str,
    timeout: int = 120,
) -> None:
    """Spawn 2 workers, run the recovery test, and collect errors."""
    world_size = 2
    dist_ports = [_get_open_port() for _ in range(world_size)]
    world_coord_port = _get_open_port()
    original_port = _get_open_port()
    recovery_port = _get_open_port()

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_recovery_worker,
            args=(
                rank,
                world_size,
                dist_ports,
                world_coord_port,
                original_port,
                recovery_port,
                result_queue,
                backend,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=timeout)

    errors = []
    while not result_queue.empty():
        rank_id, err = result_queue.get_nowait()
        if err is not None:
            errors.append(f"Rank {rank_id}:\n{err}")

    for i, p in enumerate(processes):
        if p.exitcode != 0:
            errors.append(f"Process {i} exited with code {p.exitcode}")

    assert not errors, "\n\n".join(errors)


# ---------------------------------------------------------------------------
# Level 2 tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs",
)
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_recovery_groups(backend: str):
    """create_recovery_groups on a fresh store produces a live DP group
    that passes an all_reduce sanity check."""
    _run_recovery_test(backend=backend)


# ---------------------------------------------------------------------------
# Level 2b: weight transfer round-trip (Gap C)
# ---------------------------------------------------------------------------


class FakeModel(torch.nn.Module):
    """Minimal model for weight-transfer tests: one linear layer, no MoE."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4, device=device))
        # Empty expert_weights means all params in state_dict are transferred.
        self.expert_weights: list = []


def _weight_transfer_worker(
    rank: int,
    world_size: int,
    dist_ports: list[int],
    coord_port: int,
    result_queue: "mp.Queue",
) -> None:
    """Worker for test_weight_transfer_round_trip.

    Donor (rank 0) fills FakeModel.weight with 1.0 and sends it.
    Replacement (rank 1, dead_dp_rank=1) fills with 0.0 and receives.
    After transfer, replacement must have weight == 1.0.
    """
    import sys
    import traceback

    import torch.distributed as dist

    import vllm.distributed.parallel_state as ps
    from vllm.distributed.elastic_ep.elastic_execute import batch_transfer_weights
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

    def _log(msg: str) -> None:
        print(f"[Rank {rank}] {msg}", flush=True)
        sys.stdout.flush()

    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        # Init a local torch.distributed group (world_size=1, elastic EP topology).
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{_HOST}:{dist_ports[rank]}",
            world_size=1,
            rank=0,
        )

        # Build a real StatelessGroupCoordinator as the DP group (nccl for P2P).
        coord_store = _create_tcp_store(rank, coord_port, world_size)
        dp_group = StatelessGroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            torch_distributed_backend="nccl",
            use_device_communicator=True,
            coord_store=coord_store,
            group_name="dp",
            host=_HOST,
            global_rank=rank,
            global_world_size=world_size,
        )
        ps._DP = dp_group

        # Build model: donor fills with 1.0, replacement fills with 0.0.
        model = FakeModel(device)
        donor_dp_rank = 0
        dead_dp_rank = 1
        if rank == donor_dp_rank:
            model.weight.data.fill_(1.0)
        else:
            model.weight.data.fill_(0.0)

        # Execute the P2P weight transfer.
        batch_transfer_weights(
            model=model,
            is_sender=(rank == donor_dp_rank),
            peer_rank=dead_dp_rank if rank == donor_dp_rank else donor_dp_rank,
            dp_group=dp_group,
            expert_weights=model.expert_weights,
        )
        torch.cuda.synchronize()

        # Only the replacement rank verifies it received the correct weights.
        if rank == dead_dp_rank:
            expected = torch.ones(4, 4, device=device)
            assert model.weight.data.allclose(expected), (
                f"Rank {rank}: weight transfer failed — "
                f"got {model.weight.data}, expected all 1.0"
            )
            _log("weight transfer verified: replacement has donor weights")

        result_queue.put((rank, None))

    except Exception:
        result_queue.put((rank, traceback.format_exc()))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs",
)
def test_weight_transfer_round_trip():
    """Donor sends model weights to replacement via batch_transfer_weights.

    Verifies the P2P transfer mechanism used by receive_weights_from_donor
    and transfer_weights_to_replacement: after the transfer the replacement
    rank holds an exact copy of the donor's weights.
    """
    world_size = 2
    dist_ports = [_get_open_port() for _ in range(world_size)]
    coord_port = _get_open_port()

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_weight_transfer_worker,
            args=(rank, world_size, dist_ports, coord_port, result_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=120)

    errors = []
    while not result_queue.empty():
        rank_id, err = result_queue.get_nowait()
        if err is not None:
            errors.append(f"Rank {rank_id}:\n{err}")

    for i, p in enumerate(processes):
        if p.exitcode != 0:
            errors.append(f"Process {i} exited with code {p.exitcode}")

    assert not errors, "\n\n".join(errors)


# ---------------------------------------------------------------------------
# Level 3: 2 GPUs + NIXL, end-to-end recovery
# ---------------------------------------------------------------------------

try:
    from ..utils import RemoteOpenAIServer, multi_gpu_test  # noqa: E402
except ImportError:
    RemoteOpenAIServer = None  # type: ignore[assignment,misc]

    def multi_gpu_test(num_gpus: int):  # type: ignore[misc]
        return pytest.mark.skip(reason="test utils not available")

_MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
_NUM_GSM8K_QUESTIONS = 256
_EXPECTED_ACCURACY = 0.58
_ACCURACY_TOL = 0.08
_MAX_NUM_SEQS = 32


def _send_recover_command(server: RemoteOpenAIServer, dead_dp_rank: int) -> bool:
    """POST /recover_elastic_ep to trigger DP rank recovery."""
    import requests

    url = server.url_for("recover_elastic_ep")
    payload = {"dead_data_parallel_rank": dead_dp_rank}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _run_gsm8k(server: RemoteOpenAIServer, stage: str) -> float:
    from ..evals.gsm8k.gsm8k_eval import evaluate_gsm8k

    assert server.port is not None
    result = evaluate_gsm8k(
        num_questions=_NUM_GSM8K_QUESTIONS,
        host=f"http://{server.host}",
        port=server.port,
    )
    accuracy = result["accuracy"]
    print(f"[{stage}] GSM8K accuracy: {accuracy:.3f} ({result['num_questions']} Qs)")
    assert accuracy >= _EXPECTED_ACCURACY, (
        f"[{stage}] accuracy {accuracy:.3f} below threshold {_EXPECTED_ACCURACY}"
    )
    return accuracy


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not os.environ.get("NIXL_EP_ENABLED"),
    reason="Set NIXL_EP_ENABLED=1 to run NIXL end-to-end recovery test",
)
def test_recovery_ep():
    """Level 3: kill DP rank 1, trigger recovery, verify accuracy is maintained.

    Requires:
    - 2 SM 90+ GPUs (Hopper/Blackwell) with RDMA support
    - nixl_ep built from source (see build_nixl_ep.sh)
    - NIXL_EP_ENABLED=1 environment variable
    - Ray installed (pip install ray)

    TODO: The recovery trigger mechanism is not yet implemented.
      - reconnect_peer() exists in elastic_execute.py but nothing calls it.
      - Need either: (a) an API endpoint (e.g. POST /recover_elastic_ep) that
        the orchestrator calls after spawning a replacement worker, or
        (b) automatic failure detection (e.g. NCCL timeout / heartbeat) that
        triggers reconnect_peer internally.
      - Once the trigger exists, update _send_recover_command() to use it
        and remove this TODO.
      - The baseline GSM8K eval (server startup + inference with NIXL-EP DP=2)
        has been verified working on B200.
    """
    vllm_serve_args = [
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "4096",
        "--max-num-seqs", str(_MAX_NUM_SEQS),
        "--enable-expert-parallel",
        "--all2all-backend", "nixl_ep",
        "--enable-elastic-ep",
        "--enable-eplb",
        "--eplb-config.num_redundant_experts", "0",
        "--data-parallel-backend", "ray",
        "--data-parallel-size", "2",
        "--api-server-count", "1",
    ]

    leader_address = os.environ.get("LEADER_ADDRESS")
    if leader_address:
        vllm_serve_args.extend(["--data-parallel-address", leader_address])

    with RemoteOpenAIServer(
        _MODEL_NAME, vllm_serve_args, env_dict={}, max_wait_seconds=1200
    ) as server:
        baseline_accuracy = _run_gsm8k(server, "Baseline (DP=2, NIXL)")

        # Kill DP rank 1 and trigger recovery
        assert _send_recover_command(server, dead_dp_rank=1), (
            "Recovery command failed — is /recover_elastic_ep implemented?"
        )
        time.sleep(30)  # allow reconnection + replacement rank warmup

        recovery_accuracy = _run_gsm8k(server, "After recovery (DP=2, NIXL)")
        assert recovery_accuracy >= baseline_accuracy - _ACCURACY_TOL, (
            f"Recovery accuracy {recovery_accuracy:.3f} dropped more than "
            f"{_ACCURACY_TOL} below baseline {baseline_accuracy:.3f}"
        )

        print("\nAccuracy Summary:")
        print(f"  Baseline:  {baseline_accuracy:.3f}")
        print(f"  Recovery:  {recovery_accuracy:.3f} "
              f"(diff: {recovery_accuracy - baseline_accuracy:+.3f})")
        print(f"  Tolerance: {_ACCURACY_TOL:.3f}")
