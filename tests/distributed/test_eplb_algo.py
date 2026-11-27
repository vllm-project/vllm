# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import signal
import socket
import struct
import sys
import threading
import time
from contextlib import closing, suppress
from multiprocessing import get_context

import numpy as np
import psutil
import pytest
import torch

from vllm.distributed.eplb.eplb_state import (
    _compute_eplb_load_stats,  # pyright: ignore[reportPrivateUsage]
    compute_logical_maps,
)
from vllm.distributed.eplb.policy import EplbStats
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy
from vllm.distributed.eplb.policy_process import (
    EplbPolicyProcess,
    EplbPolicyProcessError,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "layers, experts, replicas, groups, nodes, ranks",
    [(2, 12, 16, 4, 2, 8), (4, 128, 132, 1, 1, 2), (58, 256, 288, 1, 2, 8)],
)
def test_policy_process_preserves_exact_placements(
    layers, experts, replicas, groups, nodes, ranks
):
    """IPC preserves placement and local-slot retention across planning cycles."""
    generator = torch.Generator().manual_seed(17)
    policy = DefaultEplbPolicy()
    old_map = None
    with closing(EplbPolicyProcess(policy)) as process:
        for load in (
            torch.zeros(layers, experts),
            torch.ones(layers, experts),
            torch.randint(0, 1000, (layers, experts), generator=generator),
        ):
            stats = EplbStats(load, replicas, groups, nodes, ranks)
            expected = policy.rebalance_experts(stats, old_map)
            actual = process.rebalance_experts(stats, old_map)
            assert torch.equal(actual, expected)
            assert actual.device.type == "cpu"
            old_map = actual


class _CountingPolicy(DefaultEplbPolicy):
    def __init__(self):
        self.calls = 0

    def rebalance_experts(self, stats, old_global_expert_indices=None):
        self.calls += 1
        return torch.tensor([os.getpid(), self.calls, torch.get_num_threads()])


class _LockedPolicy(_CountingPolicy):
    def __init__(self, calls=7):
        self.calls = calls
        self.lock = threading.Lock()

    def __getstate__(self):
        return self.calls

    def __setstate__(self, calls):
        self.calls = calls
        self.lock = threading.Lock()

    def rebalance_experts(self, stats, old_global_expert_indices=None):
        with self.lock:
            return super().rebalance_experts(stats, old_global_expert_indices)


def test_policy_process_recreates_private_resources_and_preserves_state():
    """A custom policy recreates its lock without losing its initial CPU state."""
    policy = _LockedPolicy()
    stats = EplbStats(torch.ones(1, 4), 4, 1, 1, 1)
    with closing(EplbPolicyProcess(policy)) as process:
        first = process.rebalance_experts(stats).tolist()
        second = process.rebalance_experts(stats).tolist()
        assert first == [second[0], 8, 1]
        assert second == [first[0], 9, 1]
        assert first[0] != os.getpid()
    assert policy.calls == 7
    assert not psutil.pid_exists(first[0])


def test_policy_process_does_not_initialize_accelerators_or_distributed(monkeypatch):
    """CPU planning must not create an inference device context or process group."""
    from vllm.platforms.interface import Platform, PlatformEnum

    class FakeAcceleratorPlatform(Platform):
        _enum = PlatformEnum.CUDA

    monkeypatch.setattr(
        sys.modules[__name__], "current_platform", FakeAcceleratorPlatform()
    )

    class Policy(DefaultEplbPolicy):
        def rebalance_experts(self, stats, old_global_expert_indices=None):
            import vllm.platforms as platforms

            return torch.tensor(
                [
                    torch.cuda.is_initialized(),
                    torch.xpu.is_initialized(),
                    torch.distributed.is_initialized(),
                    not platforms.current_platform.is_cpu(),
                    os.environ.get("VLLM_TARGET_DEVICE") != "cpu",
                ]
            )

    with closing(EplbPolicyProcess(Policy())) as process:
        result = process.rebalance_experts(EplbStats(torch.ones(1, 4), 4, 1, 1, 1))
    assert not result.any()


def test_policy_process_ignores_shadowing_cwd_package(tmp_path, monkeypatch):
    """Launching from a directory with another vllm/ tree keeps parent's sys.path."""
    shadow_pkg = tmp_path / "vllm"
    shadow_pkg.mkdir()
    (shadow_pkg / "__init__.py").write_text(
        'raise RuntimeError("shadowed vllm imported from cwd")\n',
        encoding="utf-8",
    )
    clean_sys_path = [
        p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(tmp_path)
    ]
    monkeypatch.setattr(sys, "path", clean_sys_path)
    monkeypatch.chdir(tmp_path)

    with closing(EplbPolicyProcess(DefaultEplbPolicy())) as process:
        result = process.rebalance_experts(EplbStats(torch.ones(1, 4), 4, 1, 1, 1))
    assert result.shape == (1, 4)


def test_policy_process_ignores_sigint():
    """Terminal Ctrl+C is handled by the parent without killing the planner first."""
    stats = EplbStats(torch.ones(1, 4), 4, 1, 1, 1)
    with closing(EplbPolicyProcess(_CountingPolicy())) as process:
        first = process.rebalance_experts(stats).tolist()
        os.kill(first[0], signal.SIGINT)
        time.sleep(0.05)
        second = process.rebalance_experts(stats).tolist()
        assert second == [first[0], 2, 1]


def test_policy_process_explains_incompatible_policy_before_starting_child():
    policy = _CountingPolicy()
    policy.lock = threading.Lock()
    children = {child.pid for child in psutil.Process().children()}
    with pytest.raises(TypeError, match="__getstate__/__setstate__ or __reduce__"):
        EplbPolicyProcess(policy)
    assert {child.pid for child in psutil.Process().children()} == children


def test_policy_process_timeout_includes_blocked_request_send():
    """A stopped child cannot leave a large snapshot blocked in send forever."""
    with closing(EplbPolicyProcess(_CountingPolicy())) as process:
        stats = EplbStats(torch.ones(1, 4), 4, 1, 1, 1)
        pid = process.rebalance_experts(stats)[0].item()
        process._timeout = 0.2
        os.kill(pid, signal.SIGSTOP)
        finished = threading.Event()
        errors = []

        def request():
            try:
                process.rebalance_experts(
                    EplbStats(torch.ones(1, 1024 * 1024), 4, 1, 1, 1)
                )
            except EplbPolicyProcessError as error:
                errors.append(error)
            finally:
                finished.set()

        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        try:
            assert finished.wait(15), "request remained blocked past its IPC budget"
            assert len(errors) == 1 and "timed out" in str(errors[0])
        finally:
            if psutil.pid_exists(pid):
                os.kill(pid, signal.SIGCONT)
            worker.join(timeout=15)
        assert not worker.is_alive()
        assert not psutil.pid_exists(pid)


def test_policy_process_timeout_includes_partial_response():
    """Read readiness does not imply that the entire response has arrived."""
    from vllm.distributed.eplb.policy_process import _receive

    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(struct.pack("!Q", 1024) + b"partial")
        with pytest.raises(TimeoutError):
            _receive(reader, time.monotonic() + 0.1)


class _OwnerDeathPolicy(DefaultEplbPolicy):
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def rebalance_experts(self, stats, old_global_expert_indices=None):
        with socket.create_connection(self.endpoint, timeout=20) as connection:
            connection.sendall(struct.pack("!Q", os.getpid()))
            threading.Event().wait()


def _plan_until_owner_exits(endpoint):
    with closing(EplbPolicyProcess(_OwnerDeathPolicy(endpoint))) as process:
        process.rebalance_experts(EplbStats(torch.ones(1, 4), 4, 1, 1, 1))


def test_policy_process_exits_if_owner_is_killed_during_planning():
    """A forced executor shutdown must not leave an orphaned, blocked policy."""
    child = None
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(30)
        owner = get_context("spawn").Process(
            target=_plan_until_owner_exits, args=(listener.getsockname(),), daemon=True
        )
        owner.start()
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(10)
                payload = bytearray()
                while len(payload) < 8:
                    chunk = connection.recv(8 - len(payload))
                    assert chunk, "planner closed before publishing its PID"
                    payload.extend(chunk)
                child = psutil.Process(struct.unpack("!Q", payload)[0])
                owner.kill()
                owner.join(timeout=5)
                assert owner.exitcode is not None
                try:
                    child.wait(timeout=5)
                except psutil.TimeoutExpired:
                    assert child.status() == psutil.STATUS_ZOMBIE
        finally:
            if owner.is_alive():
                owner.kill()
                owner.join(timeout=5)
            if child is not None:
                with suppress(psutil.NoSuchProcess):
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        child.kill()
                        child.wait(timeout=5)


def _plan_from_daemon(connection):
    try:
        with closing(EplbPolicyProcess(_CountingPolicy())) as process:
            stats = EplbStats(torch.ones(1, 4), 4, 1, 1, 1)
            first = process.rebalance_experts(stats).tolist()
            second = process.rebalance_experts(stats).tolist()
        connection.send((os.getpid(), first, second, psutil.pid_exists(first[0])))
    finally:
        connection.close()


def test_policy_process_runs_persistently_inside_daemon_worker():
    """A daemon executor can reuse and reap its independently launched planner."""
    context = get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=_plan_from_daemon, args=(child,), daemon=True)
    worker.start()
    child.close()
    try:
        assert parent.poll(60), "daemon worker did not return a policy result"
        worker_pid, first, second, child_alive = parent.recv()
        worker.join(timeout=10)
        assert worker.exitcode == 0
        assert first == [second[0], 1, 1]
        assert second == [first[0], 2, 1]
        assert first[0] != worker_pid
        assert not child_alive
    finally:
        parent.close()
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=10)


def test_policy_process_reports_policy_exception():
    """A failed policy is reported instead of leaving IPC waiting forever."""
    with (
        closing(EplbPolicyProcess(DefaultEplbPolicy())) as process,
        pytest.raises(RuntimeError, match="AssertionError"),
    ):
        process.rebalance_experts(EplbStats(torch.ones(1, 4), 3, 1, 1, 1))


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_policy_process_reports_unresponsive_child(failure):
    """Child death and a stuck policy both terminate the planning request."""

    class Policy(DefaultEplbPolicy):
        def rebalance_experts(self, stats, old_global_expert_indices=None):
            if failure == "exit":
                os._exit(7)
            threading.Event().wait()

    message = "disconnected" if failure == "exit" else "timed out"
    with closing(EplbPolicyProcess(Policy())) as process:
        process._timeout = 0.2
        with pytest.raises(EplbPolicyProcessError, match=message):
            process.rebalance_experts(EplbStats(torch.ones(1, 4), 4, 1, 1, 1))


def test_eplb_load_stats_reduce_across_ranks():
    num_tokens_per_rank = torch.tensor(
        [
            [100, 0, 0, 0],
            [100, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    avg_tokens, max_tokens = _compute_eplb_load_stats(num_tokens_per_rank)

    assert avg_tokens.item() == 50
    assert max_tokens.item() == 200
    assert (avg_tokens / max_tokens).item() == 0.25


def test_basic_rebalance():
    """Test basic rebalancing functionality."""
    # Example from https://github.com/deepseek-ai/eplb
    weight = torch.tensor(
        [
            [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
            [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
        ]
    )

    num_layers = weight.shape[0]
    num_replicas = 16
    num_groups = 4
    num_nodes = 2
    num_gpus = 8

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    log2phy, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Verify output shapes
    assert phy2log.shape == (
        2,
        16,
    ), f"Expected `phy2log` shape (2, 16), got {phy2log.shape}"
    assert log2phy.shape[0] == 2, (
        f"Expected `log2phy` first dimension 2, got {log2phy.shape[0]}"
    )
    assert log2phy.shape[1] == 12, (
        f"Expected `log2phy` second dimension 12, got {log2phy.shape[1]}"
    )
    assert logcnt.shape == (
        2,
        12,
    ), f"Expected `logcnt` shape (2, 12), got {logcnt.shape}"

    # Verify physical to logical expert mapping range is correct
    assert torch.all(phy2log >= 0) and torch.all(phy2log < 12), (
        "Physical to logical mapping should be in range [0, 12)"
    )

    # Verify expert count reasonableness
    assert torch.all(logcnt >= 1), "Each logical expert should have at least 1 replica"
    assert torch.sum(logcnt, dim=1).sum() == num_replicas * num_layers, (
        f"Total replicas should be {num_replicas * num_layers}"
    )

    # Verify expected output
    expected_phy2log = torch.tensor(
        [
            [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
            [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1],
        ]
    )
    assert torch.all(phy2log == expected_phy2log)

    expected_logcnt = torch.tensor(
        [[1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1], [1, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1]]
    )
    assert torch.all(logcnt == expected_logcnt)


def test_single_gpu_case():
    """Test single GPU case."""
    weight = torch.tensor([[10, 20, 30, 40]])
    num_replicas = 4
    num_groups = 1
    num_nodes = 1
    num_gpus = 1

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    log2phy, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Verify shapes
    assert phy2log.shape == (1, 4)
    assert log2phy.shape[0] == 1
    assert log2phy.shape[1] == 4
    assert logcnt.shape == (1, 4)

    # Verify all logical experts are mapped
    assert set(phy2log[0].tolist()) == {0, 1, 2, 3}


def test_equal_weights():
    """Test case with equal weights."""
    weight = torch.tensor([[50, 50, 50, 50, 50, 50, 50, 50]])
    num_replicas = 8
    num_groups = 2
    num_nodes = 2
    num_gpus = 4

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Verify shapes
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 8)

    # With equal weights, each expert should have exactly one replica
    assert torch.all(logcnt == 1), (
        "With equal weights and no replication, "
        "each expert should have exactly 1 replica"
    )


def test_extreme_weight_imbalance():
    """Test extreme weight imbalance case."""
    weight = torch.tensor([[1000, 1, 1, 1, 1, 1, 1, 1]])
    num_replicas = 12
    num_groups = 2
    num_nodes = 2
    num_gpus = 4

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Verify shapes
    assert phy2log.shape == (1, 12)
    assert logcnt.shape == (1, 8)

    # Expert with highest weight (index 0) should have more replicas
    assert logcnt[0, 0] > logcnt[0, 1], (
        "Expert with highest weight should have more replicas"
    )


def test_multiple_layers():
    """Test multiple layers case."""
    weight = torch.tensor(
        [
            [10, 20, 30, 40, 50, 60],  # First layer
            [60, 50, 40, 30, 20, 10],  # Second layer (opposite weight pattern)
            [25, 25, 25, 25, 25, 25],  # Third layer (equal weights)
        ]
    )
    num_replicas = 8
    num_groups = 2
    num_nodes = 2
    num_gpus = 4

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Verify shapes
    assert phy2log.shape == (3, 8)
    assert logcnt.shape == (3, 6)

    # Verify expert allocation is reasonable for each layer
    for layer in range(3):
        assert torch.all(phy2log[layer] >= 0) and torch.all(phy2log[layer] < 6), (
            f"Layer {layer} physical to logical mappingshould be in range [0, 6)"
        )
        assert torch.sum(logcnt[layer]) == num_replicas, (
            f"Layer {layer} total replicas should be {num_replicas}"
        )


def test_parameter_validation():
    """Test parameter validation."""
    weight = torch.tensor([[10, 20, 30, 40]])

    # Test non-divisible case - this should handle normally without throwing
    # errors because the function will fall back to global load balancing
    # strategy
    phy2log = DefaultEplbPolicy().rebalance_experts(EplbStats(weight, 8, 3, 2, 4))
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 4)

    # Test cases that will actually cause errors:
    # num_physical_experts not divisible by num_gpus
    with pytest.raises(AssertionError):
        DefaultEplbPolicy().rebalance_experts(
            EplbStats(weight, 7, 2, 2, 4)
        )  # 7 not divisible by 4


def test_small_scale_hierarchical():
    """Test small-scale hierarchical load balancing."""
    weight = torch.tensor(
        [
            [100, 50, 200, 75, 150, 25, 300, 80],  # 8 experts
        ]
    )
    num_replicas = 12
    num_groups = 4  # 4 groups, 2 experts each
    num_nodes = 2  # 2 nodes
    num_gpus = 4  # 4 GPUs

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Verify basic constraints
    assert phy2log.shape == (1, 12)
    assert logcnt.shape == (1, 8)
    assert torch.sum(logcnt) == num_replicas
    assert torch.all(logcnt >= 1)

    # Expert with highest weight should have more replicas
    max_weight_expert = torch.argmax(weight[0])
    assert logcnt[0, max_weight_expert] >= 2, (
        "Highest weight expert should have multiple replicas"
    )


def test_global_load_balance_fallback():
    """Test global load balancing fallback case."""
    # When num_groups % num_nodes != 0, should fall back to global load
    # balancing
    weight = torch.tensor([[10, 20, 30, 40, 50, 60]])
    num_replicas = 8
    num_groups = 3  # Cannot be divided evenly by num_nodes=2
    num_nodes = 2
    num_gpus = 4

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Should work normally, just using global load balancing strategy
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 6)
    assert torch.sum(logcnt) == num_replicas


@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_device_compatibility(device):
    """Test device compatibility."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "xpu" and not current_platform.is_xpu():
        pytest.skip("XPU not available")

    weight = torch.tensor([[10, 20, 30, 40]], device=device)
    num_replicas = 6
    num_groups = 2
    num_nodes = 1
    num_gpus = 2

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    _, logcnt = compute_logical_maps(phy2log, weight.shape[-1])

    # Function will convert to CPU internally, but should handle different
    # device inputs normally
    assert phy2log.shape == (1, 6)
    assert logcnt.shape == (1, 4)


def test_additional_cases():
    """Test more edge cases and different parameter combinations."""
    # Test case 1: Large-scale distributed setup
    weight1 = torch.tensor(
        [[50, 100, 75, 120, 90, 60, 80, 110, 40, 70, 95, 85, 65, 55, 45, 35]]
    )
    phy2log1 = DefaultEplbPolicy().rebalance_experts(EplbStats(weight1, 24, 8, 4, 8))
    _, logcnt1 = compute_logical_maps(phy2log1, weight1.shape[-1])

    assert phy2log1.shape == (1, 24)
    assert logcnt1.shape == (1, 16)
    assert torch.sum(logcnt1) == 24

    # Test case 2: Different weight distributions
    weight2 = torch.tensor(
        [
            [200, 150, 100, 50, 25, 12],  # Decreasing weights
            [12, 25, 50, 100, 150, 200],  # Increasing weights
        ]
    )
    phy2log2 = DefaultEplbPolicy().rebalance_experts(EplbStats(weight2, 10, 3, 1, 2))
    _, logcnt2 = compute_logical_maps(phy2log2, weight2.shape[-1])

    assert phy2log2.shape == (2, 10)
    assert logcnt2.shape == (2, 6)

    # Verify high-weight experts have more replicas
    for layer in range(2):
        max_weight_idx = torch.argmax(weight2[layer])
        assert logcnt2[layer, max_weight_idx] >= 2


def test_compute_logical_maps_with_negative_indices():
    """Test that compute_logical_maps correctly handles physical slots containing
    -1 (unused slots).
    """
    # 2 layers, 6 physical slots, 4 logical experts.
    # Slots 2 and 5 are unused (-1).
    phy2log = torch.tensor(
        [
            [0, 1, -1, 2, 3, -1],
            [3, -1, 2, 1, 0, -1],
        ]
    )
    num_layers = 2
    num_logical_experts = 4

    log2phy, logcnt = compute_logical_maps(phy2log, num_logical_experts)

    assert logcnt.shape == (num_layers, num_logical_experts)
    assert log2phy.shape == (num_layers, num_logical_experts, 1)

    expected_logcnt = torch.ones(num_layers, num_logical_experts, dtype=phy2log.dtype)
    assert torch.all(logcnt == expected_logcnt), (
        f"Expected that all replica counts == 1, got {logcnt}"
    )

    assert torch.all(log2phy >= 0), (
        "log2phy should only contain valid physical indices, not -1"
    )

    assert log2phy[0, 0, 0] == 0
    assert log2phy[0, 1, 0] == 1
    assert log2phy[0, 2, 0] == 3
    assert log2phy[0, 3, 0] == 4


if __name__ == "__main__":
    weight = torch.tensor(
        [
            [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
            [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
        ]
    )

    num_replicas = 16
    num_groups = 4
    num_nodes = 2
    num_gpus = 8

    phy2log = DefaultEplbPolicy().rebalance_experts(
        EplbStats(weight, num_replicas, num_groups, num_nodes, num_gpus)
    )
    print(phy2log)

    test_basic_rebalance()


def _make_phy_replicas_idx_from_phy2log(phy2log: np.ndarray) -> np.ndarray:
    """Create replicas indices mapping from phy2log."""
    pr = np.zeros_like(phy2log, dtype=np.int64)
    for layer in range(phy2log.shape[0]):
        seen: dict[int, int] = {}
        row = phy2log[layer].tolist()
        for i, expert in enumerate(row):
            r = seen.get(expert, 0)
            pr[layer, i] = r
            seen[expert] = r + 1
    return pr


def _validate_intragpu_rearrangement(
    old_global_expert_indices: np.ndarray,
    new_phy2log: np.ndarray,
    new_phy_replicas_idx: np.ndarray,
    post_phy2log: np.ndarray,
    post_phy_replicas_idx: np.ndarray,
    num_ranks: int,
    slots_per_gpu: int,
):
    # Per-GPU checks
    for gpu_idx in range(num_ranks):
        start = gpu_idx * slots_per_gpu
        end = start + slots_per_gpu
        old_seg = old_global_expert_indices[0, start:end]
        new_seg = new_phy2log[0, start:end]
        new_rnk = new_phy_replicas_idx[0, start:end]
        post_seg = post_phy2log[0, start:end]
        post_rnk = post_phy_replicas_idx[0, start:end]

        # Pairwise equality for (expert, rank) pairs to ensure nothing is lost
        def sorted_pairs(seg, rnk):
            pairs = list(zip(seg.tolist(), rnk.tolist()))
            pairs.sort()
            return pairs

        assert sorted_pairs(post_seg, post_rnk) == sorted_pairs(new_seg, new_rnk), (
            f"Per-GPU pairs of (expert,rank) must match new mapping for GPU {gpu_idx}"
        )

        # For experts that remain on the same GPU, the old slot is preserved
        # for at least one occurrence; rank at that slot must be valid for that expert
        old_list = old_seg.tolist()
        new_list = new_seg.tolist()
        post_list = post_seg.tolist()
        remained = set(old_list) & set(new_list)
        new_ranks_for_expert: dict[int, list[int]] = {}
        for v, r in zip(new_list, new_rnk.tolist()):
            new_ranks_for_expert.setdefault(v, []).append(r)
        for expert in remained:
            old_pos = old_list.index(expert)
            assert post_list[old_pos] == expert, (
                f"Expert {expert} on GPU {gpu_idx} should stay at old slot {old_pos}"
            )
            # Rank at preserved slot must be one of the ranks
            # the expert has in new mapping
            assert post_rnk.tolist()[old_pos] in new_ranks_for_expert[expert], (
                f"Rank for expert {expert} at preserved slot on GPU {gpu_idx} "
                "must come from new mapping"
            )


@pytest.mark.parametrize(
    "num_ranks, slots_per_gpu, old_phy2log, new_phy2log",
    [
        pytest.param(
            # Setup: 2 GPUs, 4 slots each, 1 layer
            # Old mapping: GPU0 -> [0,1,2,3], GPU1 -> [4,5,6,7]
            # New mapping shuffles within GPU0 and brings 4,5 into GPU0.
            # GPU0 new -> [1,5,0,4]; GPU1 new -> [6,2,7,3]
            2,
            4,
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
            np.array([[1, 5, 0, 4, 6, 2, 7, 3]]),
            id="simple",
        ),
        pytest.param(
            # Setup: 2 GPUs, 5 slots each (total 10 physical experts), 1 layer
            # Old mapping:
            #   GPU0 -> [0, 1, 0, 2, 3]  (expert 0 duplicated)
            #   GPU1 -> [4, 5, 6, 1, 2]
            # New mapping reorders within GPUs and moves some experts across GPUs,
            # while still including duplicates:
            #   GPU0 new -> [0, 5, 4, 0, 1]  (expert 0 duplicated, 4/5 incoming)
            #   GPU1 new -> [6, 2, 3, 2, 1]  (expert 2 duplicated)
            2,
            5,
            np.array([[0, 1, 0, 2, 3, 4, 5, 6, 1, 2]]),
            np.array([[0, 5, 4, 0, 1, 6, 2, 3, 2, 1]]),
            id="duplicates",
        ),
        pytest.param(
            # Setup: 3 GPUs, 4 slots each (total 12 physical experts), 1 layer
            # Old mapping:
            #   GPU0 -> [0, 1, 2, 3]
            #   GPU1 -> [0, 1, 2, 3]
            #   GPU2 -> [0, 1, 2, 3]
            # New mapping decides to use one expert on 2 GPUs and shuffles
            # experts on the third GPU,
            #   GPU0 new -> [0, 0, 0, 0]
            #   GPU1 new -> [0, 0, 0, 0]
            #   GPU2 new -> [1, 2, 3, 0]
            3,
            4,
            np.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]),
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0]]),
            id="skewed_expert",
        ),
    ],
)
def test_preserve_intragpu_slots(
    num_ranks: int,
    slots_per_gpu: int,
    old_phy2log: torch.Tensor,
    new_phy2log: torch.Tensor,
):
    """Experts that stay on a GPU keep their old slots; incoming not lost."""
    phy_replicas_idx = _make_phy_replicas_idx_from_phy2log(new_phy2log)

    post_phy2log = DefaultEplbPolicy.preserve_intragpu_slots(
        new_phy2log, num_ranks, old_phy2log
    )
    post_phy_replicas_idx = _make_phy_replicas_idx_from_phy2log(post_phy2log)

    # Shapes preserved
    assert post_phy2log.shape == new_phy2log.shape
    assert post_phy_replicas_idx.shape == phy_replicas_idx.shape

    _validate_intragpu_rearrangement(
        old_phy2log,
        new_phy2log,
        phy_replicas_idx,
        post_phy2log,
        post_phy_replicas_idx,
        num_ranks,
        slots_per_gpu,
    )
