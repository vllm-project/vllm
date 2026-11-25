# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for parallel group device resolution across platforms."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import TCPStore

import vllm.platforms
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.utils.network_utils import get_open_port

# These tests use CPU gloo process groups with no accelerator state to clean up.
pytestmark = pytest.mark.skip_global_cleanup


class _DefaultOotPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "privateuseone"


class _SingleDeviceOotPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "cpu"

    def get_group_device(self, local_rank: int) -> torch.device:
        return torch.device(self.device_name)


class _CudaLikePlatform(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"

    @classmethod
    def logical_device_id_to_visible_device_id(cls, device_id: int) -> int:
        return device_id + 2


class _XpuLikePlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name = "xpu"


class _CpuLikePlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name = "cpu"


@pytest.fixture
def single_rank_gloo(tmp_path):
    """Initialize a single-rank gloo process group and restore after test."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path}/store",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_oot_platform_custom_group_device(monkeypatch, single_rank_gloo):
    """Out-of-tree platforms can return an unindexed device for any rank."""
    monkeypatch.setattr(vllm.platforms, "current_platform", _SingleDeviceOotPlatform())
    group = GroupCoordinator(
        group_ranks=[[0]],
        local_rank=1,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="oot_custom_device",
    )
    assert group.device == torch.device("cpu")
    assert group.device.index is None
    tensor = torch.ones(2, device=group.device)
    assert tensor.device == torch.device("cpu")


def test_oot_platform_default_group_device(monkeypatch, single_rank_gloo):
    """Default Platform.get_group_device preserves `<device_name>:<local_rank>`."""
    monkeypatch.setattr(vllm.platforms, "current_platform", _DefaultOotPlatform())
    group = GroupCoordinator(
        group_ranks=[[0]],
        local_rank=1,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="oot_default_device",
    )
    assert group.device == torch.device("privateuseone:1")
    assert group.device.index == 1


@pytest.mark.parametrize(
    ("platform_cls", "expected_device"),
    [
        (_CudaLikePlatform, torch.device("cuda:3")),
        (_XpuLikePlatform, torch.device("xpu:1")),
        (_CpuLikePlatform, torch.device("cpu")),
    ],
)
def test_builtin_platforms_preserve_group_device(
    monkeypatch,
    single_rank_gloo,
    platform_cls,
    expected_device,
):
    """Built-in CUDA, XPU, and CPU device selection stays unchanged."""
    monkeypatch.setattr(vllm.platforms, "current_platform", platform_cls())
    group = GroupCoordinator(
        group_ranks=[[0]],
        local_rank=1,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name=f"builtin_{platform_cls.__name__}",
    )
    assert group.device == expected_device


def test_stateless_coordinator_uses_oot_group_device(monkeypatch):
    """StatelessGroupCoordinator delegates out-of-tree device naming too."""
    monkeypatch.setattr(vllm.platforms, "current_platform", _SingleDeviceOotPlatform())
    store = TCPStore("127.0.0.1", get_open_port(), 1, is_master=True)
    group = StatelessGroupCoordinator(
        group_ranks=[[0]],
        local_rank=1,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        coord_store=store,
        group_name="stateless_oot_device",
    )
    try:
        assert group.device == torch.device("cpu")
        assert group.device.index is None
    finally:
        group.destroy()


def _communicator_worker(rank: int, world_size: int, port: int) -> None:
    vllm.platforms.current_platform = _SingleDeviceOotPlatform()
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            torch_distributed_backend="gloo",
            use_device_communicator=True,
            group_name="oot_comm_device",
        )
        assert group.device == torch.device("cpu")
        assert group.device.index is None
        assert group.device_communicator is not None
        assert group.device_communicator.device == torch.device("cpu")
        assert group.device_communicator.device.index is None
        tensor = torch.ones(2, device=group.device)
        out = group.device_communicator.all_reduce(tensor)
        assert torch.equal(out, torch.full((2,), float(world_size)))
    finally:
        dist.destroy_process_group()


def test_oot_platform_group_device_with_communicator():
    """Communicator initialization uses the custom group device across ranks."""
    mp.spawn(
        _communicator_worker,
        args=(2, get_open_port()),
        nprocs=2,
        join=True,
    )
