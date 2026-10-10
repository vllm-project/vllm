# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the nixl_ep_v2 all2all manager: its one buffer per process,
its domain-size, world-layout and SM-count settings, and its guards."""

import types
from typing import Any

import pytest

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)
from tests.utils import multi_gpu_marks
from vllm.config import EPLBConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_nixl_ep_v2


def _nixl_ep_v2_manager_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    domain_size: int,
) -> None:
    from vllm.distributed.device_communicators.all2all import (
        NixlEPV2All2AllManager,
    )

    world = pgi.world_size
    comm = get_ep_group().device_communicator
    assert comm is not None
    mgr = comm.all2all_manager
    assert isinstance(mgr, NixlEPV2All2AllManager)
    assert mgr.max_sms_used() is None  # no buffer yet
    kw = dict(
        num_max_tokens_per_rank=64,
        hidden=2048,
        num_topk=2,
        use_fp8_dispatch=False,
        num_local_experts=4,
    )
    if 0 < domain_size < world:
        # Domains smaller than the node's NVLink domain: every rank refuses.
        with pytest.raises(ValueError, match="smaller than the NVLink domain"):
            mgr.get_handle(kw)
        return
    buf = mgr.get_handle(kw)
    assert buf.rank_idx in buf.get_active_ranks()
    assert mgr.max_num_ep_ranks == world
    assert mgr.max_sms_used() == buf.num_launch_sms
    if domain_size == 0:
        # All GPUs of one node share one NVLink domain: direct mode.
        assert buf.get_topology_ranks() == (1, world)
        assert buf.get_topology() == ("direct", buf.num_launch_sms)
    else:
        # Declared domains: each EP rank sits at (domain, position) =
        # divmod(rank, domain_size), and only hybrid mode spans several.
        num_domains = world // domain_size
        assert (buf.max_num_scaleout_ranks, buf.max_num_scaleup_ranks) == (
            num_domains,
            domain_size,
        )
        assert (buf.scaleout_rank_idx, buf.scaleup_rank_idx) == divmod(
            pgi.rank, domain_size
        )
        mode = "hybrid" if num_domains > 1 else "direct"
        assert buf.get_topology()[0] == mode

    # Elastic EP builds the new manager before it destroys the old one; the
    # buffer must survive both.
    new_mgr = NixlEPV2All2AllManager(mgr.cpu_group, mgr.tcp_store_group)
    mgr.destroy()
    assert new_mgr.get_handle(kw) is buf
    with pytest.raises(AssertionError):
        new_mgr.get_handle({**kw, "hidden": 4096})


@pytest.mark.parametrize(
    "world, domain_size",
    [
        (2, 0),  # inferred: one domain of 2, direct mode
        (2, 2),  # one declared domain (the elastic EP default): direct mode
        (4, 2),  # two declared domains of 2 in one NVLink domain: refused
    ],
)
def test_nixl_ep_v2_manager(world: int, domain_size: int) -> None:
    """One nixl_ep_v2 buffer per process, built once and kept by a new manager.

    Assumes one node whose GPUs share one NVLink domain.

    Args:
        world: GPUs, one EP rank each.
        domain_size: VLLM_NIXL_EP_V2_DOMAIN_SIZE; 0 lets nixl_ep_v2 infer it.

    """
    if not has_nixl_ep_v2():
        pytest.skip("Requires NIXL EP v2 (nixl.runtime.ep)")
    if current_platform.device_count() < world:
        pytest.skip(f"Requires {world} GPUs")

    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=world,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="nixl_ep_v2",
        ),
    )
    env = {"VLLM_NIXL_EP_V2_DOMAIN_SIZE": str(domain_size)}
    parallel_launch_with_config(
        world, _nixl_ep_v2_manager_worker, vllm_config, env, domain_size
    )


def test_nixl_ep_v2_manager_init_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manager refuses an unusable nixl_ep_v2 and elastic EP scale-up."""
    from vllm.distributed.device_communicators import all2all

    monkeypatch.setattr(all2all, "has_nixl_ep_v2", lambda: False)
    with pytest.raises(AssertionError, match="nixl_ep_v2 needs nixl.runtime.ep"):
        all2all.NixlEPV2All2AllManager(cpu_group=None)

    monkeypatch.setattr(all2all, "has_nixl_ep_v2", lambda: True)
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "1")
    with pytest.raises(AssertionError, match="scale-up"):
        all2all.NixlEPV2All2AllManager(cpu_group=None)


# Async EPLB under elastic EP needs the nixl package; the domain size ignores it.
_ELASTIC_DP4 = {
    "enable_elastic_ep": True,
    "enable_eplb": True,
    "eplb_config": EPLBConfig(use_async=False),
    "data_parallel_size": 4,
}


@pytest.mark.parametrize(
    "env, config, expected",
    [
        # Fixed EP size: 0 lets nixl_ep_v2 infer the domains.
        (None, {"data_parallel_size": 4}, 0),
        # Elastic EP: one domain of the maximum EP size, TP x PCP x max DP.
        # ParallelConfig checks that the node has TP x PCP GPUs.
        pytest.param(
            None,
            {
                **_ELASTIC_DP4,
                "tensor_parallel_size": 2,
                "prefill_context_parallel_size": 2,
                "elastic_ep_max_dp_size": 8,
            },
            32,
            marks=multi_gpu_marks(num_gpus=4),
        ),
        # Set: used as is, but never larger than the maximum EP size.
        ("8", {**_ELASTIC_DP4, "elastic_ep_max_dp_size": 16}, 8),
        ("72", {"data_parallel_size": 4}, 4),
        # external_launcher folds DP into world_size; DP must not count twice.
        (
            None,
            {
                **_ELASTIC_DP4,
                "elastic_ep_max_dp_size": 8,
                "distributed_executor_backend": "external_launcher",
            },
            8,
        ),
    ],
)
def test_nixl_ep_v2_domain_size(
    env: str | None,
    config: dict[str, Any],
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The domain size: inferred, the elastic default, set, capped.

    Args:
        env: VLLM_NIXL_EP_V2_DOMAIN_SIZE, or None to leave it unset.
        config: ParallelConfig arguments.
        expected: The domain size nixl_ep_v2_domain_size() must return.

    """
    from vllm.distributed.elastic_ep.elastic_execute import nixl_ep_v2_domain_size

    monkeypatch.setenv("RANK", "0")  # read by external_launcher
    if env is not None:
        monkeypatch.setenv("VLLM_NIXL_EP_V2_DOMAIN_SIZE", env)
    parallel_config = ParallelConfig(enable_expert_parallel=True, **config)
    assert nixl_ep_v2_domain_size(parallel_config) == expected


@pytest.mark.parametrize(
    "rank, ep_size, max_ep_size, domain_size, expected",
    [
        # Domain size 0: nixl_ep_v2 infers the layout.
        (1, 2, 2, 0, ()),
        # Start in part of domain 0 of a 2 x 8 world; rank 5 is position 5.
        (5, 6, 16, 8, (2, 8, 0, 5, 6)),
        # Start with both whole domains; rank 13 is domain 1, position 5.
        (13, 16, 16, 8, (2, 8, 1, 5, 8)),
        # Start with exactly one whole domain of a 2 x 8 world.
        (3, 8, 16, 8, (2, 8, 0, 3, 8)),
    ],
)
def test_nixl_ep_v2_world_args(
    rank: int,
    ep_size: int,
    max_ep_size: int,
    domain_size: int,
    expected: tuple[int, ...],
) -> None:
    """The buffer layout and each rank's (domain, position) in it.

    Args:
        rank: The EP rank.
        ep_size: The starting EP size.
        max_ep_size: The largest EP size.
        domain_size: GPUs per NVLink domain.
        expected: max_num_scaleout_ranks, max_num_scaleup_ranks,
            scaleout_rank_idx, scaleup_rank_idx and num_group_scaleup_ranks;
            empty for no arguments.

    """
    from vllm.distributed.device_communicators.all2all import _nixl_ep_v2_world_args

    keys = (
        "max_num_scaleout_ranks",
        "max_num_scaleup_ranks",
        "scaleout_rank_idx",
        "scaleup_rank_idx",
        "num_group_scaleup_ranks",
    )
    args = _nixl_ep_v2_world_args(rank, ep_size, max_ep_size, domain_size)
    assert args == dict(zip(keys, expected))


@pytest.mark.parametrize(
    "ep_size, max_ep_size, domain_size, match",
    [
        # The maximum world is not whole domains.
        (4, 12, 8, "not a whole number"),
        # The starting ranks fill part of a second domain.
        (12, 16, 8, "starting EP size"),
    ],
)
def test_nixl_ep_v2_world_args_rejects(
    ep_size: int, max_ep_size: int, domain_size: int, match: str
) -> None:
    """Layouts nixl_ep_v2 cannot build fail with a clear ValueError.

    Args:
        ep_size: The starting EP size.
        max_ep_size: The largest EP size.
        domain_size: GPUs per NVLink domain.
        match: A regex the error message must match.

    """
    from vllm.distributed.device_communicators.all2all import _nixl_ep_v2_world_args

    with pytest.raises(ValueError, match=match):
        _nixl_ep_v2_world_args(0, ep_size, max_ep_size, domain_size)


def test_nixl_ep_v2_domain_size_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative VLLM_NIXL_EP_V2_DOMAIN_SIZE is a ValueError."""
    from vllm.distributed.elastic_ep.elastic_execute import nixl_ep_v2_domain_size

    monkeypatch.setenv("VLLM_NIXL_EP_V2_DOMAIN_SIZE", "-1")
    with pytest.raises(ValueError, match="VLLM_NIXL_EP_V2_DOMAIN_SIZE"):
        nixl_ep_v2_domain_size(ParallelConfig(data_parallel_size=2))


@pytest.mark.cpu_test
@pytest.mark.parametrize("env, expected", [(None, 26), ("0", 26), ("16", 16)])
def test_nixl_ep_v2_num_sms_env(
    env: str | None, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VLLM_NIXL_EP_V2_NUM_SMS, or the launch grid when unset or 0."""
    from vllm.distributed.device_communicators.all2all import _nixl_ep_v2_num_sms

    monkeypatch.delenv("VLLM_NIXL_EP_V2_NUM_SMS", raising=False)
    if env is not None:
        monkeypatch.setenv("VLLM_NIXL_EP_V2_NUM_SMS", env)
    assert _nixl_ep_v2_num_sms(26) == expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("env", ["15", "-2", "28"])
def test_nixl_ep_v2_num_sms_env_rejects(
    env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An odd, negative or larger-than-grid VLLM_NIXL_EP_V2_NUM_SMS is a
    ValueError, not a silent clamp."""
    from vllm.distributed.device_communicators.all2all import _nixl_ep_v2_num_sms

    monkeypatch.setenv("VLLM_NIXL_EP_V2_NUM_SMS", env)
    with pytest.raises(ValueError, match="VLLM_NIXL_EP_V2_NUM_SMS"):
        _nixl_ep_v2_num_sms(26)


@pytest.mark.parametrize(
    "env, flag, num_sms",
    [
        ({}, True, 26),
        (
            {
                "VLLM_NIXL_EP_V2_PREFER_OVERLAP": "0",
                "VLLM_NIXL_EP_V2_ALLOW_MULTIPLE_REDUCTION": "0",
                "VLLM_NIXL_EP_V2_NUM_SMS": "16",
            },
            False,
            16,
        ),
    ],
    ids=["unset", "set"],
)
def test_nixl_ep_v2_buffer_knobs(
    env: dict[str, str],
    flag: bool,
    num_sms: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The VLLM_NIXL_EP_V2_* knobs reach the ElasticBuffer and the manager's
    SM count; unset, they are the kernels' defaults.

    Args:
        env: The knobs to set; the others are unset.
        flag: prefer_overlap_with_compute and allow_multiple_reduction.
        num_sms: The manager's active SMs and max_sms_used().

    """
    if not has_nixl_ep_v2():
        pytest.skip("Requires NIXL EP v2 (nixl.runtime.ep)")
    from nixl.runtime import ep

    from vllm.distributed.device_communicators.all2all import NixlEPV2All2AllManager

    class _Buffer:
        num_launch_sms = 26  # NIXL's default kNumLaunchSMs

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ep, "ElasticBuffer", _Buffer)
    for name in (
        "VLLM_NIXL_EP_V2_PREFER_OVERLAP",
        "VLLM_NIXL_EP_V2_ALLOW_MULTIPLE_REDUCTION",
        "VLLM_NIXL_EP_V2_NUM_SMS",
        "VLLM_NIXL_EP_V2_DOMAIN_SIZE",  # 0: no NVLink-domain bootstrap
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    stub = types.SimpleNamespace(
        rank=0, world_size=2, tcp_store_group=types.SimpleNamespace(store=None)
    )
    kw = dict(
        num_max_tokens_per_rank=16,
        hidden=256,
        num_topk=2,
        use_fp8_dispatch=False,
        num_local_experts=4,
    )
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="nixl_ep_v2",
        ),
    )
    with set_current_vllm_config(vllm_config):
        state = NixlEPV2All2AllManager._make_state(stub, kw)
    assert state.buffer.kwargs["prefer_overlap_with_compute"] is flag
    assert state.buffer.kwargs["allow_multiple_reduction"] is flag
    assert state.num_sms == num_sms
    monkeypatch.setattr(NixlEPV2All2AllManager, "_state", state)
    assert NixlEPV2All2AllManager.max_sms_used(stub) == num_sms
