# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm latent-MoE tail must equal the replicated up-projection.

A wrong shard offset or a dropped accumulation still runs and still produces
plausible text, so these pin the arithmetic rather than the behaviour.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.multiprocessing import spawn

from tests.utils import (
    ensure_current_vllm_config,
    init_test_distributed_environment,
    multi_gpu_test,
)
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiRoutedOutputTransform
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCmLatentMoERunner is only wired up on ROCm",
)

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
EPS = 1e-5
DTYPE = torch.bfloat16


def _build_transform(device: torch.device) -> KimiRoutedOutputTransform:
    norm = RMSNorm(LATENT_SIZE, eps=EPS).to(device=device, dtype=DTYPE)
    up_proj = ReplicatedLinear(
        LATENT_SIZE,
        HIDDEN_SIZE,
        bias=False,
        params_dtype=DTYPE,
        prefix="routed_expert_up_proj",
    ).to(device=device)

    torch.manual_seed(0)
    norm.weight.data.copy_(1 + 0.1 * torch.randn_like(norm.weight))
    up_proj.weight.data.copy_(torch.randn_like(up_proj.weight) / LATENT_SIZE**0.5)
    return KimiRoutedOutputTransform(norm, up_proj)


def _tail_runner(
    transform: KimiRoutedOutputTransform, tp_size: int
) -> ROCmLatentMoERunner:
    """A runner carrying only what the tail reads, so no engine is needed.

    ``_maybe_reduce_final_output`` is left as the real base-class method, so the
    all-reduce that stitches the shards is the real collective.
    """
    runner = object.__new__(ROCmLatentMoERunner)
    attrs = {
        "routed_output_transform": transform,
        "_up_proj_shard_size": HIDDEN_SIZE // tp_size,
        "_logged_sharded_tail": False,
        "moe_config": SimpleNamespace(
            tp_size=tp_size,
            ep_size=1,
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
        ),
    }
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


def _all_reduced(tensor: torch.Tensor, group) -> torch.Tensor:
    reduced = tensor.clone()
    dist.all_reduce(reduced, group=group)
    return reduced


def _rank_partials(
    num_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    routed = torch.randn(num_tokens, LATENT_SIZE, device=device, dtype=DTYPE)
    shared = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE)
    return routed.mul_(0.01), shared


def _check_matches_replicated(device: torch.device, tp_size: int, rank: int) -> None:
    transform = _build_transform(device)
    runner = _tail_runner(transform, tp_size)
    group = get_tp_group().device_group

    for iteration, num_tokens in enumerate((1, 5, 8, 16, 5)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed_output, shared_output = _rank_partials(num_tokens, device)

        expected = F.linear(
            F.rms_norm(
                _all_reduced(routed_output, group),
                (LATENT_SIZE,),
                transform.norm.weight,
                EPS,
            ),
            transform.up_proj.weight,
        )
        expected.add_(_all_reduced(shared_output, group))

        actual = runner._shard_up_proj_tail(routed_output, shared_output, None)

        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def _check_writes_only_its_own_shard(
    device: torch.device, tp_size: int, rank: int
) -> None:
    """Two ranks that swapped both weight rows and write offsets still sum to
    the right total, so inspect each slice before the final collective."""
    transform = _build_transform(device)
    runner = _tail_runner(transform, tp_size)
    group = get_tp_group().device_group

    torch.manual_seed(rank + 1)
    routed_output, shared_output = _rank_partials(8, device)
    before = shared_output.clone()
    latent = transform.norm(_all_reduced(routed_output, group))

    captured: dict = {}
    real_reduce = runner._maybe_reduce_final_output

    def _capture(states, trunc_size, output_is_reduced=None):
        captured["states"] = states.clone()
        captured["output_is_reduced"] = output_is_reduced
        return real_reduce(states, trunc_size, output_is_reduced)

    object.__setattr__(runner, "_maybe_reduce_final_output", _capture)
    runner._shard_up_proj_tail(routed_output, shared_output, None)
    # Restoring the method breaks the runner -> _capture -> runner cycle, so the
    # captured device tensor is freed before the process group is torn down.
    object.__setattr__(runner, "_maybe_reduce_final_output", real_reduce)

    assert captured["output_is_reduced"] is False

    shard = HIDDEN_SIZE // tp_size
    start, end = rank * shard, (rank + 1) * shard
    local = captured["states"]
    projected = F.linear(latent, transform.up_proj.weight[start:end])

    torch.testing.assert_close(
        local[:, start:end], before[:, start:end] + projected, atol=8e-2, rtol=3e-2
    )
    torch.testing.assert_close(local[:, :start], before[:, :start], atol=0, rtol=0)
    torch.testing.assert_close(local[:, end:], before[:, end:], atol=0, rtol=0)


_CHECKS = {
    "matches_replicated": _check_matches_replicated,
    "own_shard_only": _check_writes_only_its_own_shard,
}


def _worker(local_rank: int, world_size: int, port: str, check: str) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    with ensure_current_vllm_config():
        init_test_distributed_environment(
            world_size, 1, local_rank, port, local_rank=local_rank
        )
        _CHECKS[check](device, world_size, local_rank)


def _run_ranks(check: str, tp_size: int) -> None:
    spawn(
        _worker,
        args=(tp_size, str(get_open_port()), check),
        nprocs=tp_size,
        join=True,
    )


@multi_gpu_test(num_gpus=4)
def test_sharded_tail_tp4_matches_replicated_projection() -> None:
    _run_ranks("matches_replicated", 4)


@multi_gpu_test(num_gpus=8)
def test_sharded_tail_tp8_matches_replicated_projection() -> None:
    _run_ranks("matches_replicated", 8)


@multi_gpu_test(num_gpus=4)
def test_sharded_tail_tp4_writes_only_its_own_shard() -> None:
    _run_ranks("own_shard_only", 4)


def _runner(**attrs) -> ROCmLatentMoERunner:
    runner = object.__new__(ROCmLatentMoERunner)
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


@pytest.mark.parametrize(
    "shardable,pre_reduced,expect_sharded",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_forward_shards_only_when_the_tail_is_valid(
    shardable: bool,
    pre_reduced: bool,
    expect_sharded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shard needs an un-reduced shared partial to accumulate into."""
    base_output, sharded_output = torch.zeros(1), torch.ones(1)
    monkeypatch.setattr(moe_runner.MoERunner, "forward", lambda *a, **k: base_output)
    monkeypatch.setattr(
        ROCmLatentMoERunner, "_fused_forward", lambda *a, **k: sharded_output
    )
    monkeypatch.setattr(
        ROCmLatentMoERunner, "_fused_output_is_reduced", property(lambda _: pre_reduced)
    )
    runner = _runner(_tail_shardable=shardable)

    result = runner.forward(torch.zeros(1), torch.zeros(1))

    assert result is (sharded_output if expect_sharded else base_output)


@pytest.fixture
def build_runner(monkeypatch: pytest.MonkeyPatch):
    """Construct through the real subclass ``__init__``, stubbing only the base."""

    def _base_init(
        self,
        *,
        tp_size: int = 8,
        hidden: int = HIDDEN_SIZE,
        is_sequence_parallel: bool = False,
        has_up_proj: bool = True,
        has_shared_experts: bool = True,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        self.moe_config = SimpleNamespace(
            tp_size=tp_size, is_sequence_parallel=is_sequence_parallel
        )
        self.routed_output_transform = SimpleNamespace(
            norm=None,
            up_proj=SimpleNamespace(weight=torch.zeros(hidden, LATENT_SIZE))
            if has_up_proj
            else None,
        )
        self.routed_scaling_factor = routed_scaling_factor
        self._shared_experts = object() if has_shared_experts else None

    monkeypatch.setattr(moe_runner.MoERunner, "__init__", _base_init)
    return ROCmLatentMoERunner


def test_shards_under_the_kimi_k3_serving_config(build_runner) -> None:
    runner = build_runner()

    assert runner._tail_shardable
    assert runner._up_proj_shard_size == HIDDEN_SIZE // 8


def test_hybrid_ep_uses_physical_tp_for_the_latent_tail(build_runner) -> None:
    # Explicit EP2 preserves the physical TP4 shard instead of flattening
    # TP into expert ownership.
    runner = build_runner(tp_size=4)

    assert runner._tail_shardable
    assert runner._up_proj_shard_size == HIDDEN_SIZE // 4


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"tp_size": 1}, id="no-tp"),
        pytest.param({"hidden": HIDDEN_SIZE + 2}, id="hidden-not-divisible-by-tp"),
        pytest.param({"has_up_proj": False}, id="no-up-proj"),
        pytest.param({"has_shared_experts": False}, id="no-shared-partial"),
        pytest.param({"is_sequence_parallel": True}, id="sequence-parallel"),
        pytest.param({"routed_scaling_factor": 2.0}, id="routed-scaling-factor"),
    ],
)
def test_falls_back_when_the_config_breaks_the_shard(
    build_runner, override: dict
) -> None:
    """Each of these makes the sharded tail wrong, not merely slower."""
    runner = build_runner(**override)

    assert not runner._tail_shardable
    assert runner._up_proj_shard_size == 0
