# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm latent-MoE tails must equal the replicated up-projection.

A wrong shard offset, a dropped accumulation, or an overlapped all-reduce that
races still runs and still produces plausible text, so these pin the arithmetic
rather than the behaviour. The unit tests cover tier selection, which decides
which tail arithmetic runs.
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
from vllm.models.kimi_k3.amd import latent_moe_runner
from vllm.models.kimi_k3.amd.latent_moe_runner import (
    ROCmLatentMoERunner,
    ROCmLatentTailTier,
)
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
    """A runner carrying only what the tails read, so no engine is needed.

    ``_maybe_reduce_final_output`` is left as the real base-class method, so the
    all-reduce that stitches the shards is the real collective.
    """
    runner = object.__new__(ROCmLatentMoERunner)
    attrs = {
        "routed_output_transform": transform,
        "_logged_column_parallel": False,
        "_logged_overlap_fallback": False,
        "_shared_ar_events": (torch.cuda.Event(), torch.cuda.Event()),
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


def _replicated_reference(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
    transform: KimiRoutedOutputTransform,
    group,
) -> torch.Tensor:
    """latent-space all-reduce, RMSNorm, replicated up-proj, plus shared."""
    expected = F.linear(
        F.rms_norm(
            _all_reduced(routed_output, group),
            (LATENT_SIZE,),
            transform.norm.weight,
            EPS,
        ),
        transform.up_proj.weight,
    )
    return expected.add_(_all_reduced(shared_output, group))


def _check_shard_matches_replicated(
    device: torch.device, tp_size: int, rank: int
) -> None:
    transform = _build_transform(device)
    runner = _tail_runner(transform, tp_size)
    group = get_tp_group().device_group

    for iteration, num_tokens in enumerate((1, 5, 8, 16, 5)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed_output, shared_output = _rank_partials(num_tokens, device)

        expected = _replicated_reference(routed_output, shared_output, transform, group)
        actual = runner._shard_up_proj_tail(routed_output, shared_output, None)

        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def _check_overlap_matches_replicated(
    device: torch.device, tp_size: int, rank: int
) -> None:
    transform = _build_transform(device)
    runner = _tail_runner(transform, tp_size)
    group = get_tp_group().device_group

    for iteration, num_tokens in enumerate((1, 5, 8, 16, 5)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed_output, shared_output = _rank_partials(num_tokens, device)

        expected = _replicated_reference(routed_output, shared_output, transform, group)
        actual = runner._overlap_allreduce_tail(routed_output, shared_output, None)

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
    "shard_matches_replicated": _check_shard_matches_replicated,
    "overlap_matches_replicated": _check_overlap_matches_replicated,
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
def test_shard_tail_tp4_matches_replicated_projection() -> None:
    _run_ranks("shard_matches_replicated", 4)


@multi_gpu_test(num_gpus=8)
def test_shard_tail_tp8_matches_replicated_projection() -> None:
    _run_ranks("shard_matches_replicated", 8)


@multi_gpu_test(num_gpus=4)
def test_overlap_tail_tp4_matches_replicated_projection() -> None:
    _run_ranks("overlap_matches_replicated", 4)


@multi_gpu_test(num_gpus=8)
def test_overlap_tail_tp8_matches_replicated_projection() -> None:
    _run_ranks("overlap_matches_replicated", 8)


@multi_gpu_test(num_gpus=4)
def test_shard_tail_tp4_writes_only_its_own_shard() -> None:
    _run_ranks("own_shard_only", 4)


def _logic_runner(
    *,
    tp_size: int = 8,
    hidden: int = HIDDEN_SIZE,
    is_sequence_parallel: bool = False,
    has_up_proj: bool = True,
    has_shared_experts: bool = True,
    routed_scaling_factor: float = 1.0,
    fused_output_is_reduced: bool = False,
) -> ROCmLatentMoERunner:
    """Build a runner with only the fields the pure-logic gates read.

    Bypasses ``__init__`` so no CUDA events are allocated; the tier gates never
    touch the device.
    """
    runner = object.__new__(ROCmLatentMoERunner)
    moe_kernel = (
        SimpleNamespace(output_is_reduced=lambda: True)
        if fused_output_is_reduced
        else None
    )
    # ``_quant_method`` is a read-only property reading ``routed_experts``;
    # ``_fused_output_is_reduced`` reaches through it to ``moe_kernel``.
    attrs = {
        "moe_config": SimpleNamespace(
            tp_size=tp_size, is_sequence_parallel=is_sequence_parallel
        ),
        "_shared_experts": object() if has_shared_experts else None,
        "routed_scaling_factor": routed_scaling_factor,
        "routed_experts": SimpleNamespace(
            quant_method=SimpleNamespace(moe_kernel=moe_kernel)
        ),
        "routed_output_transform": SimpleNamespace(
            norm=None,
            up_proj=SimpleNamespace(weight=torch.zeros(hidden, LATENT_SIZE))
            if has_up_proj
            else None,
        ),
        "_logged_overlap_fallback": False,
    }
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


def test_fused_path_under_the_kimi_k3_serving_config() -> None:
    runner = _logic_runner()

    assert runner._use_fused_path()
    assert runner._column_parallel_shardable()


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"tp_size": 1}, id="no-tp"),
        pytest.param({"has_shared_experts": False}, id="no-shared-partial"),
        pytest.param({"is_sequence_parallel": True}, id="sequence-parallel"),
        pytest.param({"routed_scaling_factor": 2.0}, id="routed-scaling-factor"),
        pytest.param({"fused_output_is_reduced": True}, id="already-reduced"),
    ],
)
def test_falls_back_to_native_path_when_fusion_is_unsafe(override: dict) -> None:
    """Each of these makes the fused tail wrong, not merely slower, so the
    runner must defer to the base combine."""
    runner = _logic_runner(**override)

    assert not runner._use_fused_path()


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"hidden": HIDDEN_SIZE + 2}, id="hidden-not-divisible-by-tp"),
        pytest.param({"has_up_proj": False}, id="no-up-proj"),
    ],
)
def test_not_column_parallel_shardable_but_still_fused(override: dict) -> None:
    """These break only the shard tail; the fused overlap tail is still correct,
    so the fused path stays on and selection avoids the column-parallel tail."""
    runner = _logic_runner(**override)

    assert runner._use_fused_path()
    assert not runner._column_parallel_shardable()
    tier = runner._select_tail_tier(torch.zeros(4096, 1))
    assert tier is ROCmLatentTailTier.ALLREDUCE_OVERLAP


def test_small_batches_pick_the_overlap_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latent_moe_runner.envs,
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
        16,
        raising=False,
    )
    monkeypatch.setattr(
        latent_moe_runner.envs,
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
        False,
        raising=False,
    )
    runner = _logic_runner()

    # Small and shardable: overlap still wins under the threshold.
    assert (
        runner._select_tail_tier(torch.zeros(8, 1))
        is ROCmLatentTailTier.ALLREDUCE_OVERLAP
    )
    # Large and shardable: fold the up-projection into the reduce instead.
    assert (
        runner._select_tail_tier(torch.zeros(64, 1))
        is ROCmLatentTailTier.COLUMN_PARALLEL
    )


def test_disabling_the_stream_forces_the_shard_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latent_moe_runner.envs,
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
        16,
        raising=False,
    )
    monkeypatch.setattr(
        latent_moe_runner.envs,
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
        True,
        raising=False,
    )
    runner = _logic_runner()

    # Even a single-token batch takes the column-parallel tail when the overlap
    # stream is disabled, as long as the up-projection is shardable.
    assert (
        runner._select_tail_tier(torch.zeros(1, 1))
        is ROCmLatentTailTier.COLUMN_PARALLEL
    )


def _runner(**attrs) -> ROCmLatentMoERunner:
    runner = object.__new__(ROCmLatentMoERunner)
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


@pytest.mark.parametrize("fused_path,expect_fused", [(True, True), (False, False)])
def test_forward_dispatches_to_fused_only_on_the_fused_path(
    fused_path: bool,
    expect_fused: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forward() routes to the fused tail exactly when _use_fused_path holds."""
    base_output, fused_output = torch.zeros(1), torch.ones(1)
    monkeypatch.setattr(moe_runner.MoERunner, "forward", lambda *a, **k: base_output)
    monkeypatch.setattr(
        ROCmLatentMoERunner, "_fused_forward", lambda *a, **k: fused_output
    )
    monkeypatch.setattr(ROCmLatentMoERunner, "_use_fused_path", lambda self: fused_path)
    runner = _runner()

    result = runner.forward(torch.zeros(1), torch.zeros(1))

    assert result is (fused_output if expect_fused else base_output)
