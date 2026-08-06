# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm latent-MoE tail: sharded output must equal the replicated one.

``ROCmLatentMoERunner`` has each rank project only its slice of the hidden dim
into its partial shared-expert output and leaves the final all-reduce to stitch
the shards. If the slicing or the accumulation is wrong the model still runs and
still produces plausible text, so these pin the arithmetic instead.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.models.kimi_k3.amd import latent_moe_runner
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner

TP_SIZE = 4
HIDDEN = 32
LATENT = 16
NUM_TOKENS = 5


class _Norm(torch.nn.Module):
    """Stands in for the transform's RMSNorm.

    The tail is exact for any normalisation, because the norm runs on the
    all-reduced latent before the shard; keeping it device-independent lets
    this test run without a GPU.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


def _runner(**attrs) -> ROCmLatentMoERunner:
    """A runner carrying only what the tail reads, so no engine is needed."""
    runner = object.__new__(ROCmLatentMoERunner)
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


@pytest.fixture
def tail_inputs():
    torch.manual_seed(0)
    transform = SimpleNamespace(
        norm=_Norm(LATENT),
        up_proj=SimpleNamespace(weight=torch.randn(HIDDEN, LATENT) / LATENT**0.5),
    )
    return SimpleNamespace(
        transform=transform,
        # What each rank holds before any collective: partial routed output in
        # latent space, partial shared-expert output at full hidden dim.
        fused=[torch.randn(NUM_TOKENS, LATENT) for _ in range(TP_SIZE)],
        shared=[torch.randn(NUM_TOKENS, HIDDEN) for _ in range(TP_SIZE)],
    )


def _run_all_ranks(tail_inputs, monkeypatch: pytest.MonkeyPatch) -> torch.Tensor:
    """Run the tail on every simulated rank and sum, as the final AR would."""
    latent_total = sum(tail_inputs.fused)
    monkeypatch.setattr(
        latent_moe_runner, "tensor_model_parallel_all_reduce", lambda _: latent_total
    )

    outputs = []
    for rank in range(TP_SIZE):
        monkeypatch.setattr(
            latent_moe_runner, "get_tensor_model_parallel_rank", lambda r=rank: r
        )
        runner = _runner(
            routed_output_transform=tail_inputs.transform,
            _up_proj_shard_size=HIDDEN // TP_SIZE,
            _logged_sharded_tail=True,
        )
        # Stand in for the final all-reduce, which the caller performs below.
        monkeypatch.setattr(
            runner, "_maybe_reduce_final_output", lambda states, *_, **__: states
        )
        outputs.append(
            runner._shard_up_proj_tail(
                tail_inputs.fused[rank], tail_inputs.shared[rank], None
            )
        )
    return sum(outputs)


def test_sharded_tail_sums_to_the_replicated_projection(
    tail_inputs, monkeypatch: pytest.MonkeyPatch
):
    transform = tail_inputs.transform
    latent = transform.norm(sum(tail_inputs.fused))
    expected = sum(tail_inputs.shared) + latent @ transform.up_proj.weight.t()

    actual = _run_all_ranks(tail_inputs, monkeypatch)

    torch.testing.assert_close(actual, expected)


def test_every_rank_writes_only_its_own_shard(
    tail_inputs, monkeypatch: pytest.MonkeyPatch
):
    """Overlapping or gapped shards would still sum to something plausible."""
    untouched = [s.clone() for s in tail_inputs.shared]
    shard = HIDDEN // TP_SIZE

    _run_all_ranks(tail_inputs, monkeypatch)

    for rank, (before, after) in enumerate(zip(untouched, tail_inputs.shared)):
        start, end = rank * shard, (rank + 1) * shard
        assert not torch.equal(after[:, start:end], before[:, start:end])
        torch.testing.assert_close(after[:, :start], before[:, :start])
        torch.testing.assert_close(after[:, end:], before[:, end:])


@pytest.mark.parametrize(
    "shardable,pre_reduced",
    [(False, False), (True, True), (False, True)],
)
def test_falls_back_to_the_base_runner(
    shardable: bool, pre_reduced: bool, monkeypatch: pytest.MonkeyPatch
):
    """The shard needs an un-reduced shared partial to accumulate into."""
    sentinel = torch.zeros(1)
    monkeypatch.setattr(moe_runner.MoERunner, "forward", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        ROCmLatentMoERunner, "_fused_output_is_reduced", property(lambda _: pre_reduced)
    )
    runner = _runner(_tail_shardable=shardable)

    assert runner.forward(torch.zeros(1), torch.zeros(1)) is sentinel
