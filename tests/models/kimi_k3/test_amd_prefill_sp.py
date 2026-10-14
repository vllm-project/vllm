# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-sharded Kimi-K3 prefill on ROCm must equal the replicated path.

Each rank ends up holding only its own token rows, so the multi-GPU checks
compare every rank's shard with the matching rows of the all-reduce result.
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
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
)
from vllm.models.kimi_k3.amd import sp
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import KimiRoutedOutputTransform
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Token-sharded Kimi-K3 prefill is only wired up on ROCm",
)

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
TOP_K = 16
NUM_EXPERTS = 896
EPS = 1e-5
DTYPE = torch.bfloat16


def _build_transform(
    device: torch.device, tp_world: int, rank: int, row_sharded: bool
) -> tuple[KimiRoutedOutputTransform, torch.Tensor]:
    """Returns the transform and the full up-projection weight."""
    norm = RMSNorm(LATENT_SIZE, eps=EPS).to(device=device, dtype=DTYPE)
    torch.manual_seed(0)
    norm.weight.data.copy_(1 + 0.1 * torch.randn_like(norm.weight))
    weight = torch.randn(HIDDEN_SIZE, LATENT_SIZE, device=device, dtype=DTYPE)
    weight /= LATENT_SIZE**0.5

    up_proj: ColumnParallelLinear | ReplicatedLinear
    if row_sharded:
        # The layout KimiMoE uses under TP: each rank stores its output rows.
        up_proj = ColumnParallelLinear(
            LATENT_SIZE,
            HIDDEN_SIZE,
            bias=False,
            gather_output=True,
            params_dtype=DTYPE,
            prefix="routed_expert_up_proj",
        ).to(device=device)
        rows = HIDDEN_SIZE // tp_world
        up_proj.weight.data.copy_(weight[rank * rows : (rank + 1) * rows])
    else:
        up_proj = ReplicatedLinear(
            LATENT_SIZE,
            HIDDEN_SIZE,
            bias=False,
            params_dtype=DTYPE,
            prefix="routed_expert_up_proj",
        ).to(device=device)
        up_proj.weight.data.copy_(weight)
    return KimiRoutedOutputTransform(norm, up_proj, row_sharded=row_sharded), weight


def _tail_runner(
    transform: KimiRoutedOutputTransform, tp_world: int
) -> ROCmLatentMoERunner:
    runner = object.__new__(ROCmLatentMoERunner)
    attrs = {
        "routed_output_transform": transform,
        "_up_proj_preshard": transform.row_sharded,
        "_up_proj_shard_size": HIDDEN_SIZE // tp_world,
        "_logged_sharded_tail": False,
        "moe_config": SimpleNamespace(
            tp_size=tp_world,
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


def _check_sharded_tail(device: torch.device, tp_world: int, rank: int) -> None:
    """For both up-projection layouts, the column-parallel tail matches the
    replicated result, and its token-sharded variant returns this rank's rows."""
    for row_sharded in (True, False):
        _check_tail_layout(device, tp_world, rank, row_sharded)


def _check_tail_layout(
    device: torch.device, tp_world: int, rank: int, row_sharded: bool
) -> None:
    transform, weight = _build_transform(device, tp_world, rank, row_sharded)
    runner = _tail_runner(transform, tp_world)
    group = get_tp_group().device_group

    for iteration, num_tokens in enumerate((tp_world, 2 * tp_world, 64)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed = 0.01 * torch.randn(num_tokens, LATENT_SIZE, device=device, dtype=DTYPE)
        shared = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE)

        expected = F.linear(
            F.rms_norm(
                _all_reduced(routed, group),
                (LATENT_SIZE,),
                transform.norm.weight,
                EPS,
            ),
            weight,
        )
        expected.add_(_all_reduced(shared, group))
        rows = num_tokens // tp_world

        full = runner._shard_up_proj_tail(routed.clone(), shared.clone(), None)
        actual = runner._shard_up_proj_tail(routed, shared, None, token_sharded=True)

        torch.testing.assert_close(full, expected, atol=8e-2, rtol=3e-2)
        assert actual.shape == (rows, HIDDEN_SIZE)
        torch.testing.assert_close(
            actual, expected[rank * rows : (rank + 1) * rows], atol=8e-2, rtol=3e-2
        )


def _check_routing_gather(device: torch.device, tp_world: int, rank: int) -> None:
    """Packed routing survives the gather bit-exactly and in rank order."""
    rows = 5
    gen = torch.Generator(device=device).manual_seed(7)
    weights = torch.rand(tp_world * rows, TOP_K, device=device, generator=gen)
    ids = torch.randint(
        NUM_EXPERTS, (tp_world * rows, TOP_K), device=device, generator=gen
    ).to(torch.int32)
    shard = slice(rank * rows, (rank + 1) * rows)

    got_weights, got_ids = sp.all_gather_routing(weights[shard], ids[shard])

    assert got_weights.dtype == weights.dtype and got_ids.dtype == ids.dtype
    torch.testing.assert_close(got_weights, weights, atol=0, rtol=0)
    torch.testing.assert_close(got_ids, ids, atol=0, rtol=0)


def _check_padded_gather_scatter(
    device: torch.device, tp_world: int, rank: int
) -> None:
    """Token counts that do not divide TP round-trip through the padded shards."""
    group = get_tp_group().device_group
    for num_tokens in (tp_world * 8 - 3, tp_world * 8 + 1, tp_world * 8):
        padded = -(-num_tokens // tp_world) * tp_world
        rows = padded // tp_world
        gen = torch.Generator(device=device).manual_seed(num_tokens)
        full = torch.randn(padded, 64, device=device, generator=gen)
        full[num_tokens:] = 0

        gathered = sp.gather_tokens(full[rank * rows : (rank + 1) * rows], num_tokens)
        torch.manual_seed(rank + 1)
        partial = torch.randn(num_tokens, 64, device=device)
        scattered = sp.scatter_tokens(partial)

        torch.testing.assert_close(gathered, full[:num_tokens], atol=0, rtol=0)
        expected = F.pad(_all_reduced(partial, group), (0, 0, 0, padded - num_tokens))
        torch.testing.assert_close(
            scattered, expected[rank * rows : (rank + 1) * rows], atol=1e-5, rtol=1e-5
        )

        data = full.to(current_platform.fp8_dtype())
        scale = torch.rand(padded, 1, device=device, generator=gen)
        mine = slice(rank * rows, (rank + 1) * rows)
        quantized = QuantizedActivation(
            data[mine], scale[mine], DTYPE, torch.Size((rows, 64)), kFp8DynamicTokenSym
        )
        for keep, expected_rows in ((num_tokens, num_tokens), (None, padded)):
            got = sp.gather_activation(quantized, keep)
            assert isinstance(got, QuantizedActivation)
            assert got.orig_shape == (expected_rows, 64)
            assert got.orig_dtype == DTYPE and got.quant_key == kFp8DynamicTokenSym
            torch.testing.assert_close(
                got.data.view(torch.uint8),
                data[:expected_rows].view(torch.uint8),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(got.scale, scale[:expected_rows], atol=0, rtol=0)


_CHECKS = {
    "sharded_tail": _check_sharded_tail,
    "routing_gather": _check_routing_gather,
    "padded_gather_scatter": _check_padded_gather_scatter,
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
def test_sharded_tail_tp4_matches_replicated_rows() -> None:
    _run_ranks("sharded_tail", 4)


@multi_gpu_test(num_gpus=8)
def test_sharded_tail_tp8_matches_replicated_rows() -> None:
    _run_ranks("sharded_tail", 8)


@multi_gpu_test(num_gpus=4)
def test_routing_gather_tp4_is_exact() -> None:
    _run_ranks("routing_gather", 4)


@multi_gpu_test(num_gpus=8)
def test_padded_gather_scatter_tp8() -> None:
    _run_ranks("padded_gather_scatter", 8)


class _Router(torch.nn.Module):
    def select_experts(self, hidden_states, router_logits, **kwargs):
        return "routed"


def test_pre_routed_overrides_then_restores() -> None:
    router = _Router()
    topk = (torch.ones(2, TOP_K), torch.zeros(2, TOP_K, dtype=torch.int32))

    with sp.pre_routed(router, topk):
        assert router.select_experts(hidden_states=None, router_logits=None) is topk
    assert router.select_experts(None, None) == "routed"
    assert "select_experts" not in router.__dict__

    with pytest.raises(RuntimeError), sp.pre_routed(router, topk):
        raise RuntimeError
    assert router.select_experts(None, None) == "routed"


def test_pre_routed_keeps_an_instance_override() -> None:
    router = _Router()
    router.__dict__["select_experts"] = lambda *args, **kwargs: "patched"

    with sp.pre_routed(router, ("w", "i")):
        assert router.select_experts(None, None) == ("w", "i")
    assert router.select_experts(None, None) == "patched"


@pytest.mark.parametrize(
    "min_tokens,tp_size,num_tokens,has_residual,has_aux,last_rank,dp_size,expected",
    [
        (1024, 8, 16384, False, False, True, 1, True),
        (1024, 8, 1024, False, False, True, 1, True),
        (0, 8, 16384, False, False, True, 1, False),
        (1024, 1, 16384, False, False, True, 1, False),
        (1024, 8, 1016, False, False, True, 1, False),
        (1024, 8, 16381, False, False, True, 1, True),
        (1024, 8, 16384, True, False, True, 1, False),
        (1024, 8, 16384, False, True, True, 1, False),
        (1024, 8, 16384, False, False, False, 1, False),
        (1024, 8, 16384, False, False, True, 2, False),
    ],
)
def test_should_shard(
    monkeypatch: pytest.MonkeyPatch,
    min_tokens: int,
    tp_size: int,
    num_tokens: int,
    has_residual: bool,
    has_aux: bool,
    last_rank: bool,
    dp_size: int,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS", str(min_tokens))
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(
        sp, "get_pp_group", lambda: SimpleNamespace(is_last_rank=last_rank)
    )
    monkeypatch.setattr(sp, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size))
    assert sp.should_shard(num_tokens, has_residual, has_aux) is expected
