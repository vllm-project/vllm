# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-sharded Kimi-K3 prefill on ROCm must equal the replicated path.

Each rank ends up holding only its own token rows, so the multi-GPU checks
compare every rank's shard with the matching rows of the all-reduce result.
"""

from collections.abc import Callable
from functools import partial
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
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
)
from vllm.models.kimi_k3.amd import sp
from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
from vllm.models.kimi_k3.amd.linear import (
    KimiDecoderLayer,
    KimiMLP,
    KimiMoE,
    KimiRoutedOutputTransform,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import init_workspace_manager

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


MLP_HIDDEN_SIZE = 512
# At least the 256 per rank that KimiMoE would otherwise pad up to.
MLP_INTERMEDIATE_PER_RANK = 256


def _build_mlp(
    device: torch.device, tp_world: int, kind: str
) -> tuple[KimiMLP | KimiMoE, VllmConfig]:
    """A small dense KimiMLP or latent KimiMoE with the same weights on every
    rank's view of the full layer.

    ``kind`` is "dense", "moe" (token-shardable tail) or "moe_no_shared"
    (without shared experts the tail is not token-shardable, so the layer
    takes the all-gather fallback).
    """
    init_workspace_manager(device)
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {}
    torch.set_default_dtype(DTYPE)
    mlp: KimiMLP | KimiMoE
    with set_current_vllm_config(vllm_config), torch.device(device):
        if kind == "dense":
            mlp = KimiMLP(
                MLP_HIDDEN_SIZE,
                MLP_INTERMEDIATE_PER_RANK * tp_world,
                "silu",
                prefix="model.layers.0.mlp",
            )
        else:
            config = KimiLinearConfig(
                hidden_size=MLP_HIDDEN_SIZE,
                hidden_act="silu",
                rms_norm_eps=EPS,
                moe_intermediate_size=MLP_INTERMEDIATE_PER_RANK * tp_world,
                num_experts=32,
                num_experts_per_token=4,
                num_shared_experts=None if kind == "moe_no_shared" else 1,
                routed_expert_hidden_size=256,
                latent_moe_use_norm=True,
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
            )
            mlp = KimiMoE(config, prefix="model.layers.1.mlp", layer_idx=1)
            assert mlp.experts.token_shardable is (kind == "moe")

    # Same seed on every rank: replicated weights match across ranks, and each
    # rank's TP shard is just a slice of some consistent full weight.
    torch.manual_seed(0)
    with torch.no_grad():
        for param in mlp.parameters():
            if param.dim() == 1:
                param.copy_(1 + 0.1 * torch.randn_like(param))
            else:
                param.copy_(torch.randn_like(param) / param.size(-1) ** 0.5)
    for module in mlp.modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)
    return mlp, vllm_config


def _check_mlp_rows(
    device: torch.device,
    tp_world: int,
    rank: int,
    kind: str,
    run_sharded: Callable[[SimpleNamespace, torch.Tensor], torch.Tensor],
) -> None:
    """``run_sharded`` on this rank's token shard returns this rank's rows of
    the unsharded layer. The token count does not divide TP, so the last
    rank's shard ends in zero padding rows."""
    mlp, vllm_config = _build_mlp(device, tp_world, kind)
    rows = 6
    num_tokens = tp_world * rows - 3
    torch.manual_seed(1)
    hidden = torch.randn(num_tokens, MLP_HIDDEN_SIZE, device=device, dtype=DTYPE)
    padded = F.pad(hidden, (0, 0, 0, tp_world * rows - num_tokens))
    mine = slice(rank * rows, (rank + 1) * rows)
    real = max(0, min(rows, num_tokens - rank * rows))
    layer = SimpleNamespace(
        mlp=mlp, _output_unreduced=KimiDecoderLayer._output_unreduced
    )
    with set_forward_context(None, vllm_config, num_tokens=num_tokens):
        expected = mlp(hidden)
        actual = run_sharded(layer, padded[mine])

    assert actual.shape == (rows, MLP_HIDDEN_SIZE)
    torch.testing.assert_close(
        actual[:real], expected[mine][:real], atol=8e-2, rtol=3e-2
    )


def _check_shard_routed_moe(device: torch.device, tp_world: int, rank: int) -> None:
    """A latent KimiMoE routed on the token shard, with the gathered top-k
    replacing routing."""
    _check_mlp_rows(
        device, tp_world, rank, "moe", KimiDecoderLayer._run_moe_sp_shard_routed
    )


def _check_mlp_sp(device: torch.device, tp_world: int, rank: int, kind: str) -> None:
    """The decoder layer's sharded MLP step: reduce-scatter of the unreduced
    down_proj for a dense MLP, all-gather then this rank's rows for an MoE
    whose tail is not token-shardable."""
    _check_mlp_rows(device, tp_world, rank, kind, KimiDecoderLayer._run_mlp_sp)


_CHECKS: dict[str, Callable[[torch.device, int, int], None]] = {
    "sharded_tail": _check_sharded_tail,
    "routing_gather": _check_routing_gather,
    "padded_gather_scatter": _check_padded_gather_scatter,
    "shard_routed_moe": _check_shard_routed_moe,
    "mlp_sp_dense": partial(_check_mlp_sp, kind="dense"),
    "mlp_sp_moe_no_shared": partial(_check_mlp_sp, kind="moe_no_shared"),
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


@multi_gpu_test(num_gpus=4)
def test_padded_gather_scatter_tp4() -> None:
    _run_ranks("padded_gather_scatter", 4)


@multi_gpu_test(num_gpus=8)
def test_padded_gather_scatter_tp8() -> None:
    _run_ranks("padded_gather_scatter", 8)


def _run_shard_routed_moe(
    tp_size: int, use_aiter: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", str(int(use_aiter)))
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", str(int(use_aiter)))
    _run_ranks("shard_routed_moe", tp_size)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("use_aiter", [False, True])
def test_shard_routed_moe_tp4_matches_unsharded_rows(
    use_aiter: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_shard_routed_moe(4, use_aiter, monkeypatch)


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("use_aiter", [False, True])
def test_shard_routed_moe_tp8_matches_unsharded_rows(
    use_aiter: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_shard_routed_moe(8, use_aiter, monkeypatch)


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("kind", ["dense", "moe_no_shared"])
def test_mlp_sp_tp4_matches_unsharded_rows(kind: str) -> None:
    _run_ranks(f"mlp_sp_{kind}", 4)


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


@pytest.mark.parametrize("calls", [0, 2])
def test_pre_routed_requires_exactly_one_route(calls: int) -> None:
    router = _Router()
    with pytest.raises(AssertionError), sp.pre_routed(router, ("w", "i")):
        for _ in range(calls):
            router.select_experts(None, None)
    assert router.select_experts(None, None) == "routed"


def test_pre_routed_keeps_an_instance_override() -> None:
    router = _Router()
    router.__dict__["select_experts"] = lambda *args, **kwargs: "patched"

    with sp.pre_routed(router, ("w", "i")):
        assert router.select_experts(None, None) == ("w", "i")
    assert router.select_experts(None, None) == "patched"


@pytest.mark.parametrize(
    "min_tokens,tp_size,num_tokens,has_residual,has_aux,last_rank,dp_size,observed,expected",
    [
        (1024, 8, 16384, False, False, True, 1, False, True),
        (1024, 8, 1024, False, False, True, 1, False, True),
        (0, 8, 16384, False, False, True, 1, False, False),
        (1024, 1, 16384, False, False, True, 1, False, False),
        (1024, 8, 1016, False, False, True, 1, False, False),
        (1024, 8, 16381, False, False, True, 1, False, True),
        (1024, 8, 16384, True, False, True, 1, False, False),
        (1024, 8, 16384, False, True, True, 1, False, False),
        (1024, 8, 16384, False, False, False, 1, False, False),
        (1024, 8, 16384, False, False, True, 2, False, False),
        (1024, 8, 16384, False, False, True, 1, True, False),
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
    observed: bool,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_KIMI_K3_AMD_PREFILL_SP_MIN_TOKENS", str(min_tokens))
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(
        sp, "get_pp_group", lambda: SimpleNamespace(is_last_rank=last_rank)
    )
    monkeypatch.setattr(sp, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size))
    assert sp.should_shard(num_tokens, has_residual, has_aux, observed) is expected
