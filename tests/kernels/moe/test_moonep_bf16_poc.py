# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test MoonEP dispatch / prefetch / combine logic (BF16).

Two paths, both compared against the pure-PyTorch reference MoE:
- MoonEPPrepareAndFinalize + a reference segment-loop expert runner over
  MoonEP's expert-grouped ``[NvS, H]`` layout;
- the full modular kernel, MoonEPPrepareAndFinalize + MoonEPExperts through
  FusedMoEKernel.
Each rank owns only its ``epn = E / world_size`` experts, placed through
``MoonEPExpertWeightPools`` exactly as the engine path does.
Requires NVSwitch multicast capable GPUs.
"""

import dataclasses
import types

import pytest
import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from tests.kernels.moe.utils import make_test_weights
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.utils.import_utils import has_moonep
from vllm.utils.torch_utils import set_random_seed

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

if has_moonep():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (
        MoonEPExpertWeightPools,
        MoonEPExpertWeights,
        MoonEPPrepareAndFinalize,
    )

requires_moonep = pytest.mark.skipif(
    not has_moonep(),
    reason="Requires MoonEP",
)


class MulticastNotAvailableError(RuntimeError):
    pass


@dataclasses.dataclass
class TestConfig:
    topk: int
    m: int
    k: int
    n: int
    num_experts: int
    router_skew: float
    apply_router_weight_on_input: bool = False


@dataclasses.dataclass
class TestTensors:
    rank_tokens: torch.Tensor
    topk: torch.Tensor
    topk_weights: torch.Tensor
    config: TestConfig

    @staticmethod
    def make(config: TestConfig) -> "TestTensors":
        rank_tokens = (
            torch.randn((config.m, config.k), device="cuda", dtype=torch.bfloat16) / 10
        )
        # Skewed routing via a per-expert bias: scaling logits by a scalar
        # preserves their ordering and would not change the selected topk
        # ids, but a shared bias makes a few experts hot on every token so
        # the planner has to fill redundant-expert prefetch slots.
        expert_bias = config.router_skew * torch.randn(
            1, config.num_experts, device="cuda", dtype=torch.float32
        )
        logits = (
            torch.randn(
                config.m, config.num_experts, device="cuda", dtype=torch.float32
            )
            + expert_bias
        )
        topk_weights, topk = torch.topk(logits, config.topk, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1)
        return TestTensors(
            rank_tokens=rank_tokens,
            topk=topk.to(dtype=torch.int64),
            topk_weights=topk_weights,
            config=config,
        )


def reference_moonep_experts(
    hidden_nvsh: torch.Tensor,
    route_weights_nvs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    weights: "MoonEPExpertWeights",
) -> torch.Tensor:
    """Segment loop over MoonEP's expert-grouped layout.

    ``prefetch_weight`` has already materialized redundant experts' weights
    in rows ``[epn, 2 * epn)``, so every segment reads its own row. Route
    weights are applied here; MoonEP's combine does the K-sum.
    """
    output = torch.empty_like(hidden_nvsh)
    prev = 0
    for row, cur in enumerate(cu_seqlens.tolist()):
        if cur == prev:
            continue
        x = hidden_nvsh[prev:cur]
        gate = F.linear(x, weights.gate[row])
        up = F.linear(x, weights.up[row])
        y = F.linear(F.silu(gate) * up, weights.down[row])
        y = y * route_weights_nvs[prev:cur].to(dtype=y.dtype).unsqueeze(-1)
        output[prev:cur].copy_(y)
        prev = cur
    if prev < hidden_nvsh.shape[0]:
        output[prev:].zero_()
    return output


def make_moonep_prepare_finalize(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    hidden_size: int,
    num_experts: int,
    topk: int,
    max_tokens_per_rank: int,
    weights: "MoonEPExpertWeights",
    pass_weights_to_pf: bool = True,
):
    from moonep._C import nvl_multicast_supported

    from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (
        MoonEPBufferPool,
    )

    if not nvl_multicast_supported():
        raise MulticastNotAvailableError("NVSwitch multicast not available")

    pool = MoonEPBufferPool(
        dict(
            H=hidden_size,
            K=topk,
            E=num_experts,
            num_ep_ranks=pgi.world_size,
            group=pg,
            explicitly_destroy=True,
        ),
        max_tokens_per_rank=max_tokens_per_rank,
    )
    return pool, MoonEPPrepareAndFinalize(
        buffer_pool=pool,
        max_tokens_per_rank=max_tokens_per_rank,
        num_dispatchers=pgi.world_size,
        num_global_experts=num_experts,
        expert_weights=weights if pass_weights_to_pf else None,
    )


def place_local_experts(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple["MoonEPExpertWeightPools", "MoonEPExpertWeights"]:
    """Place this rank's ``epn`` experts of the global ``w1``/``w2`` the way
    the engine's weight conversion does."""
    num_experts = w1.shape[0]
    assert num_experts % pgi.world_size == 0
    epn = num_experts // pgi.world_size
    local = slice(pgi.rank * epn, (pgi.rank + 1) * epn)
    pools = MoonEPExpertWeightPools(group=pg)
    weights = pools.build_expert_weights(w1[local].contiguous(), w2[local].contiguous())
    torch.testing.assert_close(weights.local_gate, w1[local, : w2.shape[2], :])
    torch.testing.assert_close(weights.local_down, w2[local])
    return pools, weights


def moonep_moe_impl(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    config = test_tensors.config
    hidden_size = test_tensors.rank_tokens.shape[1]
    max_tokens_per_rank = 128 * ((config.m + 127) // 128)

    pools, weights = place_local_experts(pg, pgi, w1, w2)
    pool, pf = make_moonep_prepare_finalize(
        pg,
        pgi,
        hidden_size,
        config.num_experts,
        config.topk,
        max_tokens_per_rank,
        weights,
    )
    try:
        hidden_nvsh, _, _, _, route_weights_nvs = pf.prepare(
            test_tensors.rank_tokens,
            test_tensors.topk_weights,
            test_tensors.topk,
            num_experts=config.num_experts,
            expert_map=None,
            apply_router_weight_on_input=config.apply_router_weight_on_input,
            quant_config=_no_quant_config(),
        )
        assert route_weights_nvs is not None
        if config.apply_router_weight_on_input:
            # prepare() already applied the weights to the inputs; the
            # runner must not apply them again.
            route_weights_nvs = torch.ones_like(route_weights_nvs)
        if config.router_skew >= 8 and config.m >= 100:
            # Heavy skew must actually engage the redundant-expert planner.
            assert (pf.plan.experts_to_copy >= 0).any(), (
                "no prefetch slot used despite heavy router skew"
            )
        expert_out = reference_moonep_experts(
            hidden_nvsh, route_weights_nvs, pf.cu_seqlens, weights
        )
        output = torch.empty_like(test_tensors.rank_tokens)
        pf.finalize(
            output,
            expert_out,
            test_tensors.topk_weights,
            test_tensors.topk,
            apply_router_weight_on_input=False,
            weight_and_reduce_impl=TopKWeightAndReduceNoOP(),
        )
        torch.accelerator.synchronize()
        return output
    finally:
        pool.destroy()
        pools.close()


def _no_quant_config():
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    return FusedMoEQuantConfig.make(quant_dtype=None)


def moonep_modular_kernel_impl(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Full modular-kernel path: MoonEPPrepareAndFinalize + MoonEPExperts."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.experts.moonep_experts import (
        MoonEPExperts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel

    config = test_tensors.config
    hidden_size = test_tensors.rank_tokens.shape[1]
    max_tokens_per_rank = 128 * ((config.m + 127) // 128)

    pools, weights = place_local_experts(pg, pgi, w1, w2)
    # weights deliberately not passed to the P/F: the engine path resolves
    # them through post_init_setup + the experts' weight hook.
    pool, pf = make_moonep_prepare_finalize(
        pg,
        pgi,
        hidden_size,
        config.num_experts,
        config.topk,
        max_tokens_per_rank,
        weights,
        pass_weights_to_pf=False,
    )
    try:
        moe_config = make_dummy_moe_config(
            num_experts=config.num_experts,
            experts_per_token=config.topk,
            hidden_dim=hidden_size,
            intermediate_size=config.n,
            max_num_tokens=max_tokens_per_rank,
        )
        experts = MoonEPExperts(moe_config=moe_config, quant_config=_no_quant_config())
        # Stand-in for the layer carrying the weights, as
        # convert_to_unquantized_kernel_format leaves it in the engine path.
        fake_layer = types.SimpleNamespace(_moonep_expert_weights=weights)
        experts.process_weights_after_loading(fake_layer)
        kernel = FusedMoEKernel(prepare_finalize=pf, fused_experts=experts)
        out = kernel.apply(
            hidden_states=test_tensors.rank_tokens,
            w1=weights.gate,
            w2=weights.down,
            topk_weights=test_tensors.topk_weights,
            topk_ids=test_tensors.topk,
            activation=MoEActivation.SILU,
            global_num_experts=config.num_experts,
            expert_map=None,
            apply_router_weight_on_input=config.apply_router_weight_on_input,
        )
        torch.accelerator.synchronize()
        return out
    finally:
        pool.destroy()
        pools.close()


def _moonep_moe(
    pgi: ProcessGroupInfo,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    use_modular_kernel: bool,
):
    from vllm.v1.worker.workspace import init_workspace_manager

    device_idx = torch.accelerator.current_device_index()
    init_workspace_manager(torch.device("cuda", device_idx))
    w1 = w1.to(device=device_idx)
    w2 = w2.to(device=device_idx)

    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    set_random_seed(7 + pgi.rank)
    test_tensors = TestTensors.make(config)

    with set_current_vllm_config(VllmConfig()):
        torch_combined = torch_experts(
            test_tensors.rank_tokens,
            w1,
            w2,
            test_tensors.topk_weights,
            test_tensors.topk,
            apply_router_weights_on_input=config.apply_router_weight_on_input,
        )
        impl = moonep_modular_kernel_impl if use_modular_kernel else moonep_moe_impl
        moonep_combined = impl(pg, pgi, test_tensors, w1, w2)

    torch.testing.assert_close(
        torch_combined,
        moonep_combined,
        atol=6e-2,
        rtol=6e-2,
    )


def _moonep_weight_pool_aliasing(pgi: ProcessGroupInfo, num_experts: int):
    """Two same-shaped layers must own independent local weights but share
    one physical prefetch pool, and the placed weights must outlive their
    source tensors."""
    device_idx = torch.accelerator.current_device_index()
    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    epn = num_experts // pgi.world_size
    n, k = 256, 512
    pools = MoonEPExpertWeightPools(group=pg)
    try:
        layers = []
        for seed in (1, 2):
            set_random_seed(seed)
            (_, w1, _, _), (_, w2, _, _) = make_test_weights(epn, n, k)
            src1 = w1.to(device=device_idx)
            src2 = w2.to(device=device_idx)
            expected = (src1.clone(), src2.clone())
            layers.append((pools.build_expert_weights(src1, src2), expected))
            del src1, src2
        torch.accelerator.empty_cache()
        (a, exp_a), (b, exp_b) = layers

        # One pool per projection, not per layer.
        assert len(pools._pools) == 3
        assert a.gate_prefetch_buffer is b.gate_prefetch_buffer

        # Local rows survive the release of their source tensors and are
        # independent between layers.
        torch.testing.assert_close(a.local_gate, exp_a[0][:, :n, :])
        torch.testing.assert_close(b.local_down, exp_b[1])
        assert a.gate.data_ptr() != b.gate.data_ptr()
        a.local_gate.fill_(3.0)
        torch.testing.assert_close(b.local_gate, exp_b[0][:, :n, :])

        # Prefetch rows are distinct virtual mappings of the same physical
        # slots: a write through layer A is visible through layer B and
        # through this rank's slice of the all-rank prefetch view.
        for proj in ("gate", "up", "down"):
            va, vb = getattr(a, proj), getattr(b, proj)
            va[epn:].fill_(float(pgi.rank + 1))
            torch.accelerator.synchronize()
            assert torch.equal(vb[epn:], va[epn:])
            assert torch.equal(
                getattr(a, f"{proj}_prefetch_buffer")[pgi.rank], va[epn:]
            )
            assert va[epn:].data_ptr() != vb[epn:].data_ptr()
        torch.distributed.barrier(group=pg)
        # Every rank sees every other rank's slots through the all-rank view.
        for r in range(pgi.world_size):
            assert torch.all(a.gate_prefetch_buffer[r] == float(r + 1))
        torch.distributed.barrier(group=pg)
    finally:
        pools.close()


@pytest.mark.parametrize("num_experts", [32])
@multi_gpu_test(num_gpus=2)
@requires_moonep
def test_moonep_weight_pool_aliasing(num_experts: int):
    try:
        parallel_launch(2, _moonep_weight_pool_aliasing, num_experts)
    except Exception as exc:
        if "MulticastNotAvailableError" in str(exc):
            pytest.skip("NVSwitch multicast not available")
        raise


MNKs = [
    (1, 256, 512),
    (37, 256, 512),
    (100, 512, 1024),
    (512, 768, 2048),
]


@pytest.mark.parametrize("m,n,k", MNKs)
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("router_skew", [1.0, 8.0])
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("use_modular_kernel", [False, True])
@multi_gpu_test(num_gpus=2)
@requires_moonep
def test_moonep_bf16_moe(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    router_skew: float,
    world_size: int,
    use_modular_kernel: bool,
):
    set_random_seed(7)
    config = TestConfig(
        topk=topk,
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
        router_skew=router_skew,
    )
    (_, w1, _, _), (_, w2, _, _) = make_test_weights(num_experts, n, k)

    try:
        parallel_launch(
            world_size,
            _moonep_moe,
            config,
            w1,
            w2,
            use_modular_kernel,
        )
    except Exception as exc:
        if "MulticastNotAvailableError" in str(exc):
            pytest.skip("NVSwitch multicast not available")
        raise


@pytest.mark.parametrize("use_modular_kernel", [False, True])
@multi_gpu_test(num_gpus=2)
@requires_moonep
def test_moonep_input_weighted_moe(use_modular_kernel: bool):
    """Llama-4-style routing applies route weights to the expert inputs."""
    set_random_seed(7)
    config = TestConfig(
        topk=1,
        m=100,
        k=1024,
        n=512,
        num_experts=32,
        router_skew=1.0,
        apply_router_weight_on_input=True,
    )
    (_, w1, _, _), (_, w2, _, _) = make_test_weights(
        config.num_experts, config.n, config.k
    )
    try:
        parallel_launch(2, _moonep_moe, config, w1, w2, use_modular_kernel)
    except Exception as exc:
        if "MulticastNotAvailableError" in str(exc):
            pytest.skip("NVSwitch multicast not available")
        raise
