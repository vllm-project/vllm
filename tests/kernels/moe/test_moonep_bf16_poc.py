# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test MoonEP dispatch / prefetch / combine logic (BF16 PoC).

Runs MoonEPPrepareAndFinalize plus a reference segment-loop expert runner
over MoonEP's expert-grouped ``[NvS, H]`` layout and compares against the
pure-PyTorch reference MoE. Requires NVSwitch multicast capable GPUs.
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from tests.kernels.moe.utils import make_test_weights
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.utils.import_utils import has_moonep
from vllm.utils.torch_utils import set_random_seed

from ...utils import multi_gpu_test
from .parallel_utils import ProcessGroupInfo, parallel_launch

if has_moonep():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (
        MoonEPExpertWeightLayout,
        MoonEPPrepareAndFinalize,
        make_moonep_weight_layout,
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
        # Skewed router logits so the planner has to duplicate hot experts.
        logits = config.router_skew * torch.randn(
            config.m, config.num_experts, device="cuda", dtype=torch.float32
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
    weight_layout: "MoonEPExpertWeightLayout",
) -> torch.Tensor:
    """Segment loop over MoonEP's expert-grouped layout.

    ``prefetch_weight`` has already materialized redundant experts' weights
    in rows ``[E, E+B)``, so every segment reads its own row. Route weights
    are applied here; MoonEP's combine does the K-sum.
    """
    output = torch.empty_like(hidden_nvsh)
    prev = 0
    for row, cur in enumerate(cu_seqlens.tolist()):
        if cur == prev:
            continue
        x = hidden_nvsh[prev:cur]
        gate = F.linear(x, weight_layout.full_gate_weight[row])
        up = F.linear(x, weight_layout.full_up_weight[row])
        y = F.linear(F.silu(gate) * up, weight_layout.full_down_weight[row])
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
    weight_layout: "MoonEPExpertWeightLayout",
):
    from moonep import Buffer
    from moonep._C import nvl_multicast_supported

    if not nvl_multicast_supported():
        raise MulticastNotAvailableError("NVSwitch multicast not available")

    buffer = Buffer(
        S=max_tokens_per_rank,
        H=hidden_size,
        K=topk,
        E=num_experts,
        num_ep_ranks=pgi.world_size,
        B=weight_layout.num_prefetch_slots,
        group=pg,
        explicitly_destroy=True,
    )
    return buffer, MoonEPPrepareAndFinalize(
        buffer=buffer,
        max_tokens_per_rank=max_tokens_per_rank,
        num_dispatchers=pgi.world_size,
        num_global_experts=num_experts,
        weight_layout=weight_layout,
    )


def moonep_moe_impl(
    pg: ProcessGroup,
    pgi: ProcessGroupInfo,
    test_tensors: TestTensors,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_prefetch_slots: int,
) -> torch.Tensor:
    config = test_tensors.config
    hidden_size = test_tensors.rank_tokens.shape[1]
    max_tokens_per_rank = 128 * ((config.m + 127) // 128)

    weight_layout = make_moonep_weight_layout(w1, w2, num_prefetch_slots)
    buffer, pf = make_moonep_prepare_finalize(
        pg,
        pgi,
        hidden_size,
        config.num_experts,
        config.topk,
        max_tokens_per_rank,
        weight_layout,
    )
    try:
        hidden_nvsh, _, _, _, route_weights_nvs = pf.prepare(
            test_tensors.rank_tokens,
            test_tensors.topk_weights,
            test_tensors.topk,
            num_experts=config.num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=_no_quant_config(),
        )
        assert route_weights_nvs is not None
        expert_out = reference_moonep_experts(
            hidden_nvsh, route_weights_nvs, pf.cu_seqlens, weight_layout
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
        buffer.destroy()


def _no_quant_config():
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    return FusedMoEQuantConfig.make(quant_dtype=None)


def _moonep_moe(
    pgi: ProcessGroupInfo,
    config: TestConfig,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_prefetch_slots: int,
):
    device_idx = torch.accelerator.current_device_index()
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
        )
        moonep_combined = moonep_moe_impl(
            pg, pgi, test_tensors, w1, w2, num_prefetch_slots
        )

    torch.testing.assert_close(
        torch_combined,
        moonep_combined,
        atol=6e-2,
        rtol=6e-2,
    )


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
@pytest.mark.parametrize("num_prefetch_slots", [4])
@pytest.mark.parametrize("world_size", [2])
@multi_gpu_test(num_gpus=2)
@requires_moonep
def test_moonep_bf16_moe(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    router_skew: float,
    num_prefetch_slots: int,
    world_size: int,
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
        parallel_launch(world_size, _moonep_moe, config, w1, w2, num_prefetch_slots)
    except Exception as exc:
        if "MulticastNotAvailableError" in str(exc):
            pytest.skip("NVSwitch multicast not available")
        raise
