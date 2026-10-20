# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the FlashInfer MoE-EP adapter on Blackwell and Hopper.

Each test spawns one process per GPU, bootstraps vLLM's expert-parallel
groups, builds a ``FlashInferMoeEp`` adapter from bf16 expert weights at the
MoE geometry of a real model, warms it up and runs a forward whose inputs are
chosen so that every quantization step in the megakernel is exact, for the
CuTeDSL kernel (NVFP4 activations and weights, E4M3 scales per 16) as well as
the DeepGEMM kernel (FP8 activations and FP4 weights, power-of-two scales per
32). The expected output is then known in closed form and compared bit for
bit. Hopper tests use packed MXFP4 weights at the DSV4.1 and DSpark geometry,
checking remote routing, clamped SwiGLU, and CUDA graph replay.

A test that needs no SM100 also checks the adapter's kwargs against the
installed FlashInfer's config types, so config drift fails on any CI runner
with FlashInfer installed.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.distributed.eplb_utils import distributed_run, set_env_vars_and_device
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
    FLASHINFER_MOE_EP_SM90,
)
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
    FlashInferMoeEpExperts,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm


@dataclass(frozen=True)
class Geometry:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


# Routed-expert geometry of the models the branch was evaluated on. DeepGEMM
# keeps one UE8M0 scale byte per 32 elements and requires the per-token scale
# rows to be TMA aligned, so hidden and intermediate sizes must be multiples
# of 512 (FlashInfer's own validation only checks 128); all four qualify.
GEOMETRIES = (
    Geometry("dsv4-pro", 7168, 3072, 384, 6),
    Geometry("dsv4-flash", 4096, 2048, 256, 6),
    Geometry("mistral-small-4", 4096, 2048, 128, 4),
    Geometry("mistral-large-3", 7168, 4096, 128, 4),
)
BACKENDS = (FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM)

NUM_TOKENS_PER_RANK = 4
MAX_TOKENS_PER_RANK = 128

# Every block scale below is either 1 or a power of two, so the FP4 codes
# and the FP8 values survive quantization exactly: NVFP4 scales a 16-block by
# absmax / 6 in E4M3, DeepGEMM scales a 32-block by absmax / 6 (FP4) or
# absmax / 448 (FP8) rounded up to a power of two. A 6.0 in a block pins the
# FP4 scale to 1; 2.0, 6.0 and 144 are exact FP8 values at any such scale.
#
# Token t carries X_SIGNAL at hidden column 16 * t and a 6.0 sentinel next to
# it; every weight column the sentinel meets is zero.
X_SIGNAL = 2.0
X_SENTINEL = 6.0
# Gate and up rows are identical, so h = silu(12) * 12 = 144 = 6 * 24
# whichever half the kernel treats as the gate, and 144 requantizes exactly.
W13_DIAG = 6.0
H_SIGNAL = 144.0
# Expert e writes E2M1_CODES[e % 7] * H_SIGNAL into output channel e. The 6.0
# sentinel sits next to a zero intermediate channel and pins the block scale.
W2_SENTINEL = 6.0
E2M1_CODES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _expert_out(expert: int) -> float:
    return E2M1_CODES[expert % len(E2M1_CODES)]


def _topk_weights(top_k: int) -> tuple[float, ...]:
    # The DeepSeek-V4 routing policy applies the router weight before fc2, so
    # weight * 144 must requantize exactly too: powers of two do.
    return tuple(2.0 ** -(k + 1) for k in range(top_k))


def _routing_method(backend: str) -> RoutingMethodType:
    if backend in (FLASHINFER_MOE_EP_DEEP_GEMM, FLASHINFER_MOE_EP_SM90):
        return RoutingMethodType.DeepseekV4
    return RoutingMethodType.Default


def _routing(
    geometry: Geometry, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """top_k distinct experts per token, strided across ranks so dispatch goes
    remote."""
    base = (
        torch.arange(NUM_TOKENS_PER_RANK, device=device) + rank * NUM_TOKENS_PER_RANK
    ) % geometry.num_experts
    stride = geometry.num_experts // geometry.top_k
    offsets = torch.arange(geometry.top_k, device=device) * stride
    topk_ids = ((base[:, None] + offsets[None, :]) % geometry.num_experts).to(
        torch.int32
    )
    topk_weights = torch.tensor(
        [_topk_weights(geometry.top_k)] * NUM_TOKENS_PER_RANK,
        dtype=torch.float32,
        device=device,
    )
    return topk_ids, topk_weights


def _hidden_states(geometry: Geometry, device: torch.device) -> torch.Tensor:
    x = torch.zeros(
        NUM_TOKENS_PER_RANK, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    for t in range(NUM_TOKENS_PER_RANK):
        x[t, 16 * t] = X_SIGNAL
        x[t, 16 * t + 1] = X_SENTINEL
    return x


def _local_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    w13 = torch.zeros(
        num_local_experts,
        2 * geometry.intermediate_size,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.zeros(
        num_local_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    token_columns = [16 * t for t in range(NUM_TOKENS_PER_RANK)]
    for local in range(num_local_experts):
        expert = rank * num_local_experts + local
        w13[local, 0, token_columns] = W13_DIAG
        w13[local, geometry.intermediate_size, token_columns] = W13_DIAG
        w2[local, expert, 0] = _expert_out(expert)
        w2[local, expert, 2] = W2_SENTINEL
    return FlashInferMoeEpWeights(w13=w13, w2=w2)


def _expected_output(
    geometry: Geometry, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    expected = torch.zeros(
        NUM_TOKENS_PER_RANK,
        geometry.hidden_size,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    for t in range(NUM_TOKENS_PER_RANK):
        for k in range(geometry.top_k):
            expert = int(topk_ids[t, k])
            weight = float(topk_weights[t, k])
            expected[t, expert] += weight * _expert_out(expert) * H_SIGNAL
    return expected.to(torch.bfloat16)


def _moe_config(
    vllm_config: VllmConfig,
    geometry: Geometry,
    world_size: int,
    device: torch.device,
    backend: str,
    sequence_parallel: bool = False,
) -> FusedMoEConfig:
    parallel = FusedMoEParallelConfig.make(
        tp_size_=world_size if sequence_parallel else 1,
        pcp_size_=1,
        dp_size_=1 if sequence_parallel else world_size,
        sp_size_=world_size if sequence_parallel else 1,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    return FusedMoEConfig(
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_local_experts=geometry.num_experts // world_size,
        num_logical_experts=geometry.num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=_routing_method(backend),
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend=backend,
        max_num_tokens=1024 if sequence_parallel else MAX_TOKENS_PER_RANK,
    )


def _check_expert_views_alias_kernel_memory(
    adapter: FlashInferMoeEp,
    geometry: Geometry,
    rank: int,
    num_local_experts: int,
    device: torch.device,
) -> None:
    """EPLB moves experts through ``expert_weight_views``; the kernel must see it.

    Swap local experts 0 and 1 in place on every rank, then route through the
    swapped physical slots: the output must be the one the unswapped logical
    experts would have produced.
    """
    for view in adapter.expert_weight_views():
        first = view[0].clone()
        view[0].copy_(view[1])
        view[1].copy_(first)
    torch.distributed.barrier()

    topk_ids, topk_weights = _routing(geometry, rank, device)
    slot = topk_ids % num_local_experts
    physical_ids = torch.where(slot == 0, topk_ids + 1, topk_ids)
    physical_ids = torch.where(slot == 1, topk_ids - 1, physical_ids)
    output = adapter(_hidden_states(geometry, device), physical_ids, topk_weights)
    torch.accelerator.synchronize()
    assert torch.equal(output, _expected_output(geometry, topk_ids, topk_weights))


@torch.inference_mode()
def _init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    backend: str,
    sequence_parallel: bool = False,
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = (
        1 if sequence_parallel else world_size
    )
    vllm_config.parallel_config.tensor_parallel_size = (
        world_size if sequence_parallel else 1
    )
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = backend

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size if sequence_parallel else 1,
            pipeline_model_parallel_size=1,
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(
            vllm_config, geometry, world_size, device, backend, sequence_parallel
        )
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            kMxfp4Static,
            None,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        hidden_states = _hidden_states(geometry, device)
        if backend == FLASHINFER_MOE_EP_SM90:
            # Pin the per-token FP8 scale to 1 so X_SIGNAL stays exact.
            hidden_states[:, 1::16] = 448
            moe.swiglu_limit = 10.0
            weights = _sm90_packed_weights(
                geometry, rank, moe.num_local_experts, device
            )
        else:
            weights = _local_expert_weights(
                geometry, rank, moe.num_local_experts, device
            )
        adapter = FlashInferMoeEp(
            moe,
            weights,
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            output = adapter(hidden_states, topk_ids, topk_weights)
            torch.accelerator.synchronize()

            expected = _expected_output(geometry, topk_ids, topk_weights)
            if backend == FLASHINFER_MOE_EP_SM90:
                # Distinct gate/up values also guard their checkpoint ordering.
                # clamp(gate=12)=10; SiLU(10)*up=4 rounds to 40 in BF16.
                expected = (expected.float() * (40.0 / H_SIGNAL)).bfloat16()
            assert output.dtype == expected.dtype and output.shape == expected.shape
            if backend == FLASHINFER_MOE_EP_SM90:
                torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
            else:
                assert torch.equal(output, expected), (
                    f"rank {rank}: max abs diff "
                    f"{(output.float() - expected.float()).abs().max().item()}, "
                    f"nonzero output channels per token "
                    f"{[torch.nonzero(row).flatten().tolist() for row in output]}"
                )

            if backend == FLASHINFER_MOE_EP_CUTEDSL:
                _check_expert_views_alias_kernel_memory(
                    adapter, geometry, rank, moe.num_local_experts, device
                )
            if backend == FLASHINFER_MOE_EP_SM90:
                if sequence_parallel:
                    # Cross the small/decode workspace boundary, then capture
                    # the smaller bucket again to check masking and reuse.
                    large = adapter(
                        hidden_states.repeat(64, 1),
                        topk_ids.repeat(64, 1),
                        topk_weights.repeat(64, 1),
                    )
                    torch.accelerator.synchronize()
                    torch.testing.assert_close(
                        large, expected.repeat(64, 1), rtol=0.01, atol=0.01
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = adapter(hidden_states, topk_ids, topk_weights)
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(captured, expected, rtol=0.01, atol=0.01)
                hidden_states.zero_()
                graph.replay()
                torch.accelerator.synchronize()
                assert torch.count_nonzero(captured) == 0
                graph.reset()
        finally:
            adapter.destroy()


def _sm90_packed_weights(geometry, rank, num_local_experts, device):
    """Exact E2M1 sparse experts with raw K32 E8M0 scales."""
    h, i = geometry.hidden_size, geometry.intermediate_size
    w13 = torch.zeros(
        (num_local_experts, 2 * i, h // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.zeros((num_local_experts, h, i // 2), dtype=torch.uint8, device=device)
    for local in range(num_local_experts):
        expert = rank * num_local_experts + local
        for t in range(NUM_TOKENS_PER_RANK):
            w13[local, 0, 8 * t] = 7
            w13[local, i, 8 * t] = 4
        w2[local, expert, 0] = expert % 7 + 1
        w2[local, expert, 1] = 7
    return FlashInferMoeEpWeights(
        w13,
        w2,
        torch.full(
            (num_local_experts, 2 * i, h // 32), 127, dtype=torch.uint8, device=device
        ),
        torch.full(
            (num_local_experts, h, i // 32), 127, dtype=torch.uint8, device=device
        ),
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires Hopper"
)
@pytest.mark.parametrize(
    "geometry,sequence_parallel",
    [
        (Geometry("dsv41", 5120, 2304, 384, 6), False),
        (Geometry("dsv41-dspark", 5120, 2304, 128, 3), False),
        (Geometry("dsv41-sp", 5120, 2304, 384, 6), True),
    ],
    ids=["dsv41", "dsv41-dspark", "dsv41-sp"],
)
def test_sm90_dsv41_packed_weights_dispatch_and_clamp(geometry, sequence_parallel):
    """Exercise real model shapes and remote routing through the adapter."""
    ep = pytest.importorskip("flashinfer.moe_ep")
    if not hasattr(ep, "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig"):
        pytest.skip("FlashInfer does not have the SM90 MXFP4 megakernel")
    if torch.accelerator.device_count() < 4:
        pytest.skip("Need four GPUs")
    distributed_run(
        _init_warmup_forward, 4, geometry, FLASHINFER_MOE_EP_SM90, sequence_parallel
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="The FlashInfer MoE-EP megakernels require SM100",
)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=[g.name for g in GEOMETRIES])
@pytest.mark.parametrize("backend", BACKENDS, ids=("cutedsl", "deep_gemm"))
@pytest.mark.parametrize("world_size", [2, 4])
def test_init_warmup_forward_is_exact_on_grid_inputs(
    world_size: int, backend: str, geometry: Geometry
):
    pytest.importorskip("flashinfer.moe_ep")
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM and not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(_init_warmup_forward, world_size, geometry, backend)


@pytest.mark.parametrize(
    "backend",
    (*BACKENDS, FLASHINFER_MOE_EP_SM90),
    ids=("cutedsl", "deep_gemm", "sm90"),
)
def test_adapter_kwargs_match_installed_flashinfer(
    monkeypatch: pytest.MonkeyPatch, backend: str
):
    """The adapter's kwargs construct the installed FlashInfer's config types.

    Needs no SM100, so config kwarg drift fails in CI rather than at model load.
    Fakes only the megakernel layer (symmetric memory), the EP group, the device
    index and the DeepGEMM import.
    """
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    if backend == FLASHINFER_MOE_EP_SM90 and not hasattr(
        moe_ep, "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig"
    ):
        pytest.skip("FlashInfer does not have the SM90 MXFP4 megakernel")
    layer_args: list[tuple] = []

    class FakeMegaLayer:
        def __init__(self, *args) -> None:
            layer_args.append(args)

        def destroy(self) -> None:
            pass

    monkeypatch.setattr(moe_ep, "MoEEpMegaLayer", FakeMegaLayer)
    monkeypatch.setattr(fi_ep, "_expose_deep_gemm_to_flashinfer", lambda: None)
    monkeypatch.setattr(
        fi_ep,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    geometry = GEOMETRIES[0]
    moe = SimpleNamespace(
        moe_backend=backend,
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        swiglu_limit=10.0,
        routing_method=_routing_method(backend),
        sp_size=1,
        dp_size=2,
        pcp_size=1,
    )
    weight = torch.zeros(1, dtype=torch.bfloat16)
    adapter = FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=weight, w2=weight),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
    )
    tensors = adapter._tensors(weight, torch.zeros(1, dtype=torch.int32), torch.ones(1))
    expected_dtype = torch.int64 if backend == FLASHINFER_MOE_EP_SM90 else torch.int32
    assert tensors.topk_ids.dtype == adapter.topk_indices_dtype == expected_dtype

    ((_, _, _, mega_config),) = layer_args
    if backend == FLASHINFER_MOE_EP_SM90:
        expected = moe_ep.Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
    else:
        expected = (
            moe_ep.Nvfp4CutedslMegaMoeConfig
            if backend == FLASHINFER_MOE_EP_CUTEDSL
            else moe_ep.DeepGemmMegaMoeConfig
        )
    assert isinstance(mega_config.megakernel, expected)
