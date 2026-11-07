# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the FlashInfer MoE-EP adapter on real Blackwell GPUs.

Each test spawns one process per GPU, bootstraps vLLM's expert-parallel
groups, builds a ``FlashInferMoeEp`` adapter from bf16 expert weights at the
MoE geometry of a real model, warms it up and runs a forward whose inputs are
chosen so that every quantization step in the megakernel is exact, for the
CuTeDSL kernel (NVFP4 activations and weights, E4M3 scales per 16) as well as
the DeepGEMM kernel (FP8 activations and FP4 weights, power-of-two scales per
32). The expected output is then known in closed form and compared bit for
bit.

On SM107 the CuTeDSL kernel instead takes checkpoint-format NVFP4 experts and
the epilogue calibrated from their activation scales, again exactly.

A test that needs no SM100 also checks the adapter's kwargs against the
installed FlashInfer's config types, so config drift fails on any CI runner
with FlashInfer installed.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.distributed.eplb_utils import distributed_run, set_env_vars_and_device
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM
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
    sm107_nvfp4_epilogue,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpEpilogue,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
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

# The SM107 test feeds checkpoint-format NVFP4 experts with distinct gate (6)
# and up (1) rows, so h = silu(12) * 2 requantizes to 24, whereas swapped
# halves would give silu(2) * 12, which requantizes to 21.
SM107_W13_GATE = 6.0
SM107_W13_UP = 1.0
SM107_H_SIGNAL = 24.0
SM107_ROUTE_CHECK_ENV = "FLASHINFER_SM107_MEGA_CHECK_ROUTES"
SM107_CASES = (
    (Geometry("dsv4.1-flash", 5120, 2304, 384, 6), RoutingMethodType.DeepseekV4),
    (GEOMETRIES[1], RoutingMethodType.Default),
)


def _expert_out(expert: int) -> float:
    return E2M1_CODES[expert % len(E2M1_CODES)]


def _topk_weights(top_k: int) -> tuple[float, ...]:
    # The DeepSeek-V4 routing policy applies the router weight before fc2, so
    # weight * 144 must requantize exactly too: powers of two do.
    return tuple(2.0 ** -(k + 1) for k in range(top_k))


def _routing_method(backend: str) -> RoutingMethodType:
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM:
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


def _unit_scale(expert: int) -> float:
    return 1.0


def _expected_output(
    geometry: Geometry,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    h_signal: float = H_SIGNAL,
    expert_scale: Callable[[int], float] = _unit_scale,
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
            expected[t, expert] += (
                weight * _expert_out(expert) * h_signal * expert_scale(expert)
            )
    return expected.to(torch.bfloat16)


def _e2m1_code(value: float) -> int:
    return fi_ep._E2M1_LUT.index(value)


def _sm107_expert_scale(expert: int) -> float:
    return 2.0 ** -(expert % 3)


def _local_expert_nvfp4_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> tuple[FlashInferMoeEpWeights, torch.Tensor, torch.Tensor]:
    """Checkpoint-format NVFP4 experts and their weight global scales.

    Every expert dequantizes to the same w13, but its block scale 2**j and
    global scale 2**-j differ per expert, so a misindexed fc1 alpha breaks
    exactness. Expert e's w2 global scale scales output channel e. Even
    columns are the low nibbles.
    """
    hidden, intermediate = geometry.hidden_size, geometry.intermediate_size
    experts = [rank * num_local_experts + local for local in range(num_local_experts)]
    w13 = torch.zeros(
        num_local_experts,
        2 * intermediate,
        hidden // 2,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.zeros(
        num_local_experts, hidden, intermediate // 2, dtype=torch.uint8, device=device
    )
    token_bytes = [16 * t // 2 for t in range(NUM_TOKENS_PER_RANK)]
    w13[:, 0, token_bytes] = _e2m1_code(SM107_W13_GATE)
    w13[:, intermediate, token_bytes] = _e2m1_code(SM107_W13_UP)
    for local, expert in enumerate(experts):
        w2[local, expert, 0] = _e2m1_code(_expert_out(expert))
        w2[local, expert, 1] = _e2m1_code(W2_SENTINEL)

    w13_block_scale = torch.tensor(
        [2.0 ** (expert % 4) for expert in experts], device=device
    )
    w13_scale = (
        w13_block_scale.view(-1, 1, 1)
        .expand(num_local_experts, 2 * intermediate, hidden // 16)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    w2_scale = (
        torch.ones((), device=device)
        .expand(num_local_experts, hidden, intermediate // 16)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    g1 = 1.0 / w13_block_scale
    g2 = torch.tensor(
        [_sm107_expert_scale(expert) for expert in experts],
        dtype=torch.float32,
        device=device,
    )
    return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale), g1, g2


def _moe_config(
    vllm_config: VllmConfig,
    geometry: Geometry,
    world_size: int,
    device: torch.device,
    backend: str,
    routing_method: RoutingMethodType | None = None,
) -> FusedMoEConfig:
    parallel = FusedMoEParallelConfig.make(
        tp_size_=1,
        pcp_size_=1,
        dp_size_=world_size,
        sp_size_=1,
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
        routing_method=(
            _routing_method(backend) if routing_method is None else routing_method
        ),
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend=backend,
        max_num_tokens=MAX_TOKENS_PER_RANK,
    )


def _check_expert_views_alias_kernel_memory(
    adapter: FlashInferMoeEp,
    geometry: Geometry,
    rank: int,
    num_local_experts: int,
    device: torch.device,
    h_signal: float = H_SIGNAL,
    expert_scale: Callable[[int], float] = _unit_scale,
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
    expected = _expected_output(
        geometry, topk_ids, topk_weights, h_signal, expert_scale
    )
    assert torch.equal(output, expected)


def _init_warmup_forward(
    env: dict[str, str], world_size: int, geometry: Geometry, backend: str
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = backend

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(vllm_config, geometry, world_size, device, backend)
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            kMxfp4Static,
            None,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        adapter = FlashInferMoeEp(
            moe,
            _local_expert_weights(geometry, rank, moe.num_local_experts, device),
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            output = adapter(_hidden_states(geometry, device), topk_ids, topk_weights)
            torch.accelerator.synchronize()

            expected = _expected_output(geometry, topk_ids, topk_weights)
            assert output.dtype == expected.dtype and output.shape == expected.shape
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
        finally:
            adapter.destroy()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or current_platform.is_device_capability(107),
    reason="The MXFP4 MoE-EP megakernels require SM100/SM103",
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


def _sm107_nvfp4_init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    routing_method: RoutingMethodType,
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = FLASHINFER_MOE_EP_CUTEDSL

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(
            vllm_config,
            geometry,
            world_size,
            device,
            FLASHINFER_MOE_EP_CUTEDSL,
            routing_method,
        )
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            kNvfp4Static,
            kNvfp4Dynamic,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        num_local_experts = moe.num_local_experts
        weights, g1, g2 = _local_expert_nvfp4_weights(
            geometry, rank, num_local_experts, device
        )
        g1_orig, g2_orig = g1.clone(), g2.clone()
        # Ranks disagree on the activation scales; the EP maximum wins.
        epilogue = sm107_nvfp4_epilogue(
            FlashInferMoeEpEpilogue(fc1_alpha=g1, fc2_alpha=g2),
            torch.full((num_local_experts, 2), 2.0 ** -(3 + rank % 2), device=device),
            torch.full((num_local_experts,), 2.0 ** -(4 + rank % 2), device=device),
        )
        assert epilogue.fc1_alpha is not None and epilogue.fc2_alpha is not None
        assert epilogue.fc1_norm_const is not None
        assert epilogue.fc1_alpha.data_ptr() == g1.data_ptr()
        assert epilogue.fc2_alpha.data_ptr() == g2.data_ptr()
        # Without the EP reduce only odd ranks fail; fail all ranks together.
        reduced = torch.tensor(
            float(
                epilogue.input_norm_const == 8.0
                and torch.equal(g1, g1_orig / 8)
                and torch.equal(g2, g2_orig / 16)
                and torch.equal(epilogue.fc1_norm_const, torch.full_like(g2, 16.0))
            ),
            device=device,
        )
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.MIN)
        assert reduced.item() == 1.0, (
            f"EP reduce of the activation scales failed on some rank; rank {rank} "
            f"has input_norm_const={epilogue.input_norm_const}"
        )

        adapter = FlashInferMoeEp(
            moe, weights, epilogue, apply_topk_in_fc1=apply_topk_in_fc1(moe)
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            output = adapter(_hidden_states(geometry, device), topk_ids, topk_weights)
            torch.accelerator.synchronize()

            expected = _expected_output(
                geometry, topk_ids, topk_weights, SM107_H_SIGNAL, _sm107_expert_scale
            )
            assert output.dtype == expected.dtype and output.shape == expected.shape
            assert torch.equal(output, expected), (
                f"rank {rank}: max abs diff "
                f"{(output.float() - expected.float()).abs().max().item()}"
            )

            _check_expert_views_alias_kernel_memory(
                adapter,
                geometry,
                rank,
                num_local_experts,
                device,
                SM107_H_SIGNAL,
                _sm107_expert_scale,
            )
        finally:
            adapter.destroy()


@pytest.mark.skipif(
    not current_platform.is_device_capability(107),
    reason="The SM107 NVFP4 MoE-EP megakernel requires SM107",
)
@pytest.mark.parametrize(
    "geometry, routing_method",
    SM107_CASES,
    ids=[geometry.name for geometry, _ in SM107_CASES],
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_sm107_nvfp4_init_warmup_forward_is_exact_on_grid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    geometry: Geometry,
    routing_method: RoutingMethodType,
):
    """Checkpoint NVFP4 experts and the calibrated epilogue on SM107.

    Covers the prequantized weight handoff, gate/up order, per-expert and
    EP-local epilogue indexing, the EP reduce of the activation scales and
    FlashInfer's owned-output forward.
    """
    pytest.importorskip("flashinfer.moe_ep")
    gap = fi_ep.flashinfer_sm107_nvfp4_gap()
    if gap is not None:
        pytest.skip(f"kernel does not support {gap}")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    # Spawned workers inherit it before CuTe DSL picks a target.
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    distributed_run(
        _sm107_nvfp4_init_warmup_forward, world_size, geometry, routing_method
    )


@pytest.mark.parametrize(
    "backend, capability, routing",
    [
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 0), RoutingMethodType.Default),
        (FLASHINFER_MOE_EP_DEEP_GEMM, (10, 0), RoutingMethodType.DeepseekV4),
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 7), RoutingMethodType.DeepseekV4),
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 7), RoutingMethodType.Default),
    ],
    ids=["cutedsl", "deep_gemm", "cutedsl-sm107", "cutedsl-sm107-default"],
)
def test_adapter_kwargs_match_installed_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    capability: tuple[int, int],
    routing: RoutingMethodType,
):
    """The adapter's kwargs construct the installed FlashInfer's config types.

    Needs no SM100 or SM107, so config kwarg drift fails in CI rather than at
    model load. Fakes only the megakernel layer (symmetric memory), the device
    capability, the EP group, the device index and the DeepGEMM import. On
    SM107 the routing flag must reach the config.
    """
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    on_sm107 = capability == (10, 7)
    layer_args: list[tuple] = []

    class FakeMegaLayer:
        def __init__(self, *args) -> None:
            layer_args.append(args)

        def destroy(self) -> None:
            pass

    monkeypatch.setattr(moe_ep, "MoEEpMegaLayer", FakeMegaLayer)
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: capability)
    monkeypatch.setattr(fi_ep, "_expose_deep_gemm_to_flashinfer", lambda: None)
    monkeypatch.setattr(
        fi_ep,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    # Record the original value so teardown also removes what the adapter sets.
    monkeypatch.setenv(SM107_ROUTE_CHECK_ENV, "x")
    monkeypatch.delenv(SM107_ROUTE_CHECK_ENV)

    geometry = GEOMETRIES[0]
    moe = SimpleNamespace(
        moe_backend=backend,
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        swiglu_limit=10.0,
        routing_method=routing,
    )
    weight = torch.zeros(1, dtype=torch.bfloat16)
    adapter = FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=weight, w2=weight),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
    )

    ((_, _, _, mega_config),) = layer_args
    megakernel = mega_config.megakernel
    if on_sm107:
        assert isinstance(
            megakernel, moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        )
        assert megakernel.knobs is None
        assert megakernel.apply_topk_in_fc1 is (routing is RoutingMethodType.DeepseekV4)
        assert adapter._return_workspace_view is False
        assert os.environ[SM107_ROUTE_CHECK_ENV] == "0"
    else:
        expected = (
            moe_ep.Nvfp4CutedslMegaMoeConfig
            if backend == FLASHINFER_MOE_EP_CUTEDSL
            else moe_ep.DeepGemmMegaMoeConfig
        )
        assert isinstance(megakernel, expected)
        assert adapter._return_workspace_view is (
            routing is RoutingMethodType.DeepseekV4
        )
        assert SM107_ROUTE_CHECK_ENV not in os.environ
