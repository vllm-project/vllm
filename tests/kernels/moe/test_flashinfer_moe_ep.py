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

On Hopper the same exact forward runs the SM90 FP8 megakernel on block-FP8
weights (power-of-two scales per 128x128 block, FP8 activations per 1x128).

Tests that need no GPU also check the adapter's kwargs against the installed
FlashInfer's config types, so config drift fails on any CI runner with
FlashInfer installed.
"""

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
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
    sm90_fp8_block_moe_ep_weights,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
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
# Routed-expert geometry of block-FP8 checkpoints for the Hopper megakernel.
BLOCK_FP8_GEOMETRIES = (
    Geometry("glm-5.3-flash", 4096, 2048, 288, 8),
    Geometry("deepseek-v3", 7168, 2048, 256, 8),
    Geometry("qwen3-235b", 4096, 1536, 128, 8),
)

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
# FP8 activations take absmax / 448 per 1x128 group: 448 pins that scale to 1.
X_SENTINEL_FP8 = 448.0
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


def _hidden_states(
    geometry: Geometry, device: torch.device, sentinel: float = X_SENTINEL
) -> torch.Tensor:
    x = torch.zeros(
        NUM_TOKENS_PER_RANK, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    for t in range(NUM_TOKENS_PER_RANK):
        x[t, 16 * t] = X_SIGNAL
        x[t, 16 * t + 1] = sentinel
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


def _block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block FP8 with power-of-two scales: exact for the grid weights."""
    experts, rows, cols = weight.shape
    blocks = weight.float().view(experts, rows // 128, 128, cols // 128, 128)
    absmax = blocks.abs().amax(dim=(2, 4))
    scale = torch.where(
        absmax > 0,
        torch.exp2(torch.ceil(torch.log2(absmax / 448.0))),
        torch.ones_like(absmax),
    )
    quantized = blocks / scale[:, :, None, :, None]
    return quantized.view(experts, rows, cols).to(torch.float8_e4m3fn), scale


def _local_block_fp8_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    """The grid weights in block FP8, in the oracle's SM90 kernel layout."""
    weights = _local_expert_weights(geometry, rank, num_local_experts, device)
    w13, w13_scale = sm90_fp8_block_moe_ep_weights(*_block_fp8(weights.w13))
    w2, w2_scale = _block_fp8(weights.w2)
    return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)


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
        routing_method=_routing_method(backend),
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
    sentinel: float = X_SENTINEL,
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
    output = adapter(
        _hidden_states(geometry, device, sentinel), physical_ids, topk_weights
    )
    torch.accelerator.synchronize()
    assert torch.equal(output, _expected_output(geometry, topk_ids, topk_weights))


def _init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    backend: str,
    block_fp8: bool = False,
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
        if block_fp8:
            weight_key, activation_key = kFp8Static128BlockSym, kFp8Dynamic128Sym
            make_weights = _local_block_fp8_expert_weights
            sentinel = X_SENTINEL_FP8
        else:
            weight_key, activation_key = kMxfp4Static, None
            make_weights = _local_expert_weights
            sentinel = X_SENTINEL
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            weight_key,
            activation_key,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        adapter = FlashInferMoeEp(
            moe,
            make_weights(geometry, rank, moe.num_local_experts, device),
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            output = adapter(
                _hidden_states(geometry, device, sentinel), topk_ids, topk_weights
            )
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
                    adapter, geometry, rank, moe.num_local_experts, device, sentinel
                )
        finally:
            adapter.destroy()


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


@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="The block-FP8 FlashInfer MoE-EP megakernel requires SM90",
)
@pytest.mark.parametrize(
    "geometry", BLOCK_FP8_GEOMETRIES, ids=[g.name for g in BLOCK_FP8_GEOMETRIES]
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_block_fp8_init_warmup_forward_is_exact_on_grid_inputs(
    world_size: int, geometry: Geometry
):
    pytest.importorskip("flashinfer.moe_ep")
    pytest.importorskip("nvshmem.core")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(
        _init_warmup_forward, world_size, geometry, FLASHINFER_MOE_EP_CUTEDSL, True
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=("cutedsl", "deep_gemm"))
def test_adapter_kwargs_match_installed_flashinfer(
    monkeypatch: pytest.MonkeyPatch, backend: str
):
    """The adapter's kwargs construct the installed FlashInfer's config types.

    Needs no SM100, so config kwarg drift fails in CI rather than at model load.
    Fakes only the megakernel layer (symmetric memory), the EP group, the device
    index and the DeepGEMM import.
    """
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    layer_args = _fake_mega_layer(monkeypatch, moe_ep)
    moe = _adapter_moe(backend, GEOMETRIES[0])
    weight = torch.zeros(1, dtype=torch.bfloat16)
    FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=weight, w2=weight),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
    )

    ((_, _, _, mega_config),) = layer_args
    expected = (
        moe_ep.Nvfp4CutedslMegaMoeConfig
        if backend == FLASHINFER_MOE_EP_CUTEDSL
        else moe_ep.DeepGemmMegaMoeConfig
    )
    assert isinstance(mega_config.megakernel, expected)


def test_block_fp8_adapter_hands_kernel_ready_weights_to_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
):
    """Block-FP8 weights build the SM90 config and skip FlashInfer's own
    preprocessing, which would requantize them from bf16."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    workspace_capacities: list[int] = []
    layer_args = _fake_mega_layer(monkeypatch, moe_ep, workspace_capacities)
    geometry = Geometry("tiny", 256, 128, 4, 2)
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, geometry)
    hidden, intermediate = geometry.hidden_size, geometry.intermediate_size
    w13 = torch.zeros(2, 2 * intermediate, hidden, dtype=torch.float8_e4m3fn)
    w2 = torch.zeros(2, hidden, intermediate, dtype=torch.float8_e4m3fn)
    w13_scale = torch.ones(2, 2 * intermediate // 128, hidden // 128)
    w2_scale = torch.ones(2, hidden // 128, intermediate // 128)
    FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
    )

    ((_, _, weight_pack, mega_config),) = layer_args
    assert weight_pack is None and not mega_config.preprocess_weights
    # Batches up to 64 tokens get their own workspace below the 128 maximum.
    assert workspace_capacities == [64]
    megakernel = mega_config.megakernel
    assert isinstance(megakernel, moe_ep.Sm90PullFp8MegaMoeConfig)
    assert megakernel.fp8_scale_mode == "blockwise"
    assert megakernel.gate_up_clamp == moe.swiglu_limit
    (fc1, fc1_scale, *_), (fc2, fc2_scale, *_) = mega_config.transformed_weights
    # (E, K, N) views over the layer's storage, K contiguous.
    assert fc1.shape == (2, hidden, 2 * intermediate) and fc1.stride(1) == 1
    assert fc2.shape == (2, intermediate, hidden) and fc2.stride(1) == 1
    assert fc1.data_ptr() == w13.data_ptr() and fc1_scale is w13_scale
    assert fc2.data_ptr() == w2.data_ptr() and fc2_scale is w2_scale


def _fake_mega_layer(
    monkeypatch: pytest.MonkeyPatch,
    moe_ep,
    workspace_capacities: list[int] | None = None,
) -> list[tuple]:
    layer_args: list[tuple] = []

    class FakeMegaLayer:
        def __init__(self, *args) -> None:
            layer_args.append(args)

        def create_workspace(self, max_tokens_per_rank: int) -> object:
            assert workspace_capacities is not None
            workspace_capacities.append(max_tokens_per_rank)
            return object()

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
    return layer_args


def _adapter_moe(backend: str, geometry: Geometry) -> SimpleNamespace:
    return SimpleNamespace(
        moe_backend=backend,
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        swiglu_limit=10.0,
        routing_method=_routing_method(backend),
    )
