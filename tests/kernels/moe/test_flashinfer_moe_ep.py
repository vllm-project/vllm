# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the FlashInfer MoE-EP adapter on Hopper and Blackwell GPUs.

Each test spawns one process per GPU, bootstraps vLLM's expert-parallel
groups, builds a ``FlashInferMoeEp`` adapter from bf16 expert weights at the
MoE geometry of a real model, warms it up and runs a forward whose inputs are
chosen so that every quantization step in the megakernel is exact, for the
CuTeDSL kernel (NVFP4 activations and weights, E4M3 scales per 16) as well as
the DeepGEMM kernel (FP8 activations and FP4 weights, power-of-two scales per
32). The expected output is then known in closed form and compared bit for
bit.

On Hopper the same exact forward runs the SM90 megakernels on block-FP8
weights (power-of-two scales per 128x128 block, FP8 activations per 1x128)
and on packed MXFP4 weights (unit E8M0 scales, FP8 activations per token).

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
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
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
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import (
    PassThroughPrepareAndFinalize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.worker.workspace import init_workspace_manager


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
# Routed-expert geometries of recent MoE models (Kimi-K3 at its 3584-wide
# latent), run with synthetic block-FP8 or MXFP4 weights on the Hopper
# megakernels.
SM90_GEOMETRIES = (
    Geometry("glm-5.3-flash", 4096, 2048, 288, 8),
    Geometry("deepseek-v4.1-flash", 5120, 2304, 384, 6),
    Geometry("kimi-k3", 3584, 3072, 896, 16),
    Geometry("qwen3.8-flash-next", 2560, 640, 512, 10),
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


def _routing_method(backend: str, sm90_format: str | None = None) -> RoutingMethodType:
    # DeepSeek-V4 models are the MXFP4 checkpoints Hopper serves.
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM or sm90_format == "mxfp4":
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


def _mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed E2M1 (low nibble first) with unit E8M0 scales: exact for the grid
    weights, which are all E2M1 values."""
    magnitudes = torch.tensor((0.0, *E2M1_CODES), device=weight.device)
    experts, rows, cols = weight.shape
    packed = torch.empty(
        experts, rows, cols // 2, dtype=torch.uint8, device=weight.device
    )
    for expert in range(experts):
        values = weight[expert].float()
        codes = torch.searchsorted(magnitudes, values)
        assert torch.equal(magnitudes[codes], values)
        packed[expert] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    # E8M0 127 encodes 2**0.
    scale = torch.full(
        (experts, rows, cols // 32), 127, dtype=torch.uint8, device=weight.device
    )
    return packed, scale


def _local_mxfp4_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    """The grid weights as an MXFP4 checkpoint stores them."""
    weights = _local_expert_weights(geometry, rank, num_local_experts, device)
    w13, w13_scale = _mxfp4(weights.w13)
    w2, w2_scale = _mxfp4(weights.w2)
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
    sm90_format: str | None = None,
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
        routing_method=_routing_method(backend, sm90_format),
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
    with torch.inference_mode():
        output = adapter(
            _hidden_states(geometry, device, sentinel), physical_ids, topk_weights
        )
    torch.accelerator.synchronize()
    assert torch.equal(output, _expected_output(geometry, topk_ids, topk_weights))


def _check_sm90_modular_kernel(
    vllm_config: VllmConfig,
    geometry: Geometry,
    moe: FusedMoEConfig,
    rank: int,
    device: torch.device,
    sm90_format: str,
) -> None:
    """Hopper weights through the loading hook and ``FusedMoEKernel``.

    Rank 1 has no tokens, but the other ranks route to its experts and it owns
    part of the collective, so it must still enter the megakernel. A full batch
    on every rank is then replayed from a CUDA graph. The DP-wide token counts
    come from the forward context, as in serving, so both use the 64-token
    workspace.
    """
    init_workspace_manager(device)
    if sm90_format == "fp8_block":
        weights = _local_block_fp8_expert_weights(
            geometry, rank, moe.num_local_experts, device
        )
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
            block_shape=[128, 128],
        )
        kernel_dtype = torch.float8_e4m3fn
    else:
        weights = _local_mxfp4_expert_weights(
            geometry, rank, moe.num_local_experts, device
        )
        quant_config = make_mxfp4_moe_quant_config(
            Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL, w1_scale=None, w2_scale=None
        )
        kernel_dtype = torch.uint8
    layer = torch.nn.Module()
    for name, tensor in zip(
        ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
        (weights.w13, weights.w13_scale, weights.w2, weights.w2_scale),
    ):
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.apply_router_weight_on_input = False
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    experts = FlashInferMoeEpExperts(moe, quant_config)
    experts.process_weights_after_loading(layer)
    assert layer.w13_weight.dtype == kernel_dtype
    kernel = mk.FusedMoEKernel(PassThroughPrepareAndFinalize(), experts)

    topk_ids, topk_weights = _routing(geometry, rank, device)
    hidden_states = _hidden_states(geometry, device, X_SENTINEL_FP8)
    expected = _expected_output(geometry, topk_ids, topk_weights)
    world_size = vllm_config.parallel_config.data_parallel_size
    full = [NUM_TOKENS_PER_RANK] * world_size
    idle = [0 if r == 1 else NUM_TOKENS_PER_RANK for r in range(world_size)]

    def forward(across_dp: list[int]) -> torch.Tensor:
        tokens = across_dp[rank]
        with (
            torch.inference_mode(),
            set_forward_context(
                None,
                vllm_config,
                num_tokens=tokens,
                num_tokens_across_dp=torch.tensor(across_dp, dtype=torch.int32),
            ),
        ):
            return kernel.apply(
                hidden_states[:tokens],
                layer.w13_weight,
                layer.w2_weight,
                topk_weights[:tokens],
                topk_ids[:tokens],
                MoEActivation.SILU,
                geometry.num_experts,
                None,
                False,
            )

    try:
        for across_dp in (full, idle):
            output = forward(across_dp)
            torch.accelerator.synchronize()
            assert torch.equal(output, expected[: across_dp[rank]])

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(full)
        with torch.inference_mode():
            output.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(output, expected)
    finally:
        assert experts._adapter is not None
        experts._adapter.destroy()


def _init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    backend: str,
    sm90_format: str | None = None,
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = backend
    if sm90_format is not None:
        # What KernelConfig binds for megakernel backends.
        vllm_config.parallel_config.all2all_backend = "passthrough"
        vllm_config.parallel_config.data_parallel_rank = int(env["RANK"])

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(
            vllm_config, geometry, world_size, device, backend, sm90_format
        )
        if sm90_format == "fp8_block":
            weight_key, activation_key = kFp8Static128BlockSym, kFp8Dynamic128Sym
            make_weights = _local_block_fp8_expert_weights
            sentinel = X_SENTINEL_FP8
        elif sm90_format == "mxfp4":
            # The kernel scales FP8 activations per token: 448 pins it to 1.
            weight_key, activation_key = kMxfp4Static, None
            make_weights = _local_mxfp4_expert_weights
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
            sm90_format=sm90_format,
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            # Serving runs the model under inference mode, where the
            # workspace warmup also allocates the kernel's buffers.
            with torch.inference_mode():
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

            # Hopper MXFP4 rejects EPLB.
            if backend == FLASHINFER_MOE_EP_CUTEDSL and sm90_format != "mxfp4":
                _check_expert_views_alias_kernel_memory(
                    adapter, geometry, rank, moe.num_local_experts, device, sentinel
                )
        finally:
            adapter.destroy()

        if sm90_format is not None:
            _check_sm90_modular_kernel(
                vllm_config, geometry, moe, rank, device, sm90_format
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


@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="The Hopper FlashInfer MoE-EP megakernels require SM90",
)
@pytest.mark.parametrize(
    "geometry", SM90_GEOMETRIES, ids=[g.name for g in SM90_GEOMETRIES]
)
@pytest.mark.parametrize("sm90_format", ["fp8_block", "mxfp4"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_sm90_init_warmup_forward_is_exact_on_grid_inputs(
    world_size: int, sm90_format: str, geometry: Geometry
):
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    pytest.importorskip("nvshmem.core")
    if sm90_format == "mxfp4" and not hasattr(
        moe_ep, "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig"
    ):
        pytest.skip("FlashInfer has no SM90 MXFP4 megakernel")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(
        _init_warmup_forward,
        world_size,
        geometry,
        FLASHINFER_MOE_EP_CUTEDSL,
        sm90_format,
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
    preprocessing, which would requantize them from bf16. The kernel tensors
    bound back to the layer keep their dtypes, so a rebuild from them (e.g.
    preprocessed-weight loading) selects the same kernel."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    workspace_capacities: list[int] = []
    layer_args = _fake_mega_layer(monkeypatch, moe_ep, workspace_capacities)
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, TINY_SM90)
    weights = _tiny_block_fp8_weights()
    adapter = FlashInferMoeEp(
        moe, weights, apply_topk_in_fc1=apply_topk_in_fc1(moe), sm90_format="fp8_block"
    )
    w13, w13_scale, w2, w2_scale = adapter.kernel_weights()
    FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        sm90_format="fp8_block",
    )

    assert [t.dtype for t in (w13, w13_scale, w2, w2_scale)] == [
        torch.float8_e4m3fn,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float32,
    ]
    (_, _, weight_pack, mega_config), (_, _, _, rebuilt) = layer_args
    assert isinstance(rebuilt.megakernel, moe_ep.Sm90PullFp8MegaMoeConfig)
    assert _kernel_storage(rebuilt) == _kernel_storage(mega_config)
    assert weight_pack is None and not mega_config.preprocess_weights
    # Batches up to 64 tokens get their own workspace below the 128 maximum.
    assert workspace_capacities == [64, 64]
    megakernel = mega_config.megakernel
    assert isinstance(megakernel, moe_ep.Sm90PullFp8MegaMoeConfig)
    assert megakernel.fp8_scale_mode == "blockwise"
    assert megakernel.gate_up_clamp == moe.swiglu_limit
    (fc1, _, *_), (fc2, _, *_) = mega_config.transformed_weights
    hidden, intermediate = TINY_SM90.hidden_size, TINY_SM90.intermediate_size
    # (E, K, N) views over the layer's storage, K contiguous.
    assert fc1.shape == (2, hidden, 2 * intermediate) and fc1.stride(1) == 1
    assert fc2.shape == (2, intermediate, hidden) and fc2.stride(1) == 1
    assert _kernel_storage(mega_config) == [
        t.data_ptr()
        for t in (weights.w13, weights.w13_scale, weights.w2, weights.w2_scale)
    ]


def test_block_fp8_workspace_follows_the_dp_wide_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every EP rank must launch on the same workspace, so the size comes from
    the DP-wide token counts, and without them the default workspace is used."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    used: list[object] = []
    _fake_mega_layer(monkeypatch, moe_ep, [], used)
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, TINY_SM90)
    adapter = FlashInferMoeEp(
        moe,
        _tiny_block_fp8_weights(),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        sm90_format="fp8_block",
    )
    hidden_states = torch.zeros(3, TINY_SM90.hidden_size, dtype=torch.bfloat16)
    topk_ids = torch.zeros(3, TINY_SM90.top_k, dtype=torch.int32)
    topk_weights = torch.ones(3, TINY_SM90.top_k)

    adapter(hidden_states, topk_ids, topk_weights)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = 2
    for across_dp in ([3, 64], [3, 65]):
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=3,
            num_tokens_across_dp=torch.tensor(across_dp, dtype=torch.int32),
        ):
            adapter(hidden_states, topk_ids, topk_weights)

    assert used == [None, ("workspace", 64), None]


def test_sm90_mxfp4_adapter_hands_packed_weights_to_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
):
    """MXFP4 weights reach the SM90 MXFP4 config as the checkpoint stores them,
    packed E2M1 and E8M0 bytes that FlashInfer folds without requantizing, and
    small batches get a workspace of their own as with block FP8."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    config_cls = getattr(moe_ep, "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig", None)
    if config_cls is None:
        pytest.skip("FlashInfer has no SM90 MXFP4 megakernel")
    workspace_capacities: list[int] = []
    layer_args = _fake_mega_layer(monkeypatch, moe_ep, workspace_capacities)
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, TINY_SM90)
    hidden, intermediate = TINY_SM90.hidden_size, TINY_SM90.intermediate_size
    weights = FlashInferMoeEpWeights(
        w13=torch.zeros(2, 2 * intermediate, hidden // 2, dtype=torch.uint8),
        w2=torch.zeros(2, hidden, intermediate // 2, dtype=torch.uint8),
        w13_scale=torch.zeros(2, 2 * intermediate, hidden // 32, dtype=torch.uint8),
        w2_scale=torch.zeros(2, hidden, intermediate // 32, dtype=torch.uint8),
    )
    FlashInferMoeEp(
        moe, weights, apply_topk_in_fc1=apply_topk_in_fc1(moe), sm90_format="mxfp4"
    )

    ((_, _, weight_pack, mega_config),) = layer_args
    assert isinstance(weight_pack, moe_ep.PrequantizedMoEWeights)
    assert all(
        getattr(weight_pack, name) is getattr(weights, name)
        for name in ("w13", "w2", "w13_scale", "w2_scale")
    )
    assert mega_config.preprocess_weights
    assert isinstance(mega_config.megakernel, config_cls)
    assert mega_config.megakernel.gate_up_clamp == moe.swiglu_limit
    assert workspace_capacities == [64]


def _quantize_block_fp8(real: torch.Tensor, power_of_two: bool):
    *lead, rows, cols = real.shape
    blocks = real.reshape(*lead, rows // 128, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(-3, -1)) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (real / _per_element(scale)).to(torch.float8_e4m3fn), scale


def _per_element(scale: torch.Tensor) -> torch.Tensor:
    return scale.repeat_interleave(128, -2).repeat_interleave(128, -1)


def _gate_up_interleaved(w13: torch.Tensor) -> torch.Tensor:
    """Rows ordered [pair, {gate, up}, 8], the SM90 megakernel's FC1 layout."""
    experts, rows, cols = w13.shape
    gate, up = w13.view(experts, 2, rows // 16, 8, cols).unbind(1)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, cols)


@pytest.mark.parametrize("power_of_two", [True, False])
def test_sm90_fp8_block_weights_keep_checkpoint_values(power_of_two: bool):
    """Interleaving gate and up rows puts two checkpoint blocks under one scale.

    Power-of-two scales survive that up to rounding below the E4M3 normal
    range; other scales round the rescaled half once.
    """
    torch.manual_seed(0)
    real = torch.randn(2, 512, 256) * 0.02
    # Gate blocks 8x the up blocks, so each interleaved block mixes scales.
    real[:, :256] *= 8.0
    w13, scale = _quantize_block_fp8(real, power_of_two)

    kernel_w13, kernel_scale = sm90_fp8_block_moe_ep_weights(w13, scale)

    assert kernel_w13.dtype == w13.dtype and kernel_w13.shape == w13.shape
    assert kernel_scale.shape == scale.shape
    restored = kernel_w13.float() * _per_element(kernel_scale)
    expected = _gate_up_interleaved(w13.float() * _per_element(scale))
    error = (restored - expected).abs()
    if power_of_two:
        # Half the smallest E4M3 subnormal step.
        assert torch.all(error <= 2.0**-10 * _per_element(kernel_scale))
    else:
        assert (error.norm() / expected.norm()).item() < 0.03


@pytest.mark.parametrize("sm90_format", ["fp8_block", "mxfp4"])
def test_sm90_megakernels_reject_input_side_router_weights(
    monkeypatch: pytest.MonkeyPatch, sm90_format: str
):
    """The SM90 kernels weight the activation before fc2, which is not the same
    as weighting the expert input (apply_router_weight_on_input)."""
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (9, 0))
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=128
    )
    if sm90_format == "fp8_block":
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=torch.ones(8, 2, 2),
            w2_scale=torch.ones(8, 2, 1),
            block_shape=[128, 128],
        )
    else:
        quant_config = make_mxfp4_moe_quant_config(
            Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL, w1_scale=None, w2_scale=None
        )
    layer = SimpleNamespace(
        expert_map_manager=SimpleNamespace(placement_strategy="linear"),
        apply_router_weight_on_input=True,
    )
    with pytest.raises(ValueError, match="apply_router_weight_on_input"):
        FlashInferMoeEpExperts(moe, quant_config).process_weights_after_loading(layer)


TINY_SM90 = Geometry("tiny", 256, 128, 4, 2)


def _tiny_block_fp8_weights() -> FlashInferMoeEpWeights:
    hidden, intermediate = TINY_SM90.hidden_size, TINY_SM90.intermediate_size
    return FlashInferMoeEpWeights(
        w13=torch.zeros(2, 2 * intermediate, hidden, dtype=torch.float8_e4m3fn),
        w2=torch.zeros(2, hidden, intermediate, dtype=torch.float8_e4m3fn),
        w13_scale=torch.ones(2, 2 * intermediate // 128, hidden // 128),
        w2_scale=torch.ones(2, hidden // 128, intermediate // 128),
    )


def _kernel_storage(mega_config) -> list[int]:
    (fc1, fc1_scale, *_), (fc2, fc2_scale, *_) = mega_config.transformed_weights
    return [t.data_ptr() for t in (fc1, fc1_scale, fc2, fc2_scale)]


def _fake_mega_layer(
    monkeypatch: pytest.MonkeyPatch,
    moe_ep,
    workspace_capacities: list[int] | None = None,
    used_workspaces: list[object] | None = None,
) -> list[tuple]:
    layer_args: list[tuple] = []

    class FakeMegaLayer:
        def __init__(self, *args) -> None:
            layer_args.append(args)
            self._transformed = args[3].transformed_weights

        def create_workspace(self, max_tokens_per_rank: int) -> object:
            assert workspace_capacities is not None
            workspace_capacities.append(max_tokens_per_rank)
            return ("workspace", max_tokens_per_rank)

        def forward(self, tensors, workspace=None, return_workspace_view=False):
            assert used_workspaces is not None
            used_workspaces.append(workspace)
            return torch.zeros_like(tensors.hidden_states)

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
