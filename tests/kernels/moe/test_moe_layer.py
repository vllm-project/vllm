# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layer.

Run `pytest tests/kernels/test_moe_layer.py`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    _set_vllm_config,
    parallel_launch_with_config,
)
from tests.kernels.moe.utils import TestMLP, make_test_weights, moe_quantize_weights
from vllm.config import (
    CompilationConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE, fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptNvFp4Config,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_all2all
from vllm.utils.import_utils import has_deep_ep, has_mori
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import cuda_device_count_stateless, set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

SHAPE_COMBOS = [
    (1, 128, 256),
    (32, 1024, 512),
    (222, 2048, 2048),  # should be big enough to exercise DP chunking
]

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

# dp_size, tp_size, use_ep
# Note: DP+TP is not yet supported in the FusedMoE layer.
PARALLEL_COMBOS = [
    [1, 2, False],
    [1, 4, False],
    [2, 1, True],
    [4, 1, True],
]

# TODO: should this even be set manually?  let oracles handle this
BACKENDS = ["allgather_reducescatter"]

if has_mori():
    BACKENDS += ["mori"]

if has_flashinfer_all2all():
    BACKENDS += ["flashinfer_all2allv"]

if has_deep_ep():
    BACKENDS += ["deepep_low_latency", "deepep_high_throughput"]

QUANT_METHODS = [
    None,
    "fp8",
    "modelopt_fp8",
    "modelopt_fp4",
]

# Which quantization methods each backend supports.
# fmt: off
BACKEND_SUPPORTED_QUANTS: dict[str, set[str | None]] = {
    "allgather_reducescatter": {None, "fp8", "modelopt_fp8", "modelopt_fp4"},
    "mori":                    {None, "fp8", "modelopt_fp8"},
    "flashinfer_all2allv":     {None,        "modelopt_fp8", "modelopt_fp4"},
    "deepep_low_latency":      {None, "fp8", "modelopt_fp8", "modelopt_fp4"},
    "deepep_high_throughput":   {None, "fp8", "modelopt_fp8", "modelopt_fp4"},
}
# fmt: on


def rank_chunk(num: int, r: int, w: int) -> int:
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(
    t: torch.Tensor,
    r: int,
    w: int,
    dim: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    chunk = cdiv(t.shape[dim], w)
    t = t.narrow(dim, r * chunk, chunk)
    if device is not None:
        t = t.to(device)
    return t


def maybe_chunk_by_rank(
    t: torch.Tensor | None,
    r: int,
    w: int,
    dim: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if t is not None:
        return chunk_by_rank(t, r, w, dim, device)
    else:
        return t


def tp_chunk_gate_up(
    w: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    dim: int,
    device: torch.device | int | None = None,
) -> torch.Tensor:
    """TP-chunk a combined [gate; up] weight, splitting each half separately
    so every rank gets a portion of both gate and up."""
    half = w.shape[dim] // 2
    gate = chunk_by_rank(
        w.narrow(dim, 0, half), tp_rank, tp_size, dim=dim, device=device
    )
    up = chunk_by_rank(
        w.narrow(dim, half, half), tp_rank, tp_size, dim=dim, device=device
    )
    return torch.cat([gate, up], dim=dim)


def apply_test_filter(
    *,
    use_routed_input_transform: bool = False,
    use_shared_experts: bool = False,
    use_gate: bool = False,
    quantization: str | None = None,
    k: int | None = None,
    world_size: int | None = None,
    num_gpus: int | None = None,
    reduce_results: bool = False,
    backend: str | None = None,
    use_ep: bool = False,
    num_experts: int | None = None,
    dp_size: int | None = None,
    enable_eplb: bool = False,
) -> None:
    """Apply common pytest.skip conditions for MOE layer tests.

    Args:
        use_routed_input_transform: Whether routed_input_transform is used
        use_shared_experts: Whether shared_experts is used
        use_gate: Whether gate is used
        quantization: Quantization method being tested
        k: Hidden dimension size
        world_size: Total number of GPUs in the test
        num_gpus: Number of available GPUs
        reduce_results: Whether reduce_results is enabled
        backend: MOE backend being tested
        use_ep: Whether expert parallelism is enabled
        num_experts: Number of experts
        dp_size: Data parallel size
    """
    # routed_input_transform only makes sense with shared_experts (latent MoE)
    if use_routed_input_transform and not use_shared_experts:
        pytest.skip("routed_input_transform requires shared_experts")

    # gate requires shared_experts (use_overlapped mode)
    if use_gate and not use_shared_experts:
        pytest.skip("gate requires shared_experts (use_overlapped mode)")

    # routed_input_transform + quantization + high hidden dimensions
    if (
        k is not None
        and use_routed_input_transform
        and quantization is not None
        and k >= 2048
    ):
        pytest.skip(
            "routed_input_transform + quantization + higher hidden dimensions "
            "leads to large differences."
        )

    # Skip modelopt_fp4 on H100 (compute capability 9.0)
    if quantization == "modelopt_fp4" and current_platform.has_device_capability(90):
        pytest.skip("modelopt_fp4 not supported on H100+ GPUs")

    # Skip modelopt_fp4 if not on B100+ (compute capability 10.0+)
    if quantization == "modelopt_fp4" and not current_platform.has_device_capability(
        100
    ):
        pytest.skip("modelopt_fp4 not supported on H100+ GPUs")

    # Skip flashinfer_all2allv if not on B100+ (compute capability 10.0+)
    if backend == "flashinfer_all2allv" and not current_platform.has_device_capability(
        100
    ):
        pytest.skip("flashinfer_all2allv not supported on H100+ GPUs")

    # Check if enough GPUs available
    if world_size is not None and num_gpus is not None and world_size > num_gpus:
        pytest.skip(f"Not enough GPUs got {num_gpus}, expected {world_size}.")

    # reduce_results incompatibilities
    if reduce_results and use_shared_experts:
        pytest.skip("reduce_results=True is not compatible with shared_experts=True")

    if reduce_results and world_size == 1:
        pytest.skip("reduce_results=True only makes sense for multi-GPU tests")

    if reduce_results and quantization is not None:
        pytest.skip(
            "reduce_results=True only tested with unquantized data types in order "
            "to limit number of tests run"
        )

    # Backend-specific checks
    if backend is not None:
        supported_quants = BACKEND_SUPPORTED_QUANTS.get(backend)
        if supported_quants is not None and quantization not in supported_quants:
            pytest.skip(f"{backend} does not support quantization={quantization}")

        if backend == "flashinfer_all2allv" and use_ep:
            pytest.skip("flashinfer_all2allv EP not yet supported.")

        if backend == "deepep_low_latency" and k is not None:
            from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (  # noqa: E501
                DeepEPLLPrepareAndFinalize,
            )

            if k not in DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES:
                pytest.skip(f"Skipping unsupported K {k} in {backend} w/o EP.")

    # EPLB-specific checks
    if (
        enable_eplb
        and num_experts is not None
        and dp_size is not None
        and num_experts % dp_size != 0
    ):
        pytest.skip("EPLB requires num_experts divisible by ep_size")


def chunk_scales_by_rank(
    t: torch.Tensor | None,
    r: int,
    w: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if t is not None and t.numel() > 1:
        # Calculate start index by summing chunk sizes for all previous ranks
        # start = sum(rank_chunk(t.shape[0], i, w) for i in range(r))
        # chunk = rank_chunk(t.shape[0], r, w)
        # t = t[start:(start + chunk)]
        chunk = rank_chunk(t.shape[0], r, w)
        t = t[(r * chunk) : max(t.shape[0], (r + 1) * chunk)]

    if t is not None and device is not None:
        t = t.to(device)

    return t


def chunk_scales(
    t: torch.Tensor | None,
    start: int,
    end: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if t is not None and t.numel() > 1:
        t = t[start:end]

    if t is not None and device is not None:
        t = t.to(device)

    return t


@dataclass
class QuantizedWeights:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor | None = None
    w2_weight_scale: torch.Tensor | None = None
    w13_weight_scale_2: torch.Tensor | None = None
    w2_weight_scale_2: torch.Tensor | None = None
    w13_input_scale: torch.Tensor | None = None
    w2_input_scale: torch.Tensor | None = None


def _quantize_fp8_halves(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> QuantizedWeights:
    """Quantize w13 gate/up halves separately to FP8, producing per-shard scales."""
    half = w1.shape[1] // 2
    w1q_a, w1s_a, _ = moe_quantize_weights(
        w1[:, :half, :], None, torch.float8_e4m3fn, False, None
    )
    w1q_b, w1s_b, _ = moe_quantize_weights(
        w1[:, half:, :], None, torch.float8_e4m3fn, False, None
    )
    assert w1s_a is not None and w1s_b is not None

    w2q, w2s, _ = moe_quantize_weights(w2, None, torch.float8_e4m3fn, False, None)
    assert w2s is not None

    return QuantizedWeights(
        w13_weight=torch.cat([w1q_a, w1q_b], dim=1),
        w2_weight=w2q,
        # Each w1s_x is (E, 1, 1) -> reshape to (E, 1), cat to (E, 2)
        w13_weight_scale=torch.cat([w1s_a.view(-1, 1), w1s_b.view(-1, 1)], dim=1),
        # w2s is (E, 1, 1) -> reshape to (E,)
        w2_weight_scale=w2s.view(-1),
    )


def make_quant_config(
    quantization: str | None,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
) -> tuple[QuantizationConfig | None, QuantizedWeights]:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if quantization is None:
        return None, QuantizedWeights(w13_weight=w1, w2_weight=w2)

    if quantization == "fp8":
        return Fp8Config(True), _quantize_fp8_halves(w1, w2)

    if quantization == "modelopt_fp8":
        qw = _quantize_fp8_halves(w1, w2)
        qw.w13_input_scale = torch.ones(
            num_experts, dtype=torch.float32, device=w1.device
        )
        qw.w2_input_scale = torch.ones(
            num_experts, dtype=torch.float32, device=w2.device
        )
        quant_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        )
        return quant_config, qw

    if quantization == "modelopt_fp4":
        # Quantize full w13 at once so both gate/up halves share the same
        # global scale per expert.  process_weights_after_loading uses
        # w13_weight_scale_2[:, 0] for the entire tensor, so the two shard
        # scales must match.
        w1q, w1s, w1gs = moe_quantize_weights(w1, None, "nvfp4", False, None)
        assert w1s is not None and w1gs is not None

        w2q, w2s, w2gs = moe_quantize_weights(w2, None, "nvfp4", False, None)
        assert w2s is not None and w2gs is not None

        qw = QuantizedWeights(
            w13_weight=w1q,
            w2_weight=w2q,
            w13_weight_scale=w1s,
            w2_weight_scale=w2s,
            # weight_scale_2 = 1/w_gs: the kernel computes
            # g_alphas = a_scale * w_scale_2, and correct dequant needs 1/w_gs.
            # Expand per-expert scalar to (E, 2) for the two shards.
            w13_weight_scale_2=(1.0 / w1gs).unsqueeze(1).expand(-1, 2).contiguous(),
            w2_weight_scale_2=1.0 / w2gs,
            w13_input_scale=torch.ones(
                (num_experts, 2), dtype=torch.float32, device=w1.device
            ),
            w2_input_scale=torch.ones(
                num_experts, dtype=torch.float32, device=w2.device
            ),
        )
        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        return quant_config, qw

    raise NotImplementedError(f"Unsupported quantization: {quantization}")


@dataclass
class SharedExpertsConfig:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_s: torch.Tensor | None = None
    w2_s: torch.Tensor | None = None
    quant_dtype: torch.dtype | str | None = None


class SimpleGate(torch.nn.Module):
    """Simple gate module for testing: computes router logits from hidden states."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype,
        device: str = "cuda",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, device=device, dtype=dtype) / 10
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Returns (router_logits, None) to match expected signature."""
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        return router_logits, None


class SimpleRoutedInputTransform(torch.nn.Module):
    """Simple linear transform for testing routed input transform
    (e.g., latent projection).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: str = "cuda",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype) / 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def make_fused_moe_layer(
    quantization: str | None,
    use_ep: bool,
    hidden_size: int,
    intermediate_size: int,
    in_dtype: torch.dtype,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    reduce_results: bool,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    renormalize: bool = False,
    shared_experts: torch.nn.Module | None = None,
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    indices_type: torch.dtype | None = None,
    expert_map: torch.Tensor | None = None,
    enable_eplb: bool = False,
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    gate: torch.nn.Module | None = None,
    routed_input_transform: torch.nn.Module | None = None,
    routed_output_transform: torch.nn.Module | None = None,
    pcp_size: int | None = 1,
) -> tuple[Callable, FusedMoE]:
    quant_config, qw = make_quant_config(quantization, w1, w2, global_num_experts)

    kwargs = dict()
    if shared_experts is None:
        builder = FusedMoE
    else:
        builder = SharedFusedMoE
        kwargs["shared_experts"] = shared_experts

    # Add gate and routed_input_transform if provided
    if gate is not None:
        kwargs["gate"] = gate
    if routed_input_transform is not None:
        kwargs["routed_input_transform"] = routed_input_transform

    layer = builder(
        num_experts=global_num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=in_dtype,
        reduce_results=reduce_results,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        quant_config=quant_config,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        pcp_size=pcp_size,
        prefix="from_forward_context",
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=activation,
        enable_eplb=enable_eplb,
        num_redundant_experts=num_redundant_experts,
        has_bias=has_bias,
        **kwargs,
    )

    for name, value in [
        ("w13_weight", qw.w13_weight),
        ("w2_weight", qw.w2_weight),
        ("w13_weight_scale", qw.w13_weight_scale),
        ("w2_weight_scale", qw.w2_weight_scale),
        ("w13_weight_scale_2", qw.w13_weight_scale_2),
        ("w2_weight_scale_2", qw.w2_weight_scale_2),
        ("w13_input_scale", qw.w13_input_scale),
        ("w2_input_scale", qw.w2_input_scale),
    ]:
        if value is not None:
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )

    layer.quant_method.process_weights_after_loading(layer)

    def _moe(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if shared_experts is None:
            final_shared_states = None
            final_hidden_states = layer(hidden_states, router_logits)
        else:
            final_shared_states, final_hidden_states = layer(
                hidden_states, router_logits
            )

        # Apply routed output transform if provided
        # (e.g., latent space -> original space)
        if routed_output_transform is not None:
            final_hidden_states = routed_output_transform(final_hidden_states)

        if shared_experts is not None:
            assert not reduce_results
            assert final_shared_states is not None
            final_hidden_states += final_shared_states

        if not reduce_results and layer.tp_size > 1:
            final_hidden_states = layer.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states

    return _moe, layer


def make_fake_moe_layer(
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    in_dtype: torch.dtype,
    quant_dtype: torch.dtype | None,
    renormalize: bool = False,
    shared_experts_config: SharedExpertsConfig | None = None,
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    indices_type: torch.dtype | None = None,
    expert_map: torch.Tensor | None = None,
    enable_eplb: bool = False,  # TODO: add eplb support
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
    gate: torch.nn.Module | None = None,
    routed_input_transform: torch.nn.Module | None = None,
    routed_output_transform: torch.nn.Module | None = None,
) -> Callable:
    activation = MoEActivation.from_str(activation)

    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        # eplb_state=None, # TODO
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        num_fused_shared_experts=0,  # TODO
        enable_eplb=enable_eplb,
        # TODO(bnell): once we can construct the MK at init time, we
        # can make this a value.
        indices_type_getter=lambda: indices_type,
    )

    if quant_dtype is not None:
        w1, w1_s, _ = moe_quantize_weights(w1, None, quant_dtype, False, None)
        w2, w2_s, _ = moe_quantize_weights(w2, None, quant_dtype, False, None)
    else:
        w1_s = None
        w2_s = None

    if shared_experts_config is not None:
        shared_experts = TestMLP(
            shared_experts_config.w1,
            shared_experts_config.w2,
            in_dtype,
        )
    else:
        shared_experts = None

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype,
        w1_scale=w1_s,
        w2_scale=w2_s,
    )

    def _moe(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Save original hidden_states for shared experts (before transform)
        original_hidden_states = hidden_states

        # Apply routed input transform if provided
        if routed_input_transform is not None:
            hidden_states = routed_input_transform(hidden_states)

        # If gate provided, compute router_logits from hidden_states
        # Note: gate operates on transformed hidden_states (after
        # routed_input_transform)
        if gate is not None:
            router_logits, _ = gate(hidden_states)

        topk_weights, topk_ids = router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # Shared experts use original (untransformed) hidden_states
        if shared_experts is not None:
            shared_output = shared_experts(original_hidden_states)
        else:
            shared_output = None

        # Routed experts use transformed hidden_states
        output = fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            quant_config=quant_config,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

        # Apply routed output transform if provided
        # (e.g., latent space -> original space)
        if routed_output_transform is not None:
            output = routed_output_transform(output)

        if shared_experts is not None:
            assert shared_output is not None
            output += shared_output

        return output

    return _moe


def _test_body_regular(
    moe_fn: Callable,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    vllm_config: VllmConfig,
    num_tokens: int,
    num_tokens_across_dp: torch.Tensor,
    baseline_output: torch.Tensor,
    device: torch.device,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regular MoE test body: compare layer output to baseline."""
    baseline_output = baseline_output.to(device)

    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        output = moe_fn(hidden_states, router_logits)

    return baseline_output, output


def _test_body_eplb(
    moe_fn: Callable,
    moe_layer: FusedMoE,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    vllm_config: VllmConfig,
    num_tokens: int,
    num_tokens_across_dp: torch.Tensor,
    cpu_group,
    device: torch.device,
    in_dtype: torch.dtype,
    quantization: str | None,
    use_ep: bool,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    top_k: int,
    shared_experts,
    reduce_results: bool,
    gate: torch.nn.Module | None,
    routed_input_transform: torch.nn.Module | None,
    routed_output_transform: torch.nn.Module | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EPLB test body: compare output before and after expert weight rearrangement."""
    # Get "before" output with original weight arrangement
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        output_before = moe_fn(hidden_states, router_logits)

    # Create a fresh FusedMoE layer with enable_eplb=True
    # Delete the original layer's registration so the constructor can
    # re-use the same "from_forward_context" prefix
    cc = vllm_config.compilation_config
    del cc.static_forward_context["from_forward_context"]
    cc.static_all_moe_layers.remove("from_forward_context")

    # Determine hidden size for MoE layer
    # When using routed_input_transform, experts operate in latent space
    hidden_size_for_layer = k // 2 if routed_input_transform is not None else k

    moe_fn, moe_layer = make_fused_moe_layer(
        quantization=quantization,
        use_ep=use_ep,
        hidden_size=hidden_size_for_layer,
        intermediate_size=n,
        in_dtype=in_dtype,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        reduce_results=reduce_results,
        w1=w1,
        w2=w2,
        top_k=top_k,
        global_num_experts=num_experts,
        shared_experts=shared_experts,
        enable_eplb=True,
        gate=gate,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
    )

    if moe_layer._expert_map is not None:
        moe_layer._expert_map = moe_layer._expert_map.to(device)

    # All ranks must generate the same permutation
    set_random_seed(42)
    initial_indices = torch.arange(num_experts, dtype=torch.long)
    shuffled_indices = initial_indices[torch.randperm(num_experts)]

    # Rearrange expert weights across EP ranks
    expert_weights = [list(moe_layer.get_expert_weights())]
    rearrange_expert_weights_inplace(
        old_global_expert_indices=initial_indices.unsqueeze(0),
        new_global_expert_indices=shuffled_indices.unsqueeze(0),
        expert_weights=expert_weights,
        ep_group=cpu_group,
    )

    # Build logical_to_physical_map from shuffled_indices
    # shuffled_indices[physical] = logical, we need the inverse
    logical_to_physical = torch.empty(num_experts, dtype=torch.int32, device=device)
    logical_to_physical[shuffled_indices.to(device)] = torch.arange(
        num_experts, dtype=torch.int32, device=device
    )

    moe_layer.set_eplb_state(
        moe_layer_idx=0,
        expert_load_view=torch.zeros(
            (1, num_experts),
            dtype=torch.int32,
            device=device,
        ),
        logical_to_physical_map=logical_to_physical.reshape(num_experts, 1).unsqueeze(
            0
        ),
        logical_replica_count=torch.ones(
            (1, num_experts),
            dtype=torch.int32,
            device=device,
        ),
    )

    # Get "after" output with rearranged weights and EPLB routing
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        output_after = moe_fn(hidden_states, router_logits)

    return output_before, output_after


def _test_loop(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    ep_size: int,
    dp_size: int,
    tp_size: int,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    m: int,
    n: int,
    k: int,
    top_k: int,
    quantization: str | None,
    shared_experts_config: SharedExpertsConfig | None,
    reduce_results: bool,
    test_body_fn: Callable,
    gate: torch.nn.Module | None,
    routed_input_transform: torch.nn.Module | None,
    routed_output_transform: torch.nn.Module | None,
    **kwargs,
) -> None:
    """Generic test loop that sets up environment and delegates to test_body_fn.

    This function is called directly by test_moe_layer and test_moe_layer_eplb
    via parallel_launch_with_config, passing either _test_body_regular or
    _test_body_eplb as the test_body_fn parameter.
    """
    world_size = tp_size * dp_size
    use_ep = ep_size > 1

    assert vllm_config.parallel_config.enable_expert_parallel == use_ep

    set_random_seed(7)

    in_dtype = hidden_states.dtype
    device = torch.cuda.current_device()
    init_workspace_manager(device)

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    tp_rank = pgi.rank % tp_size

    # Chunk weights for EP/TP
    if ep_size > 1:
        w1 = chunk_by_rank(w1, dp_rank, dp_size, dim=0, device=device)
        w2 = chunk_by_rank(w2, dp_rank, dp_size, dim=0, device=device)

    if tp_size > 1:
        w1 = tp_chunk_gate_up(w1, tp_rank, tp_size, dim=1, device=device)
        w2 = chunk_by_rank(w2, tp_rank, tp_size, dim=2, device=device)

    # TODO: are these to(device) calls needed?

    if ep_size <= 1 and tp_size <= 1:
        w1 = w1.to(device)
        w2 = w2.to(device)

    hidden_states = hidden_states.to(device)
    router_logits = router_logits.to(device)

    # Move gate and routed_input_transform to device if provided
    if gate is not None:
        gate = gate.to(device)
    if routed_input_transform is not None:
        routed_input_transform = routed_input_transform.to(device)
    if routed_output_transform is not None:
        routed_output_transform = routed_output_transform.to(device)

    with set_current_vllm_config(vllm_config):
        # Setup shared experts if needed
        if shared_experts_config is not None:
            s_w1 = shared_experts_config.w1.to(device)
            s_w2 = shared_experts_config.w2.to(device)

            if tp_size > 1:
                s_w1 = tp_chunk_gate_up(s_w1, tp_rank, tp_size, dim=1, device=device)
                s_w2 = chunk_by_rank(s_w2, tp_rank, tp_size, dim=0, device=device)

            shared_experts = TestMLP(
                w1=s_w1,
                w2=s_w2,
                out_dtype=in_dtype,
            )
        else:
            shared_experts = None

        # Determine hidden size for MoE layer
        # When using routed_input_transform, experts operate in latent space
        hidden_size_for_layer = k // 2 if routed_input_transform is not None else k

        # Create initial MoE layer
        moe_fn, moe_layer = make_fused_moe_layer(
            quantization=quantization,
            use_ep=use_ep,
            hidden_size=hidden_size_for_layer,
            intermediate_size=n,
            in_dtype=in_dtype,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            reduce_results=reduce_results,
            w1=w1,
            w2=w2,
            top_k=top_k,
            global_num_experts=num_experts,
            shared_experts=shared_experts,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
        )

        if moe_layer._expert_map is not None:
            moe_layer._expert_map = moe_layer._expert_map.to(device)

        num_tokens = m
        num_tokens_across_dp = torch.tensor(
            [num_tokens] * world_size,
            device=device,
            dtype=torch.int,
        )

        # Call the test body function with all necessary context
        expected, actual = test_body_fn(
            moe_fn=moe_fn,
            moe_layer=moe_layer,
            hidden_states=hidden_states,
            router_logits=router_logits,
            vllm_config=vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            pgi=pgi,
            cpu_group=cpu_group,
            device=device,
            in_dtype=in_dtype,
            quantization=quantization,
            use_ep=use_ep,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            w1=w1,
            w2=w2,
            num_experts=num_experts,
            k=k,
            n=n,
            m=m,
            top_k=top_k,
            shared_experts=shared_experts,
            reduce_results=reduce_results,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            **kwargs,
        )

    # Common tolerance logic
    # TODO: consider associating tolerances with quant methods.
    if quantization is None:
        atol, rtol = 3.5e-2, 3.5e-2
    elif quantization in ("fp8", "modelopt_fp8"):
        atol, rtol = 6e-2, 6e-2
    elif quantization == "modelopt_fp4":
        atol = rtol = 1e-1 + k * 5e-4
    else:
        atol, rtol = 6e-2, 6e-2

    try:
        torch.testing.assert_close(expected, actual, atol=atol, rtol=rtol)
    finally:
        # Cleanup GPU memory
        torch.cuda.synchronize()
        torch.accelerator.empty_cache()


# Test for non-parallel cases (world_size == 1) - backend doesn't matter
@pytest.mark.parametrize("m, n, k", SHAPE_COMBOS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("quantization", QUANT_METHODS)
@pytest.mark.parametrize("use_shared_experts", [False, True])
@pytest.mark.parametrize("use_gate", [False, True])
@pytest.mark.parametrize("use_routed_input_transform", [False, True])
def test_moe_layer_no_parallel(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    quantization: str | None,
    use_shared_experts: bool,
    use_gate: bool,
    use_routed_input_transform: bool,
    monkeypatch,
):
    """Test MoE layer without parallelism (dp_size=1, tp_size=1, use_ep=False).

    Backend doesn't matter when there's no parallelism, so we don't parametrize on it.
    """
    apply_test_filter(
        use_routed_input_transform=use_routed_input_transform,
        use_shared_experts=use_shared_experts,
        use_gate=use_gate,
        quantization=quantization,
        k=k,
    )

    set_random_seed(7)

    dp_size = 1
    tp_size = 1
    world_size = 1
    ep_size = 1
    use_ep = False

    parallel_config = ParallelConfig()
    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False

    vllm_config = VllmConfig(
        parallel_config=parallel_config, compilation_config=compilation_config
    )

    in_dtype = torch.bfloat16

    # Determine dimensions for routed experts (may be transformed)
    latent_size = k // 2 if use_routed_input_transform else k
    routed_expert_hidden_size = latent_size

    # Note: For latent MoE, routed experts operate entirely in latent space
    # (k//2). The routed_output_transform then projects back to k before
    # adding with shared experts.
    # w1: (E, 2*N, latent_size) - input latent_size
    # w2: (E, latent_size, N) - output latent_size (fused_experts returns
    # same shape as input)
    (w1, _, _, _), (w2, _, _, _) = make_test_weights(
        num_experts,
        n,
        routed_expert_hidden_size,  # Both w1 input and w2 output use latent_size
        in_dtype=in_dtype,
    )

    if use_shared_experts:
        shared_experts_config = SharedExpertsConfig(
            w1=torch.randn((k, n * 2), device="cuda", dtype=in_dtype) / 15,
            w2=torch.randn((n, k), device="cuda", dtype=in_dtype) / 15,
        )
    else:
        shared_experts_config = None

    # Create routed input transform if needed
    routed_input_transform = (
        SimpleRoutedInputTransform(k, latent_size, in_dtype, device="cuda")
        if use_routed_input_transform
        else None
    )

    # Create gate if needed
    # Note: gate is called AFTER routed_input_transform, so it should expect
    # the transformed dimension (latent_size) when routed_input_transform is used
    gate_input_dim = latent_size if use_routed_input_transform else k
    gate = (
        SimpleGate(gate_input_dim, num_experts, in_dtype, device="cuda")
        if use_gate
        else None
    )

    # Create routed output transform if needed (projects latent space back to original)
    routed_output_transform = (
        SimpleRoutedInputTransform(latent_size, k, in_dtype, device="cuda")
        if use_routed_input_transform
        else None
    )

    with set_current_vllm_config(vllm_config):
        baseline_layer = make_fake_moe_layer(
            w1=w1,
            w2=w2,
            top_k=top_k,
            global_num_experts=num_experts,
            in_dtype=in_dtype,
            quant_dtype=None,
            renormalize=False,
            shared_experts_config=shared_experts_config,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
        )

    hidden_states = torch.randn((m, k), device="cuda", dtype=in_dtype) / 10
    router_logits = torch.randn((m, num_experts), device="cuda", dtype=in_dtype)

    baseline_output = baseline_layer(hidden_states, router_logits)

    del baseline_layer
    torch.accelerator.empty_cache()

    # Initialize workspace manager and prepare for test (inlined from _test_loop)
    set_random_seed(7)

    # Initialize distributed environment for single GPU
    _set_vllm_config(vllm_config, world_size, rank=0, local_rank=0)

    init_workspace_manager("cuda")

    with set_current_vllm_config(vllm_config):
        # Setup shared experts if needed
        if shared_experts_config is not None:
            s_w1 = shared_experts_config.w1
            s_w2 = shared_experts_config.w2

            shared_experts = TestMLP(
                w1=s_w1,
                w2=s_w2,
                out_dtype=in_dtype,
            )
        else:
            shared_experts = None

        # Create MoE layer
        # Use routed_expert_hidden_size (not k) when routed_input_transform is used
        # to match the weight dimensions
        moe_fn, moe_layer = make_fused_moe_layer(
            quantization=quantization,
            use_ep=use_ep,
            hidden_size=routed_expert_hidden_size,
            intermediate_size=n,
            in_dtype=in_dtype,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            reduce_results=False,
            w1=w1,
            w2=w2,
            top_k=top_k,
            global_num_experts=num_experts,
            shared_experts=shared_experts,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
        )

        num_tokens = m
        num_tokens_across_dp = torch.tensor(
            [num_tokens] * world_size,
            device="cuda",
            dtype=torch.int,
        )

        # Call _test_body_regular to get expected and actual outputs
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            actual = moe_fn(hidden_states, router_logits)

    # Set tolerances based on quantization
    if quantization is None:
        atol, rtol = 3.5e-2, 3.5e-2
    elif quantization in ("fp8", "modelopt_fp8"):
        atol, rtol = 6e-2, 6e-2
    elif quantization == "modelopt_fp4":
        atol = rtol = 1e-1 + k * 5e-4
    else:
        atol, rtol = 6e-2, 6e-2

    try:
        # Compare outputs
        torch.testing.assert_close(baseline_output, actual, atol=atol, rtol=rtol)
    finally:
        # Cleanup GPU memory
        torch.cuda.synchronize()
        torch.accelerator.empty_cache()


# TODO: add cudagraphs/torch.compile tests
@pytest.mark.parametrize("m, n, k", SHAPE_COMBOS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("quantization", QUANT_METHODS)
@pytest.mark.parametrize("dp_size, tp_size, use_ep", PARALLEL_COMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_shared_experts", [False, True])
@pytest.mark.parametrize("reduce_results", [False, True])
@pytest.mark.parametrize("use_gate", [False, True])
@pytest.mark.parametrize("use_routed_input_transform", [False, True])
def test_moe_layer(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    quantization: str | None,
    dp_size: int,
    tp_size: int,
    use_ep: bool,
    backend: str,
    use_shared_experts: bool,
    reduce_results: bool,
    use_gate: bool,
    use_routed_input_transform: bool,
    monkeypatch,
):
    """Test MoE layer with parallelism (multi-GPU or TP/EP enabled).

    For non-parallel cases (world_size == 1), use test_moe_layer_no_parallel instead.
    """
    set_random_seed(7)

    num_gpus = cuda_device_count_stateless()
    world_size = tp_size * dp_size

    assert world_size > 1

    apply_test_filter(
        use_routed_input_transform=use_routed_input_transform,
        use_shared_experts=use_shared_experts,
        use_gate=use_gate,
        quantization=quantization,
        k=k,
        world_size=world_size,
        num_gpus=num_gpus,
        reduce_results=reduce_results,
        backend=backend,
        use_ep=use_ep,
    )

    test_env = dict()
    test_env["VLLM_MOE_DP_CHUNK_SIZE"] = "128"
    monkeypatch.setenv("VLLM_MOE_DP_CHUNK_SIZE", "128")

    # TODO
    # VLLM_FLASHINFER_MOE_BACKEND=latency
    # VLLM_USE_FLASHINFER_MOE_FP16=1
    # VLLM_USE_FLASHINFER_MOE_FP8
    # VLLM_USE_FLASHINFER_MOE_FP4
    # VLLM_USE_FLASHINFER_MOE_INT4

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=use_ep,
        all2all_backend=backend,
    )

    compilation_config = CompilationConfig()
    # compilation_config.mode = CompilationMode.NONE  # for now
    compilation_config.pass_config.fuse_allreduce_rms = False  # for now

    vllm_config = VllmConfig(
        parallel_config=parallel_config, compilation_config=compilation_config
    )

    in_dtype = torch.bfloat16

    # Determine dimensions for routed experts (may be transformed)
    latent_size = k // 2 if use_routed_input_transform else k
    routed_expert_hidden_size = latent_size

    # Note: For latent MoE, routed experts operate entirely in latent space
    # (k//2). The routed_output_transform then projects back to k before
    # adding with shared experts.
    # w1: (E, 2*N, latent_size) - input latent_size
    # w2: (E, latent_size, N) - output latent_size (fused_experts returns
    # same shape as input)
    (w1, _, _, _), (w2, _, _, _) = make_test_weights(
        num_experts,
        n,
        routed_expert_hidden_size,  # Both w1 input and w2 output use latent_size
        in_dtype=in_dtype,
    )

    # Create all tensors on CPU first (for safe pickling to subprocess workers)
    if use_shared_experts:
        shared_experts_config = SharedExpertsConfig(
            w1=torch.randn((k, n * 2), dtype=in_dtype) / 15,
            w2=torch.randn((n, k), dtype=in_dtype) / 15,
        )
    else:
        shared_experts_config = None

    # Create routed input transform if needed
    routed_input_transform = (
        SimpleRoutedInputTransform(k, latent_size, in_dtype, device="cpu")
        if use_routed_input_transform
        else None
    )

    # Create gate if needed
    # Note: gate is called AFTER routed_input_transform, so it should expect
    # the transformed dimension (latent_size) when routed_input_transform is used
    gate_input_dim = latent_size if use_routed_input_transform else k
    gate = (
        SimpleGate(gate_input_dim, num_experts, in_dtype, device="cpu")
        if use_gate
        else None
    )

    # Create routed output transform if needed (projects latent space back to original)
    routed_output_transform = (
        SimpleRoutedInputTransform(latent_size, k, in_dtype, device="cpu")
        if use_routed_input_transform
        else None
    )

    # Create test inputs
    hidden_states = torch.randn((m, k), dtype=in_dtype) / 10
    router_logits = torch.randn((m, num_experts), dtype=in_dtype)

    # For now, run baseline unquantized.
    # quant_dtype = torch.float8_e4m3fn if quantization is not None else None

    # Create baseline with tp_size=1 (single GPU, full weights)
    # to avoid TP-aware behavior when using full, unchunked weights
    baseline_parallel_config = ParallelConfig()
    baseline_vllm_config = VllmConfig(
        parallel_config=baseline_parallel_config, compilation_config=compilation_config
    )

    # Move tensors to CUDA temporarily for baseline computation
    hidden_states_cuda = hidden_states.cuda()
    router_logits_cuda = router_logits.cuda()
    w1_cuda = w1.cuda()
    w2_cuda = w2.cuda()

    # Move shared_experts_config to CUDA if present
    if shared_experts_config is not None:
        shared_experts_config_cuda = SharedExpertsConfig(
            w1=shared_experts_config.w1.cuda(),
            w2=shared_experts_config.w2.cuda(),
        )
    else:
        shared_experts_config_cuda = None

    # Move gate and transforms to CUDA if present
    gate_cuda = gate.cuda() if gate is not None else None
    routed_input_transform_cuda = (
        routed_input_transform.cuda() if routed_input_transform is not None else None
    )
    routed_output_transform_cuda = (
        routed_output_transform.cuda() if routed_output_transform is not None else None
    )

    with set_current_vllm_config(baseline_vllm_config):
        baseline_layer = make_fake_moe_layer(
            w1=w1_cuda,
            w2=w2_cuda,
            top_k=top_k,
            global_num_experts=num_experts,
            in_dtype=in_dtype,
            quant_dtype=None,  # quant_dtype,
            renormalize=False,
            shared_experts_config=shared_experts_config_cuda,
            gate=gate_cuda,
            routed_input_transform=routed_input_transform_cuda,
            routed_output_transform=routed_output_transform_cuda,
        )

    baseline_output = baseline_layer(hidden_states_cuda, router_logits_cuda)

    # Move baseline output back to CPU and detach for passing to workers
    # (detach to avoid autograd across process boundaries)
    baseline_output = baseline_output.detach().cpu()

    # Free the baseline layer and all CUDA tensors before spawn
    del baseline_layer, hidden_states_cuda, router_logits_cuda, w1_cuda, w2_cuda
    del (
        shared_experts_config_cuda,
        gate_cuda,
        routed_input_transform_cuda,
        routed_output_transform_cuda,
    )
    torch.accelerator.empty_cache()

    try:
        parallel_launch_with_config(
            world_size,
            _test_loop,
            vllm_config,
            test_env,
            1 if not use_ep else world_size,  # or dp_size?
            dp_size,
            tp_size,
            hidden_states,
            router_logits,
            w1,
            w2,
            num_experts,
            m,
            n,
            k,
            top_k,
            quantization,
            shared_experts_config,
            reduce_results,
            _test_body_regular,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            baseline_output=baseline_output,
        )
    finally:
        # Cleanup GPU memory after spawned processes complete
        torch.cuda.synchronize()
        torch.accelerator.empty_cache()


# Which quantization methods support EPLB.
# ModelOptFp8MoEMethod inherits supports_eplb=False from FusedMoEMethodBase.
# TODO: double check modelopt fp8
# modelopt_fp4 excluded: get_expert_weights() can't handle NvFP4 packed format.
EPLB_SUPPORTED_QUANTS: list[str | None] = [None, "fp8"]

# Which backends support EPLB.
# deepep backends fail in get_expert_weights / rearrange_expert_weights_inplace.
# TODO(bnell): check this
EPLB_SUPPORTED_BACKENDS: list[str] = ["allgather_reducescatter"]

# EPLB only works with EP and specific quant methods.
EPLB_PARALLEL_COMBOS = [
    [2, 1, True],
    [4, 1, True],
]


@pytest.mark.parametrize("m, n, k", SHAPE_COMBOS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("quantization", EPLB_SUPPORTED_QUANTS)
@pytest.mark.parametrize("dp_size, tp_size, use_ep", EPLB_PARALLEL_COMBOS)
@pytest.mark.parametrize("backend", EPLB_SUPPORTED_BACKENDS)
@pytest.mark.parametrize("use_shared_experts", [False, True])
@pytest.mark.parametrize("reduce_results", [False, True])
@pytest.mark.parametrize("use_gate", [False, True])
@pytest.mark.parametrize("use_routed_input_transform", [False, True])
def test_moe_layer_eplb(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    quantization: str | None,
    dp_size: int,
    tp_size: int,
    use_ep: bool,
    backend: str,
    use_shared_experts: bool,
    reduce_results: bool,
    use_gate: bool,
    use_routed_input_transform: bool,
    monkeypatch,
):
    set_random_seed(7)

    num_gpus = cuda_device_count_stateless()
    world_size = tp_size * dp_size

    assert world_size > 1

    apply_test_filter(
        use_routed_input_transform=use_routed_input_transform,
        use_shared_experts=use_shared_experts,
        use_gate=use_gate,
        quantization=quantization,
        k=k,
        world_size=world_size,
        num_gpus=num_gpus,
        reduce_results=reduce_results,
        backend=backend,
        num_experts=num_experts,
        dp_size=dp_size,
        enable_eplb=True,
    )

    test_env = dict()
    test_env["VLLM_MOE_DP_CHUNK_SIZE"] = "128"
    monkeypatch.setenv("VLLM_MOE_DP_CHUNK_SIZE", "128")

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=use_ep,
        all2all_backend=backend,
    )

    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False

    vllm_config = VllmConfig(
        parallel_config=parallel_config, compilation_config=compilation_config
    )

    in_dtype = torch.bfloat16

    # Determine dimensions for routed experts (may be transformed)
    latent_size = k // 2 if use_routed_input_transform else k
    routed_expert_hidden_size = latent_size

    # Note: For latent MoE, routed experts operate entirely in latent space
    # (k//2). The routed_output_transform then projects back to k before
    # adding with shared experts.
    # w1: (E, 2*N, latent_size) - input latent_size
    # w2: (E, latent_size, N) - output latent_size (fused_experts returns
    # same shape as input)
    (w1, _, _, _), (w2, _, _, _) = make_test_weights(
        num_experts,
        n,
        routed_expert_hidden_size,  # Both w1 input and w2 output use latent_size
        in_dtype=in_dtype,
    )
    # Keep weights on CPU — workers will .to(device).
    w1 = w1.cpu()
    w2 = w2.cpu()
    torch.accelerator.empty_cache()

    if use_shared_experts:
        shared_experts_config = SharedExpertsConfig(
            w1=torch.randn((k, n * 2), dtype=in_dtype) / 15,
            w2=torch.randn((n, k), dtype=in_dtype) / 15,
        )
    else:
        shared_experts_config = None

    # Create routed input transform if needed
    routed_input_transform = (
        SimpleRoutedInputTransform(k, latent_size, in_dtype, device="cpu")
        if use_routed_input_transform
        else None
    )

    # Create gate if needed
    # Note: gate is called AFTER routed_input_transform, so it should expect
    # the transformed dimension (latent_size) when routed_input_transform is used
    gate_input_dim = latent_size if use_routed_input_transform else k
    gate = (
        SimpleGate(gate_input_dim, num_experts, in_dtype, device="cpu")
        if use_gate
        else None
    )

    # Create routed output transform if needed (projects latent space back to original)
    routed_output_transform = (
        SimpleRoutedInputTransform(latent_size, k, in_dtype, device="cpu")
        if use_routed_input_transform
        else None
    )

    hidden_states = torch.randn((m, k), dtype=in_dtype) / 10
    router_logits = torch.randn((m, num_experts), dtype=in_dtype)

    try:
        parallel_launch_with_config(
            world_size,
            _test_loop,
            vllm_config,
            test_env,
            world_size,  # ep_size = world_size for EPLB
            dp_size,
            tp_size,
            hidden_states,
            router_logits,
            w1,
            w2,
            num_experts,
            m,
            n,
            k,
            top_k,
            quantization,
            shared_experts_config,
            reduce_results,
            _test_body_eplb,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
        )
    finally:
        # Cleanup GPU memory after spawned processes complete
        torch.cuda.synchronize()
        torch.accelerator.empty_cache()
