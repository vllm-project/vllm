#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare MXFP4 MoE backends at GLM-5.3-shaped sizes.

Every MXFP4 backend is built from the same MXFP4 checkpoint weights through the
MXFP4 oracle (``convert_weight_to_mxfp4_moe_kernel_format``,
``make_mxfp4_moe_kernel``), so each row times the kernel vLLM serves with. The
timed region is FP32 router logits to MoE output with GLM-5.3 routing (sigmoid,
correction bias, grouped top-k, routed scaling): monolithic backends route inside
the kernel as ``DeepSeekV3``, modular backends call ``grouped_topk`` first, as
``FusedMoE`` does. Before timing, each MXFP4 backend's output is compared with an
FP32 MoE on the dequantized weights.

Backends: ``CUTLASS_MXFP4_MXFP4`` (W4A4), ``FLASHINFER_TRTLLM_MXFP4_MXFP8`` and
``FLASHINFER_TRTLLM_MXFP4_BF16``, plus NVFP4 ``FLASHINFER_TRTLLM`` built the way
the ModelOpt NVFP4 MoE method does, as a timing reference on its own random
weights. Backends the GPU cannot run are skipped.

Run from a CUDA-built vLLM checkout, for example:

    .venv/bin/python benchmarks/kernels/benchmark_mxfp4_moe_backends.py
    .venv/bin/python benchmarks/kernels/benchmark_mxfp4_moe_backends.py \\
        --tp-size 4 --num-tokens 1,16,256,4096
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import vllm.model_executor.layers.fused_moe.modular_kernel as mk  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle import (  # noqa: E402
    mxfp4 as mxfp4_oracle,
)
from vllm.model_executor.layers.fused_moe.oracle import (  # noqa: E402
    nvfp4 as nvfp4_oracle,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (  # noqa: E402
    NvFp4MoeBackend,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    grouped_topk,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (  # noqa: E402
    trtllm_nvfp4_hidden_alignment,
)
from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

BACKENDS = (
    Mxfp4MoeBackend.CUTLASS_MXFP4_MXFP4,
    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
)
DEFAULT_HIDDEN_SIZE = 6144
DEFAULT_MOE_INTERMEDIATE_SIZE = 2048
DEFAULT_NUM_EXPERTS = 256
DEFAULT_TOPK = 8
DEFAULT_NUM_EXPERT_GROUP = 1
DEFAULT_TOPK_GROUP = 1
DEFAULT_ROUTED_SCALING_FACTOR = 2.5
DEFAULT_NUM_TOKENS = (1, 4, 16, 64, 256, 1024, 4096, 8192)
CHECK_TOKENS = 64
MXFP4_BLOCK = 32
NVFP4_BLOCK = 16
NVFP4_REFERENCE = "NVFP4_FLASHINFER_TRTLLM"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

MoEWeights = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Routing:
    topk: int
    num_expert_group: int
    topk_group: int
    routed_scaling_factor: float
    e_score_correction_bias: torch.Tensor

    def select_experts(
        self, hidden: torch.Tensor, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return grouped_topk(
            hidden_states=hidden,
            gating_output=logits,
            topk=self.topk,
            renormalize=True,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func="sigmoid",
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )


def make_mxfp4_weights(
    num_experts: int, n: int, k: int, device: torch.device
) -> MoEWeights:
    """Random packed E2M1 weights with E8M0 scales, in checkpoint layout."""
    w13 = torch.randint(
        0, 256, (num_experts, 2 * n, k // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (num_experts, k, n // 2), device=device, dtype=torch.uint8
    )
    w13_scale = torch.randint(
        118, 124, (num_experts, 2 * n, k // MXFP4_BLOCK), device=device
    ).byte()
    w2_scale = torch.randint(
        118, 124, (num_experts, k, n // MXFP4_BLOCK), device=device
    ).byte()
    return w13, w2, w13_scale, w2_scale


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        E2M1_VALUES + tuple(-v for v in E2M1_VALUES),
        device=packed.device,
        dtype=torch.float32,
    )
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    scales = torch.exp2(scale.float() - 127).repeat_interleave(MXFP4_BLOCK, dim=-1)
    return lut[codes] * scales


def reference_moe(
    x: torch.Tensor,
    weights: MoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    w13, w2, w13_scale, w2_scale = weights
    out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for expert in topk_ids.unique().tolist():
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        gate, up = (
            x[token].float() @ dequantize_mxfp4(w13[expert], w13_scale[expert]).t()
        ).chunk(2, dim=-1)
        y = (torch.nn.functional.silu(gate) * up) @ dequantize_mxfp4(
            w2[expert], w2_scale[expert]
        ).t()
        out.index_add_(0, token, y * topk_weights[token, slot, None].float())
    return out.to(x.dtype)


def make_moe_config(
    num_experts: int, topk: int, hidden_size: int, n: int, device: torch.device
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=n,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.DeepSeekV3,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.float32,
    )


def make_forward(
    kernel: mk.FusedMoEKernel,
    w13: torch.Tensor,
    w2: torch.Tensor,
    monolithic: bool,
    num_experts: int,
    routing: Routing,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def forward(hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if monolithic:
            out = kernel.apply_monolithic(
                hidden,
                w13,
                w2,
                logits,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
                num_expert_group=routing.num_expert_group,
                e_score_correction_bias=routing.e_score_correction_bias,
                routed_scaling_factor=routing.routed_scaling_factor,
                topk_group=routing.topk_group,
                routing_sink=None,
            )
        else:
            topk_weights, topk_ids = routing.select_experts(hidden, logits)
            out = kernel.apply(
                hidden,
                w13,
                w2,
                topk_weights,
                topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
        assert isinstance(out, torch.Tensor), "MoE finalize is never deferred here"
        return out

    return forward


def make_backend_invoke(
    backend: Mxfp4MoeBackend,
    moe_config: FusedMoEConfig,
    weights: MoEWeights,
    routing: Routing,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build ``backend`` the way the oracle does and return a forward function."""
    experts_cls = mxfp4_oracle.backend_to_kernel_cls(backend)[0]
    w13, w2, w13_scale, w2_scale = (w.clone() for w in weights)
    w13, w2, w13_scale, w2_scale, _, _ = (
        mxfp4_oracle.convert_weight_to_mxfp4_moe_kernel_format(
            backend,
            torch.nn.Module(),
            w13,
            w2,
            w13_scale,
            w2_scale,
            _cache_permute_indices={},
        )
    )
    quant_config = mxfp4_oracle.make_mxfp4_moe_quant_config(
        backend, w1_scale=w13_scale, w2_scale=w2_scale
    )
    assert quant_config is not None
    kernel = mxfp4_oracle.make_mxfp4_moe_kernel(
        quant_config, moe_config, experts_cls, backend
    )
    return make_forward(
        kernel,
        w13,
        w2,
        issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
        moe_config.num_experts,
        routing,
    )


class Nvfp4Layer(torch.nn.Module):
    """The ``RoutedExperts`` attributes the NVFP4 TRTLLM preparation reads."""

    def __init__(self, moe_config: FusedMoEConfig) -> None:
        super().__init__()
        self.activation = MoEActivation.SILU
        self.moe_config = moe_config


def make_nvfp4_invoke(
    moe_config: FusedMoEConfig, routing: Routing, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build NVFP4 FLASHINFER_TRTLLM as the ModelOpt NVFP4 MoE method does.

    Random packed weights with ModelOpt's dummy-weight scale values.
    """
    backend = NvFp4MoeBackend.FLASHINFER_TRTLLM
    experts_cls = nvfp4_oracle.backend_to_kernel_cls(backend)[0]
    e = moe_config.num_experts
    n = moe_config.intermediate_size_per_partition
    k = moe_config.hidden_dim
    layer = Nvfp4Layer(moe_config)
    w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale = (
        nvfp4_oracle.convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=backend,
            layer=layer,  # type: ignore[arg-type]
            w13=torch.randint(0, 256, (e, 2 * n, k // 2), device=device).byte(),
            w13_scale=torch.ones(
                e, 2 * n, k // NVFP4_BLOCK, device=device, dtype=torch.float8_e4m3fn
            ),
            w13_scale_2=torch.full((e,), 1e-3, device=device),
            a13_scale=torch.full((e, 2), 1e-2, device=device),
            w2=torch.randint(0, 256, (e, k, n // 2), device=device).byte(),
            w2_scale=torch.ones(
                e, k, n // NVFP4_BLOCK, device=device, dtype=torch.float8_e4m3fn
            ),
            w2_scale_2=torch.full((e,), 1e-3, device=device),
            a2_scale=torch.full((e,), 1e-2, device=device),
            is_act_and_mul=True,
            trtllm_hidden_alignment=trtllm_nvfp4_hidden_alignment(
                per_token_activation=False, is_act_and_mul=True
            ),
        )
    )
    assert a13_scale is not None and a2_scale is not None
    params = {
        name: torch.nn.Parameter(tensor, requires_grad=False)
        for name, tensor in (
            ("w13_weight", w13),
            ("w13_weight_scale", w13_scale),
            ("w13_weight_scale_2", w13_scale_2),
            ("w13_input_scale", a13_scale),
            ("w2_weight", w2),
            ("w2_weight_scale", w2_scale),
            ("w2_weight_scale_2", w2_scale_2),
            ("w2_input_scale", a2_scale),
        )
    }
    for name, param in params.items():
        layer.register_parameter(name, param)
    quant_config = nvfp4_oracle.make_nvfp4_moe_quant_config(
        backend=backend,
        w13_scale=params["w13_weight_scale"],
        w2_scale=params["w2_weight_scale"],
        w13_scale_2=params["w13_weight_scale_2"],
        w2_scale_2=params["w2_weight_scale_2"],
        a13_scale=params["w13_input_scale"],
        a2_scale=params["w2_input_scale"],
        layer=layer,
    )
    kernel = nvfp4_oracle.make_nvfp4_moe_kernel(
        quant_config, moe_config, experts_cls, backend
    )
    kernel.fused_experts.process_weights_after_loading(layer)
    return make_forward(
        kernel,
        params["w13_weight"],
        params["w2_weight"],
        issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
        e,
        routing,
    )


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(v) for v in value.split(",") if v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integers, got {value!r}") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=DEFAULT_MOE_INTERMEDIATE_SIZE,
        help="total MoE intermediate size before TP; each rank gets 1/--tp-size",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--num-expert-group", type=int, default=DEFAULT_NUM_EXPERT_GROUP
    )
    parser.add_argument("--topk-group", type=int, default=DEFAULT_TOPK_GROUP)
    parser.add_argument(
        "--routed-scaling-factor", type=float, default=DEFAULT_ROUTED_SCALING_FACTOR
    )
    parser.add_argument(
        "--num-tokens",
        type=parse_int_list,
        default=DEFAULT_NUM_TOKENS,
        help="comma-separated token counts",
    )
    parser.add_argument("--rep", type=int, default=100, help="ms of replay per case")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    if args.rep < 1:
        parser.error("--rep must be positive")
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")
    if args.moe_intermediate_size % args.tp_size != 0:
        parser.error("--moe-intermediate-size must be divisible by --tp-size")
    if (args.moe_intermediate_size // args.tp_size) % MXFP4_BLOCK != 0:
        parser.error("per-rank MoE intermediate size must be a multiple of 32")
    if args.hidden_size % MXFP4_BLOCK != 0:
        parser.error("--hidden-size must be a multiple of 32")
    if args.num_experts % args.num_expert_group != 0:
        parser.error("--num-experts must be divisible by --num-expert-group")
    if not 1 <= args.topk_group <= args.num_expert_group:
        parser.error("--topk-group must be between 1 and --num-expert-group")
    if not torch.cuda.is_available():
        parser.error("requires a CUDA GPU")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda", torch.accelerator.current_device_index())
    major, minor = torch.cuda.get_device_capability()
    n = args.moe_intermediate_size // args.tp_size
    torch.manual_seed(0)
    init_workspace_manager(device)

    print(f"GPU: {torch.cuda.get_device_name()} (SM{major}{minor})")
    print(
        f"hidden={args.hidden_size}, intermediate={n} per rank "
        f"(moe_intermediate={args.moe_intermediate_size}, tp={args.tp_size}), "
        f"experts={args.num_experts}, topk={args.topk}, "
        f"groups={args.topk_group}/{args.num_expert_group}, "
        f"routed_scaling={args.routed_scaling_factor}"
    )

    weights = make_mxfp4_weights(args.num_experts, n, args.hidden_size, device)
    moe_config = make_moe_config(
        args.num_experts, args.topk, args.hidden_size, n, device
    )
    routing = Routing(
        topk=args.topk,
        num_expert_group=args.num_expert_group,
        topk_group=args.topk_group,
        routed_scaling_factor=args.routed_scaling_factor,
        e_score_correction_bias=torch.randn(
            args.num_experts, device=device, dtype=torch.float32
        )
        * 0.01,
    )

    def make_inputs(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(num_tokens)
        hidden = torch.randn(
            num_tokens,
            args.hidden_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        logits = torch.randn(
            num_tokens,
            args.num_experts,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        return hidden, logits

    build_errors = (ValueError, NotImplementedError, RuntimeError, ImportError)
    invokes: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    with set_current_vllm_config(VllmConfig()):
        for backend in BACKENDS:
            try:
                invokes[backend.value] = make_backend_invoke(
                    backend, moe_config, weights, routing
                )
            except build_errors as exc:
                print(f"skip {backend.value}: {exc}")

        try:
            invokes[NVFP4_REFERENCE] = make_nvfp4_invoke(
                make_moe_config(
                    args.num_experts, args.topk, args.hidden_size, n, device
                ),
                routing,
                device,
            )
        except build_errors as exc:
            print(f"skip {NVFP4_REFERENCE}: {exc}")

        if not args.skip_check:
            hidden, logits = make_inputs(CHECK_TOKENS)
            topk_weights, topk_ids = routing.select_experts(hidden, logits)
            expected = reference_moe(hidden, weights, topk_weights, topk_ids)
            print(f"\nrelative error vs FP32 reference ({CHECK_TOKENS} tokens):")
            for name, forward in invokes.items():
                out = forward(hidden, logits)
                if name == NVFP4_REFERENCE:
                    finite = bool(out.isfinite().all())
                    if out.shape != hidden.shape or not finite:
                        raise RuntimeError(
                            f"{name} output is invalid: shape {tuple(out.shape)} "
                            f"(expected {tuple(hidden.shape)}), all finite: {finite}"
                        )
                    print(f"  {name:32s} n/a, own weights (shape and values ok)")
                    continue
                error = (
                    out.float() - expected.float()
                ).norm() / expected.float().norm()
                print(f"  {name:32s} {error.item():.4f}")

        names = list(invokes)
        print("\nmedian latency (us, CUDA graph replay), router logits to MoE output")
        print(f"{'tokens':>8s}  " + "  ".join(f"{name:>30s}" for name in names))
        for num_tokens in args.num_tokens:
            hidden, logits = make_inputs(num_tokens)
            row = [
                triton.testing.do_bench_cudagraph(
                    partial(forward, hidden, logits),
                    rep=args.rep,
                    quantiles=[0.5, 0.2, 0.8],
                )[0]
                * 1_000
                for forward in invokes.values()
            ]
            print(f"{num_tokens:>8d}  " + "  ".join(f"{us:>30.1f}" for us in row))


if __name__ == "__main__":
    main()
