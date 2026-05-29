#!/usr/bin/env python3
"""Mini graph reproduction for vLLM BlockScale SplitK zero-init fusion.

The full Qwen3-Next server corrupts generation when
``fuse_blockscale_splitk_zero_init`` is enabled, while direct custom-op calls
are bitwise-correct. This harness narrows the gap by compiling a tiny module
through vLLM's post-grad pass manager:

    preop(rms_norm) -> group_fp8_quant -> blockscale_gemm

or:

    group_fp8_quant -> blockscale_gemm

For the RMSNorm case, vLLM's regular ROCm AITER norm-quant pass first rewrites
``vllm_ir.rms_norm + rocm_aiter_group_fp8_quant`` into
``rocm_aiter_rmsnorm_fp8_group_quant``. Then the splitK zero-init pass should
rewrite the resulting producer+GEMM pair into:

    producer_with_zero_init(..., gemm_out_zero_init=Y)
    gemm_a8w8_blockscale_splitk(..., output=Y, y_is_zeroed=True)

The script compares eager, compiled baseline, compiled fused, and manual CUDA
graph replay outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_TARGET_DEVICE", "rocm")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_TRITON_GEMM", "1")

import torch
import torch.nn as nn

import vllm.ir.ops  # noqa: F401  Registers vllm_ir.rms_norm.
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.inductor_pass import InductorPass, pass_context
from vllm.compilation.passes.ir.inplace_functionalization import (
    VllmIRInplaceFunctionalizationPass,
)
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.config import (
    CUDAGraphMode,
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
)
from vllm.config.utils import Range


GROUP_SIZE = 128
EPS = 1e-6


class MiniPostGradPass(InductorPass):
    """Wrap vLLM's pass manager with the compile-range context it expects."""

    def __init__(self, vllm_config: VllmConfig, compile_range: Range) -> None:
        self.vllm_config = vllm_config
        self.compile_range = compile_range
        self.manager = PostGradPassManager()
        self.manager.configure(vllm_config)

    def __call__(self, graph: torch.fx.Graph) -> None:
        with pass_context(self.compile_range):
            self.manager(graph)

    def uuid(self) -> str:
        with pass_context(self.compile_range):
            return self.manager.uuid()


class MiniPreGradPass(InductorPass):
    """Wrap vLLM's pre-grad IR functionalization with compile-range context."""

    def __init__(self, vllm_config: VllmConfig, compile_range: Range) -> None:
        self.compile_range = compile_range
        self.inner = VllmIRInplaceFunctionalizationPass(vllm_config)

    def __call__(self, graph: torch.fx.Graph) -> None:
        with pass_context(self.compile_range):
            self.inner(graph)

    def uuid(self) -> str:
        with pass_context(self.compile_range):
            return self.inner.uuid()


class RawRMSNormQuantGemm(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.register_buffer("norm_weight", torch.ones((k,), dtype=torch.bfloat16))

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        normed = torch.ops.vllm_ir.rms_norm(x, self.norm_weight, EPS)
        a, a_scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(normed, GROUP_SIZE)
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            a, b, a_scale, b_scale, torch.bfloat16
        )


class GroupQuantGemm(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
    ) -> torch.Tensor:
        a, a_scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(x, GROUP_SIZE)
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            a, b, a_scale, b_scale, torch.bfloat16
        )


@dataclass
class RunResult:
    case: str
    m: int
    n: int
    k: int
    dynamic: bool
    baseline_max_abs: float
    fused_max_abs: float
    compiled_delta_max_abs: float
    baseline_cudagraph_max_abs: float
    fused_cudagraph_max_abs: float
    fused_vs_baseline_cudagraph_max_abs: float
    compiled_matches_eager: bool
    fused_matches_baseline: bool
    passed: bool


def _sync() -> None:
    torch.cuda.synchronize()


def _make_vllm_config(fuse_zero_init: bool, debug_dump_path: Path) -> VllmConfig:
    pass_config = PassConfig(
        eliminate_noops=True,
        fuse_norm_quant=True,
        fuse_act_quant=True,
        fuse_attn_quant=False,
        enable_sp=False,
        fuse_gemm_comms=False,
        fuse_allreduce_rms=False,
        fuse_minimax_qk_norm=False,
        enable_qk_norm_rope_fusion=False,
        fuse_rope_kvcache_cat_mla=False,
        fuse_act_padding=False,
        fuse_mla_dual_rms_norm=True,
        fuse_rope_kvcache=False,
        fuse_blockscale_splitk_zero_init=fuse_zero_init,
    )
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="inductor",
        debug_dump_path=debug_dump_path,
        custom_ops=["+quant_fp8"],
        ir_enable_torch_wrap=True,
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
        use_inductor_graph_partition=False,
        pass_config=pass_config,
        inductor_compile_config={
            "enable_auto_functionalized_v2": False,
            "size_asserts": False,
            "alignment_asserts": False,
            "scalar_asserts": False,
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
        },
    )
    return VllmConfig(compilation_config=compilation_config)


def _compile_module(
    module: nn.Module,
    vllm_config: VllmConfig,
    compile_range: Range,
    dynamic: bool,
) -> Any:
    options = dict(vllm_config.compilation_config.inductor_compile_config)
    pre_grad_pass_key = "pre_grad_custom_pass"
    options["pre_grad_custom_pass"] = MiniPreGradPass(vllm_config, compile_range)
    options["post_grad_custom_post_pass"] = MiniPostGradPass(vllm_config, compile_range)
    # Match VllmBackend.configure_post_pass(): this avoids trying to pickle
    # the pre-grad pass object into the Inductor cache key.
    import torch._inductor.config as inductor_config

    options["_cache_config_ignore_prefix"] = (
        inductor_config._cache_config_ignore_prefix + [pre_grad_pass_key]
    )
    return torch.compile(
        module,
        fullgraph=True,
        dynamic=dynamic,
        backend="inductor",
        options=options,
    )


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    diff = (lhs.float() - rhs.float()).abs()
    return float(diff.max().item()) if diff.numel() else 0.0


def _cudagraph_replay(
    compiled: Any,
    x: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    # vLLM captures static input buffers and replays compiled pieces. This is a
    # minimal single-piece equivalent for the GEMM block.
    static_x = x.clone()
    static_b = b.clone()
    static_b_scale = b_scale.clone()
    _ = compiled(static_x, static_b, static_b_scale)
    _sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = compiled(static_x, static_b, static_b_scale)
    _sync()

    static_x.copy_(x)
    static_b.copy_(b)
    static_b_scale.copy_(b_scale)
    graph.replay()
    _sync()
    return static_out.clone()


def _run_one(
    case: str,
    m: int,
    n: int,
    k: int,
    dynamic: bool,
    dump_root: Path,
) -> RunResult:
    if case == "rmsnorm":
        module = RawRMSNormQuantGemm(k).cuda()
    elif case == "group":
        module = GroupQuantGemm().cuda()
    else:
        raise ValueError(f"unknown case: {case}")

    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    b, b_scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(w_bf16, GROUP_SIZE)
    _sync()

    with torch.no_grad():
        eager = module(x, b, b_scale)
        _sync()

        compile_range = Range(1, max(8192, m))
        baseline_config = _make_vllm_config(
            fuse_zero_init=False,
            debug_dump_path=dump_root / f"{case}_baseline",
        )
        fused_config = _make_vllm_config(
            fuse_zero_init=True,
            debug_dump_path=dump_root / f"{case}_fused",
        )
        baseline = _compile_module(module, baseline_config, compile_range, dynamic)
        fused = _compile_module(module, fused_config, compile_range, dynamic)

        baseline_out = baseline(x, b, b_scale)
        fused_out = fused(x, b, b_scale)
        _sync()

        baseline_cg = _cudagraph_replay(baseline, x, b, b_scale)
        fused_cg = _cudagraph_replay(fused, x, b, b_scale)

    baseline_max_abs = _max_abs(baseline_out, eager)
    fused_max_abs = _max_abs(fused_out, eager)
    compiled_delta_max_abs = _max_abs(fused_out, baseline_out)
    baseline_cudagraph_max_abs = _max_abs(baseline_cg, baseline_out)
    fused_cudagraph_max_abs = _max_abs(fused_cg, fused_out)
    fused_vs_baseline_cudagraph_max_abs = _max_abs(fused_cg, baseline_cg)
    compiled_matches_eager = baseline_max_abs == 0.0 and fused_max_abs == 0.0
    fused_matches_baseline = compiled_delta_max_abs == 0.0
    passed = (
        fused_matches_baseline
        and baseline_cudagraph_max_abs == 0.0
        and fused_cudagraph_max_abs == 0.0
        and fused_vs_baseline_cudagraph_max_abs == 0.0
    )
    return RunResult(
        case=case,
        m=m,
        n=n,
        k=k,
        dynamic=dynamic,
        baseline_max_abs=baseline_max_abs,
        fused_max_abs=fused_max_abs,
        compiled_delta_max_abs=compiled_delta_max_abs,
        baseline_cudagraph_max_abs=baseline_cudagraph_max_abs,
        fused_cudagraph_max_abs=fused_cudagraph_max_abs,
        fused_vs_baseline_cudagraph_max_abs=fused_vs_baseline_cudagraph_max_abs,
        compiled_matches_eager=compiled_matches_eager,
        fused_matches_baseline=fused_matches_baseline,
        passed=passed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["rmsnorm", "group"])
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path("benchmarks/zero_init_demo_results/mini_graph_debug"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.k % GROUP_SIZE != 0:
        raise ValueError(f"k must be divisible by {GROUP_SIZE}, got {args.k}")

    torch.manual_seed(args.seed)
    rocm_aiter_ops.register_ops_once()
    args.dump_root.mkdir(parents=True, exist_ok=True)
    for child in args.dump_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    results = [
        _run_one(args_case, args.m, args.n, args.k, args.dynamic, args.dump_root)
        for args_case in args.cases
    ]
    payload = [asdict(result) for result in results]
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
