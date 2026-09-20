# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce and benchmark WNA16 MoE split-K non-determinism.

Run this script in separate processes for the CUDA and Triton backends so
``VLLM_BATCH_INVARIANT`` is applied exactly as it is in production.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "triton"), required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-output", type=Path)
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=0.0)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if not 0 < args.top_k <= args.num_experts:
        raise ValueError("--top-k must be in [1, num-experts]")
    if args.hidden_size % args.group_size != 0:
        raise ValueError("--hidden-size must be divisible by --group-size")
    if args.intermediate_size % args.group_size != 0:
        raise ValueError("--intermediate-size must be divisible by --group-size")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.accelerator.is_available():
        raise SystemExit("This benchmark requires an NVIDIA CUDA GPU.")

    os.environ["VLLM_BATCH_INVARIANT"] = "1" if args.backend == "triton" else "0"

    from vllm import __version__ as vllm_version
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        should_moe_wna16_use_cuda,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.platforms import current_platform
    from vllm.scalar_type import scalar_types

    if not current_platform.is_cuda():
        raise SystemExit("This benchmark requires an NVIDIA CUDA GPU.")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    m = args.num_tokens
    e = args.num_experts
    k = args.hidden_size
    n = args.intermediate_size
    top_k = args.top_k
    group_size = args.group_size

    selected_backend = (
        "cuda"
        if should_moe_wna16_use_cuda(m * top_k, group_size, e, bit=4)
        else "triton"
    )
    if selected_backend != args.backend:
        raise RuntimeError(
            f"Requested {args.backend}, but the dispatcher selected "
            f"{selected_backend}. The CUDA path requires tokens * top_k / "
            "experts <= 6 and group size 32, 64, or 128."
        )

    hidden_states = (
        torch.randn((m, k), device=device, dtype=dtype, generator=generator) / 10
    )
    w1 = (
        torch.randn((e, 2 * n, k), device=device, dtype=dtype, generator=generator) / 10
    )
    w2 = torch.randn((e, k, n), device=device, dtype=dtype, generator=generator) / 10
    scores = torch.randn((m, e), device=device, dtype=dtype, generator=generator)

    w1_qweight = torch.empty((e, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2_qweight = torch.empty((e, k, n // 2), device=device, dtype=torch.uint8)
    w1_scale = torch.empty((e, 2 * n, k // group_size), device=device, dtype=dtype)
    w2_scale = torch.empty((e, k, n // group_size), device=device, dtype=dtype)

    for index in range(2 * e):
        expert_id = index % e
        weight, qweight_target, scale_target = (
            (w1, w1_qweight, w1_scale) if index < e else (w2, w2_qweight, w2_scale)
        )
        _, qweight, scales, _ = quantize_weights(
            weight[expert_id].T,
            scalar_types.uint4b8,
            group_size,
            zero_points=False,
            ref_zero_points_after_scales=False,
        )
        qweight = qweight.T.contiguous().to(torch.uint8)
        qweight_target[expert_id] = qweight[:, 1::2] * 16 + qweight[:, ::2]
        scale_target[expert_id] = scales.T

    topk_weights, topk_ids = torch.topk(
        torch.softmax(scores.float(), dim=-1), top_k, dim=-1
    )
    topk_ids = topk_ids.to(torch.int32)

    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[0, group_size],
    )
    vllm_config = VllmConfig()

    def run() -> torch.Tensor:
        with set_current_vllm_config(vllm_config):
            return fused_experts(
                hidden_states,
                w1_qweight,
                w2_qweight,
                topk_weights,
                topk_ids,
                global_num_experts=e,
                quant_config=quant_config,
            )

    for _ in range(args.warmups):
        run()
    torch.accelerator.synchronize()

    outputs: list[torch.Tensor] = []
    latencies_ms: list[float] = []
    for _ in range(args.repeats):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        output = run()
        end.record()
        end.synchronize()
        outputs.append(output.detach().cpu())
        latencies_ms.append(start.elapsed_time(end))

    first = outputs[0]
    differing_elements = [
        int(torch.count_nonzero(output != first).item()) for output in outputs[1:]
    ]
    max_abs_diffs = [
        float((output.float() - first.float()).abs().max().item())
        for output in outputs[1:]
    ]
    hashes = {tensor_hash(output) for output in outputs}

    result: dict[str, object] = {
        "requested_backend": args.backend,
        "selected_backend": selected_backend,
        "batch_invariant": args.backend == "triton",
        "device": current_platform.get_device_name(),
        "compute_capability": list(current_platform.get_device_capability()),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "vllm_version": vllm_version,
        "shape": {
            "num_tokens": m,
            "num_experts": e,
            "hidden_size": k,
            "intermediate_size": n,
            "top_k": top_k,
            "group_size": group_size,
            "dtype": args.dtype,
        },
        "repeats": args.repeats,
        "changed_runs": sum(count > 0 for count in differing_elements),
        "max_differing_elements": max(differing_elements),
        "max_abs_repeat_diff": max(max_abs_diffs),
        "unique_output_hashes": len(hashes),
        "first_output_sha256": tensor_hash(first),
        "latency_ms": {
            "p20": percentile(latencies_ms, 0.2),
            "median": statistics.median(latencies_ms),
            "p80": percentile(latencies_ms, 0.8),
        },
    }

    if args.compare_output is not None:
        reference = torch.load(
            args.compare_output, map_location="cpu", weights_only=True
        )
        if not isinstance(reference, torch.Tensor):
            raise TypeError("--compare-output must contain a single tensor")
        if reference.shape != first.shape:
            raise ValueError(
                f"Reference shape {reference.shape} does not match {first.shape}"
            )
        difference = (first.float() - reference.float()).abs()
        result["reference_comparison"] = {
            "bitwise_equal": torch.equal(first, reference),
            "allclose": torch.allclose(
                first.float(), reference.float(), atol=args.atol, rtol=args.rtol
            ),
            "differing_elements": int(torch.count_nonzero(first != reference).item()),
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
        }

    if args.save_output is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(first, args.save_output)

    serialized_result = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{serialized_result}\n")
    print(serialized_result)


if __name__ == "__main__":
    main()
