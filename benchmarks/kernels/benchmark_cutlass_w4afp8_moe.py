# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cross-implementation validation and benchmark for W4AFP8 grouped MoE."""

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

SHAPES = {
    "gemm1": (512, 6144),
    "gemm2": (6144, 256),
}


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    low = values[..., 0::2] & 0x0F
    high = values[..., 1::2] & 0x0F
    return (low | (high << 4)).to(torch.uint8).view(torch.int8).contiguous()


def _interleave_scales(scales: torch.Tensor) -> torch.Tensor:
    experts, out_features, groups = scales.shape
    pack = 4 if groups % 4 == 0 else 1
    return (
        scales.reshape(experts, out_features, groups // pack, pack)
        .permute(0, 2, 1, 3)
        .reshape(experts, groups // pack, out_features * pack)
        .contiguous()
    )


def _expert_rows(total_rows: int, num_experts: int) -> torch.Tensor:
    if total_rows == 16 and num_experts == 4:
        return torch.tensor([5, 0, 7, 4], dtype=torch.int32)
    rows = torch.full(
        (num_experts,),
        total_rows // num_experts,
        dtype=torch.int32,
    )
    rows[: total_rows % num_experts] += 1
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def generate_inputs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "inputs.pt"
    if input_path.exists() and not args.force:
        raise FileExistsError(f"{input_path} exists; pass --force to replace it")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    payload: dict[str, Any] = {
        "version": 1,
        "seed": args.seed,
        "group_size": args.group_size,
        "topk": args.topk,
        "num_experts": args.num_experts,
        "shapes": {},
    }

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    for name, (n, k) in SHAPES.items():
        weights = torch.randint(
            -8,
            8,
            (args.num_experts, n, k),
            generator=generator,
            dtype=torch.int8,
        )
        packed_weights = _pack_signed_int4(weights)
        del weights

        scales = (
            torch.rand(
                (args.num_experts, n, k // args.group_size),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.015
            + 0.005
        ).to(torch.bfloat16)

        cases = {}
        for m in args.m:
            total_rows = m * args.topk
            activations = torch.randn(
                (total_rows, k),
                generator=generator,
                dtype=torch.float32,
            )
            activations = (
                (activations * 0.2)
                .clamp(min=fp8_info.min, max=fp8_info.max)
                .round()
                .to(torch.float8_e4m3fn)
            )
            cases[str(m)] = {
                "activations": activations,
                "expert_rows": _expert_rows(total_rows, args.num_experts),
            }

        payload["shapes"][name] = {
            "n": n,
            "k": k,
            "packed_weights": packed_weights,
            "weight_scales": _interleave_scales(scales),
            "activation_scale": torch.tensor([0.75], dtype=torch.float32),
            "cases": cases,
        }

    torch.save(payload, input_path)
    print(f"wrote {input_path}")
    print(f"sha256={_sha256(input_path)}")


def _load_backend(
    backend: str,
) -> Callable[..., None]:
    if backend == "vllm":
        from vllm import _custom_ops as ops

        return ops.cutlass_w4afp8_moe_mm
    if backend == "sglang":
        from sgl_kernel import cutlass_w4a8_moe_mm

        return cutlass_w4a8_moe_mm
    raise ValueError(f"unknown backend: {backend}")


def _prepare_case(
    shape: dict[str, Any],
    case: dict[str, torch.Tensor],
    num_experts: int,
) -> tuple[list[Any], torch.Tensor]:
    n = shape["n"]
    k = shape["k"]
    expert_rows = case["expert_rows"].to(device="cuda")
    expert_offsets = torch.cat(
        [
            torch.zeros(1, device="cuda", dtype=torch.int32),
            expert_rows.cumsum(0, dtype=torch.int32)[:-1],
        ]
    )
    problem_sizes = torch.stack(
        [
            torch.full_like(expert_rows, n),
            expert_rows,
            torch.full_like(expert_rows, k),
        ],
        dim=1,
    )
    a_strides = torch.full(
        (num_experts, 3),
        k,
        device="cuda",
        dtype=torch.int64,
    )
    d_strides = torch.full(
        (num_experts, 3),
        n,
        device="cuda",
        dtype=torch.int64,
    )
    activations = case["activations"].to(device="cuda")
    output = torch.empty(
        (activations.shape[0], n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    op_args = [
        output,
        activations,
        shape["packed_weights"].to(device="cuda"),
        shape["activation_scale"].to(device="cuda"),
        shape["weight_scales"].to(device="cuda"),
        expert_offsets,
        problem_sizes,
        a_strides,
        a_strides,
        d_strides,
        d_strides,
    ]
    return op_args, output


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_backend(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    output_dir = Path(args.output_dir)
    input_path = output_dir / "inputs.pt"
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    op = _load_backend(args.backend)
    group_size = payload["group_size"]
    topk = payload["topk"]
    num_experts = payload["num_experts"]
    outputs: dict[str, torch.Tensor] = {}
    measurements = []

    for shape_name, shape in payload["shapes"].items():
        for m, case in shape["cases"].items():
            op_args, output = _prepare_case(shape, case, num_experts)

            for _ in range(args.warmup):
                op(*op_args, group_size, topk)
            torch.accelerator.synchronize()

            latencies = []
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            for _ in range(args.iterations):
                start.record()
                op(*op_args, group_size, topk)
                end.record()
                end.synchronize()
                latencies.append(start.elapsed_time(end) * 1000)

            op(*op_args, group_size, topk)
            torch.accelerator.synchronize()
            key = f"{shape_name}_m{m}"
            outputs[key] = output.cpu()
            measurement = {
                "case": key,
                "n": shape["n"],
                "k": shape["k"],
                "m": int(m),
                "p20_us": _percentile(latencies, 0.2),
                "p50_us": _percentile(latencies, 0.5),
                "p80_us": _percentile(latencies, 0.8),
            }
            measurements.append(measurement)
            print(
                f"{key}: p20={measurement['p20_us']:.3f} us, "
                f"p50={measurement['p50_us']:.3f} us, "
                f"p80={measurement['p80_us']:.3f} us"
            )

    input_hash = _sha256(input_path)
    torch.save(outputs, output_dir / f"{args.backend}_outputs.pt")
    result = {
        "backend": args.backend,
        "input_sha256": input_hash,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "measurements": measurements,
    }
    result_path = output_dir / f"{args.backend}_benchmark.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {result_path}")


def compare_outputs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    input_hash = _sha256(output_dir / "inputs.pt")
    outputs = {}
    benchmarks = {}
    for backend in ("vllm", "sglang"):
        result_path = output_dir / f"{backend}_benchmark.json"
        result = json.loads(result_path.read_text())
        if result["input_sha256"] != input_hash:
            raise ValueError(f"{backend} output used a different input fixture")
        benchmarks[backend] = {item["case"]: item for item in result["measurements"]}
        outputs[backend] = torch.load(
            output_dir / f"{backend}_outputs.pt",
            map_location="cpu",
            weights_only=True,
        )

    if outputs["vllm"].keys() != outputs["sglang"].keys():
        raise ValueError("vLLM and SGLang output cases differ")

    print(f"input_sha256={input_hash}")
    for key in outputs["vllm"]:
        vllm_output = outputs["vllm"][key]
        sglang_output = outputs["sglang"][key]
        difference = (vllm_output.float() - sglang_output.float()).abs()
        torch.testing.assert_close(
            vllm_output,
            sglang_output,
            rtol=args.rtol,
            atol=args.atol,
        )
        print(
            f"{key}: max_abs_error={difference.max().item():.8f}, "
            f"mean_abs_error={difference.mean().item():.8f}, PASS"
        )

    print("\nkernel-only latency:")
    for key in benchmarks["vllm"]:
        vllm_p50 = benchmarks["vllm"][key]["p50_us"]
        sglang_p50 = benchmarks["sglang"][key]["p50_us"]
        difference = (vllm_p50 / sglang_p50 - 1) * 100
        print(
            f"{key}: vllm_p50={vllm_p50:.3f} us, "
            f"sglang_p50={sglang_p50:.3f} us, "
            f"difference={difference:+.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--num-experts", type=int, default=4)
    generate.add_argument("--group-size", type=int, default=128)
    generate.add_argument("--topk", type=int, default=8)
    generate.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=[1, 2, 4, 16, 64, 256],
    )
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=generate_inputs)

    run = subparsers.add_parser("run")
    run.add_argument("--backend", choices=["vllm", "sglang"], required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("--iterations", type=int, default=100)
    run.set_defaults(func=run_backend)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--rtol", type=float, default=1e-2)
    compare.add_argument("--atol", type=float, default=0.1)
    compare.set_defaults(func=compare_outputs)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
