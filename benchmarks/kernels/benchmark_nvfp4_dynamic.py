# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete NVFP4 linear operations on Qwen3.8-27B matrix shapes.

Example: python benchmarks/kernels/benchmark_nvfp4_dynamic.py --output results
Source manifests, checkpoint configs and clock admission receipts are optional
provenance inputs. Timed runs require CUPTI; --qualify-only checks correctness.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import flashinfer
import torch
import torch.nn.functional as F
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm.kernels.native_bf16_fp4.runner import get_runner
from flashinfer.quantization.fp4_quantization import silu_and_mul_nvfp4_quantize
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.nvfp4.cutedsl_silu import run as native_silu
from vllm.model_executor.kernels.linear.nvfp4.dynamic_cutedsl import (
    cutedsl_dynamic_nvfp4,
)
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)
SHAPES = {
    "gate_up": (34816, 5120, False),
    "down": (5120, 17408, True),
    "head": (248320, 5120, False),
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def unpack(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    lut = torch.tensor(FP4, device=packed.device, dtype=torch.float32)
    return lut[codes].reshape(packed.shape[0], -1) * scales.float().repeat_interleave(
        16, dim=1
    )


def linear_scales(swizzled, rows, k):
    return (
        swizzled.view(torch.float8_e4m3fn)
        .reshape((rows + 127) // 128, k // 64, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(((rows + 127) // 128) * 128, k // 16)[:rows]
    )


def stock_silu(x):
    output = x.new_empty((x.shape[0], x.shape[1] // 2))
    torch.ops._C.silu_and_mul(output, x)
    return output


def stock_quant(x, input_inv, fused):
    if not fused:
        return ops.scaled_fp4_quant(
            x, input_inv, is_sf_swizzled_layout=True, backend="flashinfer-cutlass"
        )
    qa, sa = ops.create_fp4_output_tensors(x.shape[0], x.shape[1] // 2, x.device, True)
    torch.ops._C.silu_and_mul_nvfp4_quant(qa, sa, x, input_inv)
    return qa, sa


def reference(x, packed, scales, weight_global, input_inv, fused, provider):
    k = packed.shape[1] * 2
    if provider == "stock_a4":
        qa, sa = stock_quant(x, input_inv, fused)
        activation = unpack(qa, linear_scales(sa, x.shape[0], k)) / input_inv
    elif provider == "cute_a4":
        if fused:
            qa, sa = silu_and_mul_nvfp4_quantize(x, input_inv)
        else:
            qa, sa = flashinfer.nvfp4_quantize(x, input_inv, backend="cute-dsl")
        activation = unpack(qa, linear_scales(sa, x.shape[0], k)) / input_inv
    else:
        activation = (F.silu(x[:, :k]) * x[:, k:] if fused else x).float()
    result = torch.empty((x.shape[0], packed.shape[0]), device=x.device)
    for start in range(0, packed.shape[0], 1024):
        end = min(start + 1024, packed.shape[0])
        weight = unpack(packed[start:end], scales[start:end])
        result[:, start:end] = (activation @ weight.T) * weight_global
    return result.bfloat16()


def check(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.01)
    diff = actual.float() - expected.float()
    return {
        "max_abs_error": diff.abs().max().item(),
        "rms_error": diff.square().mean().sqrt().item(),
    }


def verify_sources(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest["files"].items():
        assert file_hash(path) == expected, path
    return manifest


def make_variants(
    x,
    packed,
    sf,
    wg,
    ai,
    alpha,
    fused,
    stock_layer,
    marlin_layer,
    stock_kernel,
    n,
    k,
    prepared_a16,
):
    def stock_a4():
        if not fused:
            return stock_kernel.apply_weights(stock_layer, x)
        qa, sa = stock_quant(x, ai, True)
        return flashinfer_scaled_fp4_mm(
            qa, packed, sa, sf, alpha, torch.bfloat16, "cutlass"
        )

    def cute_a4():
        return cutedsl_dynamic_nvfp4(x, packed, sf, wg, ai, alpha, 0, fused)

    def native_a16():
        return cutedsl_dynamic_nvfp4(x, packed, sf, wg, ai, alpha, 16, fused)

    def marlin():
        activation = stock_silu(x) if fused else x
        return apply_fp4_marlin_linear(
            activation,
            marlin_layer.weight,
            marlin_layer.weight_scale,
            marlin_layer.weight_global_scale,
            marlin_layer.workspace,
            size_n=n,
            size_k=k,
        )

    def flashinfer_a16():
        activation = stock_silu(x) if fused else x
        return flashinfer.mm_bf16_fp4(
            activation, *prepared_a16, backend="cute-dsl", out_dtype=torch.bfloat16
        )

    variants = [
        ("stock_a4", stock_a4),
        ("cute_a4", cute_a4),
        ("flashinfer_a16", flashinfer_a16),
        ("marlin", marlin),
    ]
    if x.shape[0] <= 16:
        variants.insert(2, ("native_a16", native_a16))
    return variants


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--expected-uuid")
    parser.add_argument("--clock-admission", type=Path)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--all-a16-tactics", action="store_true")
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.accelerator.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert (props.major, props.minor) in ((12, 0), (12, 1))
    if args.expected_uuid is not None:
        assert str(props.uuid).removeprefix("GPU-") == args.expected_uuid.removeprefix(
            "GPU-"
        )
    assert all(m >= 1 for m in args.m)
    if args.clock_admission is not None:
        admission = json.loads(args.clock_admission.read_text())
        assert admission["passed"]
        assert str(admission["jobid"]) == os.environ["SLURM_JOB_ID"]
        assert str(admission["uuid"]).removeprefix("GPU-") == str(
            props.uuid
        ).removeprefix("GPU-")
    if not args.qualify_only:
        from cupti import cupti  # noqa: F401

        assert int(importlib.metadata.version("cupti-python").split(".")[0]) >= 13
    source = verify_sources(args.source_manifest) if args.source_manifest else None
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {
        "job": os.environ.get("SLURM_JOB_ID"),
        "gpu": str(props),
        "qualify_only": args.qualify_only,
        "benchmark_sha256": file_hash(__file__),
        "source_manifest_sha256": (
            file_hash(args.source_manifest) if args.source_manifest else None
        ),
        "bases": source["bases"] if source else None,
        "clock_admission_sha256": (
            file_hash(args.clock_admission) if args.clock_admission else None
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "vllm",
                "flashinfer-python",
                "numpy",
                "nvidia-cutlass-dsl",
                "cupti-python",
            )
        },
        "inputs": "Seeded synthetic NVFP4 weights, exact checkpoint matrix shapes",
        "checkpoint": "nvidia/Qwen3.8-27B-NVFP4",
        "config_sha256": (
            file_hash(args.checkpoint_config) if args.checkpoint_config else None
        ),
        "timing": "CUPTI full operation span, CUDA graph, cold L2, balanced order",
        "stock_down": "Stock vLLM fused SiLU+NVFP4 quantization and stock CUTLASS GEMM",
        "marlin_down": "Stock vLLM SiLU+multiply and stock NVFP4 Marlin",
        "flashinfer_a16_down": "Stock SiLU+multiply and prepared CuTe W4A16",
        "native_support": "M=1..16; omitted above 16, never relabeled W4A4",
        "correctness": {
            "atol": 0.03,
            "rtol": 0.01,
            "reference": "FP32 dequantized GEMM",
        },
        "m_order": args.m,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    stock_kernel = FlashInferCutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
    for name in args.shapes:
        shape_index = list(SHAPES).index(name)
        n, k, fused = SHAPES[name]
        gen = torch.Generator(device="cuda").manual_seed(42 + shape_index)
        packed = torch.randint(
            256, (n, k // 2), generator=gen, device="cuda", dtype=torch.uint8
        )
        logical_sf = (
            0.125 + torch.rand((n, k // 16), generator=gen, device="cuda") * 0.875
        ).to(torch.float8_e4m3fn)
        sf = (
            logical_sf.reshape(n // 128, 4, 32, k // 64, 4)
            .permute(0, 3, 2, 1, 4)
            .contiguous()
            .reshape(n, k // 16)
        )
        before = {"packed": tensor_hash(packed), "scales": tensor_hash(sf)}
        wg = torch.tensor([0.0625], device="cuda")
        ai = torch.tensor([8.0], device="cuda")
        alpha = wg / ai
        stock_layer = SimpleNamespace(
            weight=packed,
            weight_scale=sf,
            alpha=alpha,
            input_global_scale_inv=ai,
            output_size_per_partition=n,
            input_size_per_partition=k,
        )
        marlin_layer = torch.nn.Module()
        marlin_layer.output_size_per_partition = n
        marlin_layer.input_size_per_partition = k
        marlin_layer.params_dtype = torch.bfloat16
        marlin_layer.weight = torch.nn.Parameter(packed.clone(), requires_grad=False)
        marlin_layer.weight_scale = torch.nn.Parameter(
            logical_sf.clone(), requires_grad=False
        )
        marlin_layer.weight_global_scale = torch.nn.Parameter(
            wg.clone(), requires_grad=False
        )
        prepare_fp4_layer_for_marlin(marlin_layer)
        torch.accelerator.synchronize()
        prep_start = time.perf_counter()
        prepared_a16 = flashinfer.prepare_bf16_fp4_weights(
            packed, sf, wg, backend="cute-dsl"
        )
        torch.accelerator.synchronize()
        canonical_ptrs = {
            tensor.untyped_storage().data_ptr() for tensor in (packed, sf, wg)
        }
        extra_storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in prepared_a16
            if tensor is not None
            and tensor.untyped_storage().data_ptr() not in canonical_ptrs
        }
        (args.output / f"preparation-{name}.json").write_text(
            json.dumps(
                {
                    "backend": "cute-dsl",
                    "first_call_wall_seconds": time.perf_counter() - prep_start,
                    "additional_storage_bytes": sum(extra_storages.values()),
                    "excluded_from_operation_timing": True,
                    "tensors": [
                        {"shape": list(t.shape), "dtype": str(t.dtype)}
                        for t in prepared_a16
                        if t is not None
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        for m in args.m:
            x = torch.randn(
                (m, k * (2 if fused else 1)),
                generator=gen,
                device="cuda",
                dtype=torch.bfloat16,
            )

            variants = make_variants(
                x,
                packed,
                sf,
                wg,
                ai,
                alpha,
                fused,
                stock_layer,
                marlin_layer,
                stock_kernel,
                n,
                k,
                prepared_a16,
            )
            print(f"PREPARE shape={name} m={m}", flush=True)
            if args.qualify_only:
                for _, fn in variants:
                    fn()
            else:
                with autotune(True):
                    for _, fn in variants:
                        fn()
            if name == args.shapes[0] and m == args.m[0]:
                from flashinfer.gemm.gemm_base import (
                    get_gemm_sm120_module_cutlass_fp4,
                )

                module = get_gemm_sm120_module_cutlass_fp4()
                tactics = module.cutlass_fp4_gemm_runner().get_valid_tactics([], None)
                assert tactics == list(range(32)), tactics
                (args.output / "stock-registry.json").write_text(
                    json.dumps({"tactics": tactics, "expanded_registry": False}) + "\n"
                )
            refs = {
                label: reference(x, packed, logical_sf, wg, ai, fused, label)
                for label in ("stock_a4", "cute_a4", "native_a16")
            }
            checks = {
                label: check(
                    fn(),
                    refs[label if label in ("stock_a4", "cute_a4") else "native_a16"],
                )
                for label, fn in variants
            }
            if args.all_a16_tactics and m <= 16:
                activation = (
                    native_silu(x, block=256, vector=1, enable_pdl=True) if fused else x
                )
                out = x.new_empty((m, n))
                inputs = [activation, packed, sf, wg, out, True]
                tactics = get_runner().get_valid_tactics(inputs, None)
                for tactic in tactics:
                    get_runner()(inputs, tactic=tactic)
                    check(out, refs["native_a16"])
                checks["all_a16_tactics"] = len(tactics)
            print(f"CORRECTNESS_PASS shape={name} m={m} {checks}", flush=True)
            if args.qualify_only:
                rows.append({"shape": name, "m": m, "correctness": checks})
            else:
                for repeat in range(args.rounds):
                    order = variants if repeat % 2 == 0 else list(reversed(variants))
                    for label, fn in order:
                        times = bench_gpu_time_with_cupti(
                            fn,
                            dry_run_iters=10,
                            repeat_iters=args.samples,
                            use_cuda_graph=True,
                            cold_l2_cache=True,
                        )
                        assert len(times) == args.samples and min(times) > 0
                        row = {
                            "shape": name,
                            "m": m,
                            "n": n,
                            "k": k,
                            "variant": label,
                            "repeat": repeat,
                            "median_us": 1000 * statistics.median(times),
                            "samples_ms": times,
                            "correctness": checks[label],
                        }
                        rows.append(row)
                        with (args.output / "measurements.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        print(
                            "MEASURED "
                            + json.dumps(
                                {
                                    key: row[key]
                                    for key in (
                                        "shape",
                                        "m",
                                        "variant",
                                        "repeat",
                                        "median_us",
                                    )
                                }
                            ),
                            flush=True,
                        )
            (args.output / "progress.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            if not args.qualify_only:
                AutoTuner.get().save_configs(str(args.output / "autotune-configs.json"))
            del refs
        assert tensor_hash(packed) == before["packed"]
        assert tensor_hash(sf) == before["scales"]
        (args.output / f"weights-preserved-{name}.json").write_text(
            json.dumps(before) + "\n"
        )
        del stock_layer, marlin_layer, packed, logical_sf, sf, prepared_a16
    if args.source_manifest is not None:
        verify_sources(args.source_manifest)
    (args.output / "complete.json").write_text(
        json.dumps(
            {"passed": True, "rows": len(rows), "qualify_only": args.qualify_only}
        )
        + "\n"
    )
    print("DONE_NVFP4_DYNAMIC_COMPARISON rc=0", flush=True)


if __name__ == "__main__":
    main()
