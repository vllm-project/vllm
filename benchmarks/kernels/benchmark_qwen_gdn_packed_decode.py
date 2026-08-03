# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    _get_packed_decode_launch_config,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser


@dataclass
class Inputs:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    out: torch.Tensor
    state_indices: torch.Tensor


def make_inputs(
    batch_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
) -> Inputs:
    torch.manual_seed(0)
    h = hv = 16
    k = v = 128
    qkv_dim = 2 * h * k + hv * v
    num_states = batch_size + 1

    return Inputs(
        mixed_qkv=torch.randn(batch_size, qkv_dim, device="cuda", dtype=dtype).mul_(
            0.1
        ),
        a=torch.randn(batch_size, hv, device="cuda", dtype=dtype).mul_(0.1),
        b=torch.randn(batch_size, hv, device="cuda", dtype=dtype).mul_(0.1),
        A_log=torch.zeros(hv, device="cuda", dtype=torch.float32),
        dt_bias=torch.zeros(hv, device="cuda", dtype=torch.float32),
        state=torch.randn(num_states, hv, v, k, device="cuda", dtype=state_dtype).mul_(
            0.1
        ),
        out=torch.empty(batch_size, 1, hv, v, device="cuda", dtype=dtype),
        state_indices=torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int32),
    )


def launch(inputs: Inputs, bv: int, num_warps: int, num_stages: int) -> None:
    batch_size = inputs.mixed_qkv.shape[0]
    h = hv = 16
    k = v = bk = 128
    split_batch_head_grid = batch_size * hv > 65535
    grid = (
        (triton.cdiv(v, bv), hv, batch_size)
        if split_batch_head_grid
        else (triton.cdiv(v, bv), batch_size * hv)
    )

    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=inputs.mixed_qkv,
        a=inputs.a,
        b=inputs.b,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        o=inputs.out,
        h0=inputs.state,
        ht=inputs.state,
        ssm_state_indices=inputs.state_indices,
        scale=k**-0.5,
        stride_mixed_qkv_tok=inputs.mixed_qkv.stride(0),
        stride_a_tok=inputs.a.stride(0),
        stride_b_tok=inputs.b.stride(0),
        stride_init_state_token=inputs.state.stride(0),
        stride_final_state_token=inputs.state.stride(0),
        stride_indices_seq=inputs.state_indices.stride(0),
        H=h,
        HV=hv,
        K=k,
        V=v,
        BK=bk,
        BV=bv,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=True,
        SPLIT_BATCH_HEAD_GRID=split_batch_head_grid,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def clone_inputs(inputs: Inputs) -> Inputs:
    return Inputs(
        mixed_qkv=inputs.mixed_qkv,
        a=inputs.a,
        b=inputs.b,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        state=inputs.state.clone(),
        out=torch.empty_like(inputs.out),
        state_indices=inputs.state_indices,
    )


def benchmark(
    batch_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    bv: int,
    num_warps: int,
    num_stages: int,
    rep_ms: int,
) -> dict[str, float | int]:
    source = make_inputs(batch_size, dtype, state_dtype)
    reference = clone_inputs(source)
    candidate = clone_inputs(source)

    launch(reference, bv=32, num_warps=1, num_stages=3)
    launch(
        candidate,
        bv=bv,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    torch.accelerator.synchronize()

    out_diff = (candidate.out.float() - reference.out.float()).abs().max().item()
    state_diff = (candidate.state.float() - reference.state.float()).abs().max().item()

    bench_inputs = make_inputs(batch_size, dtype, state_dtype)

    def run() -> None:
        launch(
            bench_inputs,
            bv=bv,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    latency = triton.testing.do_bench_cudagraph(
        run,
        rep=rep_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    if isinstance(latency, float):
        median_ms = low_ms = high_ms = latency
    else:
        median_ms, low_ms, high_ms = latency

    return {
        "batch_size": batch_size,
        "bv": bv,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "median_us": median_ms * 1000,
        "p20_us": low_ms * 1000,
        "p80_us": high_ms * 1000,
        "max_out_abs_diff": out_diff,
        "max_state_abs_diff": state_diff,
    }


def validate_wrapper(
    batch_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
) -> dict[str, float | int]:
    source = make_inputs(batch_size, dtype, state_dtype)
    reference = clone_inputs(source)
    candidate = clone_inputs(source)

    launch(reference, bv=32, num_warps=1, num_stages=3)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=candidate.mixed_qkv,
        a=candidate.a,
        b=candidate.b,
        A_log=candidate.A_log,
        dt_bias=candidate.dt_bias,
        scale=128**-0.5,
        initial_state=candidate.state,
        out=candidate.out,
        ssm_state_indices=candidate.state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(candidate.out, reference.out, rtol=0, atol=0)
    torch.testing.assert_close(candidate.state, reference.state, rtol=0, atol=0)

    bv, num_warps, num_stages = _get_packed_decode_launch_config(
        batch_size,
        16,
        16,
        128,
        128,
        torch.accelerator.current_device_index(),
    )
    return {
        "batch_size": batch_size,
        "bv": bv,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "max_out_abs_diff": (candidate.out.float() - reference.out.float())
        .abs()
        .max()
        .item(),
        "max_state_abs_diff": (candidate.state.float() - reference.state.float())
        .abs()
        .max()
        .item(),
    }


def metadata(
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    rep_ms: int,
) -> dict[str, object]:
    device_index = torch.accelerator.current_device_index()
    return {
        "type": "metadata",
        "device_index": device_index,
        "device_name": current_platform.get_device_name(device_index),
        "device_capability": current_platform.get_device_capability(device_index),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "triton_version": getattr(triton, "__version__", "unknown"),
        "dtype": str(dtype),
        "state_dtype": str(state_dtype),
        "seed": 0,
        "rep_ms": rep_ms,
        "reference_config": {
            "bv": 32,
            "num_warps": 1,
            "num_stages": 3,
        },
    }


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark Qwen3.5 GDN packed-decode launch configurations."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 14, 16, 19, 24, 32, 48, 64],
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--state-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--validate-wrapper", action="store_true")
    parser.add_argument("--bvs", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--num-warps",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=1000,
        help="Measurement time in milliseconds for each launch configuration.",
    )
    args = parser.parse_args()
    if not torch.accelerator.is_available():
        parser.error("This benchmark requires a CUDA device.")
    if args.rep_ms <= 0:
        parser.error("--rep-ms must be greater than zero.")

    dtype = getattr(torch, args.dtype)
    state_dtype = getattr(torch, args.state_dtype)
    print(
        json.dumps(metadata(dtype, state_dtype, args.rep_ms), sort_keys=True),
        flush=True,
    )

    if args.validate_wrapper:
        for batch_size in args.batch_sizes:
            print(
                json.dumps(
                    validate_wrapper(batch_size, dtype, state_dtype),
                    sort_keys=True,
                ),
                flush=True,
            )
        return

    configs = [
        (bv, num_warps, num_stages)
        for bv in args.bvs
        for num_warps in args.num_warps
        for num_stages in args.num_stages
    ]
    for batch_size in args.batch_sizes:
        for bv, num_warps, num_stages in configs:
            try:
                result = benchmark(
                    batch_size,
                    dtype,
                    state_dtype,
                    bv,
                    num_warps,
                    num_stages,
                    args.rep_ms,
                )
            except Exception as exc:
                result = {
                    "batch_size": batch_size,
                    "bv": bv,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                    "error": str(exc),
                }
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
