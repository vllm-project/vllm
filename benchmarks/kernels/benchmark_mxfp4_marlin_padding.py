# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare MXFP4 MoE padding with identical logical weights and routes.

Defaults reproduce DeepSeek-V4.1-Flash TP8 shapes. Weights and routing are
synthetic; use a serving benchmark for model-level performance.
"""

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.math_utils import round_up


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=288)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 32, 128, 512, 2048]
    )
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    dtype = torch.bfloat16
    e, k, n = args.num_experts, args.hidden_size, args.intermediate_size
    assert k % 256 == 0 and n % 32 == 0
    assert 0 < args.topk <= e
    extents = [round_up(n, alignment) for alignment in (128, 64)]
    if extents[0] == extents[1]:
        raise ValueError("Choose an intermediate size with different padding extents")
    weights, scales = [], []
    for out_dim, in_dim in [(2 * n, k), (k, n)]:
        weights.append(
            torch.randint(
                256, (e, out_dim, in_dim // 2), dtype=torch.uint8, device="cuda"
            )
        )
        scales.append(
            torch.randint(
                119, 123, (e, out_dim, in_dim // 32), dtype=torch.uint8, device="cuda"
            )
        )
    packed = {}
    for padded in extents:

        def pad_gate_up(tensor, padded=padded):
            tensor = tensor.reshape(e, 2, n, -1)
            return F.pad(tensor, (0, 0, 0, padded - n)).reshape(e, 2 * padded, -1)

        packed[padded] = prepare_moe_mxfp4_layer_for_marlin(
            SimpleNamespace(params_dtype=dtype),
            pad_gate_up(weights[0]),
            F.pad(weights[1], (0, (padded - n) // 2)),
            pad_gate_up(scales[0]),
            F.pad(scales[1], (0, (padded - n) // 32)),
            None,
            None,
        )
    del weights, scales

    rows = []
    for m in args.tokens:
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        ids = torch.rand(m, e, device="cuda").topk(args.topk, dim=-1).indices.int()
        routing = torch.rand(m, args.topk, device="cuda")
        routing /= routing.sum(-1, keepdim=True)
        calls = {}
        for padded in extents:
            w13, w2, s13, s2, _, _ = packed[padded]
            workspace = marlin_make_workspace_new(x.device, 4)
            cache13 = torch.empty(
                m * args.topk * max(2 * padded, k), device=x.device, dtype=dtype
            )
            cache2 = torch.empty(m * args.topk * padded, device=x.device, dtype=dtype)
            out = torch.empty_like(x)

            def run(
                x=x,
                routing=routing,
                ids=ids,
                w13=w13,
                w2=w2,
                s13=s13,
                s2=s2,
                workspace=workspace,
                cache13=cache13,
                cache2=cache2,
                out=out,
            ):
                return fused_marlin_moe(
                    x,
                    w13,
                    w2,
                    None,
                    None,
                    s13,
                    s2,
                    routing,
                    ids,
                    quant_type_id=scalar_types.float4_e2m1f.id,
                    activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
                    workspace=workspace,
                    intermediate_cache13=cache13,
                    intermediate_cache2=cache2,
                    output=out,
                )

            calls[padded] = run
        torch.testing.assert_close(
            calls[extents[0]](), calls[extents[1]](), atol=0.03, rtol=0.03
        )
        times = {padded: [] for padded in extents}
        for order in (extents, extents[::-1]):
            for padded in order:
                times[padded].extend(
                    bench_gpu_time_with_cupti(
                        calls[padded],
                        dry_run_iters=5,
                        repeat_iters=args.repeat_iters,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                    )
                )
        medians = {p: float(statistics.median(t)) for p, t in times.items()}
        row = {
            "tokens": m,
            "median_ms": medians,
            "speedup": medians[extents[0]] / medians[extents[1]],
            # Two expert GEMMs: 2*M*topk*K*(2*N) + 2*M*topk*N*K.
            "useful_tflops": {
                p: 6 * m * args.topk * k * n / (ms * 1e9) for p, ms in medians.items()
            },
            "samples_ms": {p: list(map(float, t)) for p, t in times.items()},
        }
        rows.append(row)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "samples_ms"}
            )
        )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "rows": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
