# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the CPU GDN recurrent-update kernels used for decode and
speculative-decode verify (defaults match Qwen3.6-35B-A3B)."""

import math
import time

import numpy as np
import torch

import vllm._custom_ops as ops
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

STATE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@torch.inference_mode()
def main(args) -> None:
    set_random_seed(args.seed)
    batch_size = args.batch_size
    q_len = args.num_spec_tokens + 1 if args.mode == "spec" else 1
    num_tokens = batch_size * q_len
    hk, hv, dk, dv = args.num_k_heads, args.num_v_heads, args.k_dim, args.v_dim
    state_dtype = STATE_DTYPES[args.state_dtype]
    num_cols = q_len

    q = torch.randn(num_tokens, hk, dk, dtype=torch.bfloat16)
    k = torch.randn(num_tokens, hk, dk, dtype=torch.bfloat16)
    v = torch.randn(num_tokens, hv, dv, dtype=torch.bfloat16)
    a = torch.randn(num_tokens, hv, dtype=torch.bfloat16)
    b = torch.randn(num_tokens, hv, dtype=torch.bfloat16)
    A_log = torch.rand(hv, dtype=torch.float32)
    dt_bias = torch.rand(hv, dtype=torch.bfloat16)

    # In serving, ~10 GB of weights stream between two visits to a layer's
    # state, so the state is always cold. Rotate through a footprint far larger
    # than the LLC so reads and write-backs of the state hit DRAM here as well.
    num_slots = batch_size * num_cols
    pool_bytes = num_slots * hv * dk * dv * state_dtype.itemsize
    num_pools = max(
        args.num_layers, math.ceil(args.footprint_gb * 1024**3 / pool_bytes)
    )
    pools = [
        torch.randn(num_slots, hv, dk, dv).to(state_dtype) for _ in range(num_pools)
    ]
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32) * q_len
    spec_indices = torch.arange(num_slots, dtype=torch.int32).view(
        batch_size, num_cols
    )
    num_accepted = torch.full((batch_size,), max(1, q_len // 2), dtype=torch.int32)
    state_indices = torch.arange(batch_size, dtype=torch.int32)
    decode_cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32)

    def run_layer(pool: torch.Tensor) -> None:
        if args.mode == "spec":
            ops.fused_sigmoid_gating_delta_rule_update_spec_cpu(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                initial_state_source=pool,
                spec_state_indices=spec_indices,
                num_accepted_tokens=num_accepted,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            ops.fused_sigmoid_gating_delta_rule_update_cpu(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q.unsqueeze(0),
                k=k.unsqueeze(0),
                v=v.unsqueeze(0),
                a=a,
                b=b,
                initial_state_source=pool,
                initial_state_indices=state_indices,
                cu_seqlens=decode_cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )

    next_pool = [0]

    def run(iters: int) -> list[float]:
        times = []
        for _ in range(iters):
            start = time.perf_counter_ns()
            for _ in range(args.num_layers):
                run_layer(pools[next_pool[0]])
                next_pool[0] = (next_pool[0] + 1) % num_pools
            times.append((time.perf_counter_ns() - start) / 1e6)
        return times

    run(args.warmup)
    times = run(args.iters)

    state_bytes = hv * dk * dv * state_dtype.itemsize * batch_size
    # Spec: read one slot, write q_len slots. Decode: read and write in place.
    bytes_per_layer = state_bytes * (1 + q_len)
    mean_ms = float(np.mean(times))
    gbps = bytes_per_layer * args.num_layers / (mean_ms * 1e-3) / 1e9
    print(
        f"mode={args.mode} batch={batch_size} q_len={q_len} "
        f"state={args.state_dtype} layers={args.num_layers} pools={num_pools} "
        f"threads={torch.get_num_threads()}"
    )
    print(
        f"  per step (all layers): mean {mean_ms:.3f} ms, "
        f"median {float(np.median(times)):.3f} ms, min {min(times):.3f} ms"
    )
    print(f"  per layer: {mean_ms / args.num_layers * 1e3:.1f} us")
    print(f"  state traffic: {gbps:.1f} GB/s")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["spec", "decode"], default="spec")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=7)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--k-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument(
        "--state-dtype", choices=list(STATE_DTYPES), default="float32"
    )
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument(
        "--footprint-gb",
        type=float,
        default=4.0,
        help="Total state memory rotated through; keep well above the LLC size.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
