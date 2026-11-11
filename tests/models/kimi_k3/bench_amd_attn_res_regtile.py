# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark: AttnRes decode kernel, online-softmax vs register tile.

Times CUDA-graph replays of many launches (rotating input buffers so the bank
is not L2-resident), reported as us per launch.

python tests/models/kimi_k3/bench_amd_attn_res_regtile.py [--sweep]
"""

import argparse
import functools
import itertools

import torch

from vllm.models.kimi_k3.amd.ops.attn_res import _launch_attn_res
from vllm.platforms import current_platform

H = 7168
BANK = 8  # cdiv(93 layers, attn_res_block_size 12)


def make_inputs(t, copies):
    g = torch.Generator(device="cuda").manual_seed(0)

    def rn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=g)

    sets = []
    for _ in range(copies):
        sets.append(
            dict(
                prefix=rn(t, H),
                delta=rn(t, H),
                blocks=rn(t, BANK + 1, H),
                out_bf16=torch.empty(t, H, device="cuda", dtype=torch.bfloat16),
                out_fp8=torch.empty(
                    t, H, device="cuda", dtype=current_platform.fp8_dtype()
                ),
                scale=torch.empty(t, 1, device="cuda", dtype=torch.float32),
            )
        )
    w = dict(
        norm=1 + 0.1 * rn(H),
        qk=rn(H) / H**0.5,
        onorm=1 + 0.1 * rn(H),
    )
    return sets, w


def launch(s, w, nvb, quant, regtile, block_l, num_warps, write):
    _launch_attn_res(
        s["prefix"],
        s["delta"],
        s["blocks"],
        w["norm"],
        w["qk"],
        w["onorm"],
        s["out_fp8"] if quant else s["out_bf16"],
        s["scale"] if quant else None,
        nvb,
        nvb if write else -1,
        1e-5,
        1e-5,
        current_platform.fp8_dtype() if quant else None,
        num_rows=s["prefix"].shape[0],
        regtile=regtile,
        block_l=block_l,
        num_warps=num_warps,
    )


def time_us(fn_list, iters):
    # Warm up / compile.
    for f in fn_list:
        f()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(iters):
            for f in fn_list:
                f()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / (iters * len(fn_list)))
    return best


CONFIGS = {
    "old L1 w4": (False, 1, 4),
    "old L4 w8": (False, 4, 8),
    "old L2 w8": (False, 2, 8),
    "old L8 w8": (False, 8, 8),
    "old L4 w16": (False, 4, 16),
    "reg w4": (True, 0, 4),
    "reg w8": (True, 0, 8),
    "reg w16": (True, 0, 16),
}


def make_step(nvb, quant):
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

    def step(prefix, delta, blocks, w):
        return attn_res(
            prefix,
            delta,
            blocks,
            w["norm"],
            w["qk"],
            w["onorm"],
            nvb,
            -1,
            1e-5,
            1e-5,
            quant_dtype=current_platform.fp8_dtype() if quant else None,
        )

    return step


def bench_public(args):
    """attn_res() as the model calls it: eager, and torch.compile'd the way
    vLLM does it (one dynamic trace at a large size, shape guards dropped)."""
    import os

    tokens = list(map(int, args.tokens.split(",")))
    print(
        "tokens nvb quant | flag0 eager | flag0 compiled | flag1 eager | flag1 compiled"
    )
    for nvb, quant in itertools.product(
        map(int, args.nvb.split(",")), map(int, args.quant.split(","))
    ):
        results = {t: [] for t in tokens}
        for flag in ("0", "1"):
            os.environ["VLLM_K3_ATTN_RES_REGTILE"] = flag
            torch._dynamo.reset()

            step = make_step(nvb, quant)

            compiled = torch.compile(
                step,
                fullgraph=True,
                dynamic=True,
                options={"guard_filter_fn": lambda g: [False for _ in g]},
            )
            sets, w = make_inputs(8192, 1)
            s = sets[0]
            compiled(s["prefix"], s["delta"], s["blocks"], w)
            for t in tokens:
                copies = max(1, min(32, (256 << 20) // (t * (BANK + 3) * H * 2)))
                sets, w = make_inputs(t, copies)
                iters = max(4, 400 // copies)
                for fn in (step, compiled):
                    fns = [
                        functools.partial(fn, s["prefix"], s["delta"], s["blocks"], w)
                        for s in sets
                    ]
                    results[t].append(time_us(fns, iters))
        for t in tokens:
            print(
                f"{t:6d} {nvb:3d} {quant:5d} | "
                + " | ".join(f"{x:11.2f}" for x in results[t]),
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tokens", default="4,16,64,192")
    ap.add_argument("--nvb", default="1,2,4,8")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--quant", default="1")
    ap.add_argument("--write", type=int, default=0)
    # 0 = rotate enough input sets to defeat L2; 1 = L2-hot single set.
    ap.add_argument("--copies", type=int, default=0)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    if args.public:
        bench_public(args)
        return
    configs = args.configs.split(",")
    print("tokens nvb quant | " + " | ".join(f"{c:>11}" for c in configs))
    for t, nvb, quant in itertools.product(
        map(int, args.tokens.split(",")),
        map(int, args.nvb.split(",")),
        map(int, args.quant.split(",")),
    ):
        bytes_per_set = t * (BANK + 3) * H * 2
        copies = args.copies or max(1, min(32, (256 << 20) // bytes_per_set))
        sets, w = make_inputs(t, copies)
        iters = max(4, 400 // copies)
        row = []
        for c in configs:
            regtile, block_l, warps = CONFIGS[c]
            fns = [
                functools.partial(
                    launch, s, w, nvb, quant, regtile, block_l, warps, args.write
                )
                for s in sets
            ]
            row.append(time_us(fns, iters))
        print(
            f"{t:6d} {nvb:3d} {quant:5d} | " + " | ".join(f"{x:11.2f}" for x in row),
            flush=True,
        )


if __name__ == "__main__":
    main()
