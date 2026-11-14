# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark direct E4M3 policies and compare modular versus frozen PTX.

Run benchmark_fp8e4nv_policies.py with the repository installed.
The production implementation is measured only for policies it supports. No NaN
handling or flushing is enabled unless explicitly requested.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv import (
    FP8E4NV_EXTERN_LIBS,
    convert_from_fp8e4m3,
    convert_to_fp8e4m3,
)
from vllm.v1.attention.ops.fp8e4nv_ptx import (
    convert_from_fp8e4m3 as ptx_from,
)
from vllm.v1.attention.ops.fp8e4nv_ptx import (
    convert_to_fp8e4m3 as ptx_to,
)
from vllm.v1.attention.ops.fp8e4nv_ptx import (
    ptx,
)


@triton.jit
def _bench(
    src,
    dst,
    n,
    ENCODE: tl.constexpr,
    BF16: tl.constexpr,
    PACK: tl.constexpr,
    MODE: tl.constexpr,
    ASM: tl.constexpr,
    CONSTRAINTS: tl.constexpr,
    propagate_nan: tl.constexpr,
    enable_ftz: tl.constexpr,
    ITERS: tl.constexpr,
):
    """Measure conversion chains so load/store bandwidth cannot hide their cost."""
    offsets = tl.program_id(0) * 512 + tl.arange(0, 512)
    x = tl.load(src + offsets, offsets < n, other=0)
    dtype = tl.bfloat16 if BF16 else tl.float16
    y = tl.full((512,), 0, tl.uint8 if ENCODE else dtype)
    for _ in range(ITERS):
        if MODE == 2:
            y = tl.inline_asm_elementwise(
                ASM,
                CONSTRAINTS,
                [x],
                dtype=tl.uint8 if ENCODE else dtype,
                is_pure=True,
                pack=PACK,
            )
        elif ENCODE:
            if MODE == 0:
                y = convert_to_fp8e4m3(x, propagate_nan, True)
            else:
                y = ptx_to(x, PACK, propagate_nan, True, enable_ftz)
        else:
            if MODE == 0:
                y = convert_from_fp8e4m3(x, dtype, propagate_nan, True)
            else:
                y = ptx_from(
                    x,
                    dtype,
                    PACK,
                    propagate_nan,
                    True,
                    enable_ftz,
                )
        # A dependent finite input for the next iteration prevents conversion CSE.
        if ENCODE:
            x = ((y.to(tl.uint16) << 7) ^ 0x3C00).to(dtype, bitcast=True)
        else:
            bits = y.to(tl.uint16, bitcast=True)
            x = ((bits ^ (bits >> 7)) & 0x7E).to(tl.uint8)
    tl.store(dst + offsets, y, offsets < n)


def main():
    """Compare all packed widths with identical data and launch configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--iterations", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--propagate-nan", action="store_true")
    parser.add_argument("--enable-ftz", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    print(
        json.dumps(
            {
                "gpu": current_platform.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
            }
        ),
        flush=True,
    )
    for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        for encode in [False, True]:
            direction = "encode" if encode else "decode"
            raw = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
            raw = torch.where((raw & 0x7F) == 0x7F, 0, raw)
            source = raw.repeat(triton.cdiv(args.elements, 256))[: args.elements]
            if encode:
                source = source.view(torch.float8_e4m3fn).to(dtype)
            for pack in [1, 2, 4]:
                asm = ptx(
                    direction,
                    dtype_name,
                    pack,
                    args.propagate_nan,
                    args.enable_ftz,
                )
                constraints = (
                    ("=h,h" if pack == 1 else "=r,r" if pack == 2 else "=r,r,r")
                    if encode
                    else ("=h,r" if pack == 1 else "=r,r" if pack == 2 else "=r,=r,r")
                )
                modes = [1, 2]
                if not args.enable_ftz:
                    modes.insert(0, 0)
                for iters in args.iterations:
                    calls, compiled, outputs = {}, {}, {}
                    for mode in modes:
                        out = torch.empty(
                            args.elements,
                            device="cuda",
                            dtype=torch.uint8 if encode else dtype,
                        )
                        outputs[mode] = out

                        def run(
                            mode=mode,
                            out=out,
                            source=source,
                            encode=encode,
                            dtype=dtype,
                            pack=pack,
                            asm=asm,
                            constraints=constraints,
                            iters=iters,
                        ):
                            return _bench[(triton.cdiv(args.elements, 512),)](
                                source,
                                out,
                                args.elements,
                                encode,
                                dtype == torch.bfloat16,
                                pack,
                                mode,
                                asm,
                                constraints,
                                args.propagate_nan,
                                args.enable_ftz,
                                iters,
                                num_warps=4,
                                extern_libs=FP8E4NV_EXTERN_LIBS,
                            )

                        calls[mode] = run
                        compiled[mode] = run()
                        prefix = f"{dtype_name}-{direction}-x{pack}-i{iters}-mode{mode}"
                        (args.output / f"{prefix}.cubin").write_bytes(
                            compiled[mode].asm["cubin"]
                        )
                        (args.output / f"{prefix}.ptx").write_text(
                            compiled[mode].asm["ptx"]
                        )
                    torch.accelerator.synchronize()
                    assert all(
                        torch.equal(outputs[1].view(torch.uint8), o.view(torch.uint8))
                        for o in outputs.values()
                    )
                    assert compiled[1].n_regs == compiled[2].n_regs
                    samples = {mode: [] for mode in modes}
                    for r in range(args.rounds):
                        order = modes[r % len(modes) :] + modes[: r % len(modes)]
                        for mode in order:
                            samples[mode].append(
                                triton.testing.do_bench(calls[mode], warmup=25, rep=100)
                            )
                    row = {
                        "dtype": dtype_name,
                        "direction": direction,
                        "pack": pack,
                        "iterations": iters,
                        "registers": {m: c.n_regs for m, c in compiled.items()},
                        "median_us": {
                            m: statistics.median(s) * 1000 for m, s in samples.items()
                        },
                        "propagate_nan": args.propagate_nan,
                        "enable_ftz": args.enable_ftz,
                    }
                    rows.append(row)
                    (args.output / "measurements.json").write_text(
                        json.dumps(rows, indent=2)
                    )
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
