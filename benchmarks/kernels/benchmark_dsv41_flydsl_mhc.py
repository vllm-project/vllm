# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""DeepSeek-V4.1's mHC call sites on ROCm (gfx950, hidden 5120, hc_mult 4):
what vLLM main runs against the VLLM_ROCM_USE_AITER_FLYDSL_MHC call sites.

Paths (each the vLLM-level call sequence, not the bare kernel):

- `seam`, `seam_mxfp8`: a delayed seam with a post-mix. main: `MHCPreDelayedOp`
  with the Mega-MHC pack (`mhc_pre_delayed_mxfp8` for the MXFP8 layer input).
  flydsl: `mhc_pre_delayed_flydsl`, which keeps main's seam outside AITER's rule.
- `aux_seam`, `aux_seam_mxfp8`: a DSpark aux layer. main: `MHCPostOp` + mean
  (`DeepseekV4Model._run_layers`), then the next layer's seam. flydsl: the seam
  with `aux_mean`.
- `tail`: the step end. main: `MHCPostOp` + `hc_collapse_triton` + AITER
  `rms_norm` (what `RMSNorm` dispatches to on ROCm with CUDA graphs and
  VLLM_ROCM_USE_AITER=1). flydsl: `mhc_tail_flydsl`.
- `tail_nopost`: the step end after decoder replay (already post-mixed): main
  `hc_collapse_triton` + `rms_norm`; flydsl `mhc_tail_flydsl` without post.

With an AITER that has `mhc_post_reduce` (ROCm/aiter#8155), a `post_reduce`
column runs it in place of main's `MHCPostOp` + reduction: the aux mean before
main's seam, and the pre-mix collapse before `rms_norm` in the tail. Columns
whose kernels the installed AITER lacks are skipped.

Timing: each candidate is captured as one CUDA graph of `--calls` calls over
`--sets` distinct input sets, warmed up and captured on one stream (Mega-MHC and
the FlyDSL seam keep per-stream scratch), and replayed `--replays` times; the
best of `--rounds` rounds, candidates alternating, in us per call. Another job
on the same GPU shows up as noise.

Example:
    VLLM_ROCM_USE_AITER=1 python benchmarks/kernels/benchmark_dsv41_flydsl_mhc.py
    python benchmarks/kernels/benchmark_dsv41_flydsl_mhc.py --paths tail \
        --tokens 16 192 1024

"""

import argparse
import json
import os
from collections.abc import Callable
from functools import partial

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_FLYDSL_MHC", "1")

import torch  # noqa: E402

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mhc import (  # noqa: E402
    MHCPostOp,
    MHCPreDelayedOp,
)
from vllm.models.deepseek_v41.amd import mhc  # noqa: E402

H, HC, MIX = 5120, 4, 24
GATES = (1e-6, 1e-6, 1e-6, 2.0, 20)
DEFAULT_TOKENS = {
    "seam": (128, 160, 192, 256, 288),
    "seam_mxfp8": (160, 192, 256),
    "aux_seam": (128, 160, 192, 256, 384, 512, 1024),
    "aux_seam_mxfp8": (128, 192, 384, 1024),
    "tail": (16, 160, 192, 256, 512, 1024),
    "tail_nopost": (16, 160, 192, 256, 1024),
}


class Paths:
    def __init__(self):
        rocm_aiter_ops.refresh_env_variables()
        with set_current_vllm_config(VllmConfig()):
            self.pre_delayed = MHCPreDelayedOp()
            self.post_op = MHCPostOp()
        self.ops = mhc.get_flydsl_mhc_ops(H, HC)
        import aiter

        self.rms_norm = aiter.rms_norm
        self.post_reduce = getattr(aiter, "mhc_post_reduce", None)
        g = torch.Generator("cuda").manual_seed(0)
        fn = torch.randn(MIX, HC * H, device="cuda", generator=g) * 0.01
        self.fn = fn
        self.fn_mega = rocm_aiter_ops.mega_mhc_pack_fn(fn)
        self.fn_flydsl = None if self.ops is None else self.ops.flydsl_mhc_pack_fn(fn)
        self.scale = torch.rand(3, device="cuda", generator=g)
        self.base = torch.randn(MIX, device="cuda", generator=g) * 0.1
        self.w = (torch.rand(H, device="cuda", generator=g) + 0.5).to(torch.bfloat16)

    @staticmethod
    def inputs(t: int, seed: int) -> dict[str, torch.Tensor]:
        g = torch.Generator("cuda").manual_seed(seed)
        return dict(
            residual=torch.randn(t, HC, H, device="cuda", generator=g).to(
                torch.bfloat16
            ),
            x=torch.randn(t, H, device="cuda", generator=g).to(torch.bfloat16),
            post=torch.rand(t, HC, 1, device="cuda", generator=g),
            comb=torch.rand(t, HC, HC, device="cuda", generator=g),
            pre=torch.softmax(torch.randn(t, HC, device="cuda", generator=g), -1),
        )

    def _seam_kwargs(self, d):
        return dict(
            pre_mix=d["pre"],
            norm_weight=self.w,
            norm_eps=1e-6,
            sublayer_out=d["x"],
            post_layer_mix=d["post"],
            comb_res_mix=d["comb"],
            fn_packed=self.fn_mega,
        )

    def _main_seam_fn(self, mxfp8: bool):
        if mxfp8:
            return partial(mhc.mhc_pre_delayed_mxfp8, self.pre_delayed)
        return self.pre_delayed

    def main_seam(self, d, mxfp8: bool):
        seam = self._main_seam_fn(mxfp8)
        return seam(
            d["residual"], self.fn, self.scale, self.base, *GATES,
            **self._seam_kwargs(d),
        )  # fmt: skip

    def flydsl_seam(self, d, mxfp8: bool, aux=None):
        return mhc.mhc_pre_delayed_flydsl(
            self._main_seam_fn(mxfp8),
            d["residual"], self.fn, self.scale, self.base, *GATES,
            **self._seam_kwargs(d),
            ops=self.ops, fn_flydsl=self.fn_flydsl, mxfp8=mxfp8, aux_out=aux,
        )  # fmt: skip

    def main_aux_seam(self, d, mxfp8: bool):
        aux = self.post_op(d["x"], d["residual"], d["post"], d["comb"]).mean(dim=1)
        return aux, self.main_seam(d, mxfp8)

    def post_reduce_aux_seam(self, d, mxfp8: bool):
        aux = torch.empty(d["x"].shape, dtype=torch.bfloat16, device="cuda")
        self.post_reduce(aux, d["x"], d["residual"], d["post"], d["comb"])
        return aux, self.main_seam(d, mxfp8)

    def flydsl_aux_seam(self, d, mxfp8: bool):
        aux: list[torch.Tensor] = []
        return aux, self.flydsl_seam(d, mxfp8, aux)

    def main_tail(self, d):
        h = self.post_op(d["x"], d["residual"], d["post"], d["comb"])
        h = torch.ops.vllm.hc_collapse_triton(h, d["pre"])
        return self.rms_norm(h, self.w, 1e-6)

    def post_reduce_tail(self, d):
        h = torch.empty(d["x"].shape, dtype=torch.bfloat16, device="cuda")
        self.post_reduce(h, d["x"], d["residual"], d["post"], d["comb"], d["pre"])
        return self.rms_norm(h, self.w, 1e-6)

    def flydsl_tail(self, d):
        return mhc.mhc_tail_flydsl(
            self.ops, d["residual"], d["pre"], self.w, 1e-6,
            d["x"], d["post"], d["comb"],
        )  # fmt: skip

    def main_tail_nopost(self, d):
        h = torch.ops.vllm.hc_collapse_triton(d["residual"], d["pre"])
        return self.rms_norm(h, self.w, 1e-6)

    def flydsl_tail_nopost(self, d):
        return mhc.mhc_tail_flydsl(self.ops, d["residual"], d["pre"], self.w, 1e-6)

    def candidates(self, path: str) -> dict[str, Callable]:
        mxfp8 = path.endswith("_mxfp8")
        base = path.removesuffix("_mxfp8")
        cands: dict[str, Callable | None] = {}
        if base == "seam":
            cands["main"] = partial(self.main_seam, mxfp8=mxfp8)
            cands["flydsl"] = partial(self.flydsl_seam, mxfp8=mxfp8)
        elif base == "aux_seam":
            cands["main"] = partial(self.main_aux_seam, mxfp8=mxfp8)
            if self.post_reduce is not None:
                cands["post_reduce"] = partial(self.post_reduce_aux_seam, mxfp8=mxfp8)
            cands["flydsl"] = partial(self.flydsl_aux_seam, mxfp8=mxfp8)
        elif base == "tail":
            cands["main"] = self.main_tail
            if self.post_reduce is not None:
                cands["post_reduce"] = self.post_reduce_tail
            cands["flydsl"] = self.flydsl_tail
        elif base == "tail_nopost":
            cands["main"] = self.main_tail_nopost
            cands["flydsl"] = self.flydsl_tail_nopost
        else:
            raise ValueError(path)
        if self.ops is None:
            cands.pop("flydsl")
        return cands


def time_graph(call, sets, calls: int, replays: int) -> float:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for d in sets:
            call(d)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(calls):
                call(sets[i % len(sets)])
    torch.accelerator.synchronize()
    graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1e3 / (replays * calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--tokens", type=int, nargs="+", default=None)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--sets", type=int, default=4)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json", default=None, help="write the rows here")
    args = parser.parse_args()

    paths = Paths()
    print(
        f"# {torch.cuda.get_device_name()}; FlyDSL mHC: "
        f"{'yes' if paths.ops is not None else 'no'}; "
        f"mhc_post_reduce: {'yes' if paths.post_reduce is not None else 'no'}; "
        f"us per call, best of {args.rounds} rounds",
        flush=True,
    )
    rows = []
    for path in args.paths:
        cands = paths.candidates(path)
        for t in args.tokens or DEFAULT_TOKENS[path]:
            sets = [Paths.inputs(t, seed) for seed in range(args.sets)]
            best = {k: float("inf") for k in cands}
            for _ in range(args.rounds):
                for name, call in cands.items():
                    us = time_graph(call, sets, args.calls, args.replays)
                    best[name] = min(best[name], us)
            row = {
                "path": path,
                "tokens": t,
                **{k: round(v, 2) for k, v in best.items()},
            }
            speedups = "".join(
                f"\t{k}_speedup={best['main'] / v:.2f}x"
                for k, v in best.items()
                if k != "main"
            )
            print(
                f"{path}\t{t}\t"
                + "\t".join(f"{k}={v:.2f}" for k, v in best.items())
                + speedups,
                flush=True,
            )
            rows.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
