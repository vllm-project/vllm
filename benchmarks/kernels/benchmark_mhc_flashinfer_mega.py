# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-call time of GLM-5.3-Flash's fused mHC step: TileLang vs FlashInfer mega.

Times `mhc_fused_post_pre_tilelang` (TileLang + DeepGEMM's TF32 projection
GEMM, the production path) and `mhc_fused_post_pre_flashinfer_mega` for each
token count, as 100 [predecessor -> mHC] pairs replayed from a CUDA graph,
minus the predecessor alone. Each pair uses its own copy of the weights, so they
come from HBM as in a model, where every layer has its own. The predecessor
stands for what precedes the step in serving:

- plain: a tiny kernel that does not allow PDL overlap, as after a
  tensor-parallel all-reduce
- pdl: a PDL-launched TileLang kernel the mHC launch can overlap, as in data
  or expert parallel serving

The token count is the per-GPU count: the batch size under TP, about
batch / DP size under data parallelism. `--pad-nan K` fills the last K rows
with NaN, as the rows CUDA graphs pad a batch with can hold.

    python benchmarks/kernels/benchmark_mhc_flashinfer_mega.py --predecessor pdl
    python benchmarks/kernels/benchmark_mhc_flashinfer_mega.py --predecessor plain

`--nvfp4` instead compares the two ways an NVFP4 MoE can get its input: the
fused op followed by `scaled_fp4_quant`, and the fused op writing it directly.
"""

import argparse

import torch

import vllm.model_executor.kernels.mhc  # noqa: F401  (registers the custom ops)
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.mhc.flashinfer_mega import split_mhc_fn
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.workspace import init_workspace_manager

H, HC = 4096, 4
RMS_EPS, NORM_EPS, HC_EPS, POST_MULT, REPEAT = 1e-5, 1e-5, 1e-6, 2.0, 20
PAIRS = 100
DEFAULT_TOKENS = "1,2,3,4,5,6,8,12,16,24,32,48,64,96,128,256,512,1024,2048,4096,8192"


def make_inputs(num_tokens: int, pad_nan: int, g: torch.Generator):
    x = (torch.randn(num_tokens, H, generator=g, device="cuda") * 0.5).bfloat16()
    residual = torch.randn(num_tokens, HC, H, generator=g, device="cuda").bfloat16()
    post = torch.rand(num_tokens, HC, 1, generator=g, device="cuda") * POST_MULT
    comb = torch.softmax(
        torch.randn(num_tokens, HC, HC, generator=g, device="cuda"), dim=-1
    )
    if 0 < pad_nan < num_tokens:
        for t in (x, residual, post, comb):
            t[-pad_nan:] = float("nan")
    return x, residual, post, comb


def make_weights(g: torch.Generator):
    fn = torch.randn(HC * (2 + HC), HC * H, generator=g, device="cuda") * 1e-2
    fn = fn.bfloat16().float()
    scale = torch.rand(3, generator=g, device="cuda") + 0.5
    base = torch.randn(HC * (2 + HC), generator=g, device="cuda") * 0.1
    norm = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()
    return fn, scale, base, norm


def time_pairs(predecessor, step) -> float:
    """Microseconds per [predecessor -> step(i)] pair, from a CUDA graph replay."""
    for i in range(PAIRS):
        predecessor()
        step(i)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(PAIRS):
            predecessor()
            step(i)
    graph.replay()
    torch.accelerator.synchronize()
    start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        torch.accelerator.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / PAIRS)
    return best


def bench_one(num_tokens, predecessor_kind, pad_nan, g, layers, nvfp4) -> str:
    """One table row: microseconds of each mHC path beyond its predecessor."""
    x, residual, post, comb = make_inputs(num_tokens, pad_nan, g)
    eps = (RMS_EPS, HC_EPS, HC_EPS, POST_MULT, REPEAT)
    marker = torch.zeros(1, device="cuda")

    def predecessor():
        if predecessor_kind == "pdl":
            torch.ops.vllm.mhc_post_tilelang(x, residual, post, comb)
        else:
            marker.add_(1)

    def tilelang(i):
        fn, _, scale, base, norm = layers[i]
        torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, post, comb, fn, scale, base, *eps, 1, 1, norm, NORM_EPS,
        )  # fmt: skip

    gscale = torch.tensor([448.0 * 6.0 / 8.0], device="cuda")
    y_fp4 = x.new_empty(num_tokens, H // 2, dtype=torch.uint8)
    y_sf = x.new_empty(num_tokens, H // 16, dtype=torch.float8_e4m3fn)

    def mega(i, *nvfp4_out):
        fn, fn_split, scale, base, norm = layers[i]
        return torch.ops.vllm.mhc_fused_post_pre_flashinfer_mega(
            x, residual, post, comb, fn, fn_split, scale, base, *eps, norm,
            NORM_EPS, *nvfp4_out,
        )  # fmt: skip

    def mega_then_quant(i):
        ops.scaled_fp4_quant(mega(i)[3], gscale, is_sf_swizzled_layout=False)

    def mega_nvfp4(i):
        mega(i, gscale, y_fp4, y_sf)

    # With --nvfp4: today's two steps vs one launch writing the NVFP4 input.
    old, new = (mega_then_quant, mega_nvfp4) if nvfp4 else (tilelang, mega)
    t_pred = time_pairs(predecessor, lambda i: None)
    t_tilelang = time_pairs(predecessor, old) - t_pred
    t_mega = time_pairs(predecessor, new) - t_pred
    return (
        f"| {num_tokens} | {t_pred:.2f} | {t_tilelang:.2f} | {t_mega:.2f} | "
        f"{t_tilelang / t_mega:.3f}x |"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--predecessor", choices=("plain", "pdl"), default="pdl")
    ap.add_argument("--pad-nan", type=int, default=0)
    ap.add_argument("--nvfp4", action="store_true")
    args = ap.parse_args()

    device = torch.accelerator.current_device_index()
    init_workspace_manager(torch.device("cuda", device))
    g = torch.Generator(device="cuda").manual_seed(0)
    layers = []
    for _ in range(PAIRS):
        fn, scale, base, norm = make_weights(g)
        layers.append((fn, split_mhc_fn(fn), scale, base, norm))

    print(
        f"# {current_platform.get_device_name(device)}, "
        f"{num_compute_units(device)} SMs; predecessor={args.predecessor}; "
        f"pad-nan={args.pad_nan}"
    )
    old, new = ("mega + quant", "mega NVFP4") if args.nvfp4 else ("TileLang", "mega")
    print(f"| tokens | predecessor, us | {old}, us | {new}, us | speedup |")
    print("|--:|--:|--:|--:|--:|")
    for num_tokens in (int(t) for t in args.tokens.split(",")):
        print(
            bench_one(num_tokens, args.predecessor, args.pad_nan, g, layers, args.nvfp4)
        )


if __name__ == "__main__":
    main()
