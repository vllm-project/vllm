# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt IQ2_XXS / Q8_0 Triton kernels on NVIDIA Nemotron Bolt shapes (TP=1).

Dense layers are compared with bf16 cuBLAS on the dequantized weights, MoE
layers with vLLM's bf16 Triton `fused_experts`. Timing: CUPTI, CUDA graph,
cold L2, median.

    python benchmarks/kernels/benchmark_modelopt_block_triton.py
    python benchmarks/kernels/benchmark_modelopt_block_triton.py \
        --layer shared_up --m 1 --sweep
"""

import argparse
import itertools
import statistics
from functools import partial

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm.model_executor.layers.fused_moe.experts.modelopt_block_triton_moe as moe
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization import modelopt_block_quant_triton as t
from vllm.triton_utils import triton
from vllm.v1.worker.workspace import init_workspace_manager

DENSE = [
    ("qkv_proj", "q8_0", 4608, 4096),
    ("o_proj", "q8_0", 4096, 4096),
    ("fc1_latent", "q8_0", 1024, 4096),
    ("fc2_latent", "q8_0", 4096, 1024),
    ("shared_up", "iq2_xxs", 5376, 4096),
    ("shared_down", "iq2_xxs", 4096, 5376),
    ("lm_head", "q8_0", 131072, 4096),
]
DENSE_M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
MOE = [("moe_240x2048", 240, 2048), ("moe_432x2560", 432, 2560)]
MOE += [("moe_496x2560", 496, 2560)]
MOE_M = [1, 2, 4, 8, 16, 64, 256, 1024, 4096]
BLOCK = {"iq2_xxs": (256, 66), "q8_0": (32, 34)}


def bench_us(fn):
    for _ in range(25):
        fn()
    torch.accelerator.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn, use_cuda_graph=True)) * 1e3


def random_blocks(lead, k, codec):
    size, nbytes = BLOCK[codec]
    raw = torch.randint(0, 256, (*lead, k // size, nbytes), dtype=torch.uint8)
    scales = (torch.randn(*lead, k // size) * 2e-3).half()
    raw[..., :2] = scales.unsqueeze(-1).view(torch.uint8)
    return t.repack_blocks(raw, codec)


def try_time(fn, ref):
    try:
        out = fn()
    except triton.runtime.errors.OutOfResources:
        return None
    err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err < 2e-2, f"wrong result, relative error {err:.3g}"
    return bench_us(fn)


def linear_sweep(M, K):
    # Odd splits too: shared_down has 5376 = 21 * 256 steps, so only 3 and 7 fit.
    for bn, bk, split, warps, stages in itertools.product(
        (8, 16, 32, 64, 128),
        (64, 128, 256),
        (1, 2, 3, 4, 7, 8, 16),
        (1, 2, 4, 8),
        (1, 2, 3),
    ):
        if K % (split * bk):
            continue
        gemv = (triton.next_power_of_2(M),) if M <= 8 else ()
        gemm = (16, 32, 64, 128) if M > 1 and bn >= 16 else ()
        for bm in gemv + gemm:
            yield {"BM": bm, "BN": bn, "BK": bk, "SPLIT": split}, warps, stages


def bench_dense(args, props, rows):
    for name, codec, N, K in DENSE:
        if args.layer not in (None, name):
            continue
        qweight, scales = random_blocks((N,), K, codec)
        lut = t.iq2_lut(qweight.device) if codec == "iq2_xxs" else None
        w = t.dequantize_blocks(qweight, scales, lut, codec)
        bpw = BLOCK[codec][1] / BLOCK[codec][0]
        for M in args.m or DENSE_M:
            if name == "lm_head" and M > 512:
                continue
            x = torch.randn(M, K, dtype=torch.bfloat16)
            ref = x @ w.T
            dequant = t.dequant_min_m(codec, torch.accelerator.current_device_index())
            if dequant <= M and N * K <= t.DEQUANT_MAX_NUMEL:
                cfg = "dequant+cuBLAS"
            else:
                cfg = t._linear_config(
                    M, N, K, codec, props.multi_processor_count,
                    props.shared_memory_per_block_optin, props.major,
                )  # fmt: skip
            run = partial(t._modelopt_block_linear, x, qweight, scales, lut, codec)
            timed = [(try_time(run, ref), cfg)]
            for c, wp, st in linear_sweep(M, K) if args.sweep else ():
                c = {**c, "num_warps": wp, "num_stages": st}
                run = partial(t.launch_linear, x, qweight, scales, lut, codec, c)
                if (us := try_time(run, ref)) is not None:
                    timed.append((us, c))
            us, cfg = min(timed, key=lambda r: r[0])
            cublas = bench_us(partial(torch.matmul, x, w.T))
            nbytes = N * K * bpw + 2 * M * K + 2 * M * N
            rows.append((name, M, us, cublas, nbytes, 2 * M * N * K, cfg))


def bench_moe(args, rows):
    for name, E, inter in MOE:
        if args.layer not in (None, name):
            continue
        w1q, s1 = random_blocks((E, inter), 1024, "iq2_xxs")
        w2q, s2 = random_blocks((E, 1024), inter, "iq2_xxs")
        lut = t.iq2_lut(w1q.device)
        w1 = torch.stack(
            [t.dequantize_blocks(w1q[e], s1[e], lut, "iq2_xxs") for e in range(E)]
        )
        w2 = torch.stack(
            [t.dequantize_blocks(w2q[e], s2[e], lut, "iq2_xxs") for e in range(E)]
        )
        relu2 = MoEActivation.RELU2_NO_MUL
        moe_config = FusedMoEConfig(
            num_experts=E, experts_per_token=9, hidden_dim=1024,
            intermediate_size=inter, num_local_experts=E, num_logical_experts=E,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            activation=relu2, in_dtype=torch.bfloat16, device="cuda",
            routing_method=RoutingMethodType.TopK, max_num_tokens=max(MOE_M),
        )  # fmt: skip
        quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(),
            _a2=FusedMoEQuantDesc(),
            _w1=FusedMoEQuantDesc(dtype="iq2_xxs", scale=s1),
            _w2=FusedMoEQuantDesc(dtype="iq2_xxs", scale=s2),
        )
        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            moe.ModelOptBlockTritonExperts(moe_config, quant_config, "iq2_xxs", lut),
        )
        default_config = moe.moe_config
        default_gemv = moe.moe_gemv_config
        gen = torch.Generator(device="cuda").manual_seed(0)
        for M in args.m or MOE_M:
            x = torch.randn(M, 1024, dtype=torch.bfloat16)
            topk_ids = torch.rand(M, E, generator=gen, device="cuda").topk(9).indices
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = torch.rand(M, 9, device="cuda").softmax(-1)
            ours = partial(
                kernel.apply, x, w1q, w2q, topk_weights, topk_ids,
                activation=relu2, global_num_experts=E, expert_map=None,
                apply_router_weight_on_input=False,
            )  # fmt: skip
            bf16 = partial(
                fused_experts, x, w1, w2, topk_weights, topk_ids, activation=relu2
            )
            ref = bf16()
            dot_cfg = default_config(M, E, 9)
            gemv_cfg = default_gemv(M, E, 9, "iq2_xxs")
            candidates = [(None, gemv_cfg) if gemv_cfg else (dot_cfg, None)]
            if args.sweep:
                candidates += [
                    ({"BLOCK_SIZE_M": bm, "BN": bn, "BK": bk, "num_warps": wp,
                      "num_stages": st}, None)
                    for bm, bn, bk, wp, st in itertools.product(
                        (16, 32, 64, 128), (32, 64, 128), (64, 128, 256), (4, 8),
                        (1, 2, 3),
                    )
                ]  # fmt: skip
                candidates += [
                    (None, {"BN": bn, "BK": bk, "num_warps": wp, "num_stages": st})
                    for bn, bk, wp, st in itertools.product(
                        (8, 16, 32, 64), (64, 128, 256), (1, 2, 4, 8), (1, 2)
                    )
                ]  # fmt: skip
            timed = []
            fallback = default_config(M, E, 9)
            for dot_c, gemv_c in candidates:
                moe.moe_config = lambda *_, c=dot_c or fallback: c
                moe.moe_gemv_config = lambda *_, c=gemv_c: c
                if (us := try_time(ours, ref)) is not None:
                    timed.append((us, gemv_c or {**dot_c, "gemv": False}))
            moe.moe_config = default_config
            moe.moe_gemv_config = default_gemv
            active = topk_ids.unique().numel()
            nbytes = active * 2 * inter * 1024 * 66 / 256 + 4 * M * 1024
            flops = 2 * M * 9 * 2 * inter * 1024
            us, cfg = min(timed, key=lambda r: r[0])
            rows.append((name, M, us, bench_us(bf16), nbytes, flops, cfg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", help="one row name, e.g. shared_up")
    parser.add_argument("--m", type=int, nargs="*", help="token counts")
    parser.add_argument("--sweep", action="store_true", help="search tile configs")
    args = parser.parse_args()
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    init_workspace_manager(torch.device("cuda"))
    props = torch.cuda.get_device_properties()

    src = torch.empty(1 << 30, dtype=torch.uint8)
    copy_gbps = 2 * src.numel() / (bench_us(partial(torch.clone, src)) * 1e3)
    del src
    print(f"{props.name}, copy bandwidth {copy_gbps:.0f} GB/s")
    print("layer M ours_us baseline_us ratio GB/s TFLOP/s config")

    rows = []
    with set_current_vllm_config(VllmConfig()):
        bench_dense(args, props, rows)
        bench_moe(args, rows)
    for name, M, us, base, nbytes, flops, cfg in rows:
        print(
            f"{name} {M} {us:.1f} {base:.1f} {us / base:.2f} "
            f"{nbytes / us / 1e3:.0f} {flops / us / 1e6:.1f} {cfg}"
        )


if __name__ == "__main__":
    main()
