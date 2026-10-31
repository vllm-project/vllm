# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete paged sparse decode calls against an explicitly pinned baseline.

Extract the base revision's sparse_attn.py with git show and pass its path using
--baseline. Run from a working vLLM CUDA installation; no platform shims are used.
The benchmark checks both implementations against a dense FP32 reference before
alternating CUDA Graph timing rounds. Output is JSON Lines. This measures the
attention operator, not index selection, a model, or serving throughput.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

from vllm.models.minimax_m3.common.ops import sparse_attn as candidate
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def case(batch, heads, length, dql, mode, capacity=None, seed=781):
    torch.manual_seed(seed)
    pages = triton.cdiv(length, 128)
    capacity = capacity or pages
    q = torch.randn(batch * dql, heads * 16, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(
        batch * pages, heads, 128, 256, device="cuda", dtype=torch.bfloat16
    )
    table = torch.zeros(batch, capacity, device="cuda", dtype=torch.int32)
    table[:, :pages] = torch.randperm(
        batch * pages, device="cuda", dtype=torch.int32
    ).reshape(batch, pages)
    lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    top = torch.full(
        (heads, batch * dql, 16), 2147483647, device="cuda", dtype=torch.int32
    )
    for t in range(batch * dql):
        visible = length - dql + t % dql + 1
        n = triton.cdiv(visible, 128)
        for h in range(heads):
            top[h, t, : min(16, n)] = torch.randperm(
                n, device="cuda", dtype=torch.int32
            )[:16]
    ks = vs = None
    if mode != "bf16":
        kv = kv.to(torch.float8_e4m3fn)
        if mode == "scalar":
            ks = torch.tensor(0.7, device="cuda")
            vs = torch.tensor(1.3, device="cuda")
        else:
            ks = 0.5 + torch.rand(heads, batch * pages * 128, device="cuda")
            vs = 0.5 + torch.rand_like(ks)
    return dict(
        q=q,
        kv_cache=kv,
        topk_idx=top,
        block_table=table,
        seq_lens=lens,
        num_kv_heads=heads,
        sm_scale=128**-0.5,
        decode_query_len=dql,
        k_scale=ks,
        v_scale=vs,
    )


def dense(c):
    q, kv = c["q"], c["kv_cache"]
    out = torch.empty_like(q)
    for t in range(q.shape[0]):
        req, local = divmod(t, c["decode_query_len"])
        length = int(c["seq_lens"][req]) - c["decode_query_len"] + local + 1
        for h in range(c["num_kv_heads"]):
            kk = []
            vv = []
            for logical in c["topk_idx"][
                h, t, : min(16, triton.cdiv(length, 128))
            ].tolist():
                p = int(c["block_table"][req, logical])
                n = min(128, length - logical * 128)
                k = kv[p, h, :n, :128].bfloat16()
                v = kv[p, h, :n, 128:].bfloat16()
                if c["k_scale"] is not None:
                    ks = c["k_scale"]
                    vs = c["v_scale"]
                    if ks.numel() > 1:
                        ks = ks[h, p * 128 : p * 128 + n, None]
                        vs = vs[h, p * 128 : p * 128 + n, None]
                    k = (k.float() * ks).bfloat16()
                    v = (v.float() * vs).bfloat16()
                kk.append(k)
                vv.append(v)
            k = torch.cat(kk).float()
            v = torch.cat(vv).float()
            out[t, h * 16 : (h + 1) * 16] = (
                torch.softmax(
                    q[t, h * 16 : (h + 1) * 16].float() @ k.T * c["sm_scale"], -1
                )
                @ v
            ).bfloat16()
    return out


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Unmodified sparse_attn.py extracted from the base commit",
    )
    parser.add_argument(
        "--matrix", choices=("development", "heldout"), default="development"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[781, 997])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep-ms", type=int, default=60)
    parser.add_argument("--pad-capacity", type=int, default=0)
    args = parser.parse_args()
    if args.rounds < 1 or args.rep_ms < 1:
        parser.error("rounds and rep-ms must be positive")
    spec = importlib.util.spec_from_file_location(
        "msa_benchmark_baseline", args.baseline
    )
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    print(
        json.dumps(
            dict(
                torch=torch.__version__,
                triton=triton.__version__,
                gpu=torch.cuda.get_device_name(),
                baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                matrix=args.matrix,
                seeds=args.seeds,
                pdl=current_platform.is_arch_support_pdl(),
                rounds=args.rounds,
                rep_ms=args.rep_ms,
                pad_capacity=args.pad_capacity,
            )
        ),
        flush=True,
    )
    shapes = (
        [
            (1, 1, 73, 1),
            (4, 1, 127, 1),
            (2, 2, 113, 3),
            (16, 2, 127, 1),
            (33, 1, 127, 1),
            (1, 1, 383, 1),
            (4, 1, 1025, 1),
            (2, 4, 2047, 3),
            (8, 4, 8192, 1),
            (1, 1, 16385, 1),
        ]
        if args.matrix == "development"
        else [
            (2, 1, 73, 1),
            (3, 1, 127, 2),
            (2, 4, 113, 1),
            (5, 4, 127, 1),
            (2, 1, 193, 1),
            (3, 1, 449, 1),
            (5, 1, 129, 1),
            (2, 4, 1025, 2),
            (3, 1, 2047, 4),
            (7, 4, 4095, 1),
            (9, 1, 16385, 2),
            (5, 4, 16385, 1),
        ]
    )
    for seed in args.seeds:
        for shape in shapes:
            for mode in ["bf16", "scalar", "per_token_head"]:
                capacity = max(triton.cdiv(shape[2], 128), args.pad_capacity)
                inputs = case(*shape, mode, capacity=capacity, seed=seed)
                base_output = torch.empty_like(inputs["q"])
                candidate_output = torch.empty_like(base_output)

                run_base = partial(
                    baseline.minimax_m3_sparse_attn_decode, **inputs, output=base_output
                )
                run_candidate = partial(
                    candidate.minimax_m3_sparse_attn_decode,
                    **inputs,
                    output=candidate_output,
                )
                run_base()
                run_candidate()
                reference = dense(inputs)
                torch.testing.assert_close(base_output, reference, atol=0.02, rtol=0.02)
                torch.testing.assert_close(
                    candidate_output, reference, atol=0.02, rtol=0.02
                )
                rel_rms = float(
                    (candidate_output.float() - reference.float()).norm()
                    / reference.float().norm().clamp_min(1e-12)
                )
                samples = [[], []]
                for repeat in range(args.rounds):
                    for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                        samples[index].append(
                            triton.testing.do_bench_cudagraph(
                                [run_base, run_candidate][index], rep=args.rep_ms
                            )
                            * 1000
                        )
                base_us, candidate_us = [
                    statistics.median(values) for values in samples
                ]
                print(
                    json.dumps(
                        dict(
                            shape=shape,
                            seed=seed,
                            mode=mode,
                            capacity=capacity,
                            rel_rms=rel_rms,
                            baseline_us=base_us,
                            candidate_us=candidate_us,
                            speedup=base_us / candidate_us,
                            samples_us=samples,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
