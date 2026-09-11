# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TopK kernel microbenchmark: DeepSelect vs vLLM vs FlashInfer vs torch.topk.

Shapes are taken from the DeepSeek-V4.1-Flash Lightning Indexer
(config.json): index_topk=512, logits shaped (num_decode_tokens,
kv_context_length up to 1M). bf16 is DeepSelect's native dtype; fp32
matches vLLM's indexer logits.

Metric: effective memory bandwidth assuming the input is read exactly once
from global memory, plus the topk (value, int32 index) outputs written once.

Run: .venv/bin/python benchmarks/kernels/bench_topk_deepselect.py
"""

import statistics

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

# DeepSeek-V4.1-Flash config value
INDEX_TOPK = 512

WARMUP = 10


def bench_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.accelerator.synchronize()
    return (
        statistics.median(
            bench_gpu_time_with_cupti(fn, cold_l2_cache=True, use_cuda_graph=False)
        )
        * 1e3
    )


def check_topk(x, k, indices, values=None):
    """Validate topk indices against torch.topk (order-insensitive)."""
    ref_vals, _ = torch.topk(x.float(), k, dim=-1)
    got = x.float().gather(-1, indices.long().clamp(min=0))
    ref_sorted = ref_vals.sort(dim=-1, descending=True).values
    got_sorted = got.sort(dim=-1, descending=True).values
    torch.testing.assert_close(got_sorted, ref_sorted, atol=0, rtol=0)
    if values is not None:
        v_sorted = values.float().sort(dim=-1, descending=True).values
        torch.testing.assert_close(v_sorted, ref_sorted, atol=0, rtol=0)


def make_candidates(x, k):
    """Build (name, callable) candidates for input x. Missing deps are skipped."""
    cands = {}

    cands["torch.topk"] = lambda: torch.topk(x, k, dim=-1, sorted=False)

    try:
        import deep_select

        def run_deep_select():
            return deep_select.topk(
                x,
                k,
                sorted_index=False,
                indices_type=torch.int32,
                return_value=False,
            )

        cands["deep_select"] = run_deep_select
    except ImportError:
        pass

    try:
        from flashinfer.topk import top_k as fi_top_k

        cands["flashinfer.top_k"] = lambda: fi_top_k(x, k)
    except Exception:
        pass

    if x.dtype == torch.float32:
        try:
            import vllm._custom_ops as ops

            num_rows, vocab = x.shape
            seq_lens = torch.full(
                (num_rows,), vocab, dtype=torch.int32, device=x.device
            )
            out_idx = torch.empty(num_rows, k, dtype=torch.int32, device=x.device)

            def run_vllm():
                ops.top_k_per_row_decode(
                    x,
                    1,
                    seq_lens,
                    out_idx,
                    num_rows,
                    x.stride(0),
                    x.stride(1),
                    k,
                )
                return out_idx

            cands["vllm.top_k_per_row_decode"] = run_vllm
        except Exception:
            pass

    return cands


def resolve_indices(out):
    """Extract the index tensor from a candidate's return value."""
    if isinstance(out, tuple):
        # torch returns (values, indices); deep_select (values, indices) or
        # indices only when return_value=False; flashinfer (values, indices)
        if len(out) == 2:
            return out[1], out[0]
        return out[0], None
    return out, None


def bench_scenario(name, dtype, batch_sizes, vocab_sizes, k, results):
    esize = torch.tensor([], dtype=dtype).element_size()
    for vocab in vocab_sizes:
        for bs in batch_sizes:
            torch.manual_seed(0)
            x = torch.randn(bs, vocab, dtype=dtype, device="cuda")
            cands = make_candidates(x, k)
            for cname, fn in cands.items():
                try:
                    out = fn()
                    torch.accelerator.synchronize()
                    idx, vals = resolve_indices(out)
                    check_topk(x, k, idx, vals)
                    us = bench_us(fn)
                except Exception as e:
                    print(f"  [skip] {cname} bs={bs} vocab={vocab}: {e}")
                    continue
                bytes_moved = bs * vocab * esize + bs * k * (esize + 4)
                results.append(
                    {
                        "scenario": name,
                        "dtype": str(dtype).split(".")[-1],
                        "batch": bs,
                        "vocab": vocab,
                        "topk": k,
                        "kernel": cname,
                        "us": us,
                        "GB/s": bytes_moved / (us * 1e3),
                    }
                )
            del x, cands
            torch.accelerator.empty_cache()


def main():
    assert torch.accelerator.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    results = []
    for dtype in (torch.bfloat16, torch.float32):
        bench_scenario(
            "indexer",
            dtype,
            batch_sizes=[1, 8, 64, 256],
            vocab_sizes=[4096, 32768, 131072, 524288, 1048576],
            k=INDEX_TOPK,
            results=results,
        )

    df = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    for (scen, dt), grp in df.groupby(["scenario", "dtype"]):
        print(f"\n=== {scen} ({dt}) ===")
        piv_lat = grp.pivot_table(
            index=["batch", "vocab"], columns="kernel", values="us"
        )
        piv_bw = grp.pivot_table(
            index=["batch", "vocab"], columns="kernel", values="GB/s"
        )
        print("latency (us):")
        print(piv_lat.to_string(float_format=lambda v: f"{v:.1f}"))
        print("effective bandwidth (GB/s):")
        print(piv_bw.to_string(float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main()
