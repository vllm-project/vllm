# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare per-layer and batched native context KV writes under CUDA graphs."""

import json
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.testing.utils import get_l2_cache_size

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm.v1.attention.ops.batched_kv_cache import try_batched_kv_cache_update


def bench_with_cuda_events(fn, iterations: int = 500) -> list[float]:
    """Time graph replay, flushing L2 before the start event of each sample."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    flush = torch.empty(2 * get_l2_cache_size("cuda"), dtype=torch.uint8, device="cuda")
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for start, end in events:
        flush.zero_()
        start.record()
        graph.replay()
        end.record()
    events[-1][1].synchronize()
    return [start.elapsed_time(end) for start, end in events]


@torch.inference_mode()
def main():
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 512, 8192])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--timer", choices=["cupti", "cuda-event"], default="cupti")
    args = parser.parse_args()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        local_rank=0,
    )
    config = VllmConfig()
    with set_current_vllm_config(config):
        initialize_model_parallel(1, 1)
        attns = [
            Attention(
                2 * args.num_heads,
                args.head_size,
                args.head_size**-0.5,
                num_kv_heads=args.num_heads,
                cache_config=CacheConfig(cache_dtype="auto"),
                attn_backend=FlashAttentionBackend,
                prefix=f"batched_context.layers.{i}",
            )
            for i in range(args.num_layers)
        ]
        blocks = (max(args.tokens) + 15) // 16
        for attn in attns:
            attn.bind_kv_cache(
                torch.zeros(blocks, args.num_heads, 16, 2 * args.head_size)
            )
        results = []
        for tokens in args.tokens:
            key = torch.randn(args.num_layers, tokens, args.num_heads, args.head_size)
            value = torch.randn_like(key)
            slots = torch.arange(tokens, dtype=torch.int64)

            caches = [attn.kv_cache for attn in attns]

            def baseline(key=key, value=value, slots=slots, caches=caches):
                for i, (attn, cache) in enumerate(zip(attns, caches)):
                    assert isinstance(attn.impl, FlashAttentionImpl)
                    attn.impl.do_kv_cache_update(attn, key[i], value[i], cache, slots)

            def batched(key=key, value=value, slots=slots, caches=caches):
                for attn, cache in zip(attns, caches):
                    attn.bind_kv_cache(cache)
                assert try_batched_kv_cache_update(attns, key, value, slots)

            baseline()
            expected = [attn.kv_cache.clone() for attn in attns]
            for attn in attns:
                attn.kv_cache.zero_()
            batched()
            for attn, ref in zip(attns, expected):
                torch.testing.assert_close(attn.kv_cache, ref, rtol=0, atol=0)
            timings = {}
            for name, fn in (("baseline", baseline), ("batched", batched)):
                for _ in range(25):
                    fn()
                torch.accelerator.synchronize()
                if args.timer == "cupti":
                    samples = bench_gpu_time_with_cupti(
                        fn,
                        input_args=(key, value, slots, caches),
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        aggregate_op=sum,
                    )
                else:
                    samples = bench_with_cuda_events(fn)
                timings[f"{name}_us"] = statistics.median(samples) * 1000
                deciles = statistics.quantiles(samples, n=10)
                timings[f"{name}_p10_us"] = deciles[0] * 1000
                timings[f"{name}_p90_us"] = deciles[-1] * 1000
            row = {"tokens": tokens, **timings}
            row["speedup"] = timings["baseline_us"] / timings["batched_us"]
            bytes_moved = 4 * key.numel() * key.element_size()
            row["batched_gbps"] = bytes_moved / timings["batched_us"] / 1000
            results.append(row)
            print(json.dumps(row), flush=True)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(
                    {
                        "gpu": current_platform.get_device_name(),
                        "config": vars(args),
                        "results": results,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
