#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare legacy per-range grouped GDN with batched producer/consumer phases.

Run from the repository root on a CUDA host:

    python -m benchmarks.kernels.benchmark_grouped_gdn_prefill
"""

import argparse
import statistics
import time
import types

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    causal_conv1d_fn,
    fused_post_conv_prep,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.attention.backends.gdn_attn import (
    CausalConv1dMetadata,
    GDNAttentionMetadata,
)
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata


def make_metadata(
    num_producers: int,
    num_consumers: int,
    producer_len: int,
    consumer_len: int,
    device: torch.device,
) -> GDNAttentionMetadata:
    producer_ranges = []
    consumer_ranges = []
    cursor = 0
    for _ in range(num_producers):
        producer_ranges.append([cursor, cursor + producer_len])
        cursor += producer_len
    for _ in range(num_consumers):
        consumer_ranges.append([cursor, cursor + consumer_len])
        cursor += consumer_len

    def pack(ranges: list[list[int]]) -> tuple[list[int], list[int]]:
        token_indices = [token for start, end in ranges for token in range(start, end)]
        query_start_loc = [0]
        for start, end in ranges:
            query_start_loc.append(query_start_loc[-1] + end - start)
        return token_indices, query_start_loc

    def conv_metadata(starts: list[int]) -> CausalConv1dMetadata | None:
        if len(starts) == 1:
            return None
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            torch.tensor(starts, dtype=torch.int32),
            device=device,
        )
        return CausalConv1dMetadata(
            nums_dict,
            batch_ptr,
            token_chunk_offset_ptr,
        )

    producer_tokens, producer_starts = pack(producer_ranges)
    consumer_tokens, consumer_starts = pack(consumer_ranges)
    shared_destinations = list(range(1, num_producers + 1))
    private_destinations = list(
        range(num_producers + 1, num_producers + num_consumers + 1)
    )
    consumer_sources = [
        shared_destinations[index % num_producers] for index in range(num_consumers)
    ]
    total_tokens = cursor
    return GDNAttentionMetadata(
        num_prefills=num_producers + num_consumers,
        num_prefill_tokens=total_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=total_tokens,
        prefix_producer_ranges=torch.tensor(
            producer_ranges, dtype=torch.int32, device=device
        ).reshape(-1, 2),
        producer_token_indices=torch.tensor(
            producer_tokens, dtype=torch.long, device=device
        ),
        producer_query_start_loc=torch.tensor(
            producer_starts, dtype=torch.int32, device=device
        ),
        producer_conv_metadata=conv_metadata(producer_starts),
        consumer_ranges=torch.tensor(
            consumer_ranges, dtype=torch.int32, device=device
        ).reshape(-1, 2),
        consumer_token_indices=torch.tensor(
            consumer_tokens, dtype=torch.long, device=device
        ),
        consumer_query_start_loc=torch.tensor(
            consumer_starts, dtype=torch.int32, device=device
        ),
        consumer_conv_metadata=conv_metadata(consumer_starts),
        shared_initial_state_source=torch.zeros(
            num_producers, dtype=torch.int32, device=device
        ),
        shared_state_destinations=torch.tensor(
            shared_destinations, dtype=torch.int32, device=device
        ),
        consumer_shared_state_sources=torch.tensor(
            consumer_sources, dtype=torch.int32, device=device
        ),
        private_final_state_destination=torch.tensor(
            private_destinations, dtype=torch.int32, device=device
        ),
    )


def run_legacy(
    layer,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    output: torch.Tensor,
    metadata: GDNAttentionMetadata,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_weights: torch.Tensor,
) -> None:
    one = torch.ones(1, dtype=torch.bool, device=mixed_qkv.device)

    def copy_state(source: int, destination: int) -> None:
        if source != destination:
            conv_state[destination].copy_(conv_state[source])
            ssm_state[destination].copy_(ssm_state[source])

    def run_range(token_range: torch.Tensor, destination: int) -> None:
        start, end = (int(value) for value in token_range.tolist())
        cu_seqlens = torch.tensor(
            [0, end - start], dtype=torch.int32, device=mixed_qkv.device
        )
        conv_output = causal_conv1d_fn(
            mixed_qkv[start:end].transpose(0, 1),
            conv_weights,
            layer.conv1d.bias,
            conv_states=conv_state,
            has_initial_state=one,
            cache_indices=torch.tensor(
                [destination], dtype=torch.int32, device=mixed_qkv.device
            ),
            query_start_loc=cu_seqlens,
            metadata=None,
        ).transpose(0, 1)
        query, key, value, _, _ = fused_post_conv_prep(
            conv_output=conv_output,
            a=a[start:end],
            b=b[start:end],
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            num_k_heads=layer.num_k_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        range_output, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            a=a[start:end],
            b=b[start:end],
            dt_bias=layer.dt_bias,
            q=query.unsqueeze(0),
            k=key.unsqueeze(0),
            v=value.unsqueeze(0),
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=torch.full(
                (1, max(1, end - start)),
                destination,
                dtype=torch.int32,
                device=mixed_qkv.device,
            ),
            use_qk_l2norm_in_kernel=False,
        )
        output[start:end] = range_output.squeeze(0)

    initial_sources = metadata.shared_initial_state_source
    shared_destinations = metadata.shared_state_destinations
    consumer_sources = metadata.consumer_shared_state_sources
    private_destinations = metadata.private_final_state_destination
    assert initial_sources is not None
    assert shared_destinations is not None
    assert consumer_sources is not None
    assert private_destinations is not None
    assert metadata.prefix_producer_ranges is not None
    assert metadata.consumer_ranges is not None

    for index, token_range in enumerate(metadata.prefix_producer_ranges):
        source = int(initial_sources[min(index, initial_sources.numel() - 1)].item())
        destination = int(shared_destinations[index].item())
        copy_state(source, destination)
        run_range(token_range, destination)
    for index in range(metadata.consumer_ranges.shape[0]):
        copy_state(
            int(consumer_sources[index].item()),
            int(private_destinations[index].item()),
        )
    for index, token_range in enumerate(metadata.consumer_ranges):
        run_range(token_range, int(private_destinations[index].item()))


def measure(operation, reset) -> float:
    reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6


def benchmark_pair(
    legacy_operation,
    legacy_reset,
    batched_operation,
    batched_reset,
    warmup: int,
    repetitions: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        legacy_reset()
        legacy_operation()
        batched_reset()
        batched_operation()
    torch.cuda.synchronize()

    legacy_samples = []
    batched_samples = []
    for index in range(repetitions):
        if index % 2 == 0:
            legacy_samples.append(measure(legacy_operation, legacy_reset))
            batched_samples.append(measure(batched_operation, batched_reset))
        else:
            batched_samples.append(measure(batched_operation, batched_reset))
            legacy_samples.append(measure(legacy_operation, legacy_reset))
    return legacy_samples, batched_samples


def summarize(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        ordered[len(ordered) // 10],
        ordered[-max(1, len(ordered) // 10)],
    )


@torch.inference_mode()
def run_case(args: argparse.Namespace, num_consumers: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    total_tokens = (
        args.num_producers * args.producer_len + num_consumers * args.consumer_len
    )
    conv_dim = (
        2 * args.num_key_heads * args.head_dim + args.num_value_heads * args.value_dim
    )
    pool_size = args.num_producers + num_consumers + 1

    torch.manual_seed(args.seed)
    mixed_qkv = torch.randn(total_tokens, conv_dim, dtype=dtype, device=device)
    a = torch.randn(total_tokens, args.num_value_heads, dtype=dtype, device=device)
    b = torch.randn_like(a)
    conv_weights = (
        torch.randn(conv_dim, args.conv_kernel, dtype=dtype, device=device) * 0.02
    )
    conv_bias = torch.randn(conv_dim, dtype=dtype, device=device) * 0.02
    base_conv = (
        torch.randn(
            pool_size,
            conv_dim,
            args.conv_kernel - 1,
            dtype=dtype,
            device=device,
        )
        * 0.02
    )
    base_ssm = (
        torch.randn(
            pool_size,
            args.num_value_heads,
            args.value_dim,
            args.head_dim,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    metadata = make_metadata(
        args.num_producers,
        num_consumers,
        args.producer_len,
        args.consumer_len,
        device,
    )
    layer = types.SimpleNamespace(
        conv1d=types.SimpleNamespace(bias=conv_bias),
        A_log=torch.zeros(args.num_value_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(args.num_value_heads, dtype=torch.float32, device=device),
        num_k_heads=args.num_key_heads,
        tp_size=1,
        head_k_dim=args.head_dim,
        head_v_dim=args.value_dim,
    )

    legacy_conv = base_conv.clone()
    legacy_ssm = base_ssm.clone()
    legacy_output = torch.zeros(
        total_tokens,
        args.num_value_heads,
        args.value_dim,
        dtype=dtype,
        device=device,
    )
    batched_conv = base_conv.clone()
    batched_ssm = base_ssm.clone()
    batched_output = torch.zeros_like(legacy_output)

    run_legacy(
        layer,
        mixed_qkv,
        b,
        a,
        legacy_output,
        metadata,
        legacy_conv,
        legacy_ssm,
        conv_weights,
    )
    QwenGatedDeltaNetAttention._forward_core_grouped_prefill(
        layer,
        mixed_qkv,
        b,
        a,
        batched_output,
        metadata,
        batched_conv,
        batched_ssm,
        conv_weights,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(legacy_output, batched_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(legacy_ssm, batched_ssm, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(legacy_conv, batched_conv, atol=0, rtol=0)

    def reset_legacy() -> None:
        legacy_conv.copy_(base_conv)
        legacy_ssm.copy_(base_ssm)
        legacy_output.zero_()

    def reset_batched() -> None:
        batched_conv.copy_(base_conv)
        batched_ssm.copy_(base_ssm)
        batched_output.zero_()

    def legacy_operation() -> None:
        run_legacy(
            layer,
            mixed_qkv,
            b,
            a,
            legacy_output,
            metadata,
            legacy_conv,
            legacy_ssm,
            conv_weights,
        )

    def batched_operation() -> None:
        QwenGatedDeltaNetAttention._forward_core_grouped_prefill(
            layer,
            mixed_qkv,
            b,
            a,
            batched_output,
            metadata,
            batched_conv,
            batched_ssm,
            conv_weights,
        )

    legacy_samples, batched_samples = benchmark_pair(
        legacy_operation,
        reset_legacy,
        batched_operation,
        reset_batched,
        args.warmup,
        args.repetitions,
    )
    legacy_us, legacy_p10, legacy_p90 = summarize(legacy_samples)
    batched_us, batched_p10, batched_p90 = summarize(batched_samples)
    print(
        f"{args.num_producers},{num_consumers},{args.producer_len},"
        f"{args.consumer_len},{total_tokens},{legacy_us:.2f},{batched_us:.2f},"
        f"{legacy_us / batched_us:.3f},{legacy_p10:.2f},{legacy_p90:.2f},"
        f"{batched_p10:.2f},{batched_p90:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-producers", type=int, default=1)
    parser.add_argument(
        "--num-consumers", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--producer-len", type=int, default=512)
    parser.add_argument("--consumer-len", type=int, default=64)
    parser.add_argument("--num-key-heads", type=int, default=4)
    parser.add_argument("--num-value-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.num_producers <= 0 or any(value <= 0 for value in args.num_consumers):
        raise ValueError("producer and consumer counts must be positive")
    if args.producer_len <= 0 or args.consumer_len <= 0:
        raise ValueError("sequence lengths must be positive")

    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    print(
        "producers,consumers,producer_len,consumer_len,total_tokens,"
        "legacy_us,batched_us,speedup,legacy_p10_us,legacy_p90_us,"
        "batched_p10_us,batched_p90_us"
    )
    for num_consumers in args.num_consumers:
        run_case(args, num_consumers)


if __name__ == "__main__":
    main()
