# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark mixed decode/prefill GDN overlap through the real forward core."""

import argparse
import dataclasses
import statistics
import types
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

H = 16
HV = 64
K = 128
V = 128
CONV_KERNEL = 4
KEY_DIM = H * K
VALUE_DIM = HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"


def measure_us(
    modes: dict[str, Callable[[], Any]], warmup: int, repeats: int
) -> dict[str, float]:
    names = list(modes)
    for _ in range(warmup):
        for fn in modes.values():
            fn()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(repeats):
        rotated = names[repeat % len(names) :] + names[: repeat % len(names)]
        for name in rotated:
            start.record()
            modes[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    return {name: statistics.median(values) for name, values in samples.items()}


def build_layer(
    vllm_config,
    conv_state,
    ssm_state,
    a_log,
    dt_bias,
    conv_weight,
    conv_bias,
    enable_overlap: bool,
):
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = H
    layer.num_v_heads = HV
    layer.head_k_dim = K
    layer.head_v_dim = V
    layer.key_dim = KEY_DIM
    layer.value_dim = VALUE_DIM
    layer.activation = "silu"
    layer.A_log = a_log
    layer.dt_bias = dt_bias
    layer.conv1d = types.SimpleNamespace(weight=conv_weight, bias=conv_bias)
    layer.kv_cache = (conv_state, ssm_state)
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    layer.gdn_prefill_backend = layer.chunk_gated_delta_rule.gdn_prefill_backend
    layer.gdn_decode_prefill_overlap = enable_overlap
    layer.gdn_overlap_stream = torch.cuda.Stream() if enable_overlap else None
    layer.gdn_overlap_start_event = torch.cuda.Event() if enable_overlap else None
    layer.gdn_overlap_done_event = torch.cuda.Event() if enable_overlap else None
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def main() -> None:
    global H, HV, KEY_DIM, VALUE_DIM, CONV_DIM

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-decodes", type=int, default=63)
    parser.add_argument("--prefill-tokens", type=int, default=8192)
    parser.add_argument("--state-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-k-heads", type=int, default=H)
    parser.add_argument("--num-v-heads", type=int, default=HV)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trace-path")
    args = parser.parse_args()

    H = args.num_k_heads
    HV = args.num_v_heads
    KEY_DIM = H * K
    VALUE_DIM = HV * V
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM

    torch.manual_seed(0)
    device = torch.device("cuda")
    state_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.state_dtype]

    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    vllm_config.additional_config = {"gdn_prefill_backend": "flashinfer"}

    seq_lens = [64] * args.num_decodes + [args.prefill_tokens + 37]
    query_lens = [1] * args.num_decodes + [args.prefill_tokens]
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    with set_current_vllm_config(vllm_config):
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
    compact_state_indices = torch.arange(
        1, len(query_lens) + 1, dtype=torch.int32, device=device
    )
    metadata = dataclasses.replace(
        metadata,
        non_spec_state_indices_tensor=compact_state_indices,
        prefill_state_indices=compact_state_indices[args.num_decodes :],
    )

    num_tokens = sum(query_lens)
    pool_size = len(query_lens) + 1
    conv_shape, state_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=0
    )
    conv_state = (
        torch.randn(pool_size, *conv_shape, dtype=torch.bfloat16, device=device) * 0.05
    )
    ssm_state = (
        torch.randn(pool_size, *state_shape, dtype=state_dtype, device=device) * 0.05
    )
    a_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    conv_weight = (
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device) * 0.1
    )
    conv_bias = torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1
    mixed_qkv = (
        torch.randn(num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1
    )
    a = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1

    serial_layer = build_layer(
        vllm_config,
        conv_state.clone(),
        ssm_state.clone(),
        a_log,
        dt_bias,
        conv_weight,
        conv_bias,
        False,
    )
    overlap_layer = build_layer(
        vllm_config,
        conv_state.clone(),
        ssm_state.clone(),
        a_log,
        dt_bias,
        conv_weight,
        conv_bias,
        True,
    )
    serial_output = torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    overlap_output = torch.zeros_like(serial_output)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})

    def run(layer, output):
        layer._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=output,
        )

    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        run(serial_layer, serial_output)
        run(overlap_layer, overlap_output)
        torch.accelerator.synchronize()
        torch.testing.assert_close(serial_output, overlap_output, atol=0, rtol=0)
        torch.testing.assert_close(
            serial_layer.kv_cache[0], overlap_layer.kv_cache[0], atol=0, rtol=0
        )
        torch.testing.assert_close(
            serial_layer.kv_cache[1], overlap_layer.kv_cache[1], atol=0, rtol=0
        )
        timings = measure_us(
            {
                "serial": lambda: run(serial_layer, serial_output),
                "overlap": lambda: run(overlap_layer, overlap_output),
            },
            args.warmup,
            args.repeats,
        )
        if args.trace_path:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                run(overlap_layer, overlap_output)
            torch.accelerator.synchronize()
            profiler.export_chrome_trace(args.trace_path)

    serial_us = timings["serial"]
    overlap_us = timings["overlap"]
    hidden_us = serial_us - overlap_us
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"shape: decodes={args.num_decodes}, prefill_tokens={args.prefill_tokens}, "
        f"state_dtype={args.state_dtype}, H={H}, HV={HV}"
    )
    print(f"serial:  {serial_us:10.2f} us")
    print(f"overlap: {overlap_us:10.2f} us")
    print(f"hidden:  {hidden_us:10.2f} us")
    print(f"speedup: {hidden_us / serial_us:10.2%}")


if __name__ == "__main__":
    main()
