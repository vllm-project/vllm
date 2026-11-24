# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform and dequantization helpers shared by the Inkling2 kernel tests
(CUDA and XPU)."""

from typing import Any

import torch

from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def device_graph():
    """A device graph and the context manager that captures it."""
    if current_platform.is_xpu():
        graph = torch.xpu.XPUGraph()
        return graph, torch.xpu.graph(graph)
    graph = torch.cuda.CUDAGraph()
    return graph, torch.cuda.graph(graph)


# e2m1 values of the 16 FP4 codes.
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 `[..., K]` of packed e2m1 `values` `[..., K / 2]` (low nibble
    first) times their e4m3 block `scales` `[..., K / 16]`."""
    lut = torch.tensor(E2M1, device=values.device)
    v = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    return v.flatten(-2) * scales.float().repeat_interleave(16, -1)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 `[..., K]` of e4m3 `values` times their e8m0 block `scales`
    `[..., K / 32]` (uint8)."""
    exp2 = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return values.float() * exp2


# A tiny Inkling2 text model (16 routed experts, no quantization config,
# so NVFP4 experts) for engine and model-init tests.
TINY_CONFIG: dict[str, Any] = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
    "mtp_config": {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "max_position_embeddings": 4096,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "block_size": 8,
        "markov_rank": 32,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 2],
    },
}
