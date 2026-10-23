# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures of the Muse Spark specialized decode tests: a supported
``VllmConfig`` stand-in and the MI355X platform for CPU runs."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationMode, CUDAGraphMode

LAYERS = 62


def muse_text_config(**over) -> SimpleNamespace:
    """Muse Spark Open's text config (checkpoint c7bc74fd): 62 layers, every MLP
    sparse, full attention (NoPE) on every fourth layer from layer 1."""
    kinds = [
        "full_attention" if i % 4 == 1 else "sliding_attention" for i in range(LAYERS)
    ]
    text = SimpleNamespace(
        hidden_size=8192,
        num_attention_heads=128,
        num_key_value_heads=16,
        head_dim=64,
        num_local_experts=256,
        num_experts_per_tok=8,
        moe_hidden_size=4096,
        moe_intermediate_size=4096,
        num_shared_experts=0,
        sliding_window=2048,
        num_hidden_layers=LAYERS,
        layer_types=kinds,
        mlp_layer_types=["sparse"] * LAYERS,
        layer_rope_theta=[0 if k == "full_attention" else 500000.0 for k in kinds],
        gated_delta_net=None,
    )
    for k, v in over.items():
        setattr(text, k, v)
    return text


def dspark(**over) -> SimpleNamespace:
    """A supported speculative config: DSpark, 7 static speculative tokens."""
    spec = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=7,
        enable_adaptive_verification=False,
        dynamic=False,
    )
    spec.uses_dynamic_speculative_decoding = lambda: spec.dynamic
    for k, v in over.items():
        setattr(spec, k, v)
    return spec


def quark_config():
    """A ``QuarkConfig`` instance (``isinstance`` is all stage 1 asks)."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    return QuarkConfig.__new__(QuarkConfig)


def vllm_config(**over) -> SimpleNamespace:
    """A supported config: TP8, MRV2, plain decode, FULL_AND_PIECEWISE graphs,
    ROCM_AITER_FA, bf16 KV in blocks of 16, the Quark checkpoint, the switch on
    through ``decode_backend``. ``over`` sets dotted paths, e.g.
    ``{"parallel_config.data_parallel_size": 2}``."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    c = SimpleNamespace(
        additional_config={"decode_backend": "latency_specialized"},
        kernel_config=SimpleNamespace(monokernel=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            expert_placement_strategy="linear",
            enable_eplb=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
            use_sequence_parallel_moe=False,
            enable_batch_sharded_sampling=None,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enable_sleep_mode=False,
            enable_prompt_embeds=False,
            hf_text_config=muse_text_config(),
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(max_num_seqs=128),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.ROCM_AITER_FA, backend_per_kind={}
        ),
        quant_config=quark_config(),
        lora_config=None,
        kv_transfer_config=None,
        weight_transfer_config=None,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        speculative_config=None,
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
            max_cudagraph_capture_size=16,
        ),
    )
    for path, v in over.items():
        obj = c
        *head, last = path.split(".")
        for h in head:
            obj = getattr(obj, h)
        setattr(obj, last, v)
    return c


@pytest.fixture
def on_mi355x(monkeypatch):
    """The platform stage 1 requires (gfx950, 256 CUs, FlyDSL), on any host;
    MRV2's default env (pad masks maintained, fp32 reduce)."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.muse_spark.amd.specialized_decode import runtime

    monkeypatch.setattr(runtime, "platform_reasons", lambda: [])
    monkeypatch.setattr(
        rocm_aiter_ops, "is_shuffle_kv_cache_enabled", classmethod(lambda cls: False)
    )
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    monkeypatch.setenv("VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE", "fp32")
