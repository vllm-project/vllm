# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3.1 MoE block on TRTLLM-Gen NVFP4 experts: when captured into a
CUDA graph, the routed input is quantized on the aux stream while the router
GEMM runs; run eagerly, the experts quantize it on the current stream. Either
way the block's output is bitwise the one of the router and the MoE call run
one after the other."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="TRTLLM-Gen NVFP4 MoE requires SM100",
)


def _trtllm_moe_block(hidden_size=1024, intermediate_size=512):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.models.minimax_m3.nvidia.model import MiniMaxM3MoE
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.0,
        use_routing_bias=False,
        hidden_act="swigluoai",
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        moe_quant_algo="NVFP4",
    )
    vllm_config = VllmConfig(
        kernel_config=KernelConfig(moe_backend="flashinfer_trtllm")
    )
    vllm_config.scheduler_config.max_num_batched_tokens = 256
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        moe = MiniMaxM3MoE(
            config, layer_id=0, quant_config=quant_config, prefix="layers.0.moe"
        )
        # Random weights and scales within the formats' finite ranges.
        for name, param in moe.named_parameters():
            if param.dtype == torch.uint8:
                if name.endswith("weight"):  # packed FP4 / MXFP8 values
                    param.data.random_(0, 256)
                else:  # UE8M0 block scales around 1
                    param.data.random_(124, 130)
            elif param.dtype == torch.float8_e4m3fn:
                param.data.copy_(torch.rand_like(param, dtype=torch.float32) + 0.5)
            elif "scale" in name:
                param.data.fill_(1.0)
            else:
                param.data.normal_(0, param.shape[-1] ** -0.5)
        for module in moe.modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)
    return moe, vllm_config


def _count_provides(handoff) -> list[int]:
    """Count `handoff.provide` calls, i.e. MoE calls given an input
    quantization made on the aux stream."""
    calls = [0]
    provide = handoff.provide

    @contextlib.contextmanager
    def counting_provide(*args, **kwargs):
        calls[0] += 1
        with provide(*args, **kwargs):
            yield

    handoff.provide = counting_provide
    return calls


@pytest.mark.parametrize("num_tokens", [1, 16, 200])
@torch.inference_mode()
def test_routed_input_quantized_during_the_router_gemm(dist_init, num_tokens):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsBase,
    )

    torch.manual_seed(0)
    moe, vllm_config = _trtllm_moe_block()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        fused_experts = moe.experts.routed_experts.quant_method.moe_kernel.fused_experts
        assert isinstance(fused_experts, TrtLlmNvFp4ExpertsBase)
        assert fused_experts.expects_unquantized_inputs
        x = torch.randn(num_tokens, 1024, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(num_tokens, 1024, device="cuda", dtype=torch.bfloat16)
        assert moe._routed_input_handoff(x) is fused_experts.input_handoff
        # Not for an input of another dtype.
        assert moe._routed_input_handoff(x.half()) is None
        provides = _count_provides(fused_experts.input_handoff)

        # Reference: route, then let the experts quantize their input.
        def reference(h):
            return moe.experts(hidden_states=h, router_logits=moe.gate(h)[0])

        expected_x, expected_y = reference(x), reference(y)
        assert torch.isfinite(expected_x).all()
        assert not torch.equal(expected_x, expected_y)

        # Eager: the experts quantize on the current stream, no aux stream.
        for _ in range(3):
            assert torch.equal(moe(x), expected_x)
        assert provides[0] == 0

        # Captured: the quantization moves to the aux stream, recorded in the
        # graph, and each replay quantizes the input it is given.
        static_x = x.clone()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            moe(static_x)  # warm up on the capture stream
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            static_out = moe(static_x)
        assert provides[0] == 1
        for h, expected in ((x, expected_x), (y, expected_y), (x, expected_x)):
            static_x.copy_(h)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(static_out, expected)
        assert provides[0] == 1
        assert torch.equal(moe(y), expected_y)
        assert provides[0] == 1
