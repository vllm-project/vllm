# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests online quantization."""

import pytest
import torch

from tests.quantization.utils import (
    _test_online_quant_peak_mem_impl,
    is_quant_method_supported,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "quant_scheme,online_quant_args,expected_linear_cls,expected_moe_cls",
    [
        # simple case - quantization='fp8_per_tensor'
        (
            "fp8_per_tensor",
            None,
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        # simple case - quantization='fp8_per_block'
        (
            "fp8_per_block",
            None,
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerBlockOnlineMoEMethod,
        ),
        # quantization='online' with per-layer-kind overrides
        (
            "online",
            {
                "linear": "fp8_per_block",
                "moe": "fp8_per_tensor",
            },
            Fp8PerBlockOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
        # ignore with direct layer name
        (
            "fp8_per_tensor",
            # qkv_proj is fused from q_proj/k_proj/v_proj, so currently the
            # ignore regex must match the unfused shard names
            # TODO(future PR): also make 're:.*qkv_proj.*' work
            {"ignore": ["model.layers.1.self_attn.o_proj", "re:.*[qkv]_proj"]},
            Fp8PerTensorOnlineLinearMethod,
            Fp8PerTensorOnlineMoEMethod,
        ),
    ],
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_quantization(
    vllm_runner,
    quant_scheme: str,
    online_quant_args: dict | None,
    expected_linear_cls,
    expected_moe_cls,
    use_rocm_aiter: bool,
    monkeypatch,
) -> None:
    """
    Tests that online quantization frontend configuration works -
    selecting quant schemes, overriding quant schemes by type, ignoring
    layers.

    Does not test performance, peak memory usage, etc.
    """

    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    if current_platform.is_xpu() and quant_scheme == "fp8_per_block":
        pytest.skip("Skip test for online fp8_per_block on XPU platform.")

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # a tiny model with both dense and MoE layers
    model_name = "ibm-granite/granite-3.0-1b-a400m-base"

    runner_kwargs = dict(
        quantization=quant_scheme,
        enforce_eager=True,
    )
    if online_quant_args is not None:
        runner_kwargs["quantization_config"] = online_quant_args

    with vllm_runner(
        model_name,
        **runner_kwargs,
    ) as llm:

        def check_model(model):
            # checks further down in the test case are hardcoded for this
            # model
            assert model_name == "ibm-granite/granite-3.0-1b-a400m-base"

            o_proj = model.model.layers[0].self_attn.o_proj
            moe = model.model.layers[0].block_sparse_moe.experts

            # o_proj and moe in layer 0 are always quantized (never ignored)
            # because of how we craft the test case inputs
            assert isinstance(o_proj.quant_method, expected_linear_cls)
            if moe is not None:
                assert isinstance(moe._quant_method, expected_moe_cls)

            if current_platform.is_cuda() or current_platform.is_xpu():
                assert o_proj.weight.dtype == torch.float8_e4m3fn
            elif current_platform.is_rocm():
                assert o_proj.weight.dtype == current_platform.fp8_dtype()
            else:
                pytest.skip("Only runs on CUDA and ROCm.")

            # Verify ignored layers are unquantized.
            if isinstance(online_quant_args, dict) and "ignore" in online_quant_args:
                # only .*1.self_attn_o_proj is skipped
                for layer_idx in range(len(model.model.layers)):
                    o_proj = model.model.layers[layer_idx].self_attn.o_proj
                    if layer_idx == 1:
                        assert isinstance(o_proj.quant_method, UnquantizedLinearMethod)
                    else:
                        assert isinstance(o_proj.quant_method, expected_linear_cls)

                # every .*self_attn.qkv_proj is skipped
                for layer_idx in range(len(model.model.layers)):
                    qkv_proj = model.model.layers[layer_idx].self_attn.qkv_proj
                    assert isinstance(qkv_proj.quant_method, UnquantizedLinearMethod)

        llm.apply_model(check_model)

        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer_trtllm_fused_moe()
    ),
    reason="nvfp4_per_token needs a Blackwell (SM100) GPU + FlashInfer TRTLLM MoE.",
)
def test_online_nvfp4_per_token_moe(vllm_runner, monkeypatch) -> None:
    """Online NVFP4 quantizes the MoE and leaves dense layers unquantized."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        "ibm-granite/granite-3.0-1b-a400m-base",
        quantization="nvfp4_per_token",
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            layer = model.model.layers[0]
            assert isinstance(
                layer.block_sparse_moe.experts._quant_method, Nvfp4OnlineMoEMethod
            )
            assert isinstance(
                layer.self_attn.o_proj.quant_method, UnquantizedLinearMethod
            )

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quant_peak_mem(
    vllm_runner,
    caplog_mp_spawn,
    monkeypatch,
) -> None:
    _test_online_quant_peak_mem_impl(
        "fp8_per_tensor", vllm_runner, caplog_mp_spawn, monkeypatch
    )


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_online_quant_load_format_dummy(
    vllm_runner,
    monkeypatch,
    caplog,
) -> None:
    with vllm_runner(
        "ibm-granite/granite-3.0-1b-a400m-base",
        quantization="fp8_per_tensor",
        enforce_eager=True,
        load_format="dummy",
    ) as llm:
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("gated", [True, False])
def test_online_moe_fp8_per_block_end_to_end(
    gated: bool, dist_init, workspace_init, monkeypatch
) -> None:
    """Online fp8 MoE: weight creation, weight loading, post-loading
    processing and a kernel run, for gated and non-gated layouts.

    Non-gated MoE (``is_act_and_mul=False``, e.g. NemotronH relu2) keeps only
    the up projection in w13, so the online allocation is N rows rather than
    2N and the per-block pad rows must be zeroed per single shard. An
    intermediate size that is not block-aligned exercises that padding.

    Pinned to the Marlin MoE backend: it is the fp8 backend that supports
    non-gated MoE, and it is the one whose tile padding this covers.
    """
    monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")
    from tests.kernels.utils import torch_experts
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    e, top_k, k, n, m = 4, 2, 256, 96, 8
    dtype = torch.bfloat16
    device = torch.device("cuda")
    activation = MoEActivation.SILU if gated else MoEActivation.RELU2_NO_MUL

    quant_config = OnlineQuantizationConfig(QuantizationConfigArgs(moe="fp8_per_block"))
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=e,
            top_k=top_k,
            hidden_size=k,
            intermediate_size=n,
            params_dtype=dtype,
            renormalize=False,
            quant_config=quant_config,
            activation=activation.value,
            tp_size=1,
            dp_size=1,
            prefix="online_moe",
        )
        experts = layer.routed_experts

        # Weight creation: w13 holds one shard per projection, so a non-gated
        # layer allocates half the rows of a gated one. The intermediate size is
        # rounded up to the quant block, so compare against the allocated size.
        num_shards = 2 if gated else 1
        n_alloc = experts.w2_weight.shape[2]
        assert n_alloc >= n
        assert experts.w13_weight.shape == (e, num_shards * n_alloc, k)

        # Weight loading: fill the real rows of each shard and leave the roundup
        # pad rows non-zero, so the kernel result is only correct if post-loading
        # processing zeroes them (otherwise they contaminate the block scales).
        torch.manual_seed(0)
        w13 = torch.randn(e, num_shards * n, k, dtype=dtype, device=device) / k**0.5
        w2 = torch.randn(e, k, n, dtype=dtype, device=device) / n**0.5
        w13_buf = torch.full(
            (e, num_shards * n_alloc, k), 3.0, dtype=dtype, device=device
        )
        for s in range(num_shards):
            w13_buf[:, s * n_alloc : s * n_alloc + n] = w13[:, s * n : (s + 1) * n]
        w2_buf = torch.full((e, k, n_alloc), 3.0, dtype=dtype, device=device)
        w2_buf[:, :, :n] = w2
        experts.register_parameter(
            "w13_weight", torch.nn.Parameter(w13_buf, requires_grad=False)
        )
        experts.register_parameter(
            "w2_weight", torch.nn.Parameter(w2_buf, requires_grad=False)
        )

        # Post-loading processing: zero the pad rows and quantize to fp8.
        # Check the zeroing directly rather than through the kernel: w2's pad
        # columns are zeroed unconditionally, so a non-zeroed w13 pad row is
        # multiplied by zero and would not show up in the output, but it does
        # contaminate the shared per-block weight scale.
        quant_method = layer._quant_method
        quant_method._zero_padding(experts)
        w13_shard = experts.w13_weight.shape[1] // num_shards
        for s in range(num_shards):
            pad_rows = experts.w13_weight[:, s * w13_shard + n : (s + 1) * w13_shard]
            assert pad_rows.abs().max().item() == 0.0, (
                f"w13 shard {s} pad rows were not zeroed"
            )
        quant_method.process_weights_after_loading(experts)

        # Kernel run against a bf16 reference over the unpadded weights.
        x = torch.randn(m, k, dtype=dtype, device=device) / 10
        router_logits = torch.randn(m, e, dtype=dtype, device=device)
        with set_forward_context(None, vllm_config, num_tokens=m):
            out = layer(x, router_logits)

        topk_weights, topk_ids, _ = fused_topk(
            x, router_logits.float(), top_k, renormalize=False
        )
        ref = torch_experts(x, w13, w2, topk_weights, topk_ids, activation=activation)
        # Keep the comparison non-vacuous: the tolerance must stay well below
        # the signal, so a zeroed or dropped w13 shard cannot pass. Measured
        # fp8 error here is ~1.4e-3 against a ~1.2e-2 signal.
        assert ref.float().abs().max() > 1e-2
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-2)
