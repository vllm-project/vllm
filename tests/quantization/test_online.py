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
    Nvfp4W4A4OnlineLinearMethod,
    Nvfp4W4A16OnlineLinearMethod,
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

            if current_platform.is_cuda():
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
    not (current_platform.is_cuda() and current_platform.has_device_capability(75)),
    reason="Online NVFP4 W4A16 needs the FP4 Marlin kernel (CUDA SM>=75).",
)
def test_online_nvfp4_w4a16_linear(vllm_runner, monkeypatch) -> None:
    """Online NVFP4 W4A16 quantizes dense linear layers to FP4 at load time.

    Weight-only path: activations stay bf16/fp16 and Marlin dequantizes on the
    fly, so this runs on any SM>=75 GPU (not just Blackwell).
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(
        "ibm-granite/granite-3.0-1b-a400m-base",
        quantization="nvfp4",
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            o_proj = model.model.layers[0].self_attn.o_proj
            assert isinstance(o_proj.quant_method, Nvfp4W4A16OnlineLinearMethod)
            # Weights are packed FP4 (two values per uint8 byte).
            assert o_proj.weight.dtype == torch.uint8
            assert hasattr(o_proj, "weight_global_scale")
            # MoE stays unquantized under the linear-only "nvfp4" shorthand.
            moe = model.model.layers[0].block_sparse_moe.experts
            if moe is not None:
                assert not isinstance(moe._quant_method, Nvfp4OnlineMoEMethod)

        llm.apply_model(check_model)
        outputs = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        print(outputs[0][1])


def _nvfp4_w4a4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            cutlass_fp4_supported,
        )
    except Exception:
        return False
    return cutlass_fp4_supported()


@pytest.mark.skipif(
    not _nvfp4_w4a4_supported(),
    reason="Online NVFP4 W4A4 needs a native FP4 W4A4 GEMM kernel (Blackwell).",
)
def test_online_nvfp4_w4a4_cudagraph_dynamic_scale() -> None:
    """W4A4's dynamic activation scale must replay against the live input.

    The old ``.data.copy_`` refresh captured the warmup activation's scale once
    and replayed it stale under CUDA graphs, yielding garbage. The explicit
    dataflow fix computes the scale as fresh graph-visible ops threaded into the
    kernel, so a captured graph replayed on a materially different input must
    match eager on that same input (a stale scale would not).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    out_features, in_features = 256, 512

    layer = torch.nn.Module()
    layer.output_size_per_partition = out_features
    layer.input_size_per_partition = in_features
    layer.params_dtype = torch.bfloat16
    weight = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)

    method = Nvfp4W4A4OnlineLinearMethod()
    method.process_weights_after_loading(layer)

    # Two inputs with materially different amax -> different dynamic scale.
    x_small = torch.randn(8, in_features, device=device, dtype=torch.bfloat16) * 0.1
    x_large = torch.randn(8, in_features, device=device, dtype=torch.bfloat16) * 8.0

    eager_small = method.apply(layer, x_small).clone()
    eager_large = method.apply(layer, x_large).clone()
    # Distinct scales must produce meaningfully distinct outputs.
    assert not torch.allclose(eager_small, eager_large, atol=1e-2)

    # Capture a graph on x_small, then replay on x_large's data.
    static_in = x_small.clone()
    method.apply(layer, static_in)  # warmup
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = method.apply(layer, static_in)

    static_in.copy_(x_large)
    graph.replay()
    torch.accelerator.synchronize()

    # If the scale replayed stale (bug), the graph output would match the
    # captured x_small scale instead of x_large's eager result.
    assert torch.allclose(static_out, eager_large, atol=5e-2, rtol=5e-2)
    assert not torch.allclose(static_out, eager_small, atol=1e-2)


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
