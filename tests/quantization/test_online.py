# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests online quantization."""

from typing import cast

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
    _quantize_moe_weight_per_tensor_fp8,
)
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
    _quantize_moe_weight_to_nvfp4,
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


@pytest.mark.parametrize(
    "quantization",
    [
        pytest.param(
            "fp8_per_tensor",
            marks=pytest.mark.skipif(
                not is_quant_method_supported("fp8"),
                reason="FP8 is not supported on this GPU type.",
            ),
        ),
        pytest.param(
            "nvfp4",
            marks=pytest.mark.skipif(
                not (
                    current_platform.is_cuda()
                    and current_platform.is_device_capability_family(100)
                ),
                reason="NVFP4 weight quantization needs a Blackwell (SM100) GPU.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("shard_dim", [1, 2])
def test_online_moe_tp_weight_quant_matches_ep(
    monkeypatch, quantization: str, shard_dim: int
) -> None:
    """TP shards use the same global scale and quantized values as a full expert."""
    torch.manual_seed(0)
    weight = torch.randn(2, 32, 32, device="cuda", dtype=torch.bfloat16)
    weight[:, -1, -1] = torch.tensor([32.0, 64.0], device="cuda")

    if quantization == "nvfp4":
        ep_weight, ep_block_scale, ep_global_scale = _quantize_moe_weight_to_nvfp4(
            weight
        )
    else:
        ep_weight, ep_global_scale = _quantize_moe_weight_per_tensor_fp8(weight)
    full_amax = weight.abs().amax(dim=(1, 2)).to(torch.float32)
    tp_group = cast(torch.distributed.ProcessGroup, object())

    def fake_all_reduce(tensor, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is tp_group
        tensor.copy_(full_amax)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    shard_size = weight.shape[shard_dim] // 2
    shard = weight.narrow(shard_dim, 0, shard_size)
    if quantization == "nvfp4":
        tp_weight, tp_block_scale, tp_global_scale = _quantize_moe_weight_to_nvfp4(
            shard, tp_group
        )
    else:
        tp_weight, tp_global_scale = _quantize_moe_weight_per_tensor_fp8(
            shard, tp_group
        )

    if shard_dim == 1:
        expected_weight = ep_weight[:, :shard_size]
        if quantization == "nvfp4":
            expected_block_scale = ep_block_scale[:, :shard_size]
    else:
        packing = 2 if quantization == "nvfp4" else 1
        expected_weight = ep_weight[:, :, : shard_size // packing]
        if quantization == "nvfp4":
            expected_block_scale = ep_block_scale[:, :, : shard_size // 16]

    assert torch.equal(tp_weight, expected_weight)
    if quantization == "nvfp4":
        assert torch.equal(tp_block_scale, expected_block_scale)
    assert torch.equal(tp_global_scale, ep_global_scale)


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
