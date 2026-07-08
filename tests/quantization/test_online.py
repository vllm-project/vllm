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
from vllm.platforms import current_platform


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


@pytest.mark.parametrize("is_act_and_mul", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_online_moe_create_weights_w13_dim(is_act_and_mul, has_bias):
    """w13 holds gate+up (2N) for gated MoE and up only (N) for non-gated MoE
    (is_act_and_mul=False, e.g. NemotronH relu2)."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    class _Method(OnlineMoEMethodBase):
        def __init__(self, moe):
            self.moe = moe

        def process_weights_after_loading(self, layer):
            pass

        def get_fused_moe_quant_config(self, layer):
            return None

    e, n, k = 4, 96, 64
    moe = SimpleNamespace(is_act_and_mul=is_act_and_mul, has_bias=has_bias)
    layer = torch.nn.Module()
    _Method(moe).create_weights(
        layer=layer,
        num_experts=e,
        hidden_size=k,
        intermediate_size_per_partition=n,
        params_dtype=torch.bfloat16,
    )
    w13_up_dim = 2 * n if is_act_and_mul else n
    assert layer.w13_weight.shape == (e, w13_up_dim, k)
    assert layer.w2_weight.shape == (e, k, n)
    if has_bias:
        assert layer.w13_bias.shape == (e, w13_up_dim)
        assert layer.w2_bias.shape == (e, k)


@pytest.mark.parametrize("is_act_and_mul", [True, False])
def test_fp8_per_block_zero_padding(is_act_and_mul):
    """_zero_padding must zero the roundup-padded rows of each w13 shard: two
    gate/up shards for gated MoE, a single up shard for non-gated MoE."""
    from types import SimpleNamespace

    e, n, k = 2, 96, 256  # unpadded sizes; k is already block-aligned
    n_pad = 128  # rounded up to the 128 quant block
    num_shards = 2 if is_act_and_mul else 1
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            hidden_dim_unpadded=k, intermediate_size_per_partition_unpadded=n
        ),
        w13_weight=torch.ones(e, num_shards * n_pad, k),
        w2_weight=torch.ones(e, k, n_pad),
        w13_bias=torch.ones(e, num_shards * n_pad),
        w2_bias=torch.ones(e, k),
    )
    method = SimpleNamespace(moe=SimpleNamespace(is_act_and_mul=is_act_and_mul))
    Fp8PerBlockOnlineMoEMethod._zero_padding(method, layer)

    w13 = layer.w13_weight.view(e, num_shards, n_pad, k)
    assert (w13[:, :, :n] == 1).all()
    assert (w13[:, :, n:] == 0).all()
    bias = layer.w13_bias.view(e, num_shards, n_pad)
    assert (bias[..., :n] == 1).all()
    assert (bias[..., n:] == 0).all()
    assert (layer.w2_weight[:, :, :n] == 1).all()
    assert (layer.w2_weight[:, :, n:] == 0).all()
