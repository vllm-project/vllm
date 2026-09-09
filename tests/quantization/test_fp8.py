# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether FP8 computation is enabled correctly.

Run `pytest tests/quantization/test_fp8.py --forked`.
"""

import logging
from types import SimpleNamespace

import pytest
import regex as re
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm import _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.config.kernel import KernelConfig
from vllm.config.model import ModelConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.linear.scaled_mm import (
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8KVCacheMethod,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils import flashinfer_utils
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    prepare_fp8_moe_layer_for_fi,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type

MODELS = [
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    # The checkpoint below was removed from the HF.
    # TODO: add a small replacement checkpoint.
    pytest.param(
        "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
        marks=pytest.mark.skip(reason="Checkpoint removed from HF."),
    ),
]


def test_prepare_gated_trtllm_fp8_moe_weights_pads_each_projection(monkeypatch):
    monkeypatch.setattr(
        flashinfer_utils,
        "rotate_weights_for_fi_trtllm_fp8_per_tensor_moe",
        lambda *args: None,
    )
    intermediate = 17
    padded_intermediate = 32
    hidden_size = 4
    gate = torch.ones((1, intermediate, hidden_size), dtype=torch.float8_e4m3fn)
    up = torch.full_like(gate, 2)
    w13 = torch.cat((gate, up), dim=1)
    w2 = torch.ones((1, hidden_size, intermediate), dtype=torch.float8_e4m3fn)
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=True),
        moe_config=SimpleNamespace(
            is_act_and_mul=True,
            intermediate_size_per_partition=intermediate,
        ),
    )

    padded_w31, _, _, _ = prepare_fp8_moe_layer_for_fi(
        layer,
        w13,
        w2,
        w13_scale=torch.ones(1),
        w13_input_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        w2_input_scale=torch.ones(1),
        is_trtllm=True,
    )

    expected = w13.new_zeros((1, 2 * padded_intermediate, hidden_size))
    expected[:, :intermediate] = up
    expected[:, padded_intermediate : padded_intermediate + intermediate] = gate
    assert layer.moe_config.intermediate_size_per_partition == padded_intermediate
    assert torch.equal(padded_w31, expected)


def test_static_fp8_moe_input_scales_remain_scalar() -> None:
    a1_scale, a2_scale = process_fp8_input_tensor_strategy_moe(
        torch.tensor([0.25, 0.5]),
        torch.tensor([0.75, 0.6]),
        enable_eplb=False,
    )

    assert a1_scale.ndim == a2_scale.ndim == 0


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    "force_marlin", [True, False] if current_platform.is_cuda() else [False]
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_model_load_and_run(
    model_id: str,
    force_marlin: bool,
    use_rocm_aiter: bool,
    monkeypatch,
    dist_init,
    workspace_init,
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    kernel_config = KernelConfig(
        linear_backend="marlin" if force_marlin else "auto",
        moe_backend="marlin" if force_marlin else "auto",
    )
    model, vllm_config = load_model_without_vllm_runner(
        model_id,
        model_config_kwargs={"hf_overrides": {"num_hidden_layers": 3}},
        vllm_config_kwargs={"kernel_config": kernel_config},
    )
    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        model(input_ids, positions, None)


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "force_marlin", [True, False] if current_platform.is_cuda() else [False]
)
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False]
)
def test_online_quantization(
    vllm_runner,
    kv_cache_dtype: str,
    force_marlin: bool,
    use_rocm_aiter: bool,
    monkeypatch,
) -> None:
    if use_rocm_aiter:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    kwargs = {}
    if force_marlin:
        kwargs["linear_backend"] = "marlin"
        kwargs["moe_backend"] = "marlin"

    model_dtype = "auto"
    if kv_cache_dtype == "fp8" and current_platform.is_device_capability_family(90):
        # FA3 requires BF16 output when the query input is FP8.
        model_dtype = "bfloat16"

    with vllm_runner(
        "facebook/opt-125m",
        quantization="fp8",
        dtype=model_dtype,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        **kwargs,
    ) as llm:

        def check_model(model):
            fc1 = model.model.decoder.layers[0].fc1
            assert isinstance(fc1.quant_method, Fp8PerTensorOnlineLinearMethod)
            if kv_cache_dtype == "fp8":
                attn = model.model.decoder.layers[0].self_attn.attn
                assert isinstance(attn.quant_method, Fp8KVCacheMethod)
                assert attn._k_scale == 1.0
                assert attn._v_scale == 1.0

            if current_platform.is_cuda() or current_platform.is_xpu():
                if current_platform.supports_fp8() and not force_marlin:
                    # For GPUs with hardware support, we keep weights in fp8
                    assert fc1.weight.dtype == torch.float8_e4m3fn
                    assert not isinstance(
                        fc1.quant_method.fp8_linear, MarlinFP8ScaledMMLinearKernel
                    )
                else:
                    # For GPUs without hardware support, we pack the fp8 weights
                    # for weight-only quantization using Marlin kernels
                    assert fc1.weight.dtype == torch.int32
                    assert isinstance(
                        fc1.quant_method.fp8_linear, MarlinFP8ScaledMMLinearKernel
                    )
            elif current_platform.is_rocm():
                if current_platform.supports_fp8() and not force_marlin:
                    # For GPUs with hardware support, we keep weights in fp8
                    assert fc1.weight.dtype == current_platform.fp8_dtype()
                else:  # unsupported ROCm platform
                    pytest.skip(
                        "Skip `test_load_fp16_model`. "
                        "It only runs on ROCm platform with FP8 compute."
                        " e.g. MI300X and above."
                    )
            else:  # unsupported platform
                pytest.skip(
                    "Skip `test_load_fp16_model`. "
                    "It only runs on CUDA and ROCm platform."
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
    # Note: `allenai/OLMoE-1B-7B-0125-Instruct` was selected because:
    # 1. it covers both Linear and MoE paths
    # 2. it is already used by other tests in CI, so adding it here
    #    does not increase disk space for CI runners
    # I really wanted to use `ibm-granite/granite-3.0-1b-a400m-base`
    # which I think is the smallest MoE model in vLLM (2.5 GiB bf16,
    # 1.3 GiB fp8), but could not as adding one more model makes CI
    # run out of disk space.
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"

    # Force spawn to ensure caplog_mp_spawn works consistently
    # (it relies on VLLM_LOGGING_CONFIG_PATH which spawn reads but fork ignores)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    with (
        caplog_mp_spawn(logging.DEBUG) as log_holder,
        vllm_runner(
            model_name,
            quantization="fp8",
            enforce_eager=True,
        ) as llm,
    ):
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])

    log_text = log_holder.text

    # Parse memory usage from captured logs
    model_memory_gib = None
    peak_memory_gib = None
    for line in log_text.splitlines():
        if model_memory_gib is None:
            match = re.search(r"Model loading took ([\d.]+) GiB memory", line)
            if match:
                model_memory_gib = float(match.group(1))
        if peak_memory_gib is None:
            match = re.search(
                r"Peak GPU memory after loading weights: ([\d.]+) GiB", line
            )
            if match:
                peak_memory_gib = float(match.group(1))

    assert model_memory_gib is not None, "Could not find model loading memory log"
    assert peak_memory_gib is not None, "Could not find peak memory log"
    print(f"GPU memory used after loading weights: {model_memory_gib} GiB")
    print(f"Peak GPU memory usage while loading weights: {peak_memory_gib} GiB")

    # model specific, allenai/OLMoE-1B-7B-0125-Instruct fp8 online quant
    # uses 6.65 GiB for weight loading (bf16 checkpoint is ~12.89 GiB)
    expected_model_memory_gib = 6.7

    # for allenai/OLMoE-1B-7B-0125-Instruct the number we see today is 9.06
    # GiB, which is 1.36x above model_memory_gib. A slightly higher number is
    # expected as when we load and quantize weights in a streaming fashion we
    # need to have individual weights in bf16 + fp8 alive at the same time.
    expected_peak_memory_gib = expected_model_memory_gib * 1.4

    assert model_memory_gib < expected_model_memory_gib, (
        f"{model_memory_gib=} higher than {expected_model_memory_gib}"
    )
    assert peak_memory_gib < expected_peak_memory_gib, (
        f"{peak_memory_gib=} higher than {expected_peak_memory_gib}"
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
        quantization="fp8",
        enforce_eager=True,
        load_format="dummy",
    ) as llm:
        outputs = llm.generate_greedy(["The future of AI is"], max_tokens=4)
        print(outputs[0][1])


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant(dtype) -> None:
    def quantize_ref(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(current_platform.fp8_dtype())
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(current_platform.fp8_dtype())
        return qweight

    def per_tensor_dequantize(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight

    # Note that we use a shape % 4 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 4.
    x = (torch.randn(size=(11, 11), device=DEVICE_TYPE) * 13).to(dtype)

    # Dynamic quantization
    ref_y, inv_scale = ops.scaled_fp8_quant(x, None)
    ref_y = per_tensor_dequantize(ref_y, inv_scale, dtype)

    # Reference dynamic quantization
    y = quantize_ref(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Static quantization
    y, _ = ops.scaled_fp8_quant(x, inv_scale)
    torch.testing.assert_close(ref_y, per_tensor_dequantize(y, inv_scale, dtype))

    # Padding
    y, _ = ops.scaled_fp8_quant(x, inv_scale, num_token_padding=17)
    assert y.shape[0] == 17
    torch.testing.assert_close(
        ref_y,
        per_tensor_dequantize(torch.narrow(y, 0, 0, x.shape[0]), inv_scale, dtype),
    )

    # non-contiguous input with padding
    m, n, padded_stride = 975, 512, 576
    padded_tensor = (torch.randn(size=(m, padded_stride), device=DEVICE_TYPE) * 13).to(
        dtype
    )
    x_nc = padded_tensor[:, :n]  # shape (m, n) with stride (padded_stride, 1)

    assert not x_nc.is_contiguous()
    assert x_nc.stride(0) == padded_stride

    # dynamic quantization
    ref_y_nc, inv_scale_nc = ops.scaled_fp8_quant(x_nc, None)
    ref_y_nc = per_tensor_dequantize(ref_y_nc, inv_scale_nc, dtype)

    # reference dynamic quantization
    y_nc = quantize_ref(x_nc, inv_scale_nc)
    torch.testing.assert_close(
        ref_y_nc, per_tensor_dequantize(y_nc, inv_scale_nc, dtype)
    )

    # static quantization
    y_nc, _ = ops.scaled_fp8_quant(x_nc, inv_scale_nc)
    torch.testing.assert_close(
        ref_y_nc, per_tensor_dequantize(y_nc, inv_scale_nc, dtype)
    )

    # padding after non-contiguous input quantization
    y_nc_pad, _ = ops.scaled_fp8_quant(x_nc, inv_scale_nc, num_token_padding=m + 10)
    assert y_nc_pad.shape[0] == m + 10
    torch.testing.assert_close(
        ref_y_nc,
        per_tensor_dequantize(
            torch.narrow(y_nc_pad, 0, 0, x_nc.shape[0]), inv_scale_nc, dtype
        ),
    )


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="FP8 e4m3fn weight reloading is not supported on e4m3fnuz platforms",
)
@pytest.mark.parametrize("method_cls", [Fp8LinearMethod, Fp8MoEMethod])
# FP8 weight reloading does not support online quantization
@pytest.mark.parametrize("is_checkpoint_fp8_serialized", [True])  # skip False
@pytest.mark.parametrize("weight_block_size", [None, [128, 128]])
# any postprocessing that is applied to the weights such as padding and repacking
# (excluding device sharding) must also be applied to the reloaded weights
#
# this is the case for marlin as well as per-tensor Fp8MoEMethod
@pytest.mark.parametrize("use_marlin", [False])  # skip True
def test_fp8_reloading(
    default_vllm_config,
    method_cls,
    is_checkpoint_fp8_serialized,
    weight_block_size,
    use_marlin,
    dist_init,
    monkeypatch,
):
    # NOTE(rob): this test fails when using DeepGEMM because the
    # shapes are invalid. Previously the test was passing because
    # we set fp8_backend to None, which sidestepped the issue.
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")

    if is_checkpoint_fp8_serialized is False:
        pytest.skip("FP8 weight reloading does not support online quantization")

    if method_cls is Fp8MoEMethod and weight_block_size is None:
        pytest.skip(
            "FP8 Tensor weight reloading does not support fusing w13_weight_scale. "
            "If this is your use case, consider using a restore function like #26327"
        )

    # Set model config as model_config.dtype is required in Fp8LinearMethod.
    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.moe_backend = "triton"
    layer_size = 128 if weight_block_size is not None else 1
    with torch.device(f"{DEVICE_TYPE}:0"):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            weight_block_size=weight_block_size,
        )

        if method_cls is Fp8LinearMethod:
            layer = torch.nn.Linear(layer_size, layer_size)
            method = method_cls(config)
            method.create_weights(
                layer=layer,
                input_size_per_partition=layer_size,
                output_partition_sizes=[layer_size],
                input_size=layer_size,
                output_size=layer_size,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )
            method.use_marlin = use_marlin

        else:
            layer = FusedMoEFactory(
                num_experts=1,
                top_k=1,
                hidden_size=layer_size,
                intermediate_size=layer_size,
            )
            layer = layer.routed_experts
            method = method_cls(config, layer)
            method.create_weights(
                layer=layer,
                num_experts=1,
                hidden_size=layer_size,
                intermediate_size_per_partition=layer_size,
                params_dtype=torch.bfloat16,
                weight_loader=default_weight_loader,
            )

    # capture weights format during loading
    original_metadata = [
        (name, param.shape, getattr(param, "weight_loader", default_weight_loader))
        for name, param in layer.named_parameters()
    ]

    # test loading
    for name, shape, _ in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)

    # test reloading works after loading
    for name, shape, _ in original_metadata:
        param = getattr(layer, name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, torch.zeros(shape))  # cannot use empty

    method.process_weights_after_loading(layer)


def test_kv_cache_scale_sync_to_host_copies():
    """Test device-to-host sync of the k/v quantization scales, for both the
    checkpoint-load and runtime-calc paths that produce them.
    """
    layer = torch.nn.Module()
    set_default_quant_scales(layer, register_buffer=True)
    layer.kv_cache_dtype = "fp8"

    method = BaseKVCacheMethod(quant_config=None)
    method.create_weights(layer)
    # 0.3 stays != 1.0 even after the fp8_fnuz x2 rescale.
    checkpoint_scale = torch.tensor(0.3, dtype=torch.float32)
    layer.k_scale.weight_loader(layer.k_scale, checkpoint_scale)
    layer.v_scale.weight_loader(layer.v_scale, checkpoint_scale)
    method.process_weights_after_loading(layer)

    assert layer._k_scale_float != 1.0
    assert layer._v_scale_float != 1.0
    # Host copy must mirror both the float and the device scale tensor.
    assert layer._k_scale_cpu.item() == pytest.approx(layer._k_scale_float)
    assert layer._v_scale_cpu.item() == pytest.approx(layer._v_scale_float)
    assert layer._k_scale_cpu.item() == pytest.approx(layer._k_scale.item())
    assert layer._v_scale_cpu.item() == pytest.approx(layer._v_scale.item())


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
def test_kv_cache_dtype_skip_layers(monkeypatch, dist_init, workspace_init):
    """Test that kv_cache_dtype_skip_layers skips quantization for specified layers."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    model, _ = load_model_without_vllm_runner(
        "facebook/opt-125m",
        vllm_config_kwargs={
            "cache_config": CacheConfig(
                cache_dtype="fp8", kv_cache_dtype_skip_layers=["0", "2"]
            )
        },
    )
    for i, layer in enumerate(model.model.decoder.layers):
        expected = "auto" if str(i) in ["0", "2"] else "fp8"
        assert layer.self_attn.attn.kv_cache_dtype == expected
