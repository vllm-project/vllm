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


@pytest.mark.parametrize("activation_scheme", ["dynamic", "static"])
@pytest.mark.parametrize("width", [16, 17])
@pytest.mark.parametrize(
    "shard_case", ["complete", "missing", "duplicate", "overlap", "missing_scale"]
)
def test_per_tensor_refresh_without_pwal(
    default_vllm_config, dist_init, monkeypatch, activation_scheme, width, shard_case
):
    """Fused shard scales refresh twice without rebuilding runtime objects."""
    from unittest.mock import Mock

    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    default_vllm_config.kernel_config.linear_backend = "cutlass"

    def weight_loader(param, loaded_weight, shard_id=None, offset=None):
        if offset is not None:
            param.data.narrow(0, offset, loaded_weight.shape[0]).copy_(loaded_weight)
        elif shard_id is None:
            default_weight_loader(param, loaded_weight)
        else:
            param.data.narrow(0, shard_id * width, width).copy_(loaded_weight)

    with torch.device("cuda:0"):
        method = Fp8LinearMethod(Fp8Config(True, activation_scheme))
        layer = torch.nn.Module()
        layer.quant_method = method
        method.create_weights(
            layer,
            32,
            [width, width],
            32,
            2 * width,
            torch.bfloat16,
            weight_loader=weight_loader,
        )
        record_metadata_for_reloading(layer)
        layer.weight.data.fill_(1)
        layer.weight_scale.data.fill_(1)
        if activation_scheme == "static":
            layer.input_scale.data.fill_(1)
        method.process_weights_after_loading(layer)
        assert method.supports_selective_reload()
        originals = dict(layer.named_parameters())
        pointers = {k: v.data_ptr() for k, v in originals.items()}
        kernel = method.fp8_linear
        monkeypatch.setattr(
            method,
            "process_weights_after_loading",
            Mock(side_effect=AssertionError("method PWAL called during reload")),
        )
        monkeypatch.setattr(
            kernel,
            "process_weights_after_loading",
            Mock(side_effect=AssertionError("kernel PWAL called during reload")),
        )
        for generation in (2, 4):
            initialize_layerwise_reload(layer)
            source = torch.full((width, 32), 8, dtype=torch.float8_e4m3fn)
            values = {
                "weight_scale": torch.tensor([generation / 2, generation]),
            }
            if activation_scheme == "static":
                values["input_scale"] = torch.tensor([0.5, 2.0])
            if shard_case == "missing_scale":
                values.pop("weight_scale")
            for name, value in values.items():
                param = getattr(layer, name)
                param.weight_loader(param, value)
            param = layer.weight
            param.weight_loader(param, source, shard_id=0)
            if shard_case in ("duplicate", "overlap"):
                before = originals["weight"].float().clone()
                with pytest.raises(ValueError, match="overlapping shards"):
                    param.weight_loader(
                        param,
                        source,
                        offset=0 if shard_case == "duplicate" else width // 2,
                    )
                torch.testing.assert_close(originals["weight"].float(), before)
            if shard_case in ("complete", "missing_scale"):
                param.weight_loader(param, source, shard_id=1)
            # Runtime is not mutated before FINISH.
            assert originals["weight_scale"].item() == generation / 2
            if shard_case != "complete":
                before = originals["weight"].float().clone()
                error = (
                    "requires weight and all scale parameters"
                    if shard_case == "missing_scale"
                    else "incomplete shards for weight"
                )
                with pytest.raises(ValueError, match=error):
                    finalize_layerwise_reload(layer, None)
                torch.testing.assert_close(layer.weight.float(), before, rtol=0, atol=0)
                assert layer.weight is originals["weight"]
                break
            finalize_layerwise_reload(layer, None)
            method.refresh_derived_state(layer)
            assert method.fp8_linear is kernel
            assert layer.weight_scale.item() == generation
            expected = torch.zeros(layer.weight.shape, dtype=torch.float32)
            expected[:, :width] = 4
            expected[:, width : 2 * width] = 8
            torch.testing.assert_close(layer.weight.float(), expected, rtol=0, atol=0)
            for name, original in originals.items():
                assert getattr(layer, name) is original
                assert original.data_ptr() == pointers[name]
            if activation_scheme == "static":
                assert layer.input_scale.item() == 2
        method.process_weights_after_loading.assert_not_called()
        kernel.process_weights_after_loading.assert_not_called()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("fused", [False, True])
def test_moe_reload_tracker_covers_expert_column_shards(fused):
    """Expert and fused-column shards are tracked without overlap."""
    from vllm.model_executor.model_loader.reload.meta import ReloadMoETracker

    target = torch.zeros(2, 8, 4, device="cuda")
    regions = []
    with ReloadMoETracker(target, regions):
        if fused:
            target[:, :, :2].copy_(torch.ones(2, 8, 2, device="cuda"))
            with pytest.raises(ValueError, match="Overlapping"):
                target[:, :, 1:3].copy_(torch.full((2, 8, 2), 9.0, device="cuda"))
            target[:, :, 2:].copy_(torch.ones(2, 8, 2, device="cuda"))
            torch.testing.assert_close(target, torch.ones_like(target))
            return
        target[0, :, :2].copy_(torch.ones(8, 2, device="cuda"))
        target[0, :, 2:].copy_(torch.ones(8, 2, device="cuda"))
        with pytest.raises(ValueError, match="Overlapping"):
            target[0, :, 1:3].copy_(torch.ones(8, 2, device="cuda"))
        target[1].copy_(torch.ones(8, 4, device="cuda"))
    torch.testing.assert_close(target, torch.ones_like(target))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("block_quant", [False, True])
@pytest.mark.parametrize("failure", [None, "layout", "missing_expert", "missing_scale"])
def test_moe_finish_validates_all_layouts_before_write(
    monkeypatch, block_quant, failure
):
    """FINISH counts expert rectangles and validates every target before copying."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.quant_config = SimpleNamespace(activation_scheme="dynamic")
    method.weight_scale_name = "weight_scale_inv" if block_quant else "weight_scale"
    names = (
        "w13_weight",
        "w2_weight",
        f"w13_{method.weight_scale_name}",
        f"w2_{method.weight_scale_name}",
    )
    shapes = (
        (2, 8, 4),
        (2, 4, 4),
        (2, 2, 1) if block_quant else (2, 2),
        (2, 1, 1) if block_quant else (2,),
    )
    layer = torch.nn.Module()
    staging, regions = {}, {}
    for name, shape in zip(names, shapes):
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.zeros(shape, device="cuda"), requires_grad=False),
        )
        staging[name] = torch.ones(shape, device="cuda")
        regions[name] = (
            [(e, 0, shape[1], 0, shape[2]) for e in range(shape[0])]
            if len(shape) == 3
            else [(0, staging[name].numel())]
        )
    layer._fp8_moe_reload_staging = staging
    layer._fp8_moe_reload_regions = regions
    original = dict(layer.named_parameters())
    pointers = {name: value.data_ptr() for name, value in original.items()}
    if failure == "missing_expert":
        regions["w13_weight"].pop()
    elif failure == "missing_scale":
        staging.pop(names[-1])
    converted = []

    def convert(runtime_layer, w13, w2, s13, s2, *unused):
        converted.append(True)
        w13.add_(1)
        assert torch.equal(staging["w13_weight"], torch.ones_like(w13))
        return w13, w2, s13, s2.flatten()[:0] if failure == "layout" else s2

    monkeypatch.setattr(method, "_convert_moe_runtime", convert)
    if failure:
        message = {
            "layout": "runtime layout changed",
            "missing_expert": "incomplete shards",
            "missing_scale": "requires all expert weights and scales",
        }[failure]
        with pytest.raises(ValueError, match=message):
            method.refresh_derived_state(layer)
        assert staging
        if failure != "layout":
            assert not converted
    else:
        method.refresh_derived_state(layer)
        method.refresh_derived_state(layer)
        assert not staging and not regions
    for name, value in original.items():
        assert getattr(layer, name) is value and value.data_ptr() == pointers[name]
        expected = 0 if failure else (2 if name == "w13_weight" else 1)
        torch.testing.assert_close(value, torch.full_like(value, expected))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_moe_finish_refreshes_kernel_owned_scales(monkeypatch):
    """Refresh alpha and reciprocal storage retained by the CUTLASS kernel."""
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.block_quant = False
    method.weight_scale_name = "weight_scale"
    method.fp8_backend = Fp8MoeBackend.FLASHINFER_CUTLASS
    method.quant_config = SimpleNamespace(activation_scheme="static")
    layer = torch.nn.Module()
    names = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_input_scale",
        "w2_input_scale",
    )
    for name in names:
        shape = () if "input_scale" in name else (2,)
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.ones(shape, device="cuda"), requires_grad=False),
        )
    method.moe_quant_config = SimpleNamespace(
        g1_alphas=torch.ones(2, device="cuda"),
        g2_alphas=torch.ones(2, device="cuda"),
        a1_gscale=torch.ones((), device="cuda"),
        a2_gscale=torch.ones((), device="cuda"),
    )
    config = method.moe_quant_config
    original = vars(config).copy()
    pointers = {name: value.data_ptr() for name, value in original.items()}
    monkeypatch.setattr(
        method,
        "_convert_moe_runtime",
        lambda layer, w13, w2, s13, s2, *unused: (w13, w2, s13, s2),
    )
    for scale in (2.0, 4.0):
        layer._fp8_moe_reload_staging = {
            name: torch.full((2,), scale, device="cuda") for name in names
        }
        layer._fp8_moe_reload_regions = {name: [(0, 2)] for name in names}
        method.refresh_derived_state(layer)
        method.refresh_derived_state(layer)
        assert method.moe_quant_config is config
        for name, value in original.items():
            assert getattr(config, name) is value
            assert value.data_ptr() == pointers[name]
            expected = scale * scale if "alphas" in name else 1 / scale
            torch.testing.assert_close(value, torch.full_like(value, expected))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("block_quant", [False, True])
def test_moe_finish_runs_flashinfer_cutlass_conversion(block_quant):
    """Exercise real requantization/W31 conversion, not a conversion substitute."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.block_quant = block_quant
    method.weight_scale_name = "weight_scale_inv" if block_quant else "weight_scale"
    method.fp8_backend = Fp8MoeBackend.FLASHINFER_CUTLASS
    method.quant_config = SimpleNamespace(
        activation_scheme="dynamic" if block_quant else "static"
    )
    method.moe = SimpleNamespace(
        w13_num_shards=2, is_act_and_mul=True, intermediate_size_per_partition=128
    )
    layer = torch.nn.Module()
    layer.moe_config = method.moe
    layer.local_num_experts = 2
    layer.activation = MoEActivation.SILU
    layer.weight_block_size = [128, 128] if block_quant else None
    s13, s2 = f"w13_{method.weight_scale_name}", f"w2_{method.weight_scale_name}"
    with torch.device("cuda"):
        sources = {
            "w13_weight": torch.full((2, 256, 128), 8.0, dtype=torch.float8_e4m3fn),
            "w2_weight": torch.full((2, 128, 128), 8.0, dtype=torch.float8_e4m3fn),
            s13: torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            s2: torch.ones(2),
        }
        if block_quant:
            sources[s13] = sources[s13].unsqueeze(-1)
            sources[s2] = sources[s2].reshape(2, 1, 1)
        else:
            sources.update(w13_input_scale=torch.ones(2), w2_input_scale=torch.ones(2))
            method.moe_quant_config = SimpleNamespace(
                g1_alphas=torch.ones(2),
                g2_alphas=torch.ones(2),
                a1_gscale=torch.ones(()),
                a2_gscale=torch.ones(()),
            )
        for name, source in sources.items():
            shape = source.shape
            if not block_quant and name == s13:
                shape = (2,)
            elif "input_scale" in name:
                shape = ()
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.zeros(shape, dtype=source.dtype), requires_grad=False
                ),
            )
    layer._fp8_moe_reload_staging = sources
    layer._fp8_moe_reload_regions = {
        name: (
            [(e, 0, value.shape[1], 0, value.shape[2]) for e in range(2)]
            if value.ndim == 3
            else [(0, value.numel())]
        )
        for name, value in sources.items()
    }
    method.refresh_derived_state(layer)
    assert layer.moe_config is method.moe
    assert method.moe.intermediate_size_per_partition == 128
    expected = torch.full_like(layer.w13_weight.float(), 8.0)
    if not block_quant:
        expected[:, 128:] = 4.0
    torch.testing.assert_close(layer.w13_weight.float(), expected, rtol=0, atol=0)
    expected_scale = (
        torch.tensor([[[2.0], [1.0]], [[2.0], [1.0]]], device="cuda")
        if block_quant
        else torch.full((2,), 2.0, device="cuda")
    )
    torch.testing.assert_close(getattr(layer, s13), expected_scale, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("block_quant", [False, True])
def test_moe_checkpoint_staging_uses_routed_experts_loader(block_quant):
    """Use real expert/TP shard loading while keeping runtime storage unchanged."""
    import inspect

    from vllm.model_executor.layers.fused_moe import (
        FusedMoeWeightScaleSupported,
        RoutedExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
    from vllm.model_executor.model_loader.reload.meta import to_meta_tensor

    method = object.__new__(Fp8MoEMethod)
    method.fp8_backend = Fp8MoeBackend.FLASHINFER_CUTLASS
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_scale_refine = None
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.quant_config = None
    layer.quant_method = method
    layer.expert_map_manager = SimpleNamespace(map_global_to_local=lambda e: e)
    layer.moe_config = SimpleNamespace(
        is_act_and_mul=True, tp_rank=0, moe_parallel_config=SimpleNamespace(tp_size=1)
    )
    layer._fp8_moe_reload_staging = {}
    layer._fp8_moe_reload_regions = {}
    loader = layer.weight_loader
    with torch.device("cuda"):
        for name, shape, shard_shape in (
            ("w13_weight", (2, 256, 128), (128, 128)),
            ("w2_weight", (2, 128, 128), (128, 128)),
            (
                "w13_weight_scale_inv" if block_quant else "w13_weight_scale",
                (2, 2, 1) if block_quant else (2, 2),
                (1, 1) if block_quant else (),
            ),
            (
                "w2_weight_scale_inv" if block_quant else "w2_weight_scale",
                (2, 1, 1) if block_quant else (2,),
                (1, 1) if block_quant else (),
            ),
        ):
            target = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
            target.quant_method = (
                FusedMoeWeightScaleSupported.BLOCK.value
                if block_quant
                else FusedMoeWeightScaleSupported.TENSOR.value
            )
            metadata = to_meta_tensor(target)
            for expert in range(2):
                for shard in ("w1", "w3") if name.startswith("w13") else ("w2",):
                    args = inspect.signature(loader).bind(
                        metadata, torch.ones(shard_shape), name, shard, expert
                    )
                    args.apply_defaults()
                    assert method.reload_parameter(layer, name, target, args, loader)
            torch.testing.assert_close(target, torch.zeros_like(target))
            staged = layer._fp8_moe_reload_staging[name]
            torch.testing.assert_close(staged, torch.ones_like(staged))
        if not block_quant:
            target = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            metadata = to_meta_tensor(target)
            for expert in range(2):
                for shard in ("w1", "w3"):
                    args = inspect.signature(loader).bind(
                        metadata, torch.tensor(2.0), "w13_input_scale", shard, expert
                    )
                    args.apply_defaults()
                    assert method.reload_parameter(
                        layer, "w13_input_scale", target, args, loader
                    )
            with pytest.raises(ValueError, match="Duplicate"):
                method.reload_parameter(layer, "w13_input_scale", target, args, loader)
            assert layer._fp8_moe_reload_regions["w13_input_scale"] == [(0, 1), (1, 2)]
            torch.testing.assert_close(target, torch.zeros_like(target))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "block_quant,intermediate_size", [(False, 256), (False, 257), (True, 256)]
)
def test_moe_cutlass_reload_lifecycle(
    default_vllm_config,
    dist_init,
    workspace_init,
    monkeypatch,
    block_quant,
    intermediate_size,
):
    """Cold and warm CUTLASS executions agree after real expert-loader reload."""
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    default_vllm_config.kernel_config.moe_backend = "flashinfer_cutlass"
    config = Fp8Config(
        True,
        "dynamic" if block_quant else "static",
        weight_block_size=[128, 128] if block_quant else None,
    )

    def cold(generation):
        with torch.device("cuda"):
            runner = FusedMoEFactory(
                4,
                2,
                256,
                intermediate_size,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"cold_{generation}",
            )
            layer = runner.routed_experts
            method = layer.quant_method
            record_metadata_for_reloading(layer)
            sources = {}
            for name, param in layer.named_parameters(recurse=False):
                value = 0.125 * generation if "scale" in name else 0.25 * generation
                sources[name] = torch.full_like(param, value)
                param.data.copy_(sources[name])
            method.process_weights_after_loading(layer)
        return runner, layer, method, sources

    runner, layer, method, _ = cold(1)
    reference_runner, reference, reference_method, sources = cold(2)
    originals = dict(layer.named_parameters(recurse=False))
    pointers = {name: p.data_ptr() for name, p in originals.items()}
    kernel = method.moe_kernel
    quant_state = method.moe_quant_config
    derived = {
        name: getattr(quant_state, name)
        for name in ("a1_gscale", "a2_gscale", "g1_alphas", "g2_alphas")
        if getattr(quant_state, name) is not None
    }
    derived_pointers = {name: value.data_ptr() for name, value in derived.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Reload must not rerun PWAL or rebuild the MoE kernel")

    monkeypatch.setattr(method, "process_weights_after_loading", forbidden)
    monkeypatch.setattr(method, "_install_moe_kernel", forbidden)
    monkeypatch.setattr(method, "_prepare_moe_runtime", forbidden)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.make_fp8_moe_kernel", forbidden
    )
    for _ in range(2):
        before = {name: value.detach().clone() for name, value in originals.items()}
        initialize_layerwise_reload(layer)
        for name, value in sources.items():
            param = getattr(layer, name)
            for expert in range(4):
                for shard in ("w1", "w3") if name.startswith("w13") else ("w2",):
                    incoming = value[expert]
                    if name.startswith("w13") and "input_scale" not in name:
                        incoming = incoming.chunk(2, dim=0)[shard == "w3"]
                    param.weight_loader(param, incoming, name, shard, expert)
        for name, original in originals.items():
            torch.testing.assert_close(
                original.float(), before[name].float(), rtol=0, atol=0
            )
        finalize_layerwise_reload(layer, default_vllm_config.model_config)
        method.refresh_derived_state(layer)
    assert method.moe_kernel is kernel
    assert method.moe_quant_config is quant_state
    for name, original in derived.items():
        assert getattr(quant_state, name) is original
        assert original.data_ptr() == derived_pointers[name]
        torch.testing.assert_close(
            original, getattr(reference_method.moe_quant_config, name), rtol=0, atol=0
        )
    for name, original in originals.items():
        assert (
            getattr(layer, name) is original and original.data_ptr() == pointers[name]
        )
        torch.testing.assert_close(
            original.float(), getattr(reference, name).float(), rtol=0, atol=0
        )
    x = torch.full((4, 256), 0.125, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]], device="cuda", dtype=torch.int32
    )
    weights = torch.full((4, 2), 0.5, device="cuda")

    def run(target, quant):
        return quant.moe_kernel.apply(
            x.clone(),
            target.w13_weight,
            target.w2_weight,
            weights,
            ids,
            activation=target.activation,
            global_num_experts=4,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    torch.testing.assert_close(
        run(layer, method), run(reference, reference_method), rtol=0, atol=0
    )


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
