# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer/native-consumer contract on Blackwell, independent of model dispatch.

B200 CI exercises the arithmetic and QuantizedActivation consumer. The opt-in
full-model path remains SM120-only and requires separate integration validation.
"""

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    KernelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptLinearMethod,
    ModelOptMxFp8Config,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.models.qwen4_exp.nvidia.gdn_decode.kernel import GdnNormQuant
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


@pytest.fixture
def native_tail(dist_init):
    assert current_platform.is_cuda() and current_platform.has_device_capability(100)
    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all"]),
        kernel_config=KernelConfig(linear_backend="flashinfer_cutlass"),
    )
    with (
        torch.inference_mode(),
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
        set_current_vllm_config(config),
    ):
        generator = torch.Generator(device="cuda").manual_seed(17)
        norm = RMSNormGated(
            128,
            eps=1e-6,
            norm_before_gate=True,
            activation="sigmoid",
            device="cuda",
            dtype=torch.bfloat16,
        )
        norm.weight.copy_(torch.rand(128, generator=generator) + 0.5)
        linear = RowParallelLinear(
            6144,
            2560,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            disable_tp=True,
            quant_config=ModelOptMxFp8Config(True, None, []),
            prefix="test_gdn.out_proj",
        )
        weight = torch.randint(-16, 17, linear.weight.shape, generator=generator)
        scale = torch.randint(120, 128, linear.weight_scale.shape, generator=generator)
        linear.weight.weight_loader(linear.weight, weight.to(linear.weight.dtype))
        linear.weight_scale.weight_loader(
            linear.weight_scale, scale.to(linear.weight_scale.dtype)
        )
        linear.quant_method.process_weights_after_loading(linear)
        assert isinstance(linear.quant_method, ModelOptLinearMethod)
        assert type(linear.quant_method.kernel) is FlashInferCutlassMxfp8LinearKernel
        assert get_input_quant_key(linear) == kMxfp8Dynamic
        yield norm, linear


def _inputs(batch, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        batch, 48, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    packed = torch.randn(
        batch, 16384, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    return x, packed[:, 10240:].view(batch, 48, 128)


def _project(linear, data, scale):
    activation = QuantizedActivation(
        data, scale, torch.bfloat16, data.shape, kMxfp8Dynamic
    )
    return linear(activation)[0]


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seed", [0, 23, 37])
@torch.inference_mode()
def test_gdn_norm_quant_matches_native_tail(native_tail, batch, seed):
    """Include zero scale blocks and two independent normal input fixtures."""
    norm, linear = native_tail
    x, z = _inputs(batch, seed)
    if seed == 0:
        x.zero_()
    original_x, original_z = x.clone(), z.clone()
    normalized = norm(x, z)
    reference_q, reference_sf = mxfp8_e4m3_quantize(
        normalized.flatten(-2), is_sf_swizzled_layout=True
    )
    expected = linear(normalized.flatten(-2))[0].clone()
    repeated = linear(normalized.flatten(-2))[0].clone()
    producer = GdnNormQuant(batch, device=x.device)
    producer.quantized.view(torch.uint8).fill_(255)
    producer.scales.fill_(255)
    data, scale = producer.run(x, z, norm.weight)
    actual = _project(linear, data, scale).clone()
    _assert_bytes(data, reference_q)
    _assert_bytes(scale, reference_sf)  # Includes all padded scale bytes.
    _assert_bytes(actual, expected)
    _assert_bytes(repeated, expected)
    _assert_bytes(x, original_x)
    _assert_bytes(z, original_z)
    assert actual.data_ptr() != expected.data_ptr()
    assert torch.isfinite(actual).all()
    if seed:
        assert torch.count_nonzero(actual) > 0


@pytest.mark.parametrize("batch", [1, 2])
@torch.inference_mode()
def test_gdn_norm_quant_graph_replay_reads_changed_inputs(native_tail, batch):
    norm, linear = native_tail
    x, z = _inputs(batch, 41)
    producer = GdnNormQuant(batch, device=x.device)

    def fused():
        return _project(linear, *producer.run(x, z, norm.weight))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused()
            linear(norm(x, z).flatten(-2))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        native_output = linear(norm(x, z).flatten(-2))[0]
        fused_output = fused()
    assert native_output.data_ptr() != fused_output.data_ptr()
    for seed in (43, 47):
        changed_x, changed_z = _inputs(batch, seed)
        assert not torch.equal(x, changed_x) and not torch.equal(z, changed_z)
        x.copy_(changed_x)
        z.copy_(changed_z)
        normalized = norm(x, z)
        expected = linear(normalized.flatten(-2))[0].clone()
        reference_q, reference_sf = mxfp8_e4m3_quantize(
            normalized.flatten(-2), is_sf_swizzled_layout=True
        )
        producer.quantized.view(torch.uint8).fill_(255)
        producer.scales.fill_(255)
        native_output.fill_(float("nan"))
        fused_output.fill_(float("nan"))
        graph.replay()
        _assert_bytes(native_output, expected)
        _assert_bytes(fused_output, expected)
        _assert_bytes(producer.quantized, reference_q)
        _assert_bytes(producer.scales, reference_sf)
        _assert_bytes(x, changed_x)
        _assert_bytes(z, changed_z)
