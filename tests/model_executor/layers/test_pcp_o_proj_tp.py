# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.kernels.linear import (
    CutlassNvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    PCPOProjLinearMethod,
    PCPOProjRowParallelLinear,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4LinearMethod,
)


class _FakeWork:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _FakePCPGroup:
    def __init__(self, rank: int) -> None:
        self.rank_in_group = rank
        self.world_size = 2
        self.device_group = object()
        self.other_output: torch.Tensor | None = None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.other_output is not None
        return tensor + self.other_output


class _UnsupportedLinearMethod(LinearMethodBase):
    def create_weights(self, *args, **kwargs) -> None:
        raise AssertionError("weights are installed by the test")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        del layer

    def apply(self, layer, x, bias=None):
        raise AssertionError("unsupported method must fail during post-load")


def _make_layer(
    monkeypatch, pcp_rank: int
) -> tuple[PCPOProjRowParallelLinear, _FakePCPGroup]:
    pcp_group = _FakePCPGroup(pcp_rank)
    config_scope = object()
    monkeypatch.setattr(linear_module, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(linear_module, "get_current_vllm_config", lambda: config_scope)
    monkeypatch.setattr(
        linear_module.UnquantizedLinearMethod,
        "apply",
        lambda _self, layer, x, bias=None: F.linear(x, layer.weight, bias),
    )
    layer = PCPOProjRowParallelLinear(
        input_size=4,
        output_size=3,
        params_dtype=torch.float32,
        reduce_results=False,
    )
    return layer, pcp_group


def test_prefill_waits_for_async_full_weight_gather(monkeypatch):
    layer, _ = _make_layer(monkeypatch, pcp_rank=1)
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)
    torch.testing.assert_close(layer.weight, full_weight[:, 2:])

    work = _FakeWork()

    def _all_gather_into_tensor(output, input_, group, async_op):
        torch.testing.assert_close(input_, full_weight[:, 2:].transpose(0, 1))
        output.copy_(full_weight.transpose(0, 1))
        assert group is layer.pcp_group.device_group
        assert async_op
        return work

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor
    )

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4)
    layer.prefetch_full_weight_if_needed(has_prefill=True)
    assert not work.waited

    output, output_bias = layer(input_)
    assert work.waited
    assert output_bias is None
    torch.testing.assert_close(output, F.linear(input_, full_weight))


def test_forward_requires_explicit_attention_prefetch(monkeypatch):
    layer, _ = _make_layer(monkeypatch, pcp_rank=0)

    with pytest.raises(RuntimeError, match="Attention must call"):
        layer(torch.zeros(1, 4))


@pytest.mark.parametrize(
    ("global_has_prefill", "local_num_prefills", "expected"),
    [(True, 0, True), (False, 1, False), (None, 1, True), (None, 0, False)],
)
def test_prefetch_decision_uses_global_pcp_batch_type(
    monkeypatch, global_has_prefill, local_num_prefills, expected
):
    monkeypatch.setattr(
        linear_module,
        "get_forward_context",
        lambda: SimpleNamespace(global_has_prefill=global_has_prefill),
    )
    metadata = SimpleNamespace(num_prefills=local_num_prefills)

    assert linear_module.get_pcp_o_proj_batch_has_prefill(metadata) is expected


def test_decode_slices_features_and_reduces_over_pcp(monkeypatch):
    layer, pcp_group = _make_layer(monkeypatch, pcp_rank=1)
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4)
    pcp_group.other_output = F.linear(input_[:, :2], full_weight[:, :2])

    layer.prefetch_full_weight_if_needed(has_prefill=False)
    output, output_bias = layer(input_)

    assert output_bias is None
    torch.testing.assert_close(output, F.linear(input_, full_weight))


def test_quantized_o_proj_method_is_rejected(monkeypatch):
    layer, _ = _make_layer(monkeypatch, pcp_rank=0)
    layer.quant_method = PCPOProjLinearMethod(_UnsupportedLinearMethod())

    with pytest.raises(RuntimeError, match="does not support the selected O-Proj"):
        layer.quant_method.process_weights_after_loading(layer)


def _swizzle_nvfp4_scale(scale: torch.Tensor) -> torch.Tensor:
    output_size, scale_cols = scale.shape
    padded_output_size = (output_size + 127) // 128 * 128
    padded_scale_cols = (scale_cols + 3) // 4 * 4
    padded = torch.zeros((1, padded_output_size, padded_scale_cols), dtype=scale.dtype)
    padded[0, :output_size, :scale_cols] = scale
    return (
        padded.view(
            1,
            padded_output_size // 128,
            4,
            32,
            padded_scale_cols // 4,
            4,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(padded_output_size, padded_scale_cols)
    )


def test_modelopt_nvfp4_switches_packed_weight_and_swizzled_scale(monkeypatch):
    output_size = 256
    local_input_size = 64
    pcp_size = 2
    full_weight = (
        torch.arange(
            output_size * local_input_size,
            dtype=torch.int64,
        )
        .to(torch.uint8)
        .view(output_size, local_input_size)
    )
    full_scale = (
        (
            torch.arange(
                output_size * local_input_size // 8,
                dtype=torch.int64,
            )
            % 64
        )
        .to(torch.uint8)
        .view(torch.float8_e4m3fn)
        .view(output_size, 8)
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = output_size
    layer.input_size_per_partition = local_input_size
    layer.weights_padding_cols = 0
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(full_weight[:, :32], requires_grad=False),
    )
    local_swizzled_scale = _swizzle_nvfp4_scale(full_scale[:, :4])
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(local_swizzled_scale, requires_grad=False),
    )
    layer.register_parameter(
        "input_global_scale_inv",
        torch.nn.Parameter(torch.tensor(2.0), requires_grad=False),
    )
    layer.register_parameter(
        "alpha",
        torch.nn.Parameter(torch.tensor(3.0), requires_grad=False),
    )

    method = ModelOptNvFp4LinearMethod.__new__(ModelOptNvFp4LinearMethod)
    method.quant_config = SimpleNamespace(group_size=16)
    method.kernel = CutlassNvFp4LinearKernel.__new__(CutlassNvFp4LinearKernel)
    state = method.enable_tp_weight_switch(layer, pcp_size)

    works: list[_FakeWork] = []

    def _all_gather_into_tensor(output, input_, group, async_op):
        del group
        assert async_op
        if input_.shape[0] == full_weight.shape[1] // pcp_size:
            output.copy_(full_weight.T)
        else:
            torch.testing.assert_close(input_, full_scale[:, :4].view(torch.uint8).T)
            output.copy_(full_scale.view(torch.uint8).T)
        work = _FakeWork()
        works.append(work)
        return work

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor
    )

    input_global_scale_inv = layer.input_global_scale_inv
    alpha = layer.alpha
    method.all_gather_tp_weight(state, group=object())
    method.wait_tp_weight_all_gather(state)
    assert all(work.waited for work in works)

    method.switch_tp_weight(layer, state, use_full_weight=True)
    torch.testing.assert_close(layer.weight, full_weight)
    torch.testing.assert_close(layer.weight_scale, _swizzle_nvfp4_scale(full_scale))
    assert layer.input_global_scale_inv is input_global_scale_inv
    assert layer.alpha is alpha

    method.switch_tp_weight(layer, state, use_full_weight=False)
    torch.testing.assert_close(layer.weight, full_weight[:, :32])
    torch.testing.assert_close(layer.weight_scale, local_swizzled_scale)


def test_modelopt_nvfp4_rejects_non_cutlass_post_load_layout():
    method = ModelOptNvFp4LinearMethod.__new__(ModelOptNvFp4LinearMethod)
    method.kernel = MarlinNvFp4LinearKernel.__new__(MarlinNvFp4LinearKernel)

    with pytest.raises(RuntimeError, match="CUTLASS-compatible"):
        method.get_tp_weight_switch_specs(torch.nn.Module())


def test_unquantized_method_restores_local_weight_after_kernel_failure(monkeypatch):
    layer, _ = _make_layer(monkeypatch, pcp_rank=0)
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)

    def _raise_on_apply(_self, layer, x, bias=None):
        raise RuntimeError("fake linear kernel failure")

    monkeypatch.setattr(linear_module.UnquantizedLinearMethod, "apply", _raise_on_apply)

    def _all_gather_into_tensor(output, input_, group, async_op):
        output.copy_(full_weight.transpose(0, 1))
        return _FakeWork()

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor
    )

    layer.prefetch_full_weight_if_needed(has_prefill=True)
    with pytest.raises(RuntimeError, match="fake linear kernel failure"):
        layer(torch.zeros(1, 4))

    assert layer.weight.shape == (3, 2)
    assert layer._use_full_weight is None
