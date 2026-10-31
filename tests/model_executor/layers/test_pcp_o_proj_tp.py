# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.layers.pcp_o_proj_tp as pcp_o_proj_module
import vllm.model_executor.parameter as parameter_module
from vllm.config.quantization import QuantSpec
from vllm.model_executor.kernels.linear import (
    CutlassNvFp4LinearKernel,
    MarlinNvFp4LinearKernel,
)
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.pcp_o_proj_tp import PCPOProjPath, PCPOProjTP
from vllm.model_executor.layers.quantization.modelopt import (
    CkptCtx,
    ModelOptLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
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


def _make_layer(monkeypatch, pcp_rank: int) -> tuple[PCPOProjTP, _FakePCPGroup]:
    pcp_group = _FakePCPGroup(pcp_rank)
    config_scope = object()
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_pcp_o_proj_tp=True)
    )
    for module in (linear_module, pcp_o_proj_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pcp_o_proj_module, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(
        pcp_o_proj_module, "get_current_vllm_config", lambda: config_scope
    )

    def _apply(_self, layer, x, bias=None):
        assert layer.input_size_per_partition == x.shape[-1]
        return F.linear(x, layer.weight, bias)

    monkeypatch.setattr(linear_module.UnquantizedLinearMethod, "apply", _apply)
    linear = RowParallelLinear(
        input_size=4,
        output_size=3,
        bias=False,
        params_dtype=torch.float32,
        reduce_results=False,
        **PCPOProjTP.linear_shard_kwargs(vllm_config),
    )
    o_proj = PCPOProjTP.maybe_create(linear, vllm_config)
    assert o_proj is not None
    return o_proj, pcp_group


def test_disabled_feature_keeps_plain_linear():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_pcp_o_proj_tp=False)
    )
    assert PCPOProjTP.linear_shard_kwargs(vllm_config) == {}
    assert PCPOProjTP.maybe_create(None, vllm_config) is None


@pytest.mark.parametrize(
    ("has_prefill", "expected"),
    [
        (True, PCPOProjPath.WEIGHT_ALL_GATHER),
        (False, PCPOProjPath.TOKEN_EXCHANGE),
    ],
)
def test_select_path(has_prefill, expected):
    assert PCPOProjTP.select_path(has_prefill) is expected


def test_prefill_waits_for_async_full_weight_gather(monkeypatch):
    o_proj, _ = _make_layer(monkeypatch, pcp_rank=1)
    layer = o_proj.linear
    assert layer.quant_method.supports_pre_processed_weights
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)
    torch.testing.assert_close(layer.weight, full_weight[:, 2:])

    work = _FakeWork()

    def _all_gather_into_tensor(output, input_, group, async_op):
        torch.testing.assert_close(input_, full_weight[:, 2:].transpose(0, 1))
        output.copy_(full_weight.transpose(0, 1))
        assert group is o_proj.pcp_group.device_group
        assert async_op
        return work

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor
    )

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4)
    monkeypatch.setattr(
        pcp_o_proj_module,
        "get_forward_context",
        lambda: SimpleNamespace(global_has_prefill=True),
    )
    assert o_proj.pre_attention_hook() is PCPOProjPath.WEIGHT_ALL_GATHER
    assert not work.waited

    output, output_bias = o_proj(input_)
    assert work.waited
    assert output_bias is None
    torch.testing.assert_close(output, F.linear(input_, full_weight))
    assert layer.input_size_per_partition == 2


def test_forward_requires_explicit_attention_prefetch(monkeypatch):
    o_proj, _ = _make_layer(monkeypatch, pcp_rank=0)

    with pytest.raises(RuntimeError, match="Attention must call"):
        o_proj(torch.zeros(1, 4))


@pytest.mark.parametrize(
    ("global_has_prefill", "local_num_prefills", "expected"),
    [(True, 0, True), (False, 1, False), (None, 1, True), (None, 0, False)],
)
def test_prefetch_decision_uses_global_pcp_batch_type(
    monkeypatch, global_has_prefill, local_num_prefills, expected
):
    monkeypatch.setattr(
        pcp_o_proj_module,
        "get_forward_context",
        lambda: SimpleNamespace(global_has_prefill=global_has_prefill),
    )
    metadata = SimpleNamespace(num_prefills=local_num_prefills)

    assert pcp_o_proj_module.get_pcp_batch_has_prefill(metadata) is expected


def test_decode_slices_features_and_reduces_over_pcp(monkeypatch):
    o_proj, pcp_group = _make_layer(monkeypatch, pcp_rank=1)
    layer = o_proj.linear
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4)
    pcp_group.other_output = F.linear(input_[:, :2], full_weight[:, :2])

    def _no_gather(*args, **kwargs):
        raise AssertionError("decode must not gather O-Proj weights")

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _no_gather)
    monkeypatch.setattr(
        pcp_o_proj_module,
        "get_forward_context",
        lambda: SimpleNamespace(global_has_prefill=False),
    )
    assert o_proj.pre_attention_hook() is PCPOProjPath.TOKEN_EXCHANGE
    output, output_bias = o_proj(input_)

    assert output_bias is None
    torch.testing.assert_close(output, F.linear(input_, full_weight))


def test_quantized_o_proj_method_is_rejected(monkeypatch):
    o_proj, _ = _make_layer(monkeypatch, pcp_rank=0)
    layer = o_proj.linear
    layer.quant_method.quant_method = _UnsupportedLinearMethod()  # type: ignore[attr-defined]

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

    method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    method.spec = QuantSpec(weight=kNvfp4Static, activation=kNvfp4Dynamic)
    method.ctx = CkptCtx(group_size=16)
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
    method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    method.spec = QuantSpec(weight=kNvfp4Static, activation=kNvfp4Dynamic)
    method.ctx = CkptCtx(group_size=16)
    method.kernel = MarlinNvFp4LinearKernel.__new__(MarlinNvFp4LinearKernel)

    with pytest.raises(RuntimeError, match="CUTLASS-compatible"):
        method.get_tp_weight_switch_specs(torch.nn.Module())


@pytest.mark.parametrize("activation,group_size", [(None, 16), (kNvfp4Dynamic, 32)])
def test_modelopt_weight_switch_rejects_unsupported_quant_spec(activation, group_size):
    method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    method.spec = QuantSpec(weight=kNvfp4Static, activation=activation)
    method.ctx = CkptCtx(group_size=group_size)
    method.kernel = CutlassNvFp4LinearKernel.__new__(CutlassNvFp4LinearKernel)

    with pytest.raises(RuntimeError, match="only ModelOpt W4A4 NVFP4"):
        method.get_tp_weight_switch_specs(torch.nn.Module())


def test_modelopt_nvfp4_rejects_padded_local_weight():
    method = ModelOptLinearMethod.__new__(ModelOptLinearMethod)
    method.spec = QuantSpec(weight=kNvfp4Static, activation=kNvfp4Dynamic)
    method.ctx = CkptCtx(group_size=16)
    method.kernel = CutlassNvFp4LinearKernel.__new__(CutlassNvFp4LinearKernel)
    layer = torch.nn.Module()
    layer.input_size_per_partition = 64
    layer.weight = torch.nn.Parameter(
        torch.zeros(128, 64, dtype=torch.uint8), requires_grad=False
    )

    with pytest.raises(RuntimeError, match="no input-axis padding"):
        method.get_tp_weight_switch_specs(layer)


def test_unquantized_method_restores_local_weight_after_kernel_failure(monkeypatch):
    o_proj, _ = _make_layer(monkeypatch, pcp_rank=0)
    layer = o_proj.linear
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)

    def _raise_on_apply(_self, layer, x, bias=None):
        assert layer.input_size_per_partition == x.shape[-1]
        raise RuntimeError("fake linear kernel failure")

    monkeypatch.setattr(linear_module.UnquantizedLinearMethod, "apply", _raise_on_apply)

    def _all_gather_into_tensor(output, input_, group, async_op):
        output.copy_(full_weight.transpose(0, 1))
        return _FakeWork()

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor
    )

    monkeypatch.setattr(
        pcp_o_proj_module,
        "get_forward_context",
        lambda: SimpleNamespace(global_has_prefill=True),
    )
    o_proj.pre_attention_hook()
    with pytest.raises(RuntimeError, match="fake linear kernel failure"):
        o_proj(torch.zeros(1, 4))

    assert layer.weight.shape == (3, 2)
    assert layer.input_size_per_partition == 2
    assert o_proj._path is None
