# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PWAL keeps the parameter objects (class and loader attributes) it converts."""

from collections.abc import Callable

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.distributed.localization import LocalizedResources
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    _LOCALIZED_SHARD_PARAMS,
    _shard_nvfp4_moe_weights_per_die,
)
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.reload.meta import materialize_meta_tensor
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="NVFP4/MXFP8 cases select the converted kernels on SM100.",
)

SIZE = 512


def _load(layer: torch.nn.Module) -> None:
    """Stand in for the weight loader: materialize and fill with finite values."""
    for name, param in list(layer.named_parameters(recurse=False)):
        if param.is_meta:
            param = materialize_meta_tensor(param)
            setattr(layer, name, param)
        if param.dtype == torch.uint8:  # packed FP4 values or E8M0 scales
            param.data.copy_(torch.randint(120, 130, param.shape))
        elif param.dtype.is_floating_point:
            param.data.copy_(torch.rand(param.shape) + 0.5)


def _nbytes(param: torch.Tensor) -> int:
    return param.numel() * param.element_size()


def _dims(param: torch.Tensor) -> tuple:
    return tuple(
        getattr(param, k, None) for k in ("input_dim", "output_dim", "packed_dim")
    )


def _linear(
    quant_config: QuantizationConfig,
    output_size: int = SIZE,
    params_dtype: torch.dtype | None = None,
):
    layer = ReplicatedLinear(
        SIZE,
        output_size,
        bias=False,
        quant_config=quant_config,
        params_dtype=params_dtype,
    )
    return layer, layer.quant_method


def _online_linear():
    layer = torch.nn.Module()
    layer.input_size = layer.input_size_per_partition = SIZE
    layer.output_size = layer.output_size_per_partition = SIZE
    layer.params_dtype = torch.bfloat16
    method = Mxfp4OnlineLinearMethod()
    method.create_weights(
        layer,
        input_size_per_partition=SIZE,
        output_partition_sizes=[SIZE],
        input_size=SIZE,
        output_size=SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    return layer, method


def _moe(quant_config: QuantizationConfig | None, **kwargs):
    layer = FusedMoEFactory(
        num_experts=8,
        top_k=2,
        hidden_size=SIZE,
        intermediate_size=SIZE,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix="model.layers.0.mlp.experts",
        **kwargs,
    ).routed_experts
    return layer, layer.quant_method


def _online_moe():
    layer, _ = _moe(None)
    method = Mxfp4OnlineMoEMethod(moe=layer.moe_config)
    method.create_weights(
        layer=layer,
        num_experts=8,
        hidden_size=SIZE,
        intermediate_size_per_partition=SIZE,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    return layer, method


def _ct(fmt: str, group: str, group_size: int, input_activations: bool):
    args = dict(num_bits=4, type="float", strategy=group, group_size=group_size)
    args |= dict(symmetric=True, dynamic=False)
    acts = args | dict(dynamic="local") if input_activations else None
    group_0 = dict(targets=["Linear"], weights=args, input_activations=acts)
    return CompressedTensorsConfig.from_config(
        dict(format=fmt, config_groups=dict(group_0=group_0), ignore=[])
    )


NVFP4 = dict(is_checkpoint_nvfp4_serialized=True, kv_cache_quant_algo=None)
CT_NVFP4 = ("nvfp4-pack-quantized", "tensor_group", 16, True)
CT_MXFP4 = ("mxfp4-pack-quantized", "group", 32, False)

CASES: list[Callable] = [
    lambda: _linear(Fp8Config(weight_block_size=[128, 128])),
    # A partial trailing 128-block is padded in post-processing (GLM-style).
    lambda: _linear(
        ModelOptFp8Config(
            quant_method="FP8_PB_WO",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        ),
        output_size=SIZE + 64,
    ),
    lambda: _linear(ModelOptMxFp8Config.from_config({"quant_method": "mxfp8"})),
    # The consumed per-tensor scales stay registered with their data.
    lambda: _linear(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
    # The packed weights are also registered under their runtime names.
    lambda: _linear(_ct(*CT_NVFP4)),
    lambda: _linear(_ct(*CT_MXFP4)),
    _online_linear,
    lambda: _moe(None),
    lambda: _moe(Fp8Config(weight_block_size=[128, 128])),
    lambda: _moe(ModelOptMxFp8Config.from_config({"quant_method": "mxfp8"})),
    lambda: _moe(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
    lambda: _moe(_ct(*CT_NVFP4)),
    lambda: _moe(_ct(*CT_MXFP4)),
    lambda: _moe(Mxfp4Config()),
    _online_moe,
]
IDS = [
    "fp8-block-linear",
    "modelopt-fp8-pb-wo-linear",
    "modelopt-mxfp8-linear",
    "modelopt-nvfp4-linear",
    "ct-nvfp4-linear",
    "ct-mxfp4-linear",
    "online-mxfp4-linear",
    "unquantized-moe",
    "fp8-block-moe",
    "modelopt-mxfp8-moe",
    "modelopt-nvfp4-moe",
    "ct-nvfp4-moe",
    "ct-mxfp4-moe",
    "mxfp4-moe",
    "online-mxfp4-moe",
]


def _check_keeps_param_objects(build: Callable, vllm_config) -> None:
    vllm_config.model_config = ModelConfig()
    with torch.device("cuda"):
        layer, method = build()
        _load(layer)
        before = {
            name: (param, set(param.__dict__), _dims(param), _nbytes(param))
            for name, param in layer.named_parameters(recurse=False)
        }
        method.process_weights_after_loading(layer)

    after = dict(layer.named_parameters(recurse=False, remove_duplicate=False))
    for name, (param, attrs, dims, nbytes) in before.items():
        assert after.get(name) is param, name
        assert attrs <= param.__dict__.keys(), name
        # The loader shards by these on reload, so they keep the checkpoint layout.
        assert _dims(param) == dims, name
        assert callable(getattr(param, "weight_loader", None)), name
        # Online quantization frees the high-precision source by design.
        if not method.uses_meta_device:
            kept = after[name]
            offset = kept.storage_offset() * kept.element_size()
            assert kept.untyped_storage().nbytes() - offset >= nbytes, name
    # Two names share storage only as one param object.
    objs = {id(p): p for p in after.values() if p.numel()}
    ptrs = [p.untyped_storage().data_ptr() for p in objs.values()]
    assert len(ptrs) == len(set(ptrs))


@pytest.mark.parametrize("build", CASES, ids=IDS)
def test_process_weights_after_loading_keeps_param_objects(
    build: Callable, default_vllm_config, dist_init
) -> None:
    _check_keeps_param_objects(build, default_vllm_config)


FORCED_CASES: list[Callable] = [
    lambda: _linear(
        ModelOptNvFp4Config(**NVFP4, exclude_modules=[]), params_dtype=torch.bfloat16
    ),
    lambda: _linear(_ct(*CT_NVFP4), params_dtype=torch.bfloat16),
    lambda: _moe(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
    lambda: _moe(_ct(*CT_NVFP4)),
]
FORCED_IDS = [
    "modelopt-nvfp4-linear",
    "ct-nvfp4-linear",
    "modelopt-nvfp4-moe",
    "ct-nvfp4-moe",
]


@pytest.mark.parametrize("backend", ["marlin", "humming"])
@pytest.mark.parametrize("build", FORCED_CASES, ids=FORCED_IDS)
def test_forced_backend_keeps_param_objects(
    build: Callable, backend: str, default_vllm_config, dist_init
) -> None:
    """Weight-only backends leave the input scales unused but registered."""
    if backend == "humming" and not has_humming():
        pytest.skip("humming is not installed")
    default_vllm_config.kernel_config.linear_backend = backend
    default_vllm_config.kernel_config.moe_backend = backend
    _check_keeps_param_objects(build, default_vllm_config)


def _kernel_tensor_ptrs(layer: torch.nn.Module, method) -> dict[str, int]:
    """Addresses of the tensors the MoE kernel reads: the layer's, the experts
    object's and the quant config's."""
    config = method.moe_quant_config
    holders = {
        "layer": layer,
        "experts": method.moe_kernel.fused_experts,
        **{f"config.{q}": getattr(config, q) for q in ("_a1", "_a2", "_w1", "_w2")},
    }
    ptrs = {f"layer.{n}": p.data_ptr() for n, p in layer.named_parameters()}
    for holder_name, holder in holders.items():
        for attr, value in vars(holder).items():
            if isinstance(value, torch.Tensor) and value.numel():
                ptrs[f"{holder_name}.{attr}"] = value.data_ptr()
    return ptrs


GPT_OSS_SWIGLU = dict(
    activation="swigluoai", swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0
)
RELOAD_CASES = [
    pytest.param(
        lambda: _moe(Mxfp4Config(), **GPT_OSS_SWIGLU),
        "flashinfer_trtllm",
        id="mxfp4-trtllm",
    ),
    pytest.param(
        lambda: _moe(_ct(*CT_NVFP4)), "flashinfer_cutedsl", id="ct-nvfp4-cutedsl"
    ),
    pytest.param(
        lambda: _moe(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
        "flashinfer_trtllm",
        id="modelopt-nvfp4-trtllm",
    ),
    pytest.param(
        lambda: _moe(ModelOptNvFp4Config(**NVFP4, exclude_modules=[])),
        "flashinfer_cutlass",
        id="modelopt-nvfp4-cutlass",
    ),
    pytest.param(
        lambda: _moe(Fp8Config(activation_scheme="static")),
        "flashinfer_cutlass",
        id="fp8-static-cutlass",
    ),
    pytest.param(
        lambda: _moe(Fp8Config(weight_block_size=[128, 128])),
        "flashinfer_trtllm",
        id="fp8-block-trtllm",
    ),
]


@pytest.mark.parametrize("build,backend", RELOAD_CASES)
def test_direct_reload_keeps_kernel_tensor_addresses(
    build: Callable, backend: str, default_vllm_config, dist_init, monkeypatch
) -> None:
    """Post-load processing reruns on every reload and rebuilds the experts
    object; every tensor its kernel reads keeps the cold-start address."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.moe_backend = backend
    model_config, device = default_vllm_config.model_config, torch.device("cuda")
    with device:
        layer, method = build()
    _load(layer)
    process_weights_after_loading(layer, model_config, device)
    cold = _kernel_tensor_ptrs(layer, method)

    begin_direct_reload(layer)
    _load(layer)
    finish_direct_reload(layer, model_config, device)
    assert _kernel_tensor_ptrs(layer, method) == cold


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA mem pools")
def test_localized_nvfp4_shards_refilled_in_place():
    """CuteDSL NVFP4 per-locality-domain shards: a rerun refills the registered
    shard parameters, and so the MMA views of their scales, in place."""
    res = LocalizedResources(
        num_split=2,
        contexts=[],
        streams=[torch.cuda.Stream(), torch.cuda.Stream()],
        pools=[torch.cuda.MemPool(), torch.cuda.MemPool()],
    )
    layer = torch.nn.Module()

    def shard(fill: int) -> list[dict]:
        def full(shape, dtype):
            t = torch.full(shape, fill, dtype=torch.uint8, device="cuda")
            return t.view(dtype)

        return _shard_nvfp4_moe_weights_per_die(
            layer,
            res,
            full((2, 256, 32), torch.uint8),
            full((2, 256, 4), torch.float8_e4m3fn),
            full((2, 256, 64), torch.uint8),
            full((2, 256, 8), torch.float8_e4m3fn),
        )

    first = shard(1)
    names = [f"localized{i}_{n}" for i in range(2) for n in _LOCALIZED_SHARD_PARAMS]
    params = {n: getattr(layer, n) for n in names}
    ptrs = {n: p.data_ptr() for n, p in params.items()}
    views = [s["w13_weight_scale"].data_ptr() for s in first]
    second = shard(2)
    assert [n for n in names if getattr(layer, n) is not params[n]] == []
    assert {n: p.data_ptr() for n, p in params.items()} == ptrs
    assert [s["w13_weight_scale"].data_ptr() for s in second] == views
    assert torch.equal(
        layer.localized0_w13_weight, torch.full_like(layer.localized0_w13_weight, 2)
    )
