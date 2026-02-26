# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE model layers.

Validates vLLM MoE layer implementations against HuggingFace reference models.

Run `pytest tests/kernels/moe/test_moe_model_layers.py`.
"""

import importlib
import importlib.machinery
import sys
import types

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.model_executor.model_loader.weight_utils import (
    get_quant_config,
    initialize_single_dummy_weight,
)
from vllm.model_executor.models.nemotron_h import NemotronHMoE
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype, set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager


def _install_mamba_ssm_stub_if_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal mamba_ssm stub for HF Nemotron import in MoE-only tests."""
    if "mamba_ssm.ops.triton.layernorm_gated" in sys.modules:
        return

    try:
        importlib.import_module("mamba_ssm.ops.triton.layernorm_gated")
        return
    except ImportError:
        pass

    from transformers.utils import import_utils as hf_import_utils

    monkeypatch.setattr(hf_import_utils, "is_mamba_2_ssm_available", lambda: False)

    def _new_pkg_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=True)
        spec.submodule_search_locations = []
        module.__spec__ = spec
        module.__dict__["__path__"] = []
        module.__package__ = name
        return module

    def _new_module(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(
            name=name, loader=None, is_package=False
        )
        module.__package__ = name.rpartition(".")[0]
        return module

    def _unsupported_rmsnorm_fn(*args, **kwargs):
        raise RuntimeError("mamba_ssm stub was invoked in a non-MoE code path")

    mamba_ssm_module = _new_pkg_module("mamba_ssm")
    ops_module = _new_pkg_module("mamba_ssm.ops")
    triton_module = _new_pkg_module("mamba_ssm.ops.triton")
    layernorm_module = _new_module("mamba_ssm.ops.triton.layernorm_gated")
    layernorm_module.__dict__["rmsnorm_fn"] = _unsupported_rmsnorm_fn

    mamba_ssm_module.__dict__["ops"] = ops_module
    ops_module.__dict__["triton"] = triton_module
    triton_module.__dict__["layernorm_gated"] = layernorm_module

    monkeypatch.setitem(sys.modules, "mamba_ssm", mamba_ssm_module)
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", ops_module)
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton", triton_module)
    monkeypatch.setitem(
        sys.modules, "mamba_ssm.ops.triton.layernorm_gated", layernorm_module
    )


def _initialize_nemotron_dummy_weights(vllm_layer, model_name: str) -> None:
    for name, param in vllm_layer.named_parameters():
        if model_name.endswith("NVFP4"):
            if param.dtype == torch.uint8:
                param.data.copy_(
                    torch.randint(
                        low=0,
                        high=256,
                        size=tuple(param.shape),
                        dtype=torch.uint8,
                        device=param.device,
                    )
                )
                continue

            if "input_scale" in name:
                param.data.fill_(1.0)
                continue

            if "scale" in name:
                param.data.copy_(
                    (
                        0.05
                        + (
                            torch.rand_like(
                                param, dtype=torch.float32, device=param.device
                            )
                            / 10
                        )
                    ).to(param.dtype)  # noqa: E501
                )
                continue

        initialize_single_dummy_weight(param, low=-3e-1, high=3e-1)


def _load_nemotron_vllm_weights_to_hf_model_fp8(vllm_layer, hf_layer):
    hf_layer.gate.weight.data[:] = vllm_layer.gate.weight.data
    hf_layer.gate.e_score_correction_bias.data[:] = (
        vllm_layer.gate.e_score_correction_bias.data
    )

    for i in range(vllm_layer.experts.w13_weight.shape[0]):
        w13 = (
            vllm_layer.experts.w13_weight[i].to(torch.float)
            * vllm_layer.experts.w13_weight_scale[i]
        )
        w2 = (
            vllm_layer.experts.w2_weight[i].to(torch.float)
            * vllm_layer.experts.w2_weight_scale[i]
        )
        hf_layer.experts[i].up_proj.weight.data[:] = w13
        hf_layer.experts[i].down_proj.weight.data[:] = w2

    up_proj = (
        vllm_layer.shared_experts.up_proj.weight.data.to(torch.float)
        * vllm_layer.shared_experts.up_proj.weight_scale
    )
    down_proj = (
        vllm_layer.shared_experts.down_proj.weight.data.to(torch.float)
        * vllm_layer.shared_experts.down_proj.weight_scale
    )
    hf_layer.shared_experts.up_proj.weight.data[:] = up_proj
    hf_layer.shared_experts.down_proj.weight.data[:] = down_proj


def _dquantize_nvfp4(weights, scales, global_scales, dest_shape):
    quant_blocksize = 16
    rows = dest_shape[-2]
    cols = dest_shape[-1]
    return (
        (
            break_fp4_bytes(weights, torch.float32).reshape(
                rows, cols // quant_blocksize, quant_blocksize
            )
            * (
                scales.view(torch.float8_e4m3fn).to(torch.float32) / (1 / global_scales)
            ).unsqueeze(-1)
        )
        .reshape(rows, cols)
        .to(torch.float32)
    )


def _load_nemotron_vllm_weights_to_hf_model_nvfp4(vllm_layer, hf_layer):
    hf_layer.gate.weight.data[:] = vllm_layer.gate.weight.data
    hf_layer.gate.e_score_correction_bias.data[:] = (
        vllm_layer.gate.e_score_correction_bias.data
    )

    for i in range(vllm_layer.experts.w13_weight.shape[0]):
        w13 = _dquantize_nvfp4(
            weights=vllm_layer.experts.w13_weight[i],
            scales=vllm_layer.experts.w13_weight_scale[i],
            global_scales=vllm_layer.experts.w13_weight_scale_2[i],
            dest_shape=hf_layer.experts[i].up_proj.weight.shape,
        )
        w2 = _dquantize_nvfp4(
            weights=vllm_layer.experts.w2_weight[i],
            scales=vllm_layer.experts.w2_weight_scale[i],
            global_scales=vllm_layer.experts.w2_weight_scale_2[i],
            dest_shape=hf_layer.experts[i].down_proj.weight.shape,
        )
        hf_layer.experts[i].up_proj.weight.data[:] = w13
        hf_layer.experts[i].down_proj.weight.data[:] = w2

    up_proj = _dquantize_nvfp4(
        weights=vllm_layer.shared_experts.up_proj.weight.data,
        scales=vllm_layer.shared_experts.up_proj.weight_scale.data,
        global_scales=vllm_layer.shared_experts.up_proj.weight_scale_2.data,
        dest_shape=hf_layer.shared_experts.up_proj.weight.shape,
    )
    down_proj = _dquantize_nvfp4(
        weights=vllm_layer.shared_experts.down_proj.weight.data,
        scales=vllm_layer.shared_experts.down_proj.weight_scale.data,
        global_scales=vllm_layer.shared_experts.down_proj.weight_scale_2.data,
        dest_shape=hf_layer.shared_experts.down_proj.weight.shape,
    )
    hf_layer.shared_experts.up_proj.weight.data[:] = up_proj
    hf_layer.shared_experts.down_proj.weight.data[:] = down_proj


@pytest.mark.parametrize(
    "model_name",
    [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    ],
)
@pytest.mark.parametrize(
    "flashinfer_backend", ["latency", "throughput"], ids=["trtllm", "cutlass"]
)
@torch.inference_mode()
def test_nemotron_flashinfer_moe(model_name, flashinfer_backend, monkeypatch):
    batch_size = 256

    if not current_platform.has_device_capability(100):
        pytest.skip("Test is only supported for sm >= 100")
    if (
        model_name == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
        and flashinfer_backend != "throughput"
    ):
        pytest.skip("BF16 model only supported with throughput backend")
    set_random_seed(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    if model_name.endswith("FP8"):
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    if model_name.endswith("NVFP4"):
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", flashinfer_backend)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12345")
    init_distributed_environment()
    init_workspace_manager(torch.cuda.current_device())
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )

    model_config = ModelConfig(
        model=model_name,
        trust_remote_code=True,
    )
    nemotron_config = model_config.hf_config
    load_config = LoadConfig(load_format="dummy")
    layer_quant_config = None
    if model_config.quantization is not None:
        layer_quant_config = get_quant_config(model_config, load_config)

    inner_vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1, tensor_parallel_size=1
        ),
        load_config=load_config,
        quant_config=layer_quant_config,
    )
    inner_vllm_config.compilation_config.fast_moe_cold_start = False
    with (
        set_forward_context({}, inner_vllm_config),
        set_current_vllm_config(inner_vllm_config),
    ):
        hidden_states = (
            torch.randn(
                (batch_size, nemotron_config.hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 10
        )

        with set_default_torch_dtype(torch.bfloat16):
            vllm_layer = NemotronHMoE(
                nemotron_config,
                quant_config=inner_vllm_config.quant_config,
                parallel_config=inner_vllm_config.parallel_config,
            ).cuda()
        _initialize_nemotron_dummy_weights(vllm_layer, model_name)

        # Mock mambas_ssm module instead of installing it.
        _install_mamba_ssm_stub_if_missing(monkeypatch)
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
        first_expert_layer_index = hf_config.hybrid_override_pattern.find("E")
        ref_hf_layer = hf_model.backbone.layers[first_expert_layer_index].mixer.cuda()

        # Load weights from our layer to the reference hf layer before
        # post-processing, since some of the weights get padded and size changes.
        if model_name.endswith("NVFP4"):
            _load_nemotron_vllm_weights_to_hf_model_nvfp4(vllm_layer, ref_hf_layer)
        else:
            _load_nemotron_vllm_weights_to_hf_model_fp8(vllm_layer, ref_hf_layer)
        vllm_layer.gate.quant_method.process_weights_after_loading(vllm_layer.gate)
        vllm_layer.experts.quant_method.process_weights_after_loading(
            vllm_layer.experts
        )
        vllm_layer.shared_experts.up_proj.quant_method.process_weights_after_loading(
            vllm_layer.shared_experts.up_proj
        )
        vllm_layer.shared_experts.down_proj.quant_method.process_weights_after_loading(
            vllm_layer.shared_experts.down_proj
        )

        actaul_output = vllm_layer(hidden_states)

        ref_output = ref_hf_layer(hidden_states)
        torch.testing.assert_close(ref_output, actaul_output, rtol=1e-2, atol=1e-1)
