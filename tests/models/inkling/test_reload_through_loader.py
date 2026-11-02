# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling expert writes go through the parameter's ``weight_loader``, so a
layerwise reload (which restores layers on meta and replays wrapped loaders)
sees them: after a reload with new weights the layer equals a cold start on
those weights."""

import pytest
import torch
from torch import nn

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    get_layerwise_info,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.inkling.common.weights import (
    copy_padded,
    deinterleave_into,
    write_loader,
)
from vllm.models.inkling.configs import InklingModelConfig
from vllm.models.inkling.nvidia import moe as nvidia_moe
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

N_ROUTED, N_SINK, HIDDEN = 4, 1, 128
Checkpoint = dict[str, torch.Tensor]


class InklingMoELayer:
    """A real ``InklingMoE`` as the model builds it, plus its loader calls."""

    def __init__(
        self,
        moe_cls: type[nn.Module],
        intermediate: int,
        quant_config: QuantizationConfig | None,
        moe_backend: str,
    ) -> None:
        self.vllm_config = VllmConfig(
            kernel_config=KernelConfig(moe_backend=moe_backend)
        )
        config = InklingModelConfig(
            hidden_size=HIDDEN,
            intermediate_size=intermediate,
            n_routed_experts=N_ROUTED,
            n_shared_experts=N_SINK,
            num_experts_per_tok=2,
            shared_expert_sink=True,
        )
        with (
            set_current_vllm_config(self.vllm_config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
        ):
            self.moe = moe_cls(
                config, prefix="model.layers.0.mlp", quant_config=quant_config
            )
            self.model = nn.Sequential(self.moe)
            # as in model loading: a reload restores layers on this device
            record_metadata_for_reloading(self.model)

    def load(self, checkpoint: Checkpoint) -> None:
        # weights load under the model dtype, as in DefaultModelLoader
        with (
            set_current_vllm_config(self.vllm_config),
            set_default_torch_dtype(torch.bfloat16),
        ):
            for name, weight in checkpoint.items():
                self.moe.load_expert_weight(name, weight)

    def process(self) -> None:
        # the per-module part of process_weights_after_loading
        with set_current_vllm_config(self.vllm_config):
            for module in self.model.modules():
                quant_method = getattr(module, "quant_method", None)
                if isinstance(quant_method, QuantizeMethodBase):
                    quant_method.process_weights_after_loading(module)

    def reload(self, checkpoint: Checkpoint) -> None:
        with set_current_vllm_config(self.vllm_config):
            initialize_layerwise_reload(self.model)
            self.load(checkpoint)
            # every routed element was counted: processed during the load,
            # not left to finalize
            routed = self.moe.experts.routed_experts
            assert get_layerwise_info(routed).load_numel_total is None
            finalize_layerwise_reload(self.model, model_config=None)

    def tensors(self) -> dict[str, torch.Tensor]:
        """The routed and sink experts' tensors (the gate is not loaded here)."""
        modules = {"experts": self.moe.experts, "sink": self.moe.sink_experts}
        return {
            f"{prefix}.{name}": t
            for prefix, module in modules.items()
            for name, t in [*module.named_parameters(), *module.named_buffers()]
        }


NVFP4_INTERMEDIATE = 128


def _nvfp4_checkpoint(seed: int) -> Checkpoint:
    gen = torch.Generator().manual_seed(seed)
    e, h, i = N_ROUTED, HIDDEN, NVFP4_INTERMEDIATE

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    def positive(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=gen) + 0.5

    return {
        # routed experts; w13 rows interleaved [g0, u0, g1, u1, ...]
        "experts.w13_weight": packed(e, 2 * i, h // 2),
        "experts.w13_weight_scale": positive(e, 2 * i, h // 16).to(torch.float8_e4m3fn),
        "experts.w13_weight_scale_2": positive(e) / 100,
        "experts.w13_weight.input_amax": positive(e) * 7,
        "experts.w2_weight": packed(e, h, i // 2),
        "experts.w2_weight_scale": positive(e, h, i // 16).to(torch.float8_e4m3fn),
        "experts.w2_weight_scale_2": positive(e) / 100,
        "experts.w2_weight.input_amax": positive(e) * 7,
        # bf16 sink experts
        "shared_experts.shared_w13_weight": torch.randn(
            N_SINK, 2 * i, h, generator=gen
        ).bfloat16(),
        "shared_experts.shared_w2_weight": torch.randn(
            N_SINK, h, i, generator=gen
        ).bfloat16(),
    }


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_nvfp4_moe_reload_matches_cold_start(dist_init):
    """Routed NVFP4 experts (weights, block scales, per-expert global scales,
    calibrated input scales) and the sink experts take the new weights."""
    quant_config = ModelOptNvFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
            }
        }
    )

    def make_layer() -> InklingMoELayer:
        return InklingMoELayer(
            nvidia_moe.InklingMoE, NVFP4_INTERMEDIATE, quant_config, "emulation"
        )

    layer = make_layer()
    checkpoint = _nvfp4_checkpoint(0)
    layer.load(checkpoint)
    # the calibrated input scale is float32 even under a bf16 default dtype
    for proj in ("w13", "w2"):
        amax = float(checkpoint[f"experts.{proj}_weight.input_amax"].max())
        scale = getattr(layer.moe.experts.routed_experts, f"{proj}_input_scale")
        assert torch.equal(scale, torch.full_like(scale, amax / (448.0 * 6.0)))
    layer.process()
    ptrs = {name: t.data_ptr() for name, t in layer.tensors().items()}

    layer.reload(_nvfp4_checkpoint(1))

    fresh = make_layer()
    fresh.load(_nvfp4_checkpoint(1))
    fresh.process()
    got, want = layer.tensors(), fresh.tensors()
    assert got.keys() == want.keys()
    for name in want:
        assert torch.equal(got[name], want[name]), name
    assert {name: t.data_ptr() for name, t in got.items()} == ptrs


class _PaddedExperts(nn.Module):
    """AITER-style padded MXFP4 parameters (ROCm only, so not built here):
    each gate/up half and the w2 columns are wider than the checkpoint's, and
    every element counts. Weights pad with zero, scales with one."""

    E, ROWS, HALF, COLS, WIDTH = 2, 4, 6, 3, 4

    def __init__(self) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(self.E, 2 * self.HALF, self.WIDTH), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(
            torch.ones(self.E, self.WIDTH, self.WIDTH), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = write_loader

    def load(self) -> None:
        gen = torch.Generator().manual_seed(0)
        for slot in range(self.E):
            slab = torch.randn(2 * self.ROWS, self.WIDTH, generator=gen)
            self.w13_weight.weight_loader(
                self.w13_weight, slab, deinterleave_into(slot, self.HALF, 0.0)
            )
            scale = torch.rand(self.WIDTH, self.COLS, generator=gen)
            self.w2_weight_scale.weight_loader(
                self.w2_weight_scale, scale, copy_padded(slot, self.COLS, 1.0)
            )


def test_padded_slots_reload_completes_and_keeps_padding():
    experts = _PaddedExperts()
    model = nn.Sequential(experts)
    experts.load()
    cold = [p.data.clone() for p in experts.parameters()]

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    experts.load()
    # every element was written, so the layer was processed as it loaded
    # instead of being left (uninitialized padding) for finalize
    assert experts.w13_weight.device.type != "meta"
    finalize_layerwise_reload(model, model_config=None)

    for param, expected in zip(experts.parameters(), cold):
        assert torch.equal(param.data, expected)


def test_lora_sink_experts_reload_takes_new_weights(dist_init):
    """The LoRA-capable sink experts write through each linear parameter's own
    ``weight_loader`` (the one a reload wraps), not the module's."""

    def checkpoint(seed: int) -> Checkpoint:
        gen = torch.Generator().manual_seed(seed)
        return {
            "w13_weight": torch.randn(2, 6, 4, generator=gen),
            "w2_weight": torch.randn(2, 4, 3, generator=gen),
        }

    def loaded_sink(weights: Checkpoint) -> nn.Module:
        with set_default_torch_dtype(torch.float32):
            sink = nvidia_moe.InklingSinkExpertsLinear(n_experts=2, d_model=4, d_mlp=3)
        for key, weight in weights.items():
            sink.load_weight(key, weight)
        return sink

    sink = loaded_sink(checkpoint(0))
    model = nn.Sequential(sink)
    record_metadata_for_reloading(model)

    initialize_layerwise_reload(model)
    for key, weight in checkpoint(1).items():
        sink.load_weight(key, weight)
    finalize_layerwise_reload(model, model_config=None)

    want = dict(loaded_sink(checkpoint(1)).named_parameters())
    got = dict(sink.named_parameters())
    assert got.keys() == want.keys()
    for name in want:
        assert torch.equal(got[name], want[name]), name


@pytest.mark.parametrize("platform", ["nvidia", "amd"])
def test_mtp_writes_go_through_weight_loader(platform):
    import importlib

    mtp = importlib.import_module(f"vllm.models.inkling.{platform}.mtp")
    module = nn.Module()
    module.model = nn.Module()
    module.model.num_mtp_layers = 1
    module.model.chain_norm = None
    module.model.layers = nn.ModuleList([nn.Module()])
    norm = nn.Parameter(torch.zeros(4), requires_grad=False)
    module.model.layers[0].norm = nn.Module()
    module.model.layers[0].norm.weight = norm
    calls = []

    def weight_loader(param: torch.Tensor, weight: torch.Tensor) -> None:
        calls.append(param)
        param.copy_(weight)

    norm.weight_loader = weight_loader

    # same shape as the parameter: a layerwise reload wraps `weight_loader`,
    # so the write must go through it
    mtp._load_inkling_mtp_weights(
        module, [("model.mtp.layers.0.norm.weight", torch.ones(4))]
    )

    assert calls == [norm]
    torch.testing.assert_close(norm, torch.ones(4))
