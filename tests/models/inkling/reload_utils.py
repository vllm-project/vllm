# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a real Inkling MoE layer, load it, reload it layerwise, and compare it
with a cold start on the same weights."""

from collections.abc import Callable

import torch
from torch import nn

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.inkling.configs import InklingModelConfig
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
            finalize_layerwise_reload(self.model, model_config=None)

    def tensors(self) -> dict[str, torch.Tensor]:
        """The routed and sink experts' tensors (the gate is not loaded here)."""
        modules = {"experts": self.moe.experts, "sink": self.moe.sink_experts}
        return {
            f"{prefix}.{name}": t
            for prefix, module in modules.items()
            for name, t in [*module.named_parameters(), *module.named_buffers()]
        }


def assert_reload_matches_cold_start(
    make_layer: Callable[[], InklingMoELayer], checkpoint: Callable[[int], Checkpoint]
) -> InklingMoELayer:
    layer = make_layer()
    layer.load(checkpoint(0))
    layer.process()
    ptrs = {name: t.data_ptr() for name, t in layer.tensors().items()}

    layer.reload(checkpoint(1))

    fresh = make_layer()
    fresh.load(checkpoint(1))
    fresh.process()
    got, want = layer.tensors(), fresh.tensors()
    assert got.keys() == want.keys()
    for name in want:
        assert torch.equal(got[name], want[name]), name
    assert {name: t.data_ptr() for name, t in got.items()} == ptrs
    return layer


def sink_checkpoint(gen: torch.Generator, intermediate: int) -> Checkpoint:
    return {
        "shared_experts.shared_w13_weight": torch.randn(
            N_SINK, 2 * intermediate, HIDDEN, generator=gen
        ).bfloat16(),
        "shared_experts.shared_w2_weight": torch.randn(
            N_SINK, HIDDEN, intermediate, generator=gen
        ).bfloat16(),
    }
