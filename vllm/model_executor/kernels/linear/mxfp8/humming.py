# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch

from vllm.model_executor.layers.quantization.utils.humming import (
    apply_humming_linear,
    convert_linear_layer_to_humming_standard,
    get_humming_linear_compute_config,
    prepare_humming_linear_layer_config,
    quant_key_to_input_schema,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.humming import create_humming_locks
from vllm.utils.import_utils import has_humming

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class HummingMxfp8LinearKernel(Mxfp8LinearKernel):
    """Humming GEMM Kernel for MXFP8."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Humming only supported on CUDA"

        if not has_humming():
            return False, "Humming is not installed"

        if not current_platform.has_device_capability(75):
            return False, "Humming only supported on SM75+"

        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None and (
            c.bmm_batch_size <= 0 or c.weight_shape[0] % c.bmm_batch_size
        ):
            return False, "BMM groups must evenly divide the weight rows."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.config.bmm_batch_size is not None:
            self._process_bmm_weights(layer)
            return
        layer.weight_scale.data = layer.weight_scale.data.view(torch.float8_e8m0fnu)
        name_map = {"weight": "weight", "weight_scale": "weight_scale"}

        quant_config = {
            "quant_method": "humming",
            "dtype": "float8e4m3",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        }

        convert_linear_layer_to_humming_standard(layer=layer, name_map=name_map)
        input_schema = quant_key_to_input_schema(kMxfp8Dynamic)
        self.layer_config = prepare_humming_linear_layer_config(
            layer, quant_config, input_schema=input_schema
        )
        self.compute_config = get_humming_linear_compute_config()
        self.locks = create_humming_locks(layer.weight.device)

    def _process_bmm_weights(self, layer: torch.nn.Module) -> None:
        """Pack each BMM group as one Humming expert."""
        from vllm.utils.humming import (
            HummingWeightSchema,
            prepare_and_transform_humming_tensors,
        )

        groups = self.config.bmm_batch_size
        assert groups is not None
        n, k = self.config.weight_shape
        tensors = {
            "weight": layer.weight.view(groups, n // groups, k).view(torch.int32),
            "weight_scale": layer.weight_scale.view(groups, n // groups, k // 32).view(
                torch.float8_e8m0fnu
            ),
        }
        # Keep the grouped projection's BF16/FP16 activations unquantized.
        self.layer_config, tensors = prepare_and_transform_humming_tensors(
            tensors=tensors,
            shape_n=n // groups,
            shape_k=k,
            num_experts=groups,
            weight_schema=HummingWeightSchema(
                b_dtype="float8e4m3",
                bs_dtype="float8e8m0",
                weight_scale_group_size=32,
            ),
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=layer.params_dtype,
            device=layer.weight.device,
        )
        for name, tensor in tensors.items():
            replace_parameter(layer, name, tensor)
        compute_config = json.loads(get_humming_linear_compute_config())
        compute_config["gemm_type"] = "grouped_masked"
        self.compute_config = json.dumps(compute_config)
        self.locks = create_humming_locks(layer.weight.device)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.bmm_batch_size is not None:
            from vllm.utils.humming import humming_forward

            tokens, groups, k = x.shape
            assert groups == self.config.bmm_batch_size
            output = (
                humming_forward(
                    self.layer_config,
                    inputs=x.transpose(0, 1).reshape(groups * tokens, k),
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    weight_scale_2=getattr(layer, "weight_scale_2", None),
                    expert_layout=torch.full(
                        (groups,), tokens, dtype=torch.int32, device=x.device
                    ),
                    locks=self.locks,
                    compute_config=self.compute_config,
                )
                .view(groups, tokens, -1)
                .transpose(0, 1)
            )
            if bias is not None:
                output = output + bias.view(groups, -1)
            return output
        return apply_humming_linear(
            layer,
            x,
            skip_bias_add=bias is None,
            layer_config=self.layer_config,
            compute_config=self.compute_config,
            locks=self.locks,
        )
