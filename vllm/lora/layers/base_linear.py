# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device

logger = init_logger(__name__)


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        # Ensure tp_size and tp_rank consistency with the base_layer.
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(self.base_layer)
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int

        # NEW: Check if base layer is INT4 quantized
        self._is_int4_quantized = self._check_int4_quantization()
        self._materialized_weight: torch.Tensor | None = None

        if self._is_int4_quantized:
            logger.info(
                "LoRA layer initialized with INT4 quantized base layer. "
                "Materializing FP16 weights for LoRA compatibility."
            )
            # Materialize FP16 weights from packed INT4 buffers
            # This creates LoRA-compatible weight tensors alongside packed buffers
            self._materialize_int4_weights()

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_config = lora_config
        #
        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (
                lora_config.max_lora_rank
                if not lora_config.fully_sharded_loras
                else divide(lora_config.max_lora_rank, self.tp_size)
            )
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (
                self.output_size
                if not lora_config.fully_sharded_loras
                else divide(self.output_size, self.tp_size)
            )
        else:
            raise NotImplementedError

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.output_slices = (self.lora_b_stacked[0].shape[2],)

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: torch.Tensor | None,
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert (
            len(self.lora_a_stacked) == len(self.lora_b_stacked) == self.n_slices == 1
        )

        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(
            lora_a, non_blocking=True
        )
        self.lora_b_stacked[0][index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(
            lora_b, non_blocking=True
        )

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # For INT4 quantized layers:
        # 1. Materialized FP16 weights (via self.weight property) allow LoRA attachment
        # 2. Base forward pass uses optimized INT4 kernels via quant_method.apply()
        # 3. LoRA delta is computed on activations and added to INT4 kernel output
        # This hybrid approach maintains INT4 inference efficiency while supporting LoRA
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        # In Transformers modeling backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        # Apply LoRA: computes x @ lora_A @ lora_B and adds to output
        # For INT4 layers, this effectively applies: INT4_kernel(x) + x @ LoRA_AB
        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        return output

    @property
    def weight(self) -> torch.Tensor:
        # For INT4 quantized layers, return materialized FP16 weights if available
        # This allows LoRA to attach to a proper weight tensor
        if self._is_int4_quantized and self._materialized_weight is not None:
            return self._materialized_weight

        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        # HQQ marlin
        elif hasattr(self.base_layer, "W_q"):
            return self.base_layer.W_q
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None

    def _check_int4_quantization(self) -> bool:
        """
        Check if the base layer is using INT4 quantization.

        Returns:
            True if base layer has INT4 packed weights
        """
        # Check for packed weights (compressed-tensors INT4 format)
        has_packed = hasattr(self.base_layer, "weight_packed") or (
            hasattr(self.base_layer, "weight")
            and hasattr(self.base_layer.weight, "dtype")
            and self.base_layer.weight.dtype == torch.uint8
        )

        # Check for quantization scales (confirms it's quantized)
        has_scales = hasattr(self.base_layer, "weight_scale")

        return has_packed and has_scales

    def _materialize_int4_weights(self) -> None:
        """
        Materialize FP16 weights from INT4 packed buffers for LoRA compatibility.

        This creates LoRA-compatible weight tensors alongside the packed buffers.
        The materialized weights are used for LoRA attachment while the packed
        buffers continue to be used by the INT4 quantized kernels.
        """
        try:
            unpacked_weights = self.get_unpacked_weights()
            if unpacked_weights is not None:
                self._materialized_weight = unpacked_weights
                logger.info(
                    "Materialized INT4 weights to FP16: shape=%s, dtype=%s, "
                    "device=%s",
                    unpacked_weights.shape,
                    unpacked_weights.dtype,
                    unpacked_weights.device,
                )
            else:
                logger.warning(
                    "Failed to materialize INT4 weights. "
                    "LoRA may not attach correctly to this layer."
                )
        except Exception as e:
            logger.error(
                "Error during INT4 weight materialization: %s. "
                "LoRA attachment may fail for this layer.",
                e,
            )
            self._materialized_weight = None

    def get_unpacked_weights(self) -> torch.Tensor | None:
        """
        Get unpacked FP16 weights for INT4 quantized layers.

        This is useful for operations that need access to dequantized weights,
        such as merging LoRA adapters into the base weights or fine-tuning.

        For inference-only use cases, this is typically not needed since
        LoRA operates directly on the input activations.

        Returns:
            Unpacked FP16 weights, or None if layer is not INT4 quantized
        """
        if not self._is_int4_quantized:
            return None

        try:
            from vllm.lora.int4_utils import get_unpacker

            unpacker = get_unpacker()
            # Generate unique name for caching
            layer_name = f"{id(self.base_layer)}"

            unpacked = unpacker.unpack_module(
                module=self.base_layer,
                module_name=layer_name,
                output_dtype=torch.float16,
            )

            return unpacked
        except Exception as e:
            logger.warning(
                "Failed to unpack INT4 weights: %s. "
                "Inference will still work using quantized kernels.",
                e,
            )
            return None
