# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from transformers import PretrainedConfig

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import direct_register_custom_op

from .base import BaseLayerWithLoRA
from .utils import _get_lora_device

if envs.VLLM_LORA_ENABLE_DUAL_STREAM:
    _lora_aux_stream: torch.cuda.Stream | None = None

    def _get_lora_aux_stream() -> torch.cuda.Stream | None:
        global _lora_aux_stream
        if _lora_aux_stream is None and current_platform.is_cuda_alike():
            _lora_aux_stream = torch.cuda.Stream()
        return _lora_aux_stream

    def lora_linear_async(
        layer_name: str,
        output_size: int,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        self = forward_context.no_compile_layers[layer_name]
        return self._apply_async_impl(x, bias)

    def lora_linear_async_fake(
        layer_name: str,
        output_size: int,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The real function reshapes output back to the original 3D shape
        # when the input has an extra batch dimension (transformers backend).
        if x.ndim == 3:
            return torch.empty(
                (x.size(0), x.size(1), output_size),
                device=x.device,
                dtype=x.dtype,
            )
        return torch.empty(
            (x.size(0), output_size),
            device=x.device,
            dtype=x.dtype,
        )

    direct_register_custom_op(
        op_name="lora_linear_async",
        op_func=lora_linear_async,
        fake_impl=lora_linear_async_fake,
    )


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()

        self._enable_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size
        # Ensure tp_size and tp_rank consistency with the base_layer.
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(self.base_layer)
        self._init_lora_stream_context()
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int

    def _init_lora_stream_context(self) -> None:
        if not self._enable_stream:
            return
        vllm_config = get_current_vllm_config()
        self._lora_stream = _get_lora_aux_stream()

        self._events = (
            [torch.cuda.Event(), torch.cuda.Event()]
            if current_platform.is_cuda_alike()
            else [None, None]
        )
        # lora_linear avoids prefix conflicts with the base layer
        self.layer_name = self.base_layer.prefix + ".lora_linear_async"
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(self.layer_name))
        compilation_config.static_forward_context[self.layer_name] = self

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_config = lora_config
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
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert isinstance(lora_a, torch.Tensor)
        assert isinstance(lora_b, torch.Tensor)
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
        # is_forward_context_available for tower modules
        if self._enable_stream and is_forward_context_available():
            output_size = sum(self.output_slices)
            return torch.ops.vllm.lora_linear_async(
                self.layer_name, output_size, x, bias
            )
        else:
            return self._apply_sync(x, bias)

    def _apply_sync(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)

        original_shape = output.shape if output.ndim == 3 else None

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        # Reshape the flattened output back to its original shape,
        # as some MM encoders cannot handle flattened inputs.
        if original_shape is not None:
            output = output.reshape(original_shape)

        return output

    def _apply_async_impl(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass with base linear and LoRA on separate CUDA streams
        for overlap, using maybe_execute_in_parallel.
        Base layer runs on default stream; LoRA runs on aux stream.
        """
        assert envs.VLLM_LORA_ENABLE_DUAL_STREAM

        num_tokens = x.size(0) if x.ndim == 2 else x.size(1)
        output_size = sum(self.output_slices)
        lora_output = torch.empty(
            (num_tokens, output_size),
            device=self.device,
            dtype=x.dtype,
        )

        def base_fn() -> torch.Tensor:
            return self.base_layer.quant_method.apply(self.base_layer, x, bias)

        def lora_fn() -> torch.Tensor:
            # Flatten the batch dimension for the transformers backend
            # (which uses shape (1, seq_len, hidden)), matching _apply_sync.
            x_2d = x.flatten(0, 1) if x.ndim == 3 else x
            self.punica_wrapper.add_lora_linear(
                lora_output,
                x_2d,
                self.lora_a_stacked,
                self.lora_b_stacked,
                1.0,
                self.output_slices,
                add_inputs=False,
            )
            return lora_output

        output, lora_result = maybe_execute_in_parallel(
            base_fn,
            lora_fn,
            self._events[0],
            self._events[1],
            self._lora_stream,
        )

        original_shape = output.shape if output.ndim == 3 else None

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        output.add_(lora_result)

        # Reshape the flattened output back to its original shape,
        # as some MM encoders cannot handle flattened inputs.
        if original_shape is not None:
            output = output.reshape(original_shape)

        return output

    @property
    def weight(self) -> torch.Tensor:
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
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None
