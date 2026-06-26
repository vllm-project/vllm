# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

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
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import direct_register_custom_op

from .base import BaseLayerWithLoRA
from .utils import _get_lora_aux_cuda_stream, _get_lora_device

if envs.VLLM_LORA_ENABLE_DUAL_STREAM:

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

        self._enable_aux_cuda_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
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
        self._dora_active_slots: set[int] = set()

    def _init_lora_stream_context(self) -> None:
        if not self._enable_aux_cuda_stream:
            return
        vllm_config = get_current_vllm_config()
        self._lora_stream = _get_lora_aux_cuda_stream()
        assert current_platform.is_cuda_alike()
        self._events = [torch.Event(), torch.Event()]
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
        self.dora_scale_stacked = torch.ones(
            max_loras,
            lora_b_out_size,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.dora_enabled_stacked = torch.zeros(
            max_loras,
            dtype=torch.bool,
            device=self.device,
        )
        self._dora_active_slots.clear()
        self.output_slices = (self.lora_b_stacked[0].shape[2],)

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
        self.dora_scale_stacked[index] = 1
        self.dora_enabled_stacked[index] = False
        self._dora_active_slots.discard(index)

    @staticmethod
    def _raise_if_dora_unsupported(
        lora_magnitude_vector: (
            torch.Tensor | list[torch.Tensor | None] | None
        ),
        feature: str,
    ) -> None:
        if lora_magnitude_vector is not None:
            raise NotImplementedError(f"DoRA is not supported for {feature}.")

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
        lora_magnitude_vector: (
            torch.Tensor | list[torch.Tensor | None] | None
        ) = None,
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

        if lora_magnitude_vector is not None:
            assert isinstance(lora_magnitude_vector, torch.Tensor)
            self._set_dora_scale(index, lora_a, lora_b, lora_magnitude_vector)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(
            lora_a, non_blocking=True
        )
        self.lora_b_stacked[0][index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(
            lora_b, non_blocking=True
        )

    def _get_quant_method(self) -> QuantizeMethodBase:
        quant_method = self.base_layer.quant_method
        if quant_method is None:
            raise RuntimeError(
                f"{type(self.base_layer).__name__} must define quant_method for LoRA."
            )
        return quant_method

    def _set_dora_scale(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        lora_magnitude_vector: torch.Tensor,
    ) -> None:
        if self.tp_size != 1:
            raise NotImplementedError("DoRA currently only supports TP=1.")
        if self.n_slices != 1:
            raise NotImplementedError("DoRA is not supported for packed LoRA layers.")

        base_weight = self._get_dora_base_weight()
        dora_scale = self._get_dora_scale(
            base_weight,
            lora_a,
            lora_b,
            lora_magnitude_vector,
        )
        self._store_dora_scale(index, dora_scale)

    def _get_dora_base_weight(self) -> torch.Tensor:
        if not isinstance(self.base_layer.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError("DoRA is not supported for quantized layers.")
        if not hasattr(self.base_layer, "weight"):
            raise NotImplementedError(
                "DoRA requires access to the unquantized base layer weight."
            )
        return self.base_layer.weight

    def _store_dora_scale(self, index: int, dora_scale: torch.Tensor) -> None:
        self.dora_scale_stacked[index, : dora_scale.shape[0]].copy_(
            dora_scale.to(dtype=self.dora_scale_stacked.dtype),
            non_blocking=True,
        )
        self.dora_enabled_stacked[index] = True
        self._dora_active_slots.add(index)

    @staticmethod
    def _get_dora_scale(
        base_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        lora_magnitude_vector: torch.Tensor,
        norm_sq_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if base_weight.shape != (lora_b.shape[0], lora_a.shape[1]):
            raise ValueError(
                "DoRA weight shapes are incompatible with the base layer: "
                f"base_weight={tuple(base_weight.shape)}, "
                f"lora_a={tuple(lora_a.shape)}, lora_b={tuple(lora_b.shape)}."
            )
        if lora_magnitude_vector.shape[0] != lora_b.shape[0]:
            raise ValueError(
                "DoRA magnitude vector shape is incompatible with lora_b: "
                f"magnitude={tuple(lora_magnitude_vector.shape)}, "
                f"lora_b={tuple(lora_b.shape)}."
            )

        lora_a = lora_a.to(device=base_weight.device, non_blocking=True)
        lora_b = lora_b.to(device=base_weight.device, non_blocking=True)
        lora_magnitude_vector = lora_magnitude_vector.to(
            device=base_weight.device, non_blocking=True
        )
        delta_weight = lora_b.float() @ lora_a.float()
        merged_weight = base_weight.float() + delta_weight
        norm_sq = merged_weight.square().sum(dim=1)
        if norm_sq_reduce is not None:
            norm_sq = norm_sq_reduce(norm_sq)
        weight_norm = torch.sqrt(norm_sq)
        weight_norm = torch.clamp(weight_norm, min=torch.finfo(weight_norm.dtype).eps)
        return lora_magnitude_vector.float() / weight_norm

    def _get_dora_token_mask(
        self,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_lora_indices = self.punica_wrapper.token_lora_indices[:num_tokens]
        has_adapter = token_lora_indices >= 0
        safe_token_lora_indices = torch.where(
            has_adapter, token_lora_indices, torch.zeros_like(token_lora_indices)
        )
        has_dora = has_adapter & self.dora_enabled_stacked[safe_token_lora_indices]
        return token_lora_indices, has_dora

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        # is_forward_context_available for tower modules
        if (
            self._enable_aux_cuda_stream
            and is_forward_context_available()
            and not self._dora_active_slots
        ):
            output_size = sum(self.output_slices)
            return torch.ops.vllm.lora_linear_async(
                self.layer_name, output_size, x, bias
            )
        return self._apply_sync(x, bias)

    def _apply_sync(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self._get_quant_method().apply(self.base_layer, x, bias)
        return self._apply_lora_to_output(x, output, base_output_bias=bias)

    def _apply_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        output = base_output[0] if isinstance(base_output, tuple) else base_output
        base_output_bias = (
            self.bias
            if self.bias is not None and not self.base_layer.skip_bias_add
            else None
        )
        return self._apply_lora_to_output(
            x, output, base_output_bias=base_output_bias
        )

    def _apply_standard_lora(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
        )
        if current_platform.can_update_inplace():
            return output
        assert lora_output is not None
        return lora_output

    def _compute_lora_delta(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        lora_delta = torch.zeros_like(output)
        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            lora_delta,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            add_inputs=False,
        )
        if current_platform.can_update_inplace():
            return lora_delta
        assert lora_output is not None
        return lora_output

    def _apply_dora_scale(
        self,
        output: torch.Tensor,
        token_lora_indices: torch.Tensor,
        has_dora: torch.Tensor,
        base_output_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dora_scale = self.dora_scale_stacked[token_lora_indices[has_dora]].to(
            dtype=output.dtype
        )
        dora_output = output[has_dora]
        if base_output_bias is not None:
            # DoRA scales only the weight path. If the base linear already
            # added bias, remove it before scaling and add it back after.
            bias = base_output_bias.to(dtype=output.dtype)
            dora_output = (dora_output - bias) * dora_scale
            dora_output = dora_output + bias
        else:
            dora_output = dora_output * dora_scale
        output[has_dora] = dora_output
        return output

    def _apply_lora_to_output(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        *,
        base_output_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = output.shape if output.ndim == 3 else None

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        if self._dora_active_slots:
            token_lora_indices, has_dora = self._get_dora_token_mask(x.shape[0])
            if has_dora.any().item():
                if not current_platform.is_cuda_alike():
                    raise NotImplementedError("DoRA currently only supports CUDA.")

                # DoRA needs the unscaled base-plus-LoRA value before applying
                # the per-output magnitude scale for DoRA tokens only.
                output.add_(self._compute_lora_delta(x, output))
                output = self._apply_dora_scale(
                    output,
                    token_lora_indices,
                    has_dora,
                    base_output_bias=base_output_bias,
                )
            else:
                output = self._apply_standard_lora(x, output)
        else:
            output = self._apply_standard_lora(x, output)

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
        assert x.ndim in (2, 3)
        num_tokens = x.size(0) if x.ndim == 2 else x.size(1)
        output_size = sum(self.output_slices)

        def base_fn() -> torch.Tensor:
            return self._get_quant_method().apply(self.base_layer, x, bias)

        def lora_fn() -> torch.Tensor:
            # Must be zeros, not empty: _lora_expand_kernel exits early (without
            # writing) when lora_id == -1 (no active LoRA). If uninitialized,
            # output.add_(lora_result) below would corrupt the base output.
            lora_output = torch.zeros(
                (num_tokens, output_size),
                device=self.device,
                dtype=x.dtype,
            )

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
