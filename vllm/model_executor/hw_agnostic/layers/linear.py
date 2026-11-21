# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Linear layers of `vllm.model_executor.layers.linear` for the hw-agnostic path.

Unquantized only, with GEMMs dispatched through `hw_agnostic.layers.utils`.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.determinism.batch_invariant import (
    linear_batch_invariant,
)
from vllm.model_executor.hw_agnostic.custom_op import PluggableLayer
from vllm.model_executor.hw_agnostic.layers.utils import (
    dispatch_cpu_unquantized_gemm,
    dispatch_unquantized_gemm,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ModelWeightParameter,
    RowvLLMParameter,
)
from vllm.model_executor.utils import copy_weight_, set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.

        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    supports_pre_processed_weights = True

    def __init__(self) -> None:
        self._gemm_impl = dispatch_unquantized_gemm()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # This method creates unquantized linear weights.
        # The weights are not quantized, and they are not sharded.
        # The amount of memory allocated for the weights is
        # sum(output_partition_sizes) * input_size_per_partition.
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            # MLA's kv_b_proj (see `_cpu_skip_gemm_dispatch`): not
            # perf-critical, so skip packing and use a plain fallback.
            if getattr(layer, "_cpu_skip_gemm_dispatch", False):
                layer.cpu_linear = torch.nn.functional.linear
                return
            dispatch_cpu_unquantized_gemm(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT and (
            current_platform.is_cuda_alike() or current_platform.is_xpu()
        ):
            return linear_batch_invariant(x, layer.weight, bias)
        return self._gemm_impl(layer, x, layer.weight, bias)


def apply_flattened(fn: Callable[[Any], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Call `fn` on `x` flattened to 2D, if that can be done without a copy.

    Linear methods and kernels may assume 2D activations, but callers such as
    the Transformers modeling backend pass `[batch, seq, hidden]` activations.
    """
    if x.ndim <= 2 or not x.is_contiguous():
        return fn(x)
    return fn(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)


def _verify_supported(quant_config: QuantizationConfig | None, prefix: str) -> None:
    """Reject the in-tree linear features that these layers lack."""
    if quant_config is not None:
        raise ValueError(
            f"Quantization is not supported with VLLM_USE_HW_AGNOSTIC=1 ({prefix}): "
            "the hw-agnostic linear layers are unquantized. Unset "
            "VLLM_USE_HW_AGNOSTIC or serve an unquantized model."
        )
    config = get_current_vllm_config_or_none()
    if config is None:
        return
    # Drafters are built under their own model config and don't run LoRA.
    is_draft = config.model_config is not None and (
        config.model_config.runner_type == "draft"
    )
    if config.lora_config is not None and not is_draft:
        raise ValueError(
            "LoRA is not supported with VLLM_USE_HW_AGNOSTIC=1: LoRA wraps the "
            "in-tree linear layers, which the hw-agnostic layers do not derive "
            "from. Unset VLLM_USE_HW_AGNOSTIC or run without `enable_lora`."
        )
    if config.kernel_config.linear_backend != "auto":
        logger.warning_once(
            "--linear-backend=%s is ignored with VLLM_USE_HW_AGNOSTIC=1.",
            config.kernel_config.linear_backend,
        )


class LinearBase(PluggableLayer):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Must be None, quantization is not supported.
        prefix: Prefix for parameter names.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, tensor parallelism will be disabled for this layer.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        _verify_supported(quant_config, prefix)
        self.quant_method = UnquantizedLinearMethod()
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )

    def update_param_tp_status(self):
        # Single source of truth for a parameter's TP state. BasevLLMParameter
        # stamps self.tp_rank with the *global* rank in __init__; this reconciles
        # every child parameter to the *layer's* tp_rank/tp_size (which correctly
        # accounts for disable_tp -> replicated weights with tp_rank == 0).
        #
        # Must be re-run whenever parameters are (re-)created after construction,
        # e.g. after quant_method.process_weights_after_loading() swaps in fresh
        # Parameters. Otherwise a later load_weights()/weight-refit would narrow a
        # replicated weight at global_rank * shard_size and overflow.
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


# --8<-- [start:replicated_linear]
@PluggableLayer.register("replicated_linear")
class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Must be None, quantization is not supported.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.

    """

    # --8<-- [end:replicated_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}"
        )
        copy_weight_(param.data, loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        output = apply_flattened(partial(self.quant_method.apply, self, bias=bias), x)

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


# --8<-- [start:column_parallel_linear]
@PluggableLayer.register("column_parallel_linear")
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Must be None, quantization is not supported.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
        tp_rank: Override the tensor-parallel rank used for sharding. Defaults to
            the global TP rank. Used to shard at a coarser granularity than one
            shard per rank.
        tp_size: Override the tensor-parallel world size used for sharding.
            Defaults to the global TP world size.

    """

    # --8<-- [end:column_parallel_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        tp_rank: int | None = None,
        tp_size: int | None = None,
    ):
        # Divide the weight matrix along the last dimension.
        if disable_tp:
            self.tp_rank, self.tp_size = 0, 1
        else:
            self.tp_rank = (
                tp_rank if tp_rank is not None else get_tensor_model_parallel_rank()
            )
            self.tp_size = (
                tp_size
                if tp_size is not None
                else get_tensor_model_parallel_world_size()
            )
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader_v2,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        copy_weight_(param_data, loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)
        param.load_column_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        apply = partial(self.quant_method.apply, self, bias=bias)
        output_parallel = apply_flattened(apply, input_)

        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class StackedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel linear whose weight is several checkpoint tensors stacked
    along the output dimension, each addressed by a shard id.

    Keeps track of which shards a load delivered to each parameter so the loader
    can reject a parameter that was only partly filled.
    """

    shard_ids: tuple[str | int, ...]
    """Every shard this layer can receive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loaded_shard_ids: dict[str, set[str | int]] = {}

    def _get_shard_size_mapping(self, loaded_shard_id: Any) -> int | None:
        """Rows of this rank's parameter that belong to the shard."""
        raise NotImplementedError

    def validate_shard_id(self, shard_id: Any, param: Parameter | None = None) -> bool:
        """Check `shard_id` names shards of this layer and, if `param` is given,
        record them as loaded into it."""
        if shard_id is None:
            # Already stacked on disk, so one tensor fills every shard.
            shards = self.shard_ids
        elif isinstance(shard_id, tuple):
            shards = shard_id
        else:
            shards = (shard_id,)
        if any(shard not in self.shard_ids for shard in shards):
            raise ValueError(
                f"Shard id for {type(self).__name__} should be one of "
                f"{self.shard_ids}, got shard id {shard_id}."
            )
        if param is not None:
            name = next((n for n, p in self.named_parameters() if p is param), None)
            if name is not None:
                self.loaded_shard_ids.setdefault(name, set()).update(shards)
        return True

    def pop_missing_shards(self) -> list[str]:
        """`<param>.<shard>` for each shard this load never filled, then forget
        the record so the next load starts fresh.

        A shard with no rows on this rank is never expected, and a parameter
        nothing was loaded into is left to the loader's parameter-level check.
        """
        loaded, self.loaded_shard_ids = self.loaded_shard_ids, {}
        return [
            f"{name}.{shard}"
            for name, shards in loaded.items()
            for shard in self.shard_ids
            if shard not in shards and self._get_shard_size_mapping(shard)
        ]


class MergedColumnParallelLinear(StackedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Must be None, quantization is not supported.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, all weights matrix won't be sharded, this layer
                    will be treated as a "Replicated" MergedLinear.
        padding_size: Extra output rows on each TP rank, appended after the
            real output_sizes. These rows are zeroed and are not checkpoint shards.

    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        padding_size: int = 0,
    ):
        self.padding_size = padding_size
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.output_sizes = output_sizes
        if padding_size:
            self.output_sizes = output_sizes + [padding_size * self.tp_size]
        self.shard_ids = tuple(range(len(output_sizes)))

        assert all(output_size % self.tp_size == 0 for output_size in self.output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        if padding_size:
            for param in (self.weight, self.bias):
                if param is None:
                    continue
                self._zero_output_padding(param)
                set_weight_attrs(
                    param, {"weight_loader_numel": param[:-padding_size].numel()}
                )
                # The loader zeroes the padding too, but a load plan skips loader
                # fills: direct reload zero-fills it at begin instead.
                param.fill_on_reload = True

    def _zero_output_padding(self, param: torch.Tensor) -> None:
        if self.padding_size and (param is self.weight or param is self.bias):
            param.data[-self.padding_size :].zero_()

    @property
    def _checkpoint_output_sizes(self) -> list[int]:
        return self.output_sizes[:-1] if self.padding_size else self.output_sizes

    def _get_shard_size_mapping(self, loaded_shard_id: int) -> int:
        return self.output_sizes[loaded_shard_id] // self.tp_size

    def validate_shard_id(self, shard_id: Any, param: Parameter | None = None) -> bool:
        if isinstance(shard_id, tuple) and any(
            b - a != 1 for a, b in zip(shard_id[:-1], shard_id[1:])
        ):
            raise ValueError(
                "Shard id with multiple indices should be consecutive. "
                f"Got shard id {shard_id}."
            )
        return super().validate_shard_id(shard_id, param)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        self.validate_shard_id(loaded_shard_id, param)
        self._zero_output_padding(param)

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                copy_weight_(param_data, loaded_weight)
                return

            output_sizes = (
                self.output_sizes[loaded_shard_id[0] : loaded_shard_id[-1] + 1]
                if loaded_shard_id is not None
                else self._checkpoint_output_sizes
            )
            current_shard_offset = 0
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            shard_offset //= self.tp_size
            shard_size //= self.tp_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        copy_weight_(param_data, loaded_weight)

    def _load_fused_module_from_checkpoint(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        output_sizes: list[int] | None = None,
    ):
        """Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        output_sizes = output_sizes or self._checkpoint_output_sizes
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        self.validate_shard_id(loaded_shard_id, param)
        self._zero_output_padding(param)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        assert loaded_shard_id < len(self.output_sizes)

        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard_offset //= self.tp_size
        shard_size //= self.tp_size

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            # Load into self if name is not an attr of self or its submodules
            param: Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            logger.debug(
                "Loaded shard %s with shape %s into %s.%s",
                shard_id,
                loaded_weight.shape,
                self.prefix,
                name,
            )
            yield name


class QKVParallelLinear(StackedColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Must be None, quantization is not supported.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.

    """

    shard_ids: tuple[str, ...] = ("q", "k", "v")

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]
        output_size = sum(self.output_sizes)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
            + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        self.validate_shard_id(loaded_shard_id, param)
        if loaded_shard_id is None:  # special case for certain models
            if type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in self.shard_ids

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        self.validate_shard_id(loaded_shard_id, param)

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                copy_weight_(param_data, loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            for shard_id, shard_offset, shard_size in shard_offsets:
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in self.shard_ids

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                shard_rank = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_rank * shard_size

            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        copy_weight_(param_data, loaded_weight)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        for name, loaded_weight in weights:
            shard_id = getattr(loaded_weight, "shard_id", None)
            self.validate_shard_id(shard_id)
            # Load into self if name is not an attr of self or its submodules
            param: Parameter
            if "." in name:
                submodule, _, attr = name.rpartition(".")
                param = getattr(self.get_submodule(submodule), attr, self)
            else:
                param = getattr(self, name, self)
            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            logger.debug(
                "Loaded shard %s with shape %s into %s.%s",
                shard_id,
                loaded_weight.shape,
                self.prefix,
                name,
            )
            yield name


# --8<-- [start:row_parallel_linear]
@PluggableLayer.register("row_parallel_linear")
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Must be None, quantization is not supported.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.

    """

    # --8<-- [end:row_parallel_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader_v2,
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        copy_weight_(param_data, loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        apply = partial(self.quant_method.apply, self, bias=bias_)
        output_parallel = apply_flattened(apply, input_parallel)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
