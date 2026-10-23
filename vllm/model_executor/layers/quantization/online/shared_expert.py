# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.online.requantization import (
    OnlineRequantizationMixin,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_checkpoint_quant_keys,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Static,
)
from vllm.model_executor.utils import set_weight_attrs

SharedExpertQuantizer = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _param_stem(shard_id: str) -> str:
    return "w2" if shard_id == "w2" else "w13"


class OnlineSharedExpertLoader:
    """Loads the fused shared experts (FSE) of a ``RoutedExperts`` layer with online
    quantization.

    - Staged shared experts: the shared experts are loaded as stored in temporary
      slots (``online_w{13,2}_weight[_scale]``), dequantized after loading, and
      quantized into the routed experts slots: with the routed experts when they
      are requantized, else by ``finalize``.
    - Unquantized shared experts with pre-quantized routed experts: the shared
      experts are quantized on arrival by ``quantizer``, and loaded in the routed
      experts slots.
    """

    def __init__(
        self,
        quant_config,
        online_config,
        quant_method: Any,
        shared_expert_prefix: str | None,
        num_logical_experts: int,
        num_fused_shared_experts: int,
    ):
        self.shared_expert_prefix = shared_expert_prefix
        self.num_logical_experts = num_logical_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        # The routed experts are requantized online from their checkpoint format.
        self.routed_requantized = (
            isinstance(quant_method, OnlineRequantizationMixin)
            and quant_method.requantization_source is not None
        )
        # Requantized routed experts quantize the staged shared experts with them.
        self.stages_shared_experts = self.routed_requantized
        self.quantizer: SharedExpertQuantizer | None = None
        target_key = self._get_shared_target_key(online_config)
        # Format the shared experts are stored in (None: unquantized). FSE
        # compatibility checks require every projection to share it.
        self.stored_key: QuantKey | None = None
        if self.routed_requantized or target_key is not None:
            self.stored_key = get_checkpoint_quant_keys(
                quant_config, f"{shared_expert_prefix}.down_proj", LinearBase
            )[0]
        if not self.routed_requantized and target_key is not None:
            if self.stored_key != target_key:
                from vllm.model_executor.layers.quantization.online.base import (
                    ONLINE_SHARED_EXPERT_QUANTIZERS,
                )

                if target_key not in ONLINE_SHARED_EXPERT_QUANTIZERS:
                    raise NotImplementedError(
                        "Online quantization of fused shared experts is not "
                        f"supported for weight key {target_key}. Disable FSE or "
                        "open an issue."
                    )
                self.quantizer = ONLINE_SHARED_EXPERT_QUANTIZERS[target_key]
                # Quantized shared experts are staged, as their weight and scale
                # are needed together to be dequantized.
                self.stages_shared_experts = self.stored_key is not None
                from vllm.model_executor.layers.quantization.online.moe_base import (
                    OnlineMoEMethodBase,
                )

                if self.stages_shared_experts and isinstance(
                    quant_method, OnlineMoEMethodBase
                ):
                    # `finalize` writes into pre-quantized routed experts slots.
                    raise NotImplementedError(
                        "Fused shared experts stored quantized are not supported "
                        "with routed experts quantized online from high "
                        "precision. Disable FSE or open an issue."
                    )
        self.staging_parameters: dict[str, torch.nn.Parameter] = {}

    def _get_shared_target_key(self, online_config) -> QuantKey | None:
        if self.shared_expert_prefix is None:
            return None
        # FSE compatibility checks require every shared-expert projection to
        # use the routed experts' weight key.
        resolved = online_config.resolve_quant_method_cls(
            LinearBase, f"{self.shared_expert_prefix}.down_proj"
        )
        return None if resolved is None else resolved[3].weight

    def create_staging(
        self,
        layer: torch.nn.Module,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create parameters holding the fused shared experts as stored.

        The checkpoint shared experts are loaded as stored into
        ``online_w{13,2}_weight`` (plus ``online_w{13,2}_weight_scale`` for
        quantized shared experts), which are separate from the routed experts'
        parameters. They are dequantized by ``dequantize_staging``.
        """
        from vllm.model_executor.layers.quantization.online.base import (
            ONLINE_SHARED_EXPERT_DEQUANTIZERS,
        )

        moe_config = layer.moe_config
        hidden_size = moe_config.hidden_dim_unpadded
        intermediate_size = moe_config.intermediate_size_per_partition_unpadded
        checkpoint_key = self.stored_key
        if (
            checkpoint_key is not None
            and checkpoint_key not in ONLINE_SHARED_EXPERT_DEQUANTIZERS
        ):
            raise NotImplementedError(
                "Fused shared experts (FSE) with online quantization only support "
                "shared experts stored unquantized or as "
                f"{list(ONLINE_SHARED_EXPERT_DEQUANTIZERS)}, got {checkpoint_key}. "
                "Disable FSE or open an issue."
            )
        # NOTE: The staging layout is MXFP8's, the only quantized format in
        # ONLINE_SHARED_EXPERT_DEQUANTIZERS.
        is_mxfp8 = checkpoint_key == kMxfp8Static
        if is_mxfp8 and (
            hidden_size % MXFP8_BLOCK_SIZE or intermediate_size % MXFP8_BLOCK_SIZE
        ):
            raise ValueError(
                "MXFP8 shared experts require hidden/intermediate sizes "
                f"divisible by {MXFP8_BLOCK_SIZE}."
            )

        def register(name: str, shape: tuple[int, ...], dtype: torch.dtype, **attrs):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs | attrs)
            self.staging_parameters[name] = param

        w13_rows = moe_config.w13_num_shards * intermediate_size
        for stem, rows, cols in (
            ("w13", w13_rows, hidden_size),
            ("w2", hidden_size, intermediate_size),
        ):
            shape = (self.num_fused_shared_experts, rows)
            weight_dtype = MXFP8_VALUE_DTYPE if is_mxfp8 else params_dtype
            register(f"online_{stem}_weight", (*shape, cols), weight_dtype)
            if is_mxfp8:
                register(
                    f"online_{stem}_weight_scale",
                    (*shape, cols // MXFP8_BLOCK_SIZE),
                    MXFP8_SCALE_DTYPE,
                    quant_method=FusedMoeWeightScaleSupported.GROUP.value,
                )

    def dequantize_staging(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the staged ``(w13, w2)`` shared experts, dequantized if needed."""
        w13 = layer.online_w13_weight.data
        w2 = layer.online_w2_weight.data
        if self.stored_key is not None:
            from vllm.model_executor.layers.quantization.online.base import (
                ONLINE_SHARED_EXPERT_DEQUANTIZERS,
            )

            dequantize = ONLINE_SHARED_EXPERT_DEQUANTIZERS[self.stored_key]
            w13 = dequantize(w13, layer.online_w13_weight_scale.data)
            w2 = dequantize(w2, layer.online_w2_weight_scale.data)
        return w13, w2

    def release_staging(self, layer: torch.nn.Module) -> None:
        for name, param in self.staging_parameters.items():
            if layer._parameters.get(name) is param:
                delattr(layer, name)
        self.staging_parameters.clear()

    def finalize(self, layer: torch.nn.Module) -> None:
        """Quantize the staged shared experts into the pre-quantized routed experts
        slots. To call after loading, before the routed experts' processing.

        Requantized routed experts quantize the staged shared experts themselves.
        """
        if self.routed_requantized or not self.staging_parameters:
            return
        assert self.quantizer is not None
        shared_w13, shared_w2 = self.dequantize_staging(layer)
        for shared_id in range(self.num_fused_shared_experts):
            expert_id = layer._map_global_expert_id_to_local_expert_id(
                self.num_logical_experts + shared_id
            )
            if expert_id == -1:
                continue
            for stem, weight in (("w13", shared_w13), ("w2", shared_w2)):
                weight, weight_scale = self.quantizer(weight[shared_id])
                for name, tensor in (
                    (f"{stem}_weight", weight),
                    (f"{stem}_weight_scale", weight_scale),
                ):
                    rows, cols = tensor.shape
                    getattr(layer, name).data[expert_id, :rows, :cols].copy_(tensor)
        self.release_staging(layer)

    def is_fused_shared_expert(self, global_expert_id: int) -> bool:
        return (
            self.num_logical_experts
            <= global_expert_id
            < self.num_logical_experts + self.num_fused_shared_experts
        )

    @staticmethod
    def _staging_param_name(weight_name: str, shard_id: str) -> str | None:
        """Staging parameter (``online_w{13,2}_weight[_scale]``) of a shared expert
        tensor. They are separate from the routed experts' parameters. None for
        other tensors.
        """
        stem = f"online_{_param_stem(shard_id)}"
        if weight_name.endswith("_weight_scale"):
            return f"{stem}_weight_scale"
        if weight_name.endswith("_weight"):
            return f"{stem}_weight"
        return None

    def redirect_param_name(
        self, param_name: str, shard_id: str, expert_id: int
    ) -> str:
        """Parameter name a checkpoint tensor of expert `expert_id` loads into."""
        # Once the staging is released, reloads go to `load`, which raises.
        if self.staging_parameters and self.is_fused_shared_expert(expert_id):
            return self._staging_param_name(param_name, shard_id) or param_name
        return param_name

    def redirect_mapping(
        self, mapping: list[tuple[str, str, int, str]]
    ) -> list[tuple[str, str, int, str]]:
        """Point the expert params mapping of the shared experts to the staging
        parameters, as e.g. ``w13_`` -> ``online_w13_``."""
        if not self.staging_parameters:
            return mapping
        redirected = []
        for param_name, weight_name, expert_id, shard_id in mapping:
            if self.is_fused_shared_expert(expert_id) and weight_name.startswith(
                f"experts.{expert_id}."
            ):
                stem = "w2_" if param_name.endswith("w2_") else "w13_"
                param_name = f"{param_name.removesuffix(stem)}online_{stem}"
            redirected.append((param_name, weight_name, expert_id, shard_id))
        return redirected

    def load(
        self,
        layer: Any,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        global_expert_id: int,
        expert_id: int,
    ) -> tuple[torch.nn.Parameter, int, bool | None]:
        """Prepare the load of a fused shared expert tensor in
        ``RoutedExperts.weight_loader``.

        Returns:
            The parameter and local expert id to load into, and the load result if
            the tensor was already loaded (None to let the caller load it).
        """
        if not self.is_fused_shared_expert(global_expert_id):
            return param, expert_id, None

        if self.stages_shared_experts:
            if not self.staging_parameters:
                # The staged shared experts were already quantized and released.
                raise NotImplementedError(
                    "Reloading fused shared experts staged for online quantization "
                    "is not supported."
                )
            staging_name = self._staging_param_name(weight_name, shard_id)
            if staging_name is None:
                raise NotImplementedError(
                    f"Fused shared expert tensor {weight_name!r} has no online "
                    "staging parameter: only *_weight and *_weight_scale are "
                    "supported when staging shared experts."
                )
            return (
                getattr(layer, staging_name),
                global_expert_id - self.num_logical_experts,
                None,
            )

        # Only quantize high-precision shared expert weights, load anything else as is.
        if (
            self.quantizer is None
            or not weight_name.endswith("_weight")
            or not loaded_weight.is_floating_point()
            or loaded_weight.dtype == param.dtype
        ):
            return param, expert_id, None

        # Full-precision shared expert: quantize it, then load the weight and
        # scale like a pre-quantized expert.
        weight, weight_scale = self.quantizer(loaded_weight.to(layer.moe_config.device))
        layer.weight_loader(
            getattr(layer, f"{_param_stem(shard_id)}_weight_scale"),
            weight_scale,
            f"{weight_name}_scale",
            shard_id,
            global_expert_id,
            return_success=True,
        )
        loaded = layer.weight_loader(
            param, weight, weight_name, shard_id, global_expert_id, True
        )
        return param, expert_id, loaded
