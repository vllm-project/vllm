# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from enum import Enum
from typing import Literal, overload

import torch
from torch.nn.parameter import UninitializedParameter

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_dp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

logger = init_logger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class RoutedExperts(torch.nn.Module):
    """
    Container for routed expert weights and execution logic.

    This module owns the expert weight parameters (w13_weight, w2_weight, scales, etc.)
    and handles:
    - Loading checkpoint weights into parameters
    - Executing routed experts via quant_method.apply()

    Weight parameters are registered on this module via _ParameterRegistrationWrapper
    during FusedMoE initialization.
    """

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        unpadded_hidden_size: int,  # put in moe_config?
        intermediate_size: int,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        quant_method: FusedMoEMethodBase,
        expert_map_manager: ExpertMapManager,
        **kwargs,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.quant_method = quant_method
        self.expert_map_manager = expert_map_manager
        self.hidden_size = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.global_num_experts = moe_config.num_experts
        self.local_num_experts = moe_config.num_local_experts

        # Register buffers for state_dict compatibility
        if self.expert_map_manager.expert_map is not None:
            self.register_buffer("_expert_map", self.expert_map_manager.expert_map)

        if self.expert_map_manager.expert_mask is not None:
            self.register_buffer("expert_mask", self.expert_map_manager.expert_mask)

        # Bit of hack until things are settled
        self.__dict__.update(kwargs)

        moe_quant_params = {
            "num_experts": moe_config.num_local_experts,
            "hidden_size": moe_config.hidden_dim,
            "unpadded_hidden_size": unpadded_hidden_size,
            "intermediate_size_per_partition": (
                moe_config.intermediate_size_per_partition
            ),
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": moe_config.num_experts,
        }

        # need full intermediate size pre-sharding for WNA16 act order
        if self._needs_intermediate_size_param(quant_method):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        quant_method.create_weights(layer=self, **moe_quant_params)

    # TODO(bnell): make this a method on quant_method
    def _needs_intermediate_size_param(self, quant_method: FusedMoEMethodBase) -> bool:
        return quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        )

    def _ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def expert_map(self) -> torch.Tensor | None:
        return (
            self.expert_map_manager.expert_map
            if not self.rocm_aiter_fmoe_enabled
            else self.expert_map_manager.expert_mask
        )

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Get routing tables (already initialized by manager)."""
        # Return routing tables from manager
        routing_tables = self.expert_map_manager.routing_tables

        if routing_tables is None:
            return None

        # Register buffers for backward compatibility if not already registered
        if not hasattr(self, "expert_global_to_physical"):
            global_to_physical, physical_to_global, local_global = routing_tables
            self.register_buffer("expert_global_to_physical", global_to_physical)
            self.register_buffer("expert_physical_to_global", physical_to_global)
            self.register_buffer("expert_local_to_global", local_global)

        return routing_tables

    def update_expert_map(self):
        """Update expert mappings for new EP configuration."""
        # ep_size and ep_rank should already be updated in moe_parallel_config
        self.expert_map_manager.update()

        # Re-register buffers for state_dict compatibility
        self.register_buffer("_expert_map", self.expert_map_manager.expert_map)
        self.register_buffer("expert_mask", self.expert_map_manager.expert_mask)

        # Update routing table buffers if needed
        self._maybe_init_expert_routing_tables()

        # Handle AITER shared experts if needed
        if self.aiter_fmoe_shared_expert_enabled:
            self._init_aiter_shared_experts_topK_buffer(
                vllm_config=get_current_vllm_config(),
                dp_size=get_dp_group().world_size,
            )

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.expert_map_manager.map_global_to_local(expert_id)

    #
    # Weight Loading Methods
    #

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
        self,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        param: torch.Tensor,
        tp_rank: int,
    ):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor)
        # and we're not loading the full weight
        if not load_full and loaded_weight.ndim > 0:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor)
        # and we're not loading the full weight
        if not load_full and loaded_weight.ndim > 0:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self.layer._map_global_expert_id_to_local_expert_id(
            global_expert_id
        )

        use_global_sf = (
            getattr(self.quant_method, "use_global_sf", False)
            and "input_scale" in weight_name
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.layer._runner.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if quant_method_name in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            if is_transposed:
                loaded_weight = loaded_weight.t().contiguous()
            else:
                loaded_weight = loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter accounting merged weights
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            # To materialize a tensor, we must have full shape including
            # number of experts, making this portion to require `full_load`.
            assert full_load
            final_shape = list(loaded_weight.shape)
            # w1 and w3 are merged per expert.
            if shard_id in {"w1", "w3"}:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.moe_config.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.moe_config.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()
            quant_method = getattr(param, "quant_method", None)

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            # NOTE: ModelOpt MXFP8 MoE uses block scales in weight_scale
            # tensors (quant_method=BLOCK), so those must not be treated
            # as per-tensor scalars here.
            is_block_weight_scale = (
                "weight_scale" in weight_name
                and quant_method == FusedMoeWeightScaleSupported.BLOCK.value
            )
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            is_per_tensor = is_per_tensor and not is_block_weight_scale
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.moe_config.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=expert_data,
                        tp_rank=self.moe_config.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.moe_config.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.moe_config.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if (expert_mapping := self.layer.expert_mapping) is None:
            raise ValueError(
                "`self.layer.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                # Fused expert weights can be identified by their 3D tensors
                if loaded_weight.dim() == 3:
                    # Repurpose expert_id as shard_idx for deconcatenating w1 and w3
                    if shard_id in {"w1", "w3"}:
                        shard_idx = expert_id
                        experts_shard = loaded_weight.chunk(2, dim=1)[shard_idx]
                    else:
                        experts_shard = loaded_weight
                    start = 0
                else:
                    # loaded_weight is a single expert weight, so we add a dummy expert
                    # dimension to unify the loading logic with the fused case
                    experts_shard = loaded_weight.unsqueeze(0)
                    start = expert_id

                # Unified loading logic for fused and non-fused experts
                loaded_experts = experts_shard.unbind()
                for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                    success = self.weight_loader(
                        param=param,
                        loaded_weight=loaded_expert,
                        weight_name=weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        logger.debug(
                            "Loaded expert %d of shard %s into %s for layer %s",
                            expert_id,
                            shard_id,
                            param_name,
                            self.layer_name,
                        )
                        yield param_name

    #
    # Execution
    #

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        router_logits: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Execute routed experts using the quantization method's apply function.

        This is called by the runner after router selection (for modular kernels)
        or with router logits (for monolithic kernels). It delegates to
        quant_method.apply() which accesses the weights on this RoutedExperts
        instance.

        Args:
            x: Input tensor after any transforms
            topk_weights: Routing weights from router (for modular kernels)
            topk_ids: Selected expert IDs from router (for modular kernels)
            router_logits: Router logits (for monolithic kernels)
            shared_experts_input: Input for shared experts (if any)

        Returns:
            Output tensor from routed experts
        """
        quant_method = self.quant_method

        if quant_method.is_monolithic:
            # Monolithic kernels handle routing internally
            return quant_method.apply_monolithic(
                layer=self,  # Pass RoutedExperts as layer
                x=x,
                router_logits=router_logits,
            )
        else:
            # Modular kernels use pre-computed routing
            return quant_method.apply(
                layer=self,  # Pass RoutedExperts as layer
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )


# Mark the RoutedExperts weight_loader as supporting MoE-specific parameters
RoutedExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
