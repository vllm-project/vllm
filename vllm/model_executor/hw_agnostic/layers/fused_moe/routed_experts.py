# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, overload

import torch

from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.hw_agnostic.custom_op import PluggableLayer
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.expert_map_manager import (  # noqa: E501
    ExpertMapManager,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.fused_moe_method_base import (  # noqa: E501
    FusedMoEMethodBase,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.hw_agnostic.layers.fused_moe.runner.shared_experts import (  # noqa: E501
        SharedExperts,
    )


logger = init_logger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


@PluggableLayer.register("routed_experts")
class RoutedExperts(PluggableLayer):
    """
    Container for routed expert weights and execution logic.

    This module owns the expert weight parameters (w13_weight, w2_weight, scales, etc.)
    and handles:
    - Loading checkpoint weights into parameters
    - Executing routed experts via quant_method.apply()
    """

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        #
        # Extra params that are needed by quant_methods, pass along for now
        # Prefer getting these from other sources, e.g. moe_config or
        # router object
        #
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.expert_mapping = expert_mapping
        self.expert_map_manager = expert_map_manager
        self.hidden_size = moe_config.hidden_dim
        self.global_num_experts = moe_config.num_experts
        self.local_num_experts = moe_config.num_local_experts
        self.params_dtype = params_dtype

        # Register buffers for state_dict compatibility
        self.update_expert_map_info()

        # It would be good to eventually codify these in FusedMoEConfig
        # or some other config.
        self.top_k = self.moe_config.experts_per_token
        self.activation = self.moe_config.activation
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input

        self.quant_method = self._get_quant_method(
            self.layer_name,
            self.quant_config,
            self.moe_config,
        )

        # Round up hidden size and update moe_config.
        self.hidden_size, self.intermediate_size_per_partition = (
            self.quant_method.maybe_roundup_sizes(
                self.hidden_size,
                self.moe_config.intermediate_size_per_partition,
                self.moe_config.in_dtype,
                self.moe_config.moe_parallel_config,
            )
        )
        self.moe_config.hidden_dim = self.hidden_size
        self.moe_config.intermediate_size_per_partition = (
            self.intermediate_size_per_partition
        )

        if (
            self.moe_config.moe_parallel_config.enable_eplb
            and not self.quant_method.supports_eplb
        ):
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}."
            )

        moe_quant_params: dict[str, Any] = {
            "num_experts": moe_config.num_local_experts,
            "hidden_size": self.hidden_size,
            "unpadded_hidden_size": self.moe_config.hidden_dim_unpadded,
            "intermediate_size_per_partition": (
                self.moe_config.intermediate_size_per_partition
            ),
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": moe_config.num_experts,
        }

        if self._needs_intermediate_size_param(self.quant_method):
            moe_quant_params["intermediate_size_full"] = (
                self.moe_config.intermediate_size
            )

        self.quant_method.create_weights(layer=self, **moe_quant_params)

    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        # Used by FusedMoEWithLoRA to swap the underlying quant method to
        # FusedMoEModularMethod after construction.
        self.quant_method = quant_method

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        """
        Helper method to ensure quant_method is never None and
        of the proper type.
        """
        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix)
        if quant_method is None:
            quant_method = UnquantizedFusedMoEMethod(moe_config)
        assert isinstance(quant_method, FusedMoEMethodBase)
        return quant_method

    def _needs_intermediate_size_param(self, quant_method: FusedMoEMethodBase) -> bool:
        return False

    def _ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def use_ep(self) -> bool:
        return self.moe_config.moe_parallel_config.use_ep

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self._expert_map

    def update_expert_map_info(self):
        # Update local attributes from ExpertMapManager
        self.local_num_experts = self.expert_map_manager.local_num_experts
        self.expert_placement_strategy = self.expert_map_manager.placement_strategy
        self.register_buffer("_expert_map", self.expert_map_manager.expert_map)

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return None

    def update_expert_map(self):
        # Update ExpertMapManager with new EP configuration
        # The moe_parallel_config (including ep_size and ep_rank)
        # should already be updated.
        # Note: ExpertMapManager.update() recalculates expert maps and
        # reinitializes routing tables internally.
        self.expert_map_manager.update(
            self.moe_config.moe_parallel_config,
            global_num_experts=self.global_num_experts,
        )

        # Update local attributes from ExpertMapManager
        self.update_expert_map_info()

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
        if shard_id in ("w1", "w3"):
            # w1 and w3 land at indices 0 and 1 of the per-expert scale
            # tensor of shape (num_experts, 2).
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

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

        Args:
            shard_dim: dimension to shard
            expert_data: parameter for a particular expert
            shard_id: either w1, w2, or w3
            loaded_weight: checkpoint weight to load into the param
            tp_rank: tensor parallel rank
            load_full_w2: whether or not the w2 loaded should be sharded.
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

    @staticmethod
    def _get_hidden_dim(shard_dim: int, ndim: int) -> int:
        """Compute the hidden dimension index from the shard (intermediate)
        dimension and tensor rank.

        For 2D weight tensors the two data dims are (0, 1). For 3D tensors
        with an expert dimension at dim 0, they are (1, 2). ``shard_dim``
        occupies one of these; the hidden dimension is the other.
        For 1D tensors (e.g. per-channel scales) returns 0.
        """
        if ndim < 2:
            return 0
        dim_a = ndim - 2
        dim_b = ndim - 1
        if shard_dim == dim_a:
            return dim_b
        if shard_dim == dim_b:
            return dim_a
        raise ValueError(
            f"shard_dim={shard_dim} is not a valid data dimension "
            f"for a {ndim}D tensor (expected {dim_a} or {dim_b})"
        )

    @staticmethod
    def _narrow_expert_data_for_padding(
        expert_data: torch.Tensor,
        loaded_weight: torch.Tensor,
        hidden_dim: int,
        shard_dim: int | None = None,
    ) -> torch.Tensor:
        """Narrow expert_data to match loaded_weight for padded dimensions.

        When backends round up hidden_size, weight parameters are larger
        than checkpoint weights. Narrow the padded hidden dimension before
        copying. Similarly, when padding occurs on the shard (intermediate)
        dimension, narrow that dimension as well.

        Args:
            expert_data: The (possibly padded) parameter tensor to narrow.
            loaded_weight: The checkpoint weight tensor with original size.
            hidden_dim: The dimension index corresponding to hidden_size.
                Must be non-negative.
            shard_dim: The dimension index corresponding to the shard
                (intermediate) dimension. Defaults to `None`.
        """
        dims = (hidden_dim,) if shard_dim is None else (hidden_dim, shard_dim)
        if loaded_weight.ndim > 0:
            for dim in dims:
                if (
                    0 <= dim < expert_data.ndim
                    and dim < loaded_weight.ndim
                    and expert_data.shape[dim] > loaded_weight.shape[dim]
                ):
                    expert_data = expert_data.narrow(dim, 0, loaded_weight.shape[dim])
        return expert_data

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
            # When the parameter has been padded (e.g. MXFP4 rounding up
            # intermediate_size_per_partition), shard_size is the padded
            # size.  Compute the offset into the checkpoint weight using
            # the *unpadded* per-rank size so that every TP rank lands at
            # the correct slice.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # If there is no available weight to load for this TP rank
                # (can happen on last TP rank with padding), we can skip
                # loading and return early
                return
            narrow_size = min(loaded_per_rank, available)
            loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        hidden_dim = self._get_hidden_dim(shard_dim, expert_data.ndim)
        expert_data = self._narrow_expert_data_for_padding(
            expert_data,
            loaded_weight,
            hidden_dim=hidden_dim,
            shard_dim=shard_dim,
        )
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
        # Only narrow if the loaded_weight is not a scalar (0-dim tensor)
        # and we're not loading the full weight
        if not load_full and loaded_weight.ndim > 0:
            # Same padding fix as _load_w13: use unpadded per-rank size.
            tp_size = self.moe_config.moe_parallel_config.tp_size
            loaded_per_rank = loaded_weight.shape[shard_dim] // tp_size
            start_offset = loaded_per_rank * tp_rank
            available = loaded_weight.shape[shard_dim] - start_offset
            if available <= 0:
                # If there is no available weight to load for this TP rank
                # (can happen on last TP rank with padding), we can skip
                # loading and return early
                return
            narrow_size = min(loaded_per_rank, available)
            loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)
        # w2, down_proj: Load into only logical weight of w2.
        hidden_dim = self._get_hidden_dim(shard_dim, expert_data.ndim)
        expert_data = self._narrow_expert_data_for_padding(
            expert_data,
            loaded_weight,
            hidden_dim=hidden_dim,
            shard_dim=shard_dim,
        )
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

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
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)

        if expert_id == -1:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # is_transposed: shard dim is flipped for transposed weights.
        is_transposed = getattr(param, "is_transposed", False)

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            loaded_weight = loaded_weight.to(param.data.device)
            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=expert_id,
            )
            return True if return_success else None

        # Weight scales: only BLOCK and TENSOR fire on the FP8 MoE path.
        if "scale" in weight_name:
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.BLOCK.value:
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
                raise ValueError(
                    f"Unsupported FusedMoE weight-scale quant_method "
                    f"{quant_method!r}; expected 'block' or 'tensor'."
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
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
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

    @staticmethod
    def make_expert_params_mapping(
        model: torch.nn.Module,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
        routed_experts_prefix: str = "routed_experts",
    ) -> list[tuple[str, str, int, str]]:
        """
        Create expert parameter mapping for weight loading with redundant experts.

        This mapping handles the physical-to-logical expert ID conversion needed
        when loading weights with EPLB redundant experts.

        Args:
            model: The model containing the MoE layer
            ckpt_gate_proj_name: Name of gate projection in checkpoint
            ckpt_down_proj_name: Name of down projection in checkpoint
            ckpt_up_proj_name: Name of up projection in checkpoint
            num_experts: Number of logical (non-redundant) experts
            num_redundant_experts: Number of redundant experts

        Returns:
            List of tuples (param_name, weight_name, expert_id, shard_id)
            where:
            - param_name: Parameter name in the layer
            - weight_name: Weight name in checkpoint
            - expert_id: Physical expert ID
            - shard_id: Shard identifier (w1, w2, w3)
        """
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        base_layer = (
            "base_layer."
            if any(".base_layer." in name for name, _ in model.named_parameters())
            else ""
        )

        if routed_experts_prefix != "":
            routed_experts_prefix = f"{routed_experts_prefix}."

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                f"experts.{routed_experts_prefix}{base_layer}w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else f"experts.{routed_experts_prefix}{base_layer}w2_",
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.{base_layer}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer parameter as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        weights = list(self.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        # `w*_input_scale` are global activation scales (FP8 static), not
        # per-expert weights; the routing bias and hash table are likewise
        # layer-global, so exclude them from EPLB weight rearrangement.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
            "w13_input_scale",
            "w2_input_scale",
            "hash_indices_table",
        }
        NON_EXPERT_PREFIXES = ()

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not name.startswith(NON_EXPERT_PREFIXES)
            and name not in NON_EXPERT_WEIGHTS
        )

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith(NON_EXPERT_PREFIXES)
        ]

    #
    # Execution
    #

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run routed experts via the quant method's modular kernel.

        Args:
            x: Input tensor after any transforms
            topk_weights: Routing weights from the router
            topk_ids: Selected expert IDs from the router
            shared_experts: The shared experts (if any)
            shared_experts_input: Input for shared experts (if any)

        Returns:
            Output tensor from routed experts
        """
        return self.quant_method.apply(
            layer=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


# Mark the RoutedExperts weight_loader as supporting MoE-specific parameters
RoutedExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
