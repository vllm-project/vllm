# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP O-Proj tensor parallelism.

:class:`PCPOProjTP` sits between an attention implementation and its
row-parallel O-Proj linear layer. It owns every PCP-specific step: sharding the
resident weight over TP x PCP, selecting the communication path per batch,
launching the asynchronous weight gather early in attention, and restoring the
local shard after O-Proj. The linear layer and its quantization method stay
unchanged apart from declaring :class:`TPWeightGatherSpec` descriptors through
:class:`TPWeightSwitchMixin`.
"""

import enum
from typing import Any
from weakref import WeakValueDictionary

import torch
from torch.nn.parameter import Parameter

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.quantization.tp_weight_switch import (
    TPWeightSwitchMixin,
    TPWeightSwitchState,
)


class PCPOProjPath(enum.Enum):
    """Communication path used by one O-Proj invocation."""

    WEIGHT_ALL_GATHER = "weight_all_gather"
    """Batches with prefill: all-gather the TP weight shard over PCP, then run
    the standard TP-local O-Proj on the rank-local tokens."""

    TOKEN_EXCHANGE = "token_exchange"
    """Decode-only batches: keep the resident PCP weight shard and exchange
    tokens over PCP instead. Decode queries are replicated on every PCP rank,
    so each rank projects its feature shard and the partial outputs are summed
    over PCP."""


def get_layer_attn_metadata(layer_name: str) -> Any:
    """Return the attention metadata of ``layer_name`` in the current forward."""
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(layer_name)
    if isinstance(attn_metadata, list):
        return attn_metadata[0].get(layer_name)
    return attn_metadata


def get_pcp_batch_has_prefill(attn_metadata: Any = None) -> bool:
    """Resolve a PCP-group-consistent prefill decision for the current batch.

    The pre-partition batch type from the forward context is preferred so every
    PCP rank selects the same collective, even when PCP token partitioning
    leaves different local batch shapes on different ranks.
    """
    global_has_prefill = get_forward_context().global_has_prefill
    if global_has_prefill is not None:
        return global_has_prefill
    return attn_metadata is not None and getattr(attn_metadata, "num_prefills", 0) > 0


class _PostLoadHookLinearMethod(LinearMethodBase):
    """Delegate a linear method and run a hook after post-load processing."""

    def __init__(self, quant_method: LinearMethodBase, owner: "PCPOProjTP") -> None:
        self.quant_method = quant_method
        self.owner = owner

    @property
    def supports_pre_processed_weights(self) -> bool:  # type: ignore[override]
        return self.quant_method.supports_pre_processed_weights

    def __getattr__(self, name: str) -> Any:
        """Preserve access to backend-specific method configuration."""
        quant_method = self.__dict__.get("quant_method")
        if quant_method is None:
            raise AttributeError(name)
        return getattr(quant_method, name)

    def create_weights(self, *args, **kwargs) -> None:
        self.quant_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)
        self.owner.enable_weight_switch()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, x, bias)


class PCPOProjTP:
    """Intermediate O-Proj layer that shards the weight over TP x PCP.

    Usage from an attention implementation:

    1. Build the linear layer with :meth:`linear_shard_kwargs` and wrap it with
       :meth:`maybe_create`.
    2. Call :meth:`pre_attention_hook` once per forward, before the main
       attention computation, so the weight gather overlaps with it.
    3. Call the instance exactly like the original O-Proj linear layer.
    """

    _full_weight_buffers: WeakValueDictionary[tuple[Any, ...], torch.Tensor] = (
        WeakValueDictionary()
    )

    @staticmethod
    def is_enabled(vllm_config: VllmConfig) -> bool:
        return vllm_config.parallel_config.enable_pcp_o_proj_tp

    @classmethod
    def linear_shard_kwargs(cls, vllm_config: VllmConfig) -> dict[str, Any]:
        """Return ``RowParallelLinear`` kwargs that shard O-Proj over TP x PCP.

        The flattened rank ``tp_rank * pcp_size + pcp_rank`` keeps each PCP
        shard inside the corresponding TP shard, so checkpoints load unchanged.
        Returns no overrides when the feature is disabled.
        """
        if not cls.is_enabled(vllm_config):
            return {}
        pcp_group = get_pcp_group()
        return {
            "tp_rank": get_tensor_model_parallel_rank() * pcp_group.world_size
            + pcp_group.rank_in_group,
            "tp_size": get_tensor_model_parallel_world_size() * pcp_group.world_size,
        }

    @classmethod
    def maybe_create(
        cls, linear: RowParallelLinear, vllm_config: VllmConfig
    ) -> "PCPOProjTP | None":
        if not cls.is_enabled(vllm_config):
            return None
        return cls(linear)

    def __init__(self, linear: RowParallelLinear) -> None:
        if linear.bias is not None:
            raise ValueError("PCP O-Proj TP requires bias=False.")
        if not linear.input_is_parallel:
            raise ValueError("PCP O-Proj TP requires input_is_parallel=True.")

        self.linear = linear
        self.pcp_group = get_pcp_group()
        self.pcp_rank = self.pcp_group.rank_in_group
        self.pcp_size = self.pcp_group.world_size
        if self.pcp_size <= 1:
            raise ValueError("PCP O-Proj TP requires prefill context parallelism.")
        self.tp_size = get_tensor_model_parallel_world_size()
        if linear.tp_size != self.tp_size * self.pcp_size:
            raise ValueError(
                "PCP O-Proj TP requires the linear layer to be built with "
                "PCPOProjTP.linear_shard_kwargs()."
            )
        self.tp_input_size = divide(linear.input_size, self.tp_size)

        self._buffer_scope = id(get_current_vllm_config())
        self._switch_method: TPWeightSwitchMixin | None = None
        self._switch_state: TPWeightSwitchState | None = None
        self._path: PCPOProjPath | None = None

        assert isinstance(linear.quant_method, LinearMethodBase)
        linear.quant_method = _PostLoadHookLinearMethod(linear.quant_method, self)
        if linear.quant_config is None:
            # Allocate the shared buffers before memory profiling.
            self.enable_weight_switch()

    def enable_weight_switch(self) -> None:
        """(Re)build local/full weight aliases from the post-load layout."""
        if self._switch_state is not None and self._switch_state.handles:
            raise RuntimeError(
                "Cannot rebuild PCP O-Proj weight state while a gather is pending."
            )
        quant_method = self.linear.quant_method
        if isinstance(quant_method, _PostLoadHookLinearMethod):
            quant_method = quant_method.quant_method
        if not (
            isinstance(quant_method, TPWeightSwitchMixin)
            and quant_method.supports_tp_weight_switch
        ):
            raise RuntimeError(
                "PCP O-Proj TP does not support the selected O-Proj linear method "
                f"or post-load layout: {type(quant_method).__name__}. Quantized "
                "checkpoints may exclude O-Proj from weight quantization."
            )
        self._switch_state = None
        self._switch_state = quant_method.enable_tp_weight_switch(
            self.linear,
            self.pcp_size,
            pool=self._full_weight_buffers,
            pool_key_prefix=(
                self._buffer_scope,
                type(quant_method).__qualname__,
                "pcp_o_proj",
            ),
        )
        self._switch_method = quant_method

    def _weight_switch(self) -> tuple[TPWeightSwitchMixin, TPWeightSwitchState]:
        if self._switch_method is None or self._switch_state is None:
            self.enable_weight_switch()
        assert self._switch_method is not None
        assert self._switch_state is not None
        return self._switch_method, self._switch_state

    @staticmethod
    def select_path(has_prefill: bool) -> PCPOProjPath:
        """Prefill/mixed batches gather weights; decode-only batches move tokens."""
        if has_prefill:
            return PCPOProjPath.WEIGHT_ALL_GATHER
        return PCPOProjPath.TOKEN_EXCHANGE

    def pre_attention_hook(
        self,
        attn_metadata: Any = None,
        *,
        layer_name: str | None = None,
    ) -> PCPOProjPath:
        """Select this forward's path and start the weight gather if needed.

        Attention calls this before its main computation so the asynchronous
        gather overlaps with it. ``attn_metadata`` (or ``layer_name`` to look
        it up) is only a fallback when the forward context does not carry the
        PCP-global batch type.
        """
        if self._path is not None:
            raise RuntimeError(
                "PCP O-Proj pre_attention_hook() was called before the previous "
                "forward consumed it."
            )
        if attn_metadata is None and layer_name is not None:
            attn_metadata = get_layer_attn_metadata(layer_name)
        path = self.select_path(get_pcp_batch_has_prefill(attn_metadata))
        if path is PCPOProjPath.WEIGHT_ALL_GATHER:
            method, state = self._weight_switch()
            method.all_gather_tp_weight(state, self.pcp_group)
        self._path = path
        return path

    def _forward_weight_all_gather(self, input_: torch.Tensor) -> torch.Tensor:
        method, state = self._weight_switch()
        method.wait_tp_weight_all_gather(state)
        linear = self.linear
        local_input_size = linear.input_size_per_partition
        method.switch_tp_weight(linear, state, use_full_weight=True)
        linear.input_size_per_partition = self.tp_input_size
        try:
            return linear.quant_method.apply(linear, input_, None)
        finally:
            linear.input_size_per_partition = local_input_size
            method.switch_tp_weight(linear, state, use_full_weight=False)

    def _forward_token_exchange(self, input_: torch.Tensor) -> torch.Tensor:
        linear = self.linear
        shard_size = linear.input_size_per_partition
        input_parallel = input_.narrow(
            -1, self.pcp_rank * shard_size, shard_size
        ).contiguous()
        output_parallel = linear.quant_method.apply(linear, input_parallel, None)
        return self.pcp_group.all_reduce(output_parallel)

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """Drop-in replacement for ``RowParallelLinear.forward``."""
        path, self._path = self._path, None
        if path is None:
            raise RuntimeError(
                "Attention must call PCPOProjTP.pre_attention_hook() before "
                "PCP O-Proj forward."
            )
        if input_.shape[-1] != self.tp_input_size:
            if path is PCPOProjPath.WEIGHT_ALL_GATHER:
                method, state = self._weight_switch()
                method.wait_tp_weight_all_gather(state)
            raise ValueError(
                "PCP O-Proj expected a TP-sharded input with last dimension "
                f"{self.tp_input_size}, but got {input_.shape[-1]}."
            )

        if path is PCPOProjPath.WEIGHT_ALL_GATHER:
            output_parallel = self._forward_weight_all_gather(input_)
        else:
            output_parallel = self._forward_token_exchange(input_)

        if self.linear.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.linear.return_bias:
            return output
        return output, None

    __call__ = forward

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(pcp_size={self.pcp_size}, "
            f"tp_size={self.tp_size}, linear={self.linear.extra_repr()})"
        )
