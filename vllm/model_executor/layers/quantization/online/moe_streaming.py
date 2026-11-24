# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize the experts of an online MoE layer one at a time, as they load.

An online MoE layer never holds its full-precision experts. The checkpoint
tensors of each expert are buffered as they arrive, unsharded, and the expert
is quantized as soon as all of them are there:

1. Decode: dequantize the expert's checkpoint tensors (identity for
   full-precision checkpoints), still unsharded.
2. Statistics: compute the quantization statistics (e.g. an amax) on the whole
   expert, so that the scales do not depend on the TP configuration.
3. Shard: slice the expert for this TP rank and pad it, as the layer's loader
   does for full-precision weights.
4. Quantize: write the quantized expert into its slot of the layer outputs.

The peak memory is the quantized layer plus one unsharded full-precision expert.
"""

from collections.abc import Callable, Iterable
from inspect import BoundArguments
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.model_loader.reload.meta import (
    SKIP_LOAD_TENSORS,
    get_numel_loaded,
    materialize_meta_tensor,
)
from vllm.model_executor.model_loader.reload.utils import (
    get_layer_tensors,
    get_tensor_load_numel,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

# Staging parameters of the fused shared experts stored quantized.
_SHARED_PREFIX = "online_"


def _stem(param_name: str) -> str:
    return "w2" if param_name.removeprefix(_SHARED_PREFIX).startswith("w2_") else "w13"


def _shard_ids(layer: "RoutedExperts", param_name: str) -> tuple[str, ...]:
    if _stem(param_name) == "w2":
        return ("w2",)
    return ("w1", "w3") if layer.moe_config.is_act_and_mul else ("w1",)


class _ExpertCheckpoint:
    """One unsharded expert in checkpoint format, which the requantization
    source dequantizes as it would the layer (with a single expert)."""

    def __init__(self, layer: "RoutedExperts", tensors: dict[str, torch.Tensor]):
        self._layer = layer
        self.__dict__.update(tensors)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


class ExpertWeightStream:
    """Loads the experts of an online MoE layer and quantizes each of them as
    soon as its checkpoint tensors are loaded.

    Expert weights are routed here by the layerwise loader. Other tensors (e.g.
    biases) are loaded as usual, and activation scales of the checkpoint, which
    online quantization does not use, are dropped.
    """

    def __init__(self, method: "OnlineMoEMethodBase", device: torch.device) -> None:
        self.method = method
        self.device = device
        # Unsharded checkpoint tensors of the experts not quantized yet, by
        # local expert, then by `(parameter name, shard id)`.
        self.pending: dict[int, dict[tuple[str, str], torch.Tensor]] = {}
        self.num_quantized = 0
        self.outputs: tuple[torch.Tensor, ...] | None = None
        # Loaded elements of the tensors loaded as usual, and of the expert
        # weights loaded whole into the materialized layer.
        self.other_numel = 0
        self.materialized_numel = 0

    # Loading

    @property
    def has_loaded(self) -> bool:
        return (
            self.num_quantized > 0
            or bool(self.pending)
            or self.other_numel > 0
            or self.materialized_numel > 0
        )

    def load(
        self,
        layer: "RoutedExperts",
        param_name: str,
        loader: Callable,
        args: BoundArguments,
    ) -> Any:
        """Load a checkpoint tensor of the layer, through its `loader`."""
        if param_name not in self._expert_parameters(layer, param_name):
            if param_name in self.method.expert_activation_scale_names:
                return True if args.arguments.get("return_success") else None
            numel, result = self._load_materialized(layer, param_name, loader, args)
            self.other_numel += numel
            return result

        if "expert_id" not in args.arguments:
            # Not loaded per expert, e.g. the layer's tensor through the default
            # loader: the layer is quantized from its materialized weights.
            numel, result = self._load_materialized(layer, param_name, loader, args)
            self.materialized_numel += numel
            return result

        success = bool(args.arguments.get("return_success"))
        loaded_weight = args.arguments["loaded_weight"]
        shard_id = args.arguments["shard_id"]
        if loaded_weight.dim() != 3:
            loaded = self._load_expert(
                layer, param_name, args.arguments["expert_id"], shard_id, loaded_weight
            )
        else:
            # All the experts at once, as `RoutedExperts.weight_loader` tells them
            # apart: load them one by one.
            loaded = any(
                [
                    self._load_expert(layer, param_name, expert, shard_id, weight)
                    for expert, weight in enumerate(loaded_weight)
                ]
            )
        if success:
            return loaded
        return None

    def _load_expert(
        self,
        layer: "RoutedExperts",
        param_name: str,
        global_expert_id: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Buffer an unsharded checkpoint tensor of one expert, and quantize the
        expert once all of its tensors are loaded. False if it is not local."""
        expert = layer._map_global_expert_id_to_local_expert_id(global_expert_id)
        if expert == -1:
            return False
        tensors = self.pending.setdefault(expert, {})
        key = (param_name, shard_id)
        if key in tensors:
            raise ValueError(f"Expert {expert} loaded {key} twice.")
        tensors[key] = loaded_weight
        if tensors.keys() == self._expected_keys(layer, param_name):
            self.quantize_expert(layer, expert, self.pending.pop(expert))
        return True

    def _load_materialized(
        self,
        layer: "RoutedExperts",
        param_name: str,
        loader: Callable,
        args: BoundArguments,
    ) -> tuple[int, Any]:
        """Load into the parameter as usual, materializing it first. Returns the
        number of loaded elements and the loader result."""
        param = getattr(layer, param_name)
        if param.is_meta:
            with torch.device(self.device):
                param = materialize_meta_tensor(param)
            setattr(layer, param_name, param)
        args.arguments["param"] = param
        return get_numel_loaded(loader, args)

    def _expert_parameters(
        self, layer: "RoutedExperts", param_name: str
    ) -> tuple[str, ...]:
        """Expert weight parameters of the kind of expert `param_name` loads:
        fused shared experts stored quantized, else the routed experts."""
        if param_name.startswith(_SHARED_PREFIX):
            assert layer.shared_expert_online_loader is not None
            return tuple(layer.shared_expert_online_loader.staging_parameters)
        return self.method.expert_weight_names

    def _expected_keys(
        self, layer: "RoutedExperts", param_name: str
    ) -> set[tuple[str, str]]:
        return {
            (name, shard_id)
            for name in self._expert_parameters(layer, param_name)
            for shard_id in _shard_ids(layer, name)
        }

    def is_complete(self, layer: "RoutedExperts") -> bool:
        method = self.method
        experts_loaded = (
            self.num_quantized == layer.local_num_experts
            or self.materialized_numel >= self._numel(layer, method.expert_weight_names)
        )
        others = (
            name
            for name in get_layer_tensors(layer)
            if name not in method.expert_weight_names
            and name not in method.expert_activation_scale_names
            and not name.startswith(_SHARED_PREFIX)
        )
        return experts_loaded and self.other_numel >= self._numel(layer, others)

    @staticmethod
    def _numel(layer: "RoutedExperts", names: Iterable[str]) -> int:
        tensors = get_layer_tensors(layer)
        return sum(
            get_tensor_load_numel(tensors[name])
            for name in names
            if name in tensors and name not in SKIP_LOAD_TENSORS
        )

    def check_complete(self, layer: "RoutedExperts") -> None:
        if self.pending or self.num_quantized != layer.local_num_experts:
            missing = {
                expert: sorted(
                    self._expected_keys(layer, next(iter(tensors))[0]) - tensors.keys()
                )
                for expert, tensors in self.pending.items()
            }
            raise ValueError(
                f"{getattr(layer, 'layer_name', type(layer).__name__)}: the "
                "checkpoint is missing expert weights to "
                f"quantize ({self.num_quantized} of {layer.local_num_experts} "
                f"experts quantized, missing tensors: {missing})."
            )

    # Quantization

    def quantize_expert(
        self,
        layer: "RoutedExperts",
        expert: int,
        tensors: dict[tuple[str, str], torch.Tensor],
    ) -> None:
        """Quantize one local expert from its unsharded checkpoint tensors."""
        w13, w2 = self._decode(layer, tensors)
        statistics = self.method.expert_statistics(w13[None], w2[None])
        self._quantize(layer, expert, *self._shard(layer, w13, w2), statistics)

    def _quantize(
        self,
        layer: "RoutedExperts",
        expert: int,
        w13: torch.Tensor,
        w2: torch.Tensor,
        statistics: dict[str, torch.Tensor],
    ) -> None:
        """Quantize this rank's padded `(w13, w2)` of one expert into its slot."""
        quantized = self.method.quantize_experts(layer, w13[None], w2[None], statistics)
        self._store(layer, expert, tuple(tensor[0] for tensor in quantized))

    def quantized_weights(self, layer: "RoutedExperts") -> tuple[torch.Tensor, ...]:
        """The quantized outputs of all local experts."""
        if self.num_quantized == 0 and not self.pending:
            # The weights were not loaded through the layer's loaders, e.g.
            # with dummy weights: quantize the materialized layer.
            self._quantize_materialized(layer)
        self.check_complete(layer)
        assert self.outputs is not None
        return self.outputs

    def _decode(
        self, layer: "RoutedExperts", tensors: dict[tuple[str, str], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unsharded full-precision `(w13, w2)` of one expert."""
        names = {name for name, _ in tensors}
        expert = {name: self._assemble(layer, name, tensors) for name in names}

        if any(name.startswith(_SHARED_PREFIX) for name in names):
            loader = layer.shared_expert_online_loader
            assert loader is not None
            w13, w2 = (
                loader.dequantize(
                    expert[f"{_SHARED_PREFIX}{stem}_weight"],
                    expert.get(f"{_SHARED_PREFIX}{stem}_weight_scale"),
                )
                for stem in ("w13", "w2")
            )
        elif (source := self.method.requantization_source) is None:
            w13, w2 = expert["w13_weight"], expert["w2_weight"]
        else:
            checkpoint = _ExpertCheckpoint(
                layer, {name: tensor[None] for name, tensor in expert.items()}
            )
            ((_, w13, w2),) = source.dequantize_weight(checkpoint)
            w13, w2 = w13[0], w2[0]
        return w13.to(layer.orig_dtype), w2.to(layer.orig_dtype)

    def _assemble(
        self,
        layer: "RoutedExperts",
        name: str,
        tensors: dict[tuple[str, str], torch.Tensor],
    ) -> torch.Tensor:
        """One unsharded expert of parameter `name`, from its w1/w3 or w2
        checkpoint tensors, laid out as the parameter."""
        param = getattr(layer, name)
        shards = [
            tensors[(name, shard_id)].to(self.device)
            for shard_id in _shard_ids(layer, name)
        ]
        if param.dim() == 1:
            # One scalar per expert.
            return shards[0].reshape(())
        if param.shape[1:] == (len(shards),):
            # One scalar per w1/w3 shard.
            return torch.stack([shard.reshape(()) for shard in shards])
        return torch.cat(shards, dim=int(getattr(param, "is_transposed", False)))

    @staticmethod
    def _shard(
        layer: "RoutedExperts", w13: torch.Tensor, w2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This TP rank's padded `(w13, w2)` of one unsharded expert."""
        config = layer.moe_config
        shard_ids = ("w1", "w3") if config.is_act_and_mul else ("w1",)
        intermediate_size = config.intermediate_size_per_partition
        local_w13 = w13.new_zeros(len(shard_ids) * intermediate_size, config.hidden_dim)
        local_w2 = w2.new_zeros(config.hidden_dim, intermediate_size)
        for shard_id, weight in zip(shard_ids, w13.chunk(len(shard_ids)), strict=True):
            layer._load_model_weight_or_group_weight_scale(
                shard_dim=0,
                expert_data=local_w13,
                shard_id=shard_id,
                loaded_weight=weight,
                tp_rank=config.tp_rank,
            )
        layer._load_model_weight_or_group_weight_scale(
            shard_dim=1,
            expert_data=local_w2,
            shard_id="w2",
            loaded_weight=w2,
            tp_rank=config.tp_rank,
        )
        return local_w13, local_w2

    def _store(
        self,
        layer: "RoutedExperts",
        expert: int,
        quantized: tuple[torch.Tensor, ...],
    ) -> None:
        if self.outputs is None:
            self.outputs = tuple(
                tensor.new_empty((layer.local_num_experts, *tensor.shape))
                for tensor in quantized
            )
        for output, tensor in zip(self.outputs, quantized, strict=True):
            output[expert] = tensor
        self.num_quantized += 1

    def _quantize_materialized(self, layer: "RoutedExperts") -> None:
        """Quantize the experts of a layer materialized with its weights, one at
        a time. Its weights are TP-local: the statistics are those of the shard.
        """
        method = self.method
        loader = getattr(layer, "shared_expert_online_loader", None)
        num_shared = layer.expert_map_manager.num_fused_shared_experts
        num_routed = layer.local_num_experts - num_shared
        staged = loader is not None and bool(loader.staging_parameters)
        source = method.requantization_source

        def staged_shared_expert(stem: str, index: int) -> torch.Tensor:
            weight = getattr(layer, f"{_SHARED_PREFIX}{stem}_weight")
            scale = getattr(layer, f"{_SHARED_PREFIX}{stem}_weight_scale", None)
            assert loader is not None
            return loader.dequantize(
                weight[index], None if scale is None else scale[index]
            )

        if source is None:
            # Full-precision weights are created padded: zero the padding, which
            # e.g. dummy weights fill.
            method._zero_padding(layer)
        for expert in range(layer.local_num_experts):
            if staged and expert >= num_routed:
                w13 = staged_shared_expert("w13", expert - num_routed)[None]
                w2 = staged_shared_expert("w2", expert - num_routed)[None]
            elif source is None:
                w13 = layer.w13_weight[expert : expert + 1]
                w2 = layer.w2_weight[expert : expert + 1]
            else:
                # The source holds the routed experts only.
                checkpoint = _ExpertCheckpoint(
                    layer,
                    {
                        name: param[expert : expert + 1]
                        for name, param in layer.named_parameters(recurse=False)
                        if param.dim() > 0 and param.shape[0] == num_routed
                    },
                )
                ((_, w13, w2),) = source.dequantize_weight(checkpoint)
            w13, w2 = w13.to(layer.orig_dtype), w2.to(layer.orig_dtype)
            statistics = method.expert_statistics(w13, w2)
            if source is not None:
                # Dequantized experts (and staged shared experts, only with a
                # source) have the unpadded checkpoint sizes.
                w13, w2 = method._pad_requantized_weights(layer, w13, w2, 1)
            self._quantize(layer, expert, w13[0], w2[0], statistics)
