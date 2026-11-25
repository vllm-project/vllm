# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendor-neutral HYV4 Flash Engram setup and checkpoint loading."""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.hy_v4.weight_utils import HYV4_WEIGHTS_MAPPER

from .engram import FlashEngram, FlashNgramHashState

_EXPERT_WEIGHT = re.compile(
    r"(.*)\.experts\.(?:(\d+)\.)?(gate_proj|up_proj|down_proj|gate_up_proj)"
    r"(\.weight(?:_packed|_scale(?:_inv)?)?|_scale(?:_inv)?)?"
)


class HYV4FlashModelMixin(nn.Module):
    """Engram layers and audited checkpoint loading for `HYV4FlashModel`.

    Listed before the vendor `HYV4Model` base, which provides `config`,
    `layers` and `get_expert_mapping`.
    """

    config: Any
    layers: Any
    get_expert_mapping: Callable[[], list[tuple[str, str, int, str]]]

    def init_engram(self, vllm_config: VllmConfig, prefix: str) -> None:
        self.loaded_parts: dict[str, set[Any]] = {}
        self.engram_hash: FlashNgramHashState | None = None
        if self.config.engram_layer_ids:
            self.engram_hash = FlashNgramHashState(
                self.config, dummy=vllm_config.load_config.load_format == "dummy"
            )
            for layer_id in self.config.engram_layer_ids:
                self.layers[layer_id - 1].engram_layer = FlashEngram(
                    vllm_config,
                    self.engram_hash.layout,
                    layer_id,
                    f"{prefix}.layers.{layer_id - 1}.engram_layer",
                )
        self.token_lookback_depth = (
            self.config.engram_max_ngram_size - 1 if self.engram_hash is not None else 0
        )

    def _record_part(self, name: str, part: Any) -> None:
        parts = self.loaded_parts.setdefault(name, set())
        if part in parts or (None in parts) or (part is None and parts):
            raise ValueError(f"Duplicate Flash checkpoint weight or shard: {name}")
        parts.add(part)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        expert_mapping = {
            weight_name: (param_name, expert_id, shard)
            for param_name, weight_name, expert_id, shard in self.get_expert_mapping()
        }
        loaded: set[str] = set()
        for name, value in HYV4_WEIGHTS_MAPPER.apply(weights):
            engram = re.fullmatch(r"layers\.(\d+)\.engram_layer\.(.+)", name)
            if engram is not None:
                assert self.engram_hash is not None
                layer, leaf = engram.groups()
                if int(layer) + 1 not in (self.config.engram_layer_ids or ()):
                    raise ValueError(f"Undeclared Engram checkpoint weight: {name}")
                if leaf == "lookup_table":
                    self._record_part(name, None)
                    self.engram_hash.load_lookup(value)
                    continue
                filled = self.layers[int(layer)].engram_layer.load_weight(leaf, value)
                if filled is not None:
                    local_name, part = filled
                    target = f"layers.{layer}.engram_layer.{local_name}"
                    self._record_part(target, part)
                    loaded.add(target)
                continue
            expert = _EXPERT_WEIGHT.fullmatch(name)
            if expert:
                prefix, expert_id, projection, suffix = expert.groups()
                projections = (
                    ("gate_proj", "up_proj")
                    if projection == "gate_up_proj"
                    else (projection,)
                )
                suffix = (suffix or "").removeprefix(".weight")
                if expert_id is None:
                    if value.ndim != 3 or value.shape[0] != self.config.num_experts:
                        raise ValueError(
                            f"Invalid packed expert shape for {name}: {value.shape}"
                        )
                    expert_ids = list(range(self.config.num_experts))
                else:
                    expert_ids = [int(expert_id)]
                for index in expert_ids:
                    if not 0 <= index < self.config.num_experts:
                        raise ValueError(f"Expert index out of range: {name}")
                    tensor = value[index] if expert_id is None else value
                    pieces = tensor.chunk(len(projections), dim=-2)
                    for projection, piece in zip(projections, pieces):
                        param_name, mapped_expert, shard = expert_mapping[
                            f"experts.{index}.{projection}."
                        ]
                        target = f"{prefix}.{param_name}weight{suffix}"
                        if target not in parameters:
                            raise ValueError(f"Unknown Flash expert weight: {name}")
                        parameter = parameters[target]
                        success = parameter.weight_loader(
                            parameter,
                            piece,
                            target,
                            shard_id=shard,
                            expert_id=mapped_expert,
                            return_success=True,
                        )
                        if success:
                            self._record_part(target, (mapped_expert, shard))
                            loaded.add(target)
                continue
            if name not in parameters:
                raise ValueError(f"Unknown Flash checkpoint weight: {name}")
            parameter = parameters[name]
            weight_shard = getattr(value, "shard_id", None)
            self._record_part(name, weight_shard)
            loader: Any = getattr(parameter, "weight_loader", default_weight_loader)
            if weight_shard is None:
                loader(parameter, value)
            else:
                loader(parameter, value, weight_shard)
            loaded.add(name)
        return loaded

    def verify_loaded_parts(self) -> None:
        for layer_id in self.config.engram_layer_ids or ():
            prefix = f"layers.{layer_id - 1}.engram_layer."
            required = {
                prefix + "lookup_table",
                prefix + "multi_head_embedding.embedding.weight",
            }
            embedding = self.layers[layer_id - 1].engram_layer.multi_head_embedding
            if embedding.fp8:
                required.add(prefix + "multi_head_embedding.embedding.weight_scale_inv")
            missing = required - self.loaded_parts.keys()
            if missing:
                raise ValueError(
                    f"Missing Engram checkpoint weights: {sorted(missing)}"
                )
        for name, parts in self.loaded_parts.items():
            if None in parts:
                continue
            module_name, _, leaf = name.rpartition(".")
            module = self.get_submodule(module_name)
            expected: set[Any]
            if leaf.startswith(("w13_weight", "w2_weight")):
                expert_map = module.expert_map_manager.expert_map
                local_ids = (
                    range(self.config.num_experts)
                    if expert_map is None
                    else (expert_map >= 0).nonzero().flatten().tolist()
                )
                shards = ("w1", "w3") if leaf.startswith("w13_") else ("w2",)
                expected = {(i, s) for i in local_ids for s in shards}
            elif hasattr(module, "output_sizes"):
                expected = set(range(len(module.output_sizes)))
            else:
                # Engram stacks one checkpoint tensor per parameter row.
                expected = set(range(getattr(module, leaf).shape[0]))
            if parts != expected:
                raise ValueError(f"Incomplete Flash checkpoint shards: {name}")


class HYV4FlashForCausalLMMixin(nn.Module):
    """Audited weight loading and V2 model state for `HYV4FlashForCausalLM`.

    Listed before the vendor `HYV4ForCausalLM` base.
    """

    config: Any
    model: HYV4FlashModelMixin
    hf_to_vllm_mapper: WeightsMapper

    def init_weight_loading(self, vllm_config: VllmConfig) -> None:
        self.token_lookback_depth = self.model.token_lookback_depth
        self._weight_loading_depth = 0
        self._seen_weights: set[str] = set()
        self._loaded_weights: set[str] = set()
        self._load_config = vllm_config.load_config

    @staticmethod
    def get_model_state_cls():
        from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )

    @contextmanager
    def weight_loading_session(self):
        """Audit the complete stream, including interleaved VL tower weights."""
        outermost = self._weight_loading_depth == 0
        if outermost:
            self.model.loaded_parts.clear()
            self._seen_weights.clear()
            self._loaded_weights.clear()
        self._weight_loading_depth += 1
        try:
            yield
            if outermost:
                required = {
                    name
                    for name, _ in self.named_parameters()
                    if name.endswith((".weight", ".w13_weight", ".w2_weight"))
                }
                missing = required - self._loaded_weights
                if missing:
                    raise ValueError(
                        f"Missing Flash checkpoint weights: {sorted(missing)}"
                    )
                DefaultModelLoader(self._load_config).track_weights_loading(
                    self, self._loaded_weights.copy(), quantized=True
                )
                self.model.verify_loaded_parts()
        finally:
            self._weight_loading_depth -= 1
            if outermost:
                self._seen_weights.clear()
                self._loaded_weights.clear()

    def _target_weights(self, weights):
        for name, value in weights:
            if name in self._seen_weights:
                raise ValueError(f"Duplicate Flash checkpoint weight: {name}")
            self._seen_weights.add(name)
            engram = re.fullmatch(
                r"model\.layers\.(\d+)\.(engram|engram_layer)\.(.+)", name
            )
            if engram is not None:
                layer_number, prefix, leaf = engram.groups()
                layer_id = int(layer_number) + (prefix == "engram_layer")
                if layer_id not in (self.config.engram_layer_ids or ()):
                    raise ValueError(f"Undeclared Engram checkpoint weight: {name}")
                name = f"model.layers.{layer_id - 1}.engram_layer.{leaf}"
            skip = False
            if name.startswith("model.layers."):
                layer = name.removeprefix("model.layers.").split(".", 1)[0]
                first = self.config.num_hidden_layers
                if layer.isdigit() and first <= int(layer) < (
                    first + self.config.num_nextn_predict_layers
                ):
                    continue
            for prefix, count in (
                ("model.mtp_layers.", self.config.num_nextn_predict_layers),
                (
                    "model.dfly_layers.",
                    getattr(self.config, "dfly_num_hidden_layers", 0),
                ),
            ):
                if name.startswith(prefix):
                    local = name[len(prefix) :]
                    layer = local.removeprefix("layers.").split(".", 1)[0]
                    declared = count > 0 and layer.isdigit() and 0 <= int(layer) < count
                    if prefix == "model.dfly_layers." and count > 0:
                        declared |= local.startswith(
                            (
                                "mtp_fusion.",
                                "hidden_correction.",
                                "final_norm.",
                                "norm.",
                                "embed_tokens.",
                                "confidence_head.",
                            )
                        )
                    if not declared:
                        raise ValueError(f"Undeclared auxiliary Flash weight: {name}")
                    skip = True
                    break
            if not skip:
                yield name, value

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with self.weight_loading_session():
            loaded = AutoWeightsLoader(self).load_weights(
                self._target_weights(weights), mapper=self.hf_to_vllm_mapper
            )
            self._loaded_weights.update(loaded)
            return loaded
