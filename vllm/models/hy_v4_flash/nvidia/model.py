# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 Flash backbone using the public HYV4 checkpoint contract."""

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
)
from vllm.models.hy_v4.weight_utils import HYV4_MLP_MAPPER, HYV4_WEIGHTS_MAPPER
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig
from vllm.sequence import IntermediateTensors

from .attention import HYV4FlashAttention
from .moe import HYV4FlashFeedForward, HYV4FlashMoE

_EXPERT_WEIGHT = re.compile(
    r"(.*)\.experts\.(?:(\d+)\.)?(gate_proj|up_proj|down_proj|gate_up_proj)"
    r"(\.weight(?:_packed|_scale(?:_inv)?)?|_scale(?:_inv)?)?"
)


class HYV4FlashDecoderLayer(HYV4DecoderLayer):
    def __init__(
        self,
        config: HYV4FlashConfig,
        vllm_config: VllmConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        ihc_sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.enable_ihc = config.enable_ihc
        assert not ihc_sequence_parallel
        self.defer_reduction = False
        self.self_attn = HYV4FlashAttention(
            vllm_config,
            config,
            self.layer_idx,
            topk_indices_buffer,
            f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        dense = (config.mlp_layer_types or ())[self.layer_idx] == "dense"
        mlp_cls = HYV4FlashFeedForward if dense else HYV4FlashMoE
        self.mlp = mlp_cls(config, quant_config, f"{prefix}.mlp")
        self.block_type = "feedforward" if dense else "moe"
        self.hc_attn_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_attn_layer"
        )
        self.hc_mlp_layer = HYV4HCLayer(
            config, self.layer_idx, prefix=f"{prefix}.hc_mlp_layer"
        )


class HYV4FlashModel(HYV4Model):
    decoder_layer_cls = HYV4FlashDecoderLayer
    supports_ihc_sequence_parallel = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.loaded_parts: dict[str, set[Any]] = {}
        self.engram_hash = None
        if self.config.engram_layer_ids:
            from .engram import FlashEngram, FlashNgramHashState

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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.engram_hash is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        hashes = input_ids.new_empty(
            (
                len(self.config.engram_layer_ids),
                input_ids.numel(),
                self.engram_hash.offsets.shape[1],
            ),
            dtype=torch.int32,
        )
        self.engram_hash.prepare_hashes_with_output(
            input_ids,
            lookback_token_ids,
            self.layers[0].self_attn.mla_attn.layer_name,
            hashes,
        )
        hash_indices = {
            layer_id - 1: index
            for index, layer_id in enumerate(self.config.engram_layer_ids)
        }
        for index, layer in enumerate(self.layers):
            if index in hash_indices:
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                hidden = layer.engram_layer(hidden, hashes[hash_indices[index]])
            hidden, _ = layer(positions, hidden, None)
        return self.norm(self.hc_head(hidden))

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
                module = self.layers[int(layer)].engram_layer
                for local_name in module.load_weights([(leaf, value)]):
                    target = f"layers.{layer}.engram_layer.{local_name}"
                    self._record_part(target, None)
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
            else:
                expected = set(range(len(module.output_sizes)))
            if parts != expected:
                raise ValueError(f"Incomplete Flash checkpoint shards: {name}")


class HYV4FlashForCausalLM(HYV4ForCausalLM):
    model_cls = HYV4FlashModel
    supports_lora = False  # type: ignore[assignment]
    supports_pp = False  # type: ignore[assignment]
    requires_raw_input_tokens = True
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    hf_to_vllm_mapper = HYV4_MLP_MAPPER | WeightsMapper(
        orig_to_new_substr={".compressed_indexer.": ".indexer."},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
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
