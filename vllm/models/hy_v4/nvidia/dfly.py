# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from copy import deepcopy
from itertools import chain
from typing import Any

import torch
from compressed_tensors.quantization import QuantizationArgs
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_default_torch_dtype

from ..common.mrope import set_token_rope
from ..dfly_config import HYV4DFlySpeculatorConfig
from .model import HYV4DecoderLayer
from .moe import HYV4FeedForward


class HYV4DFlyQuantizationConfig(CompressedTensorsConfig):
    """Match draft linears against their checkpoint paths, without rewriting regexes."""

    model_prefix: str

    @classmethod
    def for_model(
        cls, source: CompressedTensorsConfig, model_prefix: str
    ) -> "HYV4DFlyQuantizationConfig":
        if source.config is None:
            raise ValueError("DFly requires serialized compressed-tensors metadata")
        config = cls.from_config(deepcopy(source.config))
        assert isinstance(config, cls)
        config.model_prefix = f"{model_prefix}."
        config.packed_modules_mapping = source.packed_modules_mapping.copy()
        return config

    def get_scheme_dict(self, layer: nn.Module, layer_name: str | None = None):
        if layer_name is not None and layer_name.startswith(self.model_prefix):
            local_name = layer_name.removeprefix(self.model_prefix)
            local_name = local_name.removeprefix("layers.")
            layer_name = f"model.dfly_layers.{local_name}"
        scheme = super().get_scheme_dict(layer, layer_name)
        if scheme is not None:
            weights = scheme["weights"]
            activations = scheme["input_activations"]
            if not (
                isinstance(weights, QuantizationArgs)
                and isinstance(activations, QuantizationArgs)
                and self._is_fp8_w8a8(weights, activations)
                and weights.num_bits == activations.num_bits == 8
                and weights.block_structure == [128, 128]
                and activations.dynamic
            ):
                raise ValueError("HYV4 DFly draft layers require BF16 or blockwise FP8")
        return scheme


class HYV4DFlyFusion(nn.Module):
    def __init__(
        self,
        config: HYV4Config,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * d, d, bias=False, prefix=f"{prefix}.eh_proj", quant_config=quant_config
        )
        self.gate = nn.Parameter(torch.empty(d), requires_grad=False)

    def forward(
        self, anchors: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        fused, _ = self.eh_proj(
            torch.cat((self.enorm(anchors), self.hnorm(target_hidden)), -1)
        )
        return anchors + self.gate * fused


class HYV4DFlyHiddenCorrection(nn.Module):
    def __init__(
        self,
        config: HYV4Config,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        d = config.hidden_size
        intermediate = config.hidden_correction_intermediate_size
        self.hidden_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [intermediate, intermediate],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate,
            d,
            bias=False,
            prefix=f"{prefix}.down_proj",
            quant_config=quant_config,
        )

    def forward(
        self, hidden: torch.Tensor, previous_embed: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            (self.hidden_norm(hidden), self.embed_norm(previous_embed)), -1
        )
        gate_up, _ = self.gate_up_proj(combined)
        delta, _ = self.down_proj(self.act(gate_up))
        return hidden + delta


class HYV4DFlyFeedForward(HYV4FeedForward):
    """Keep the module identity while waiting for the streamed FFN width."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self._kwargs: dict[str, Any] = dict(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
        )
        self.register_buffer("_placement", torch.empty(0), persistent=False)

    def materialize(self, intermediate_size: int) -> None:
        with (
            torch.device(self._placement.device),
            set_default_torch_dtype(self._placement.dtype),
        ):
            mlp = HYV4FeedForward(intermediate_size=intermediate_size, **self._kwargs)
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn


class HYV4DFlyDecoderLayer(HYV4DecoderLayer):
    feed_forward_cls: type[HYV4FeedForward] = HYV4DFlyFeedForward


class HYV4DFlyModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        spec = vllm_config.speculative_config
        assert spec is not None
        config = deepcopy(spec.draft_model_config.hf_config)
        if config.model_type != "hy_v4_dfly":
            raise ValueError("Expected a derived hy_v4_dfly configuration")
        self.config = config
        self._vllm_config = vllm_config
        self.ffn_width: int | None = None
        self.declared_ffn_width = getattr(config, "dfly_intermediate_size", None)
        self.declared_post_proj_norm = config.attn_post_proj_norm
        dummy = vllm_config.load_config.load_format == "dummy"
        config.attn_post_proj_norm = (
            config.attn_post_proj_norm is True
            if dummy
            else config.attn_post_proj_norm is not False
        )
        quant_config = vllm_config.quant_config
        if isinstance(quant_config, CompressedTensorsConfig):
            quant_config = HYV4DFlyQuantizationConfig.for_model(quant_config, prefix)
        elif isinstance(quant_config, Fp8Config):
            quant_config = deepcopy(quant_config)
            quant_config.apply_vllm_mapper(
                WeightsMapper(
                    orig_to_new_prefix={
                        "model.dfly_layers.mtp_fusion.": f"{prefix}.mtp_fusion.",
                        "model.dfly_layers.hidden_correction.": (
                            f"{prefix}.hidden_correction."
                        ),
                        "model.dfly_layers.": f"{prefix}.layers.",
                    }
                )
            )
        self.quant_config = quant_config
        self.query_width = spec.num_speculative_tokens + 1
        config.index_topk = round_up(
            config.target_index_topk
            + (config.context_local_window or 0)
            + self.query_width,
            128,
        )
        config.dfly_external_topk = True
        self.embed_tokens: VocabParallelEmbedding | None = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        max_rows = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices",
            torch.full(
                (config.num_hidden_layers, max_rows, config.index_topk),
                -1,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [
                HYV4DFlyDecoderLayer(
                    config,
                    vllm_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                    topk_indices_buffer=self.topk_indices[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        for layer in self.layers:
            attn = layer.self_attn
            if (
                attn.learnable_sink
                and not attn.mla_attn.get_attn_backend().supports_sink()
            ):
                raise ValueError("DFly requires a backend that applies learnable sinks")
            attn.use_hpc_gated_mla = False
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_fusion = HYV4DFlyFusion(
            config, f"{prefix}.mtp_fusion", quant_config=quant_config
        )
        self.hidden_correction = HYV4DFlyHiddenCorrection(
            config, f"{prefix}.hidden_correction", quant_config=quant_config
        )
        if dummy or self.declared_ffn_width is not None:
            self.materialize_ffn(config.intermediate_size)

    def materialize_ffn(self, width: int) -> None:
        if width < 1 or width % get_tensor_model_parallel_world_size():
            raise ValueError("DFly FFN width must be positive and divisible by TP")
        if self.declared_ffn_width not in (None, width):
            raise ValueError("dfly_intermediate_size disagrees with checkpoint")
        if self.ffn_width is not None:
            if self.ffn_width != width:
                raise ValueError("Inconsistent DFly FFN widths in checkpoint")
            return
        with set_current_vllm_config(self._vllm_config):
            for layer in self.layers:
                layer.mlp.materialize(width)
        self.ffn_width = width
        self.config.intermediate_size = width

    def resolve_post_proj_norm(self, seen: set[str]) -> set[str]:
        names = {
            f"layers.{i}.self_attn.post_proj_norm.weight"
            for i in range(len(self.layers))
        }
        loaded = names & seen
        if loaded and loaded != names:
            raise ValueError("Inconsistent DFly attention post-projection norms")
        if not loaded and self.declared_post_proj_norm is not True:
            for layer in self.layers:
                layer.self_attn.post_proj_norm = None
            self.config.attn_post_proj_norm = False
            return names
        self.config.attn_post_proj_norm = True
        return set()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "DFly embedding is not bound"
            inputs_embeds = self.embed_tokens(input_ids)
        hidden = inputs_embeds
        if len(self.layers):
            set_token_rope(self.layers[0].self_attn.rotary_emb, positions, hidden)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden + residual)


class HYV4DFlyForCausalLM(nn.Module):
    speculator_config_class = HYV4DFlySpeculatorConfig
    checkpoint_prefix = "model.dfly_layers."
    fall_back_to_pt_during_load = False
    allow_patterns_overrides = ["*.safetensors"]
    has_own_lm_head = False
    draft_id_to_target_id = None
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    _weight_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
            ".q_a_proj.": (".fused_qkv_a_proj.", 0),
            ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
        }
    )
    _ignored_prefixes = ("confidence_head.",)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if not current_platform.is_cuda():
            raise NotImplementedError("HYV4 DFly currently requires NVIDIA CUDA")
        self.model = HYV4DFlyModel(vllm_config, maybe_prefix(prefix, "model"))
        self.config = self.model.config
        self.has_own_embed_tokens = False
        self.lm_head: ParallelLMHead | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.logits_processor.use_all_gather = True

    @classmethod
    def is_unused_checkpoint_weight(cls, name: str) -> bool:
        return not name.startswith(cls.checkpoint_prefix) or name.removeprefix(
            cls.checkpoint_prefix
        ).startswith(cls._ignored_prefixes)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.model.embed_tokens is not None, "DFly embedding is not bound"
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def apply_hidden_correction(
        self, hidden: torch.Tensor, previous_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model.hidden_correction(hidden, self.embed_input_ids(previous_ids))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError(
                "DFly requires the target LM head before computing logits"
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.mla_attn.layer_name for layer in self.model.layers]

    def _required_weights(self) -> dict[str, torch.dtype]:
        required: dict[str, torch.dtype] = {}
        for name, param in self.model.named_parameters():
            # No checkpoint KV scheme is supported; these backend placeholders
            # retain their initialized unit scales and zero points.
            if ".mla_attn." in name and name.rsplit(".", 1)[-1] in (
                "q_scale",
                "k_scale",
                "v_scale",
                "q_zero_point",
                "k_zero_point",
                "v_zero_point",
            ):
                continue
            dtype = (
                param.dtype
                if param.dtype == torch.float8_e4m3fn
                or name.endswith((".weight_scale", ".weight_scale_inv"))
                else torch.bfloat16
            )
            if ".gate_up_proj." in name:
                required.update(
                    (name.replace(".gate_up_proj.", part), dtype)
                    for part in (".gate_proj.", ".up_proj.")
                )
            elif ".fused_qkv_a_proj." in name:
                required.update(
                    (name.replace(".fused_qkv_a_proj.", part), dtype)
                    for part in (".q_a_proj.", ".kv_a_proj_with_mqa.")
                )
            else:
                required[name] = dtype
        return required

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        required = self._required_weights()
        seen: set[str] = set()
        pending: list[tuple[str, torch.Tensor]] = []

        def checked_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if not name.startswith(self.checkpoint_prefix):
                    continue
                name = name.removeprefix(self.checkpoint_prefix)
                if name.startswith(self._ignored_prefixes):
                    continue
                if name.split(".", 1)[0].isdigit():
                    name = f"layers.{name}"
                elif name.startswith("final_norm."):
                    name = name.replace("final_norm.", "norm.", 1)
                if name in seen:
                    raise ValueError(f"Duplicate DFly weight {name!r}")
                seen.add(name)
                layer, is_mlp, leaf = name.partition(".mlp.")
                if is_mlp:
                    index = layer.removeprefix("layers.")
                    if not index.isdigit() or not 0 <= int(index) < len(
                        self.model.layers
                    ):
                        raise ValueError(f"Unexpected DFly weight {name!r}")
                    if leaf in (
                        "gate_proj.weight",
                        "up_proj.weight",
                        "down_proj.weight",
                    ):
                        if weight.ndim != 2 or min(weight.shape) < 1:
                            raise ValueError(f"Invalid DFly FFN weight shape: {name}")
                        down = leaf == "down_proj.weight"
                        if weight.shape[0 if down else 1] != self.config.hidden_size:
                            raise ValueError(f"Invalid DFly FFN hidden size: {name}")
                        self.model.materialize_ffn(weight.shape[1 if down else 0])
                        required.update(self._required_weights())
                    elif self.model.ffn_width is None:
                        pending.append((name, weight))
                        continue
                ready = ((name, weight),)
                entries = (
                    chain(pending, ready) if self.model.ffn_width is not None else ready
                )
                for name, weight in entries:
                    if name not in required:
                        raise ValueError(
                            f"Unexpected DFly weight {name!r}, "
                            f"shape={tuple(weight.shape)}"
                        )
                    if (
                        name.endswith(".weight_scale_inv")
                        and required[name] == torch.float32
                        and weight.dtype == torch.bfloat16
                    ):
                        weight = weight.float()
                    if weight.dtype != required[name]:
                        raise ValueError(
                            f"DFly weight {name} requires {required[name]}, "
                            f"got {weight.dtype}"
                        )
                    yield f"model.{name}", weight
                if self.model.ffn_width is not None:
                    pending.clear()
            if self.model.ffn_width is None:
                raise ValueError("Missing required DFly FFN weights")
            optional = self.model.resolve_post_proj_norm(seen)
            missing = required.keys() - seen - {"embed_tokens.weight"} - optional
            if missing:
                raise ValueError(f"Missing required DFly weights: {sorted(missing)}")

        loaded = AutoWeightsLoader(self).load_weights(
            checked_weights(), mapper=self._weight_mapper
        )
        self.has_own_embed_tokens = "embed_tokens.weight" in seen
        if not self.has_own_embed_tokens:
            self.model.embed_tokens = None
        return loaded
