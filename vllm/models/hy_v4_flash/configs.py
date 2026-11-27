# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4-compatible configuration for Flash's hybrid attention and Engram."""

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import ALLOWED_LAYER_TYPES, PreTrainedConfig

from vllm.transformers_utils.configs.hy_v4 import (
    HYV4Config,
    HYV4VLConfig,
    HYV4VLVisionConfig,
)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

SPARSE_ATTENTION = (
    "indexed_attention"
    if "indexed_attention" in ALLOWED_LAYER_TYPES
    else "deepseek_sparse_attention"
)
SLIDING_ATTENTION = "sliding_attention"
_SPARSE_ALIASES = {
    "sparse",
    "sparse_attention",
    "indexed_attention",
    "deepseek_sparse_attention",
}


@strict
class HYV4FlashConfig(HYV4Config):
    model_type = "hy_v4_flash"
    base_config_key = "text_config"

    routed_expert_hidden_size: int | None = None
    sandwich_norm: bool = True
    dsa_skip_attn_post_norm: bool = False
    swa_num_attention_heads: int | None = None
    swa_qk_nope_head_dim: int | None = None
    swa_qk_rope_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    sliding_window: int = 512
    window_size: list[int] | None = None
    use_compressed_indexer: bool = True
    compressed_indexer_compress_ratio: int = 4
    compressed_indexer_num_attention_heads: int = 16
    compressed_indexer_head_dim: int = 128
    compressed_indexer_group_topk: int = 512
    compressed_indexer_candidate_topk: int = 2048
    compressed_indexer_reuse_fine_indexer: bool = False
    compressed_indexer_skip_fine_select: bool = True
    compressed_indexer_key_pooling: str = "gated"
    engram_layer_ids: list[int] | None = None
    engram_max_ngram_size: int = 3
    engram_n_embed_per_ngram: int = 768
    engram_n_head_per_ngram: int = 8
    engram_pad_id: int = 120002
    engram_seed: int = 0
    engram_fp8_embedding: bool = False
    engram_layer_head_vocab_sizes: dict[str | int, list[list[int]]] | None = None

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs: Any
    ) -> PreTrainedConfig | tuple[PreTrainedConfig, dict[str, Any]]:
        if config_dict.get("vision_config") is not None:
            return HYV4FlashVLConfig.from_dict(deepcopy(config_dict), **kwargs)
        return super().from_dict(config_dict, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        kwargs.pop("engram_enable", None)
        if self.layer_types is None:
            self.layer_types = [SPARSE_ATTENTION] * self.num_hidden_layers
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in self.layer_types
        ]
        if self.indexer_types is None:
            self.indexer_types = ["full"] * self.num_hidden_layers
        for name in (
            "num_attention_heads",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        ):
            if getattr(self, f"swa_{name}") is None:
                setattr(self, f"swa_{name}", getattr(self, name))
        self._normalize_rope()
        super().__post_init__(**kwargs)
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in self.layer_types
        ]
        self._normalize_rope()
        self.validate_flash_config()

    def _normalize_rope(self) -> None:
        rope = deepcopy(self.rope_parameters)
        if not rope:
            return
        for alias in _SPARSE_ALIASES - {SPARSE_ATTENTION}:
            if alias in rope:
                value = rope.pop(alias)
                if SPARSE_ATTENTION in rope and rope[SPARSE_ATTENTION] != value:
                    raise ValueError("Conflicting sparse attention RoPE parameters")
                rope[SPARSE_ATTENTION] = value
        nested = [value for value in rope.values() if isinstance(value, dict)]
        if nested:
            for kind in set(self.layer_types or ()):
                if kind not in rope or not isinstance(rope[kind], dict):
                    raise ValueError(f"Missing rope_parameters for {kind}")
        sections = {
            tuple(value["mrope_section"])
            for value in (rope, *nested)
            if "mrope_section" in value
        }
        interleaved = {
            value["mrope_interleaved"]
            for value in (rope, *nested)
            if "mrope_interleaved" in value
        }
        if len(sections) > 1 or len(interleaved) > 1:
            raise ValueError("MRoPE sections and interleaving must agree across layers")
        if sections:
            section = next(iter(sections))
            if (
                len(section) != 3
                or any(type(n) is not int or n < 0 for n in section)
                or sum(section) * 2 != self.qk_rope_head_dim
                or self.swa_qk_rope_head_dim != self.qk_rope_head_dim
            ):
                raise ValueError("mrope_section must cover both rotary head dimensions")
            for parameters in nested or [rope]:
                parameters["mrope_section"] = list(section)
                if interleaved:
                    parameters["mrope_interleaved"] = next(iter(interleaved))
        if nested:
            rope.pop("mrope_section", None)
            rope.pop("mrope_interleaved", None)
            if any(not isinstance(value, dict) for value in rope.values()):
                raise ValueError("Per-layer RoPE cannot contain flat RoPE parameters")
        self.rope_parameters = rope

    def validate_layer_type(self) -> None:
        self.validate_flash_config()

    def validate_flash_config(self) -> None:
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_attention_heads",
            "swa_num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "swa_qk_rope_head_dim",
            "v_head_dim",
            "swa_v_head_dim",
            "compressed_indexer_num_attention_heads",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, choices in (
            ("layer_types", {SPARSE_ATTENTION, SLIDING_ATTENTION}),
            ("mlp_layer_types", {"dense", "sparse"}),
            ("indexer_types", {"full"}),
        ):
            values = getattr(self, name)
            if values is None or len(values) != self.num_hidden_layers:
                raise ValueError(f"{name} must have {self.num_hidden_layers} entries")
            if any(value not in choices for value in values):
                raise ValueError(f"Unsupported {name}: {values}")
        if type(self.sliding_window) is not int or self.sliding_window < 1:
            raise ValueError("sliding_window must be a positive integer")
        if self.window_size is not None and self.window_size != [
            self.sliding_window - 1,
            0,
        ]:
            raise ValueError("window_size must equal [sliding_window - 1, 0]")
        if self.routed_expert_hidden_size is not None and (
            type(self.routed_expert_hidden_size) is not int
            or self.routed_expert_hidden_size < 1
        ):
            raise ValueError("routed_expert_hidden_size must be positive")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be within the expert count")
        if not self.use_compressed_indexer:
            raise ValueError("Flash requires use_compressed_indexer=True")
        if (
            not self.compressed_indexer_skip_fine_select
            or self.compressed_indexer_reuse_fine_indexer
            or self.compressed_indexer_key_pooling != "gated"
        ):
            raise ValueError("Flash supports independent gated coarse-only IndexPool")
        ratio = self.compressed_indexer_compress_ratio
        if ratio not in (2, 4, 8, 16):
            raise ValueError("compressed_indexer_compress_ratio must be 2/4/8/16")
        if self.compressed_indexer_group_topk < 2:
            raise ValueError("compressed_indexer_group_topk must be at least 2")
        candidates = self.compressed_indexer_group_topk * ratio
        if (
            self.compressed_indexer_candidate_topk != candidates
            or self.index_topk != candidates
        ):
            raise ValueError(
                "IndexPool requires index_topk=candidate_topk=group_topk*ratio"
            )
        if self.index_topk % 128 or self.index_topk < self.sliding_window:
            raise ValueError("index_topk must be 128-aligned and cover sliding_window")
        if self.compressed_indexer_head_dim != 128 or self.index_head_dim != 128:
            raise ValueError("IndexPool requires 128-dimensional keys")
        if not self.q_lora_rank:
            raise ValueError("IndexPool requires q_lora_rank")
        if self.engram_layer_ids:
            if not self.enable_ihc:
                raise ValueError("Flash Engram requires the iHC residual stream")
            layers = self.engram_layer_ids
            if len(set(layers)) != len(layers):
                raise ValueError("engram_layer_ids must be unique")
            if any(
                type(i) is not int or not 1 <= i <= self.num_hidden_layers
                for i in layers
            ):
                raise ValueError(
                    "engram_layer_ids must be 1-based decoder layer numbers"
                )
            heads = self.engram_n_head_per_ngram
            if heads < 1 or self.engram_n_embed_per_ngram % heads:
                raise ValueError(
                    "Engram embedding dimension must be divisible by heads"
                )
            if self.engram_max_ngram_size < 2:
                raise ValueError("engram_max_ngram_size must be at least 2")
            sizes = self.engram_layer_head_vocab_sizes or {}
            for layer in layers:
                groups = sizes.get(str(layer), sizes.get(layer))
                if groups is None or len(groups) != self.engram_max_ngram_size - 1:
                    raise ValueError(
                        f"Missing Engram head vocabulary for layer {layer}"
                    )
                if any(
                    len(group) != heads
                    or any(type(v) is not int or v < 1 for v in group)
                    for group in groups
                ):
                    raise ValueError(
                        f"Invalid Engram head vocabulary for layer {layer}"
                    )

    def verify_parallel_config(self, parallel: "ParallelConfig") -> None:
        for name in (
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name) != 1:
                raise ValueError(f"HYV4 Flash requires {name}=1")
        if parallel.enable_eplb or parallel.enable_dbo:
            raise ValueError("HYV4 Flash does not support EPLB or dual batch overlap")
        tp, dp = parallel.tensor_parallel_size, parallel.data_parallel_size
        for name in ("num_attention_heads", "swa_num_attention_heads"):
            if getattr(self, name) % tp:
                raise ValueError(f"{name} must be divisible by tensor_parallel_size")
        if "dense" in (self.mlp_layer_types or ()) and self.intermediate_size % tp:
            raise ValueError("Dense intermediate_size must be divisible by TP")
        if (
            self.n_shared_experts
            and (self.moe_intermediate_size * self.n_shared_experts) % tp
        ):
            raise ValueError("Shared expert intermediate_size must be divisible by TP")
        if not parallel.enable_expert_parallel and self.moe_intermediate_size % tp:
            raise ValueError("Routed expert intermediate_size must be divisible by TP")
        if (
            self.engram_layer_ids
            and tp > (self.engram_max_ngram_size - 1) * self.engram_n_head_per_ngram
        ):
            raise ValueError(
                "Engram requires at least one complete hash head per TP rank"
            )
        ep = dp * tp if parallel.enable_expert_parallel else 1
        if parallel.enable_expert_parallel and self.n_routed_experts < ep:
            raise ValueError(
                f"Flash has {self.n_routed_experts} experts for {ep} EP ranks"
            )
        if dp == 1:
            return
        if not parallel.enable_expert_parallel:
            raise ValueError("Flash DP requires expert parallelism")
        if (
            parallel.enable_elastic_ep
            or parallel.enable_fault_tolerance
            or parallel.use_ubatching
        ):
            raise ValueError("Flash DP requires fixed ranks without microbatching")
        if parallel.nnodes != 1:
            raise ValueError("Flash DP currently requires a single-node topology")
        if parallel.expert_placement_strategy != "linear":
            raise ValueError("Flash DP requires linear expert placement")
        if self.n_routed_experts % ep:
            raise ValueError("Flash DP requires experts evenly divisible by DP * TP")
        if parallel.all2all_backend not in ("allgather_reducescatter", "passthrough"):
            raise ValueError("Flash DP requires allgather_reducescatter or passthrough")

    def rope_for_layer(self, layer_idx: int) -> dict[str, Any]:
        rope = self.rope_parameters or {}
        layer_types = self.layer_types or ()
        parameters = rope.get(layer_types[layer_idx], rope)
        result = {
            k: deepcopy(v) for k, v in parameters.items() if not isinstance(v, dict)
        }
        for name in ("mrope_section", "mrope_interleaved"):
            if name in rope:
                result.setdefault(name, deepcopy(rope[name]))
        return result

    def update(self, config_dict: dict[str, Any]) -> None:
        config_dict = deepcopy(config_dict)
        config_dict.pop("engram_enable", None)
        super().update(config_dict)
        self.layer_types = [
            SPARSE_ATTENTION if kind in _SPARSE_ALIASES else kind
            for kind in self.layer_types or ()
        ]
        self._normalize_rope()
        self.validate_flash_config()


@strict
class HYV4FlashVLConfig(HYV4VLConfig):
    model_type = "hy_v4_flash_vl"
    default_architecture = "HYV4FlashVLForConditionalGeneration"
    sub_configs = {
        "text_config": HYV4FlashConfig,
        "vision_config": HYV4VLVisionConfig,
    }

    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("model_type") == HYV4FlashConfig.model_type:
            kwargs["model_type"] = self.model_type
        if (architectures := kwargs.get("architectures")) is not None:
            kwargs["architectures"] = [
                self.default_architecture
                if name in ("HYV4FlashForConditionalGeneration", "HYV4FlashForCausalLM")
                else name
                for name in architectures
            ]
        super().__init__(**kwargs)

    def validate_layer_type(self) -> None:
        text = self.__dict__.get("text_config")
        if isinstance(text, HYV4FlashConfig):
            text.validate_flash_config()

    @staticmethod
    def _prepare_text_config(text_kwargs: dict[str, Any]) -> None:
        rope = deepcopy(text_kwargs.get("rope_parameters") or {})
        nested = [value for value in rope.values() if isinstance(value, dict)]
        for parameters in nested or [rope]:
            parameters.setdefault("rope_type", "default")
            parameters.setdefault("mrope_interleaved", True)
            parameters.setdefault("mrope_section", [12, 10, 10])
        text_kwargs["rope_parameters"] = rope
