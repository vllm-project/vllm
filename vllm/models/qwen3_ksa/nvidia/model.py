# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3 KSA model with model-internal Summary rows."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, cast

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.config import (
    CacheConfig,
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import LocalArgmaxMixin, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3MLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.qwen3_ksa.common.metadata import KSAAttentionBackend
from vllm.models.qwen3_ksa.common.reference import (
    KSAExpandedBatch,
    dense_ksa_attention,
    expand_ksa_batch,
    expand_ksa_cudagraph_decode,
    infer_query_start_loc,
)
from vllm.models.qwen3_ksa.nvidia.attention import (
    KSATextCacheLayer,
    get_ksa_attention_metadata,
    get_ksa_summary_cache_spec,
    paged_ksa_attention,
    register_ksa_cache_owner,
    register_ksa_cache_scales,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class KSASettings:
    summary_chunk_size: int
    summary_token_begin: int
    summary_token_num: int
    sliding_chunk_nums: tuple[int, ...]
    max_reference_len: int = 4096


def validate_ksa_runtime_config(vllm_config: VllmConfig) -> None:
    """Reject engine modes that the initial paged KSA path cannot preserve."""
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config

    if vllm_config.device_config.device_type != "cuda":
        raise NotImplementedError(
            "KSA kernels currently require CUDA; use a CUDA device"
        )
    if scheduler_config.disable_hybrid_kv_cache_manager:
        raise NotImplementedError(
            "KSA requires the hybrid KV cache manager to keep windowed Text "
            "and compressed Summary caches separate; remove "
            "--disable-hybrid-kv-cache-manager and use a KV connector that "
            "supports HMA"
        )
    if not model_config.enforce_eager:
        compilation_config = vllm_config.compilation_config
        if (
            compilation_config.mode != CompilationMode.NONE
            or compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
        ):
            raise NotImplementedError(
                "KSA CUDA Graph currently supports FULL_DECODE_ONLY with "
                "compilation mode NONE; use enforce_eager=True or set "
                "compilation_config={'mode': 0, "
                "'cudagraph_mode': 'FULL_DECODE_ONLY'}"
            )
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                "KSA CUDA Graph is currently validated only with TP=1 and PP=1; "
                "use enforce_eager=True for distributed execution"
            )
    if cache_config.enable_prefix_caching:
        if cache_config.block_size != 8:
            raise NotImplementedError(
                "KSA prefix caching currently requires block_size=8 so Text "
                "and Summary cache hits end on the same chunk boundary; set "
                "block_size=8 or disable prefix caching"
            )
        if not model_config.enforce_eager:
            raise NotImplementedError(
                "KSA prefix caching is currently validated only in eager mode; "
                "use enforce_eager=True or disable prefix caching"
            )
        if (
            parallel_config.tensor_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                "KSA prefix caching is currently validated only with TP=1 and "
                "PP=1; disable prefix caching for distributed execution"
            )
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "KSA speculative decoding has no Summary cache rollback semantics; "
            "disable speculative decoding"
        )
    if parallel_config.pipeline_parallel_size > 1 and not model_config.enforce_eager:
        raise NotImplementedError(
            "KSA pipeline parallelism is currently validated only in eager mode; "
            "use enforce_eager=True or PP=1"
        )
    if parallel_config.decode_context_parallel_size != 1:
        raise NotImplementedError(
            "KSA DCP does not partition Text and Summary cache ownership; set "
            "decode_context_parallel_size=1"
        )
    if parallel_config.prefill_context_parallel_size != 1:
        raise NotImplementedError(
            "KSA PCP does not partition Text and Summary cache ownership; set "
            "prefill_context_parallel_size=1"
        )
    if cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError(
            "KSA KV cache quantization has no separate Text and Summary scale "
            "policy; use kv_cache_dtype=bfloat16"
        )
    if model_config.dtype is not torch.bfloat16:
        raise NotImplementedError(
            "KSA kernels are currently validated for BF16 model weights; use "
            "dtype=bfloat16"
        )
    if vllm_config.quant_config is not None:
        raise NotImplementedError(
            "KSA weight quantization is not validated for internal Summary rows; "
            "disable model quantization"
        )
    if cache_config.block_size % 8 != 0:
        raise ValueError("KSA cache block size must be divisible by 8")
    if model_config.enable_prompt_embeds:
        raise NotImplementedError(
            "KSA Summary insertion requires logical token IDs; disable prompt "
            "embeddings and provide token IDs"
        )


def _parse_pattern_node(node: ast.AST, *, max_items: int) -> int | list[int]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("KSA layer pattern literals must be integers")
        if node.value <= 0:
            raise ValueError("KSA layer pattern values must be positive")
        return node.value

    if isinstance(node, ast.List):
        literal_values: list[int] = []
        for item in node.elts:
            value = _parse_pattern_node(item, max_items=max_items)
            if not isinstance(value, int):
                raise ValueError("nested KSA layer pattern lists are not supported")
            literal_values.append(value)
        if len(literal_values) > max_items:
            raise ValueError("KSA layer pattern is longer than num_hidden_layers")
        return literal_values

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _parse_pattern_node(node.left, max_items=max_items)
        right = _parse_pattern_node(node.right, max_items=max_items)
        if not isinstance(left, list) or not isinstance(right, list):
            raise ValueError("KSA layer pattern addition requires two lists")
        if len(left) + len(right) > max_items:
            raise ValueError("KSA layer pattern is longer than num_hidden_layers")
        return left + right

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _parse_pattern_node(node.left, max_items=max_items)
        right = _parse_pattern_node(node.right, max_items=max_items)
        repeated_values: list[int]
        repeat: int
        if isinstance(left, list) and isinstance(right, int):
            repeated_values, repeat = left, right
        elif isinstance(left, int) and isinstance(right, list):
            repeated_values, repeat = right, left
        else:
            raise ValueError("KSA layer pattern multiplication requires a list")
        if len(repeated_values) * repeat > max_items:
            raise ValueError("KSA layer pattern is longer than num_hidden_layers")
        return repeated_values * repeat

    raise ValueError(f"unsupported KSA layer pattern syntax: {type(node).__name__}")


def parse_ksa_layer_pattern(
    value: int | Sequence[int] | str,
    *,
    num_hidden_layers: int,
) -> tuple[int, ...]:
    """Parse the safe subset used by ``summary_sliding_chunk_num``."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    if isinstance(value, bool):
        raise TypeError("KSA layer pattern must not be a boolean")
    if isinstance(value, int):
        parsed: int | list[int] = value
    elif isinstance(value, str):
        try:
            expression = ast.parse(value, mode="eval")
        except SyntaxError as error:
            raise ValueError("invalid KSA layer pattern syntax") from error
        parsed = _parse_pattern_node(expression.body, max_items=num_hidden_layers)
    elif isinstance(value, Sequence):
        parsed = list(cast(Sequence[int], value))
    else:
        raise TypeError("KSA layer pattern must be an integer, list, or string")

    if isinstance(parsed, int):
        parsed = [parsed] * num_hidden_layers
    if len(parsed) != num_hidden_layers:
        raise ValueError(
            "KSA layer pattern length must equal num_hidden_layers: "
            f"{len(parsed)} != {num_hidden_layers}"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise TypeError("KSA layer pattern values must be integers")
    if any(item <= 0 for item in parsed):
        raise ValueError("KSA layer pattern values must be positive")
    return tuple(parsed)


def validate_ksa_config(config: Any) -> KSASettings:
    """Validate the released KSA configuration supported by this model."""
    required_values = {
        "use_summary_attention": True,
        "summary_chunk_size": 8,
        "summary_token_num": 1,
        "summary_chunk_position_ids_type": "origin",
        "summary_token_position_ids_type": "last_chunk_slice_right",
        "summary_independent_parameters": True,
        "summary_independent_attention_layernorm": False,
        "mix_coeff": 0,
    }
    for name, expected in required_values.items():
        actual = getattr(config, name, None)
        if actual != expected:
            raise ValueError(
                f"unsupported KSA config {name}={actual!r}; expected {expected!r}"
            )

    summary_begin = int(config.summary_token_begin)
    vocab_size = int(config.vocab_size)
    if summary_begin < 0 or summary_begin + 1 > vocab_size:
        raise ValueError(
            "KSA summary token range must be inside the physical vocabulary"
        )
    if summary_begin == 0:
        raise ValueError("KSA must retain at least one normal vocabulary token")

    num_hidden_layers = int(config.num_hidden_layers)
    sliding_chunk_nums = parse_ksa_layer_pattern(
        config.summary_sliding_chunk_num,
        num_hidden_layers=num_hidden_layers,
    )
    return KSASettings(
        summary_chunk_size=8,
        summary_token_begin=summary_begin,
        summary_token_num=1,
        sliding_chunk_nums=sliding_chunk_nums,
    )


class Qwen3KSAAttention(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(
        self,
        config: Qwen3Config,
        *,
        layer_idx: int,
        cache_config: CacheConfig,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError("KSA query heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError("KSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads != 0:
            raise ValueError("KSA TP size must replicate KV heads evenly")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.settings = validate_ksa_config(config)
        self.sliding_chunk_num = self.settings.sliding_chunk_nums[layer_idx]
        self.is_small_layer = (
            self.sliding_chunk_num * self.settings.summary_chunk_size
            < vllm_config.model_config.max_model_len
        )
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        register_ksa_cache_scales(self)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.text_cache = KSATextCacheLayer(
            prefix=f"{prefix}.text_cache",
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sliding_chunk_num=self.sliding_chunk_num,
            is_small_layer=self.is_small_layer,
            cache_config=cache_config,
            vllm_config=vllm_config,
        )
        register_ksa_cache_owner(
            prefix=prefix,
            module=self,
            vllm_config=vllm_config,
        )

    def get_attn_backend(self) -> type[KSAAttentionBackend]:
        return KSAAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if not self.is_small_layer:
            return None
        return get_ksa_summary_cache_spec(
            cache_config=self.text_cache.cache_config,
            vllm_config=vllm_config,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        expanded_batch: KSAExpandedBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if is_forward_context_available():
            attention_output = paged_ksa_attention(
                query=q,
                key=k,
                value=v,
                expanded_batch=expanded_batch,
                text_cache_layer=self.text_cache,
                summary_cache_owner=self if self.is_small_layer else None,
                summary_cache_layer_name=self.prefix if self.is_small_layer else None,
                is_small_layer=self.is_small_layer,
                scale=self.scaling,
            )
        else:
            attention_output = dense_ksa_attention(
                q,
                k,
                v,
                expanded_batch,
                summary_chunk_size=self.settings.summary_chunk_size,
                sliding_chunk_num=self.sliding_chunk_num,
                scale=self.scaling,
                max_reference_len=self.settings.max_reference_len,
            ).output
        attention_output = attention_output.reshape(-1, self.q_size)
        output, _ = self.o_proj(attention_output)
        return output


class Qwen3KSADecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if cache_config is None:
            raise ValueError("KSA decoder layers require a KV cache configuration")
        vllm_config = get_current_vllm_config()
        set_default_rope_theta(config, default_theta=1000000)
        if not getattr(config, "is_causal", True):
            raise ValueError("KSA supports causal attention only")
        layer_idx = extract_layer_index(prefix)
        self.self_attn = Qwen3KSAAttention(
            config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            vllm_config=vllm_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        expanded_batch: KSAExpandedBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, expanded_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _logical_summary_rows(expanded_batch: KSAExpandedBatch) -> torch.Tensor:
    return torch.nonzero(
        expanded_batch.logical_boundary_mask,
        as_tuple=False,
    ).flatten()


def _pack_ksa_stage_tensor(
    expanded_tensor: torch.Tensor,
    expanded_batch: KSAExpandedBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack expanded Text and Summary rows into logical PP payloads."""
    text_tensor = expanded_tensor.index_select(
        0,
        expanded_batch.text_row_indices,
    )
    summary_tensor = torch.zeros_like(text_tensor)
    summary_tensor.index_copy_(
        0,
        _logical_summary_rows(expanded_batch),
        expanded_tensor.index_select(0, expanded_batch.summary_row_indices),
    )
    return text_tensor, summary_tensor


def _unpack_ksa_stage_tensor(
    text_tensor: torch.Tensor,
    summary_tensor: torch.Tensor,
    expanded_batch: KSAExpandedBatch,
) -> torch.Tensor:
    """Restore the model-internal row order from a logical PP payload."""
    expanded_tensor = text_tensor.new_empty(
        (expanded_batch.expanded_input_ids.numel(), text_tensor.shape[-1])
    )
    expanded_tensor.index_copy_(
        0,
        expanded_batch.text_row_indices,
        text_tensor,
    )
    expanded_tensor.index_copy_(
        0,
        expanded_batch.summary_row_indices,
        summary_tensor.index_select(0, _logical_summary_rows(expanded_batch)),
    )
    return expanded_tensor


def _pack_ksa_intermediate_tensors(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    expanded_batch: KSAExpandedBatch,
) -> IntermediateTensors:
    hidden_text, hidden_summary = _pack_ksa_stage_tensor(
        hidden_states,
        expanded_batch,
    )
    residual_text, residual_summary = _pack_ksa_stage_tensor(
        residual,
        expanded_batch,
    )
    return IntermediateTensors(
        {
            "hidden_states": hidden_text,
            "residual": residual_text,
            "ksa_summary_hidden_states": hidden_summary,
            "ksa_summary_residual": residual_summary,
        }
    )


def _unpack_ksa_intermediate_tensors(
    intermediate_tensors: IntermediateTensors,
    expanded_batch: KSAExpandedBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _unpack_ksa_stage_tensor(
            intermediate_tensors["hidden_states"],
            intermediate_tensors["ksa_summary_hidden_states"],
            expanded_batch,
        ),
        _unpack_ksa_stage_tensor(
            intermediate_tensors["residual"],
            intermediate_tensors["ksa_summary_residual"],
            expanded_batch,
        ),
    )


class Qwen3KSAModel(Qwen2Model):
    hf_to_vllm_mapper = Qwen2Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config.get_text_config()
        self.ksa_settings = validate_ksa_config(config)
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3KSADecoderLayer,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            [
                "hidden_states",
                "residual",
                "ksa_summary_hidden_states",
                "ksa_summary_residual",
            ],
            config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is not None:
            raise NotImplementedError(
                "dense KSA reference model requires input_ids for Summary insertion"
            )
        if get_pp_group().is_first_rank:
            if intermediate_tensors is not None:
                raise ValueError(
                    "the first KSA PP stage must not receive hidden states"
                )
            if input_ids is None:
                raise ValueError("the first KSA PP stage requires input_ids")
            logical_input_ids = input_ids
        else:
            if intermediate_tensors is None:
                raise ValueError("a non-first KSA PP stage requires hidden states")
            logical_input_ids = torch.zeros_like(positions, dtype=torch.int32)

        expanded_batch: KSAExpandedBatch | None = None
        expansion_positions = positions
        num_computed_tokens = None
        validate_position_steps = True
        if is_forward_context_available():
            validate_position_steps = False
            context = get_forward_context()
            if context.attn_metadata is None:
                query_start_loc = torch.tensor(
                    [0, logical_input_ids.numel()],
                    device=logical_input_ids.device,
                    dtype=torch.int32,
                )
                expansion_positions = torch.arange(
                    logical_input_ids.numel(),
                    device=positions.device,
                    dtype=positions.dtype,
                )
            else:
                first_layer = self.layers[self.start_layer]
                text_metadata = get_ksa_attention_metadata(
                    first_layer.self_attn.text_cache.prefix
                )
                if text_metadata is None:
                    raise RuntimeError("KSA text metadata is missing")
                if text_metadata.is_cudagraph_capture:
                    expanded_batch = expand_ksa_cudagraph_decode(
                        logical_input_ids,
                        positions,
                        text_row_is_valid=text_metadata.slot_mapping >= 0,
                        summary_chunk_size=self.ksa_settings.summary_chunk_size,
                        summary_token_begin=self.ksa_settings.summary_token_begin,
                        summary_token_num=self.ksa_settings.summary_token_num,
                    )
                else:
                    query_start_loc = text_metadata.query_start_loc
                    num_computed_tokens = text_metadata.num_computed_tokens
        else:
            query_start_loc = infer_query_start_loc(positions)
        if expanded_batch is None:
            expanded_batch = expand_ksa_batch(
                logical_input_ids,
                expansion_positions,
                query_start_loc=query_start_loc,
                summary_chunk_size=self.ksa_settings.summary_chunk_size,
                summary_token_begin=self.ksa_settings.summary_token_begin,
                summary_token_num=self.ksa_settings.summary_token_num,
                num_computed_tokens=num_computed_tokens,
                validate_position_steps=validate_position_steps,
            )
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_input_ids(expanded_batch.expanded_input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states, residual = _unpack_ksa_intermediate_tensors(
                intermediate_tensors,
                expanded_batch,
            )
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                expanded_batch.expanded_positions,
                hidden_states,
                residual,
                expanded_batch,
            )
        if not get_pp_group().is_last_rank:
            assert residual is not None
            return _pack_ksa_intermediate_tensors(
                hidden_states,
                residual,
                expanded_batch,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states.index_select(0, expanded_batch.output_gather_indices)


class Qwen3KSAForCausalLM(LocalArgmaxMixin, nn.Module, SupportsPP):
    hf_to_vllm_mapper = Qwen3KSAModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        validate_ksa_runtime_config(vllm_config)
        config = vllm_config.model_config.hf_config.get_text_config()
        settings = validate_ksa_config(config)
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen3KSAModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            org_vocab_size=settings.summary_token_begin,
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


__all__ = [
    "KSASettings",
    "Qwen3KSAForCausalLM",
    "Qwen3KSAModel",
    "parse_ksa_layer_pattern",
    "validate_ksa_runtime_config",
    "validate_ksa_config",
]
