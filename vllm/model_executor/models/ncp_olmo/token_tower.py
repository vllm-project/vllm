# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM-native OLMo3 token encoder and decoder towers for ConceptLM."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .contract import ConceptLMBackendConfig
from .weights import required_checkpoint_shards, resolve_checkpoint_weight


def _config_value(config: Any, name: str) -> Any:
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"missing OLMo3 runtime config field: {name}")
    return value


def _yarn_mscale(factor: float, multiplier: float) -> float:
    if factor <= 1.0:
        return 1.0
    return 0.1 * multiplier * math.log(factor) + 1.0


def _rope_parameters(config: Any, *, apply_yarn: bool) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "rope_theta": float(_config_value(config, "rotary_base")),
        "partial_rotary_factor": float(_config_value(config, "rotary_percent")),
    }
    if not apply_yarn:
        parameters["rope_type"] = "default"
        return parameters

    factor = float(_config_value(config, "yarn_rotary_scaling_factor"))
    mscale = float(_config_value(config, "yarn_mscale"))
    mscale_all_dim = float(_config_value(config, "yarn_mscale_all_dim"))
    desired_mscale = _yarn_mscale(factor, mscale) / _yarn_mscale(factor, mscale_all_dim)
    parameters.update(
        {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": int(
                _config_value(config, "yarn_original_max_position_embeddings")
            ),
            "beta_fast": float(_config_value(config, "yarn_beta_fast")),
            "beta_slow": float(_config_value(config, "yarn_beta_slow")),
            "attn_factor": desired_mscale / _yarn_mscale(factor, 1.0),
            "truncate": bool(
                _config_value(config, "yarn_correction_range_round_to_int")
            ),
        }
    )
    return parameters


def _sliding_window(config: Any, layer_number: int) -> int | None:
    window_size = _config_value(config, "window_size")
    skip_frequency = _config_value(config, "window_attn_skip_freq")
    if not isinstance(window_size, (list, tuple)) or len(window_size) != 2:
        raise ValueError("window_size must contain left and right extents")
    if int(window_size[1]) != 0:
        raise ValueError("the causal OLMo3 backend requires window_size[1] == 0")
    if not isinstance(skip_frequency, int) or isinstance(skip_frequency, bool):
        raise ValueError(
            "the first OLMo3 backend requires integer window_attn_skip_freq"
        )
    if layer_number % skip_frequency == 0:
        return None
    return int(window_size[0])


def _extend_yarn_cache(rotary_emb: nn.Module, required_positions: int) -> None:
    """Extend vLLM's YaRN lookup table without changing YaRN frequencies.

    vLLM 0.13 sizes the table as ``original_length * factor`` and ignores the
    separately requested runtime maximum.  Explicit extrapolation beyond that
    trained window therefore indexes past the table even though the scheduler
    and KV cache admit the request.  Reusing the module's existing inverse
    frequencies preserves the configured YaRN factor while extending only the
    lookup domain.
    """

    cache = rotary_emb.cos_sin_cache
    if int(cache.shape[0]) >= int(required_positions):
        return
    scaling_factor = float(rotary_emb.scaling_factor)
    inv_freq = rotary_emb._compute_inv_freq(scaling_factor)
    positions = torch.arange(int(required_positions), dtype=torch.float32)
    frequencies = torch.einsum("i,j -> ij", positions, inv_freq)
    extended = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    extended = extended * float(rotary_emb.mscale)
    rotary_emb.cos_sin_cache = extended.to(dtype=rotary_emb.dtype)
    if int(rotary_emb.cos_sin_cache.shape[0]) < int(required_positions):
        raise RuntimeError("failed to extend the vLLM YaRN rotary cache")


class ConceptLMOlmo3Attention(nn.Module):
    """OLMo3 MHA using vLLM Attention and fused runtime parameters."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        backend_config: ConceptLMBackendConfig,
        layer_number: int,
        prefix: str,
    ) -> None:
        super().__init__()
        raw_config = vllm_config.model_config.hf_config
        hidden_size = backend_config.hidden_size
        self.total_num_heads = backend_config.num_attention_heads
        self.total_num_kv_heads = backend_config.num_key_value_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if hidden_size % self.total_num_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.total_num_heads % self.tp_size:
            raise ValueError(
                "num_attention_heads must be divisible by tensor parallel size"
            )

        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads // self.tp_size
        if self.total_num_kv_heads >= self.tp_size:
            if self.total_num_kv_heads % self.tp_size:
                raise ValueError(
                    "num_query_groups must be divisible by tensor parallel size"
                )
            self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        else:
            if self.tp_size % self.total_num_kv_heads:
                raise ValueError(
                    "tensor parallel size must be divisible by num_query_groups"
                )
            self.num_kv_heads = 1
        self.query_size = self.num_heads * self.head_dim
        self.key_value_size = self.num_kv_heads * self.head_dim

        self.linear_qkv = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.linear_qkv",
        )
        self.qk_norm_mode = backend_config.qk_norm_mode
        self.q_layernorm = RMSNorm(
            backend_config.qk_norm_weight_size,
            eps=float(_config_value(raw_config, "layernorm_epsilon")),
        )
        self.k_layernorm = RMSNorm(
            backend_config.qk_norm_weight_size,
            eps=float(_config_value(raw_config, "layernorm_epsilon")),
        )

        sliding_window = _sliding_window(raw_config, layer_number)
        self.core_attention = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.core_attention",
        )
        use_yarn = str(
            _config_value(raw_config, "position_embedding_type")
        ) == "yarn" and (
            sliding_window is None
            or not bool(_config_value(raw_config, "yarn_full_attn_layers_only"))
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=backend_config.max_model_len,
            is_neox_style=not bool(_config_value(raw_config, "rotary_interleaved")),
            rope_parameters=_rope_parameters(raw_config, apply_yarn=use_yarn),
        )
        if use_yarn:
            _extend_yarn_cache(self.rotary_emb, backend_config.max_model_len)
        self.linear_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.linear_proj",
        )

    def _apply_qk_norm(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm_mode == "per_head":
            query_shape = query.shape
            key_shape = key.shape
            query = self.q_layernorm(
                query.reshape(*query_shape[:-1], self.num_heads, self.head_dim)
            ).reshape(query_shape)
            key = self.k_layernorm(
                key.reshape(*key_shape[:-1], self.num_kv_heads, self.head_dim)
            ).reshape(key_shape)
            return query, key
        if self.tp_size > 1:
            query = tensor_model_parallel_all_gather(query.contiguous())
            key = tensor_model_parallel_all_gather(key.contiguous())
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)
        if self.tp_size > 1:
            splitter = partial(
                split_tensor_along_last_dim,
                num_partitions=self.tp_size,
            )
            query = splitter(query)[self.tp_rank]
            key = splitter(key)[self.tp_rank]
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run one cached OLMo3 self-attention layer."""

        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = mixed_qkv.split(
            [self.query_size, self.key_value_size, self.key_value_size],
            dim=-1,
        )
        query, key = self._apply_qk_norm(query, key)
        query, key = self.rotary_emb(positions, query, key)
        attention_output = self.core_attention(query, key, value)
        output, _ = self.linear_proj(attention_output)
        return output


class ConceptLMOlmo3MLP(nn.Module):
    """OLMo3 SwiGLU MLP with vLLM's fused gate/up runtime parameter."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        backend_config: ConceptLMBackendConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self._trace_layer_index: int | None = None
        marker = "concept_predictor.hlm_block.layers."
        if os.environ.get("CONCEPTLM_VLLM_LOGITS_TRACE_DIR") and marker in prefix:
            self._trace_layer_index = int(
                prefix.split(marker, maxsplit=1)[1].split(".", maxsplit=1)[0]
            )
        self._latest_stage_trace: dict[str, torch.Tensor] = {}
        self.linear_fc1 = MergedColumnParallelLinear(
            backend_config.hidden_size,
            [backend_config.intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.linear_fc1",
        )
        self.activation = SiluAndMul()
        self.linear_fc2 = RowParallelLinear(
            backend_config.intermediate_size,
            backend_config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.linear_fc2",
        )

    def _record_stage(self, name: str, value: torch.Tensor) -> None:
        if self._trace_layer_index is not None:
            self._latest_stage_trace[f"hlm.mlp.{name}.{self._trace_layer_index}"] = (
                value[-1].detach().float().cpu()
            )

    def stage_trace(self) -> dict[str, torch.Tensor]:
        """Return the latest MLP vectors used by the parity harness."""

        return dict(self._latest_stage_trace)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the fused SwiGLU MLP."""

        gate_and_up, _ = self.linear_fc1(hidden_states)
        self._record_stage("fc1", gate_and_up)
        hidden_states = self.activation(gate_and_up)
        self._record_stage("activation", hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        self._record_stage("fc2", hidden_states)
        return hidden_states


class ConceptLMOlmo3Layer(nn.Module):
    """One post-norm OLMo3 layer matching the Megatron training graph."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        backend_config: ConceptLMBackendConfig,
        layer_number: int,
        prefix: str,
    ) -> None:
        super().__init__()
        raw_config = vllm_config.model_config.hf_config
        epsilon = float(_config_value(raw_config, "layernorm_epsilon"))
        self.self_attention = ConceptLMOlmo3Attention(
            vllm_config=vllm_config,
            backend_config=backend_config,
            layer_number=layer_number,
            prefix=f"{prefix}.self_attention",
        )
        self.mlp = ConceptLMOlmo3MLP(
            vllm_config=vllm_config,
            backend_config=backend_config,
            prefix=f"{prefix}.mlp",
        )
        self.post_attention_layernorm = RMSNorm(
            backend_config.hidden_size,
            eps=epsilon,
        )
        self.post_feedforward_layernorm = RMSNorm(
            backend_config.hidden_size,
            eps=epsilon,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply post-attention and post-MLP RMSNorm residual blocks."""

        residual = hidden_states
        hidden_states = self.self_attention(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class ConceptLMOlmo3Tower(nn.Module):
    """A locally numbered encoder or decoder tower."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        backend_config: ConceptLMBackendConfig,
        num_layers: int,
        final_layernorm: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConceptLMOlmo3Layer(
                    vllm_config=vllm_config,
                    backend_config=backend_config,
                    layer_number=layer_index + 1,
                    prefix=f"{prefix}.layers.{layer_index}",
                )
                for layer_index in range(num_layers)
            ]
        )
        if final_layernorm:
            self.final_layernorm: RMSNorm | None = RMSNorm(
                backend_config.hidden_size,
                eps=float(
                    _config_value(
                        vllm_config.model_config.hf_config,
                        "layernorm_epsilon",
                    )
                ),
            )
        else:
            self.final_layernorm = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run all local layers and the optional decoder final norm."""

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class ConceptLMEmbedding(nn.Module):
    """Megatron-compatible word embedding wrapper."""

    def __init__(self, *, backend_config: ConceptLMBackendConfig, prefix: str) -> None:
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            backend_config.vocab_size,
            backend_config.hidden_size,
            prefix=f"{prefix}.word_embeddings",
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs."""

        return self.word_embeddings(input_ids)


class ConceptLMTokenBackbone(nn.Module):
    """Token-only NCP-OLMo backbone loaded from split pure-HF tensors."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        backend_config: ConceptLMBackendConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.quant_config is not None:
            raise NotImplementedError(
                "NCP-OLMo pure-HF weights are initially supported unquantized"
            )
        root = f"{prefix}." if prefix else ""
        self.backend_config = backend_config
        self.embedding = ConceptLMEmbedding(
            backend_config=backend_config,
            prefix=f"{root}embedding".removesuffix("."),
        )
        self.encoder = ConceptLMOlmo3Tower(
            vllm_config=vllm_config,
            backend_config=backend_config,
            num_layers=backend_config.encoder_layers,
            final_layernorm=False,
            prefix=f"{root}encoder".removesuffix("."),
        )
        self.decoder = ConceptLMOlmo3Tower(
            vllm_config=vllm_config,
            backend_config=backend_config,
            num_layers=backend_config.decoder_layers,
            final_layernorm=True,
            prefix=f"{root}decoder".removesuffix("."),
        )
        self.output_layer = ParallelLMHead(
            backend_config.vocab_size,
            backend_config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{root}output_layer".removesuffix("."),
        )
        self.logits_processor = LogitsProcessor(backend_config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply NCP-OLMo token embeddings."""

        return self.embedding(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Project token-decoder states to vocabulary logits."""

        return self.logits_processor(self.output_layer, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load token parameters from pure-HF checkpoint keys."""

        params = dict(self.named_parameters(remove_duplicate=False))
        parameter_names = frozenset(params)
        loaded_targets: dict[str, set[str | int | None]] = {}
        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name.endswith("._extra_state"):
                continue
            target = resolve_checkpoint_weight(checkpoint_name, parameter_names)
            if target is None:
                continue
            target_shards = loaded_targets.setdefault(target.parameter_name, set())
            if target.shard_id in target_shards:
                raise ValueError(
                    "duplicate token-tower checkpoint target: "
                    f"{checkpoint_name} -> {target.parameter_name}"
                )
            if (target.shard_id is None and target_shards) or (
                target.shard_id is not None and None in target_shards
            ):
                raise ValueError(
                    "cannot mix full and split token-tower tensors for "
                    f"{target.parameter_name}"
                )
            parameter = params[target.parameter_name]
            weight_loader = getattr(
                parameter,
                "weight_loader",
                default_weight_loader,
            )
            if target.shard_id is not None:
                weight_loader(parameter, loaded_weight, target.shard_id)
            else:
                weight_loader(parameter, loaded_weight)
            target_shards.add(target.shard_id)

        complete: set[str] = set()
        for parameter_name, loaded_shards in loaded_targets.items():
            required = required_checkpoint_shards(parameter_name)
            if loaded_shards == {None} or (
                required is not None and loaded_shards == required
            ):
                complete.add(parameter_name)
        missing = sorted(set(params) - complete)
        if missing:
            raise ValueError(
                "missing token-tower checkpoint parameters: " + ", ".join(missing)
            )
        return complete
