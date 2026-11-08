# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Flint model for AMD ROCm."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Static
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from ..common.gguf_nvfp4 import (
    find_gguf_file,
    gguf_nvfp4_weights_iterator,
)
from ..configs import FlintConfig


def _flint_gguf_quant_config(
    quant_config: QuantizationConfig,
) -> ModelOptNvFp4Config:
    online_config = getattr(quant_config, "online_quantization_config", None)
    if online_config is None and isinstance(quant_config, OnlineQuantizationConfig):
        online_config = quant_config
    if not isinstance(online_config, OnlineQuantizationConfig):
        raise ValueError(
            "Flint NVFP4 GGUF requires online MXFP8 conversion; "
            f"got {quant_config.get_name()}."
        )
    linear_spec = online_config.args.linear
    moe_spec = online_config.args.moe
    if (
        linear_spec is None
        or linear_spec.weight != kMxfp8Static
        or moe_spec is None
        or moe_spec.weight != kMxfp8Static
    ):
        raise ValueError(
            "Flint NVFP4 GGUF requires online MXFP8 conversion for linear and "
            "MoE weights."
        )
    if isinstance(quant_config, ModelOptNvFp4Config):
        if quant_config.quant_method != "W4A16_NVFP4":
            raise ValueError(
                "Flint NVFP4 GGUF requires a ModelOpt W4A16 source; "
                f"got {quant_config.quant_method}."
            )
        return quant_config

    source_config = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        exclude_modules=["model.embed_tokens", "*.gate", "lm_head"],
        group_size=16,
    )
    source_config.online_quantization_config = online_config
    return source_config


class FlintMLP(nn.Module):
    """Dense SwiGLU MLP used on ``mlp_only_layers`` and as the shared expert.

    Uses standard vLLM ``MergedColumnParallelLinear`` / ``RowParallelLinear``
    with the generic ``SiluAndMul`` activation (no NVIDIA fast-path).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class FlintSparseMoeBlock(nn.Module):
    """Sigmoid-routed MoE block with a shared expert.

    Always uses ``FusedMoEFactory`` (vLLM's portable fused-MoE layer) which
    automatically selects the best backend for the current platform (including
    AITER on ROCm when ``VLLM_ROCM_USE_AITER=1``).
    """

    def __init__(
        self,
        config: FlintConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert = FlintMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_expert",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            scoring_func=config.router_score_function,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)


class FlintAttention(nn.Module):
    """Multi-head attention with per-head QK RMSNorm, optional RoPE, and
    learned attention sinks.

    Hybrid attention pattern:
    - Sliding-window layers use RoPE and a 513-token window.
    - Full-attention layers (in ``no_rope_layer_indices``) use no RoPE.

    Uses vLLM's standard ``Attention`` layer which dispatches to the best
    available backend on the current platform (FlashAttention, CK, Triton, …).
    The ``sinks`` parameter passes learned softmax offsets to the backend.
    """

    def __init__(
        self,
        config: FlintConfig,
        layer_idx: int,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = None
        if layer_idx not in config.no_rope_layer_indices:
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
            )

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        self.softmax_offset = make_attention_sink(self.num_heads, torch.float32)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=(config.sliding_window if is_sliding else None),
            prefix=f"{prefix}.attn",
            sinks=self.softmax_offset,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
            assert k is not None

        q = q.reshape(num_tokens, self.q_size)
        k = k.reshape(num_tokens, self.kv_size)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class FlintDecoderLayer(nn.Module):
    def __init__(
        self,
        config: FlintConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.self_attn = FlintAttention(
            config,
            layer_idx,
            cache_config,
            quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp: FlintMLP | FlintSparseMoeBlock
        if layer_idx in config.mlp_only_layers:
            self.mlp = FlintMLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = FlintSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class FlintModel(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlintDecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        else:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = None

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if self.end_layer != len(self.layers):
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class FlintForCausalLM(nn.Module, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: FlintConfig = vllm_config.model_config.hf_config
        self.config = config

        model_config = vllm_config.model_config
        self.gguf_path = find_gguf_file(
            model_config.model,
            model_config.revision,
            vllm_config.load_config.download_dir,
        )
        if self.gguf_path is not None:
            self.allow_patterns_overrides = ["*.gguf"]

        quant_config = vllm_config.quant_config
        if self.gguf_path is not None:
            if quant_config is None:
                raise ValueError(
                    "Flint NVFP4 GGUF on ROCm requires online MXFP8 conversion. "
                    "Pass --quantization mxfp8."
                )
            quant_config = _flint_gguf_quant_config(quant_config)
            vllm_config.quant_config = quant_config

        if quant_config is not None:
            quant_config.packed_modules_mapping = self.packed_modules_mapping
        self.quant_config = quant_config

        self.model = FlintModel(
            vllm_config,
            quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
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
        if self.gguf_path is not None:
            device = next(self.parameters()).device
            weights = self._prepare_gguf_weights(
                gguf_nvfp4_weights_iterator(
                    self.gguf_path, device, dequantize=frozenset({"lm_head"})
                )
            )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def _prepare_gguf_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name == "lm_head.weight_scale_2":
                self.logits_processor.scale = float(tensor)
            else:
                yield name, tensor
