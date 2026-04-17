# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only BitNet b1.58 model compatible with HuggingFace weights.

Architecture: LLaMA-like with three key differences:
  1. SubLN: Extra RMSNorm layers inside attention (attn_sub_norm) and MLP
     (ffn_sub_norm) blocks.
  2. Activation: Uses ReLU² instead of SiLU.
  3. BitNet quantization: All linear layers use online 1.58-bit weight
     quantization (ternary: -1, 0, +1 via WeightQuant) and 8-bit activation
     quantization (ActQuant) on every forward pass.

Reference checkpoint: microsoft/bitnet-b1.58-2B-4T-bf16
Reference: transformers.integrations.bitnet.AutoBitLinear
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors


# ---------------------------------------------------------------------------
# BitNet quantization functions (replicate AutoBitLinear behavior)
# ---------------------------------------------------------------------------

def weight_quant(weight: torch.Tensor) -> torch.Tensor:
    """Ternary weight quantization (BitNet b1.58 WeightQuant).

    Quantizes weights to {-1, 0, +1} scaled by mean absolute value.
    Formula: round(w / mean_abs(w)).clamp(-1, 1) * mean_abs(w)

    This is deterministic, so it can be precomputed at weight loading time.
    """
    dtype = weight.dtype
    weight = weight.float()
    scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    weight = (weight * scale).round().clamp(-1, 1) / scale
    return weight.to(dtype)


def act_quant(x: torch.Tensor) -> torch.Tensor:
    """Symmetric 8-bit activation quantization (BitNet b1.58 ActQuant).

    Quantizes activations to [-128, 127] using per-token scaling.
    Formula: round(x * (127 / max_abs(x))).clamp(-128, 127) / (127 / max_abs(x))

    Must run on every forward pass since activations change per input.
    """
    dtype = x.dtype
    x = x.float()
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    x = (x * scale).round().clamp(-128, 127) / scale
    return x.to(dtype)


class BitNetMLP(nn.Module):
    """BitNet MLP with ReLU² activation and SubLN (ffn_sub_norm).

    Structure:
        x → ActQuant → gate_up_proj(WeightQuant) → relu²(gate) * up
          → ffn_sub_norm → ActQuant → down_proj(WeightQuant)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # SubLN: extra norm between gated activation and down_proj
        self.ffn_sub_norm = RMSNorm(intermediate_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ActQuant on input to gate_up_proj
        x = act_quant(x)
        gate_up, _ = self.gate_up_proj(x)
        # Split into gate and up halves, apply relu² to gate, multiply
        d = gate_up.shape[-1] // 2
        gate = gate_up[..., :d]
        up = gate_up[..., d:]
        # ReLU² activation: relu(x)^2
        x = torch.square(F.relu(gate)) * up
        # SubLN
        x = self.ffn_sub_norm(x)
        # ActQuant on input to down_proj
        x = act_quant(x)
        x, _ = self.down_proj(x)
        return x


class BitNetAttention(nn.Module):
    """BitNet attention with SubLN (attn_sub_norm).

    Structure:
        hidden → ActQuant → QKV(WeightQuant) → RoPE → attention
               → attn_sub_norm → ActQuant → o_proj(WeightQuant)
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 4096,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # SubLN: extra norm between attention output and o_proj
        self.attn_sub_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # ActQuant on input to QKV projection
        hidden_states = act_quant(hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        # SubLN: normalize before output projection
        attn_output = self.attn_sub_norm(attn_output)
        # ActQuant on input to o_proj
        attn_output = act_quant(attn_output)
        output, _ = self.o_proj(attn_output)
        return output


class BitNetDecoderLayer(nn.Module):
    """BitNet decoder layer.

    Structure:
        x → input_layernorm → attention → residual_add
          → post_attention_layernorm → mlp → residual_add
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        # BitNet BF16 mode: no quantization
        quant_config = None

        self.hidden_size = config.hidden_size
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)

        self.self_attn = BitNetAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=getattr(
                config, "max_position_embeddings", 4096
            ),
            quant_config=quant_config,
            bias=getattr(config, "attention_bias", False),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = BitNetMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
            rms_norm_eps=rms_norm_eps,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class BitNetModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BitNetDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (
                "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
            ):
                continue

            # Apply BitNet ternary weight quantization at load time.
            # This replicates the online WeightQuant from AutoBitLinear,
            # but we do it once at load time since it's deterministic.
            # Only apply to projection weights (not norms or embeddings).
            is_proj_weight = (
                any(proj in name for proj in [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ])
                and name.endswith(".weight")
            )
            if is_proj_weight:
                loaded_weight = weight_quant(loaded_weight)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class BitNetForCausalLM(nn.Module, SupportsPP):
    """BitNet b1.58 BF16 model for causal language modeling."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.model = BitNetModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens
                )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

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
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."]
                if self.config.tie_word_embeddings
                else None
            ),
        )
        return loader.load_weights(weights)
