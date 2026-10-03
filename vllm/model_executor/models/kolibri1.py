# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kolibri 1 model."""

from functools import partial

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported

from .qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from .utils import PPMissingLayer, WeightsMapper, extract_layer_index, maybe_prefix


class Kolibri1Attention(nn.Module):
    """GQA with qk-norm. Sliding-window layers use RoPE, full-attention
    layers use no positional encoding (RNoPE)."""

    def __init__(
        self, vllm_config: VllmConfig, prefix: str, is_full_attention: bool
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
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

        if is_full_attention:
            sliding_window = None
            self.rotary_emb = None
        else:
            sliding_window = config.sliding_window
            if sliding_window is None or sliding_window <= 0:
                raise ValueError(
                    "Kolibri 1 sliding-attention layers need a positive "
                    f"sliding_window, got {sliding_window}."
                )
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


@torch.compile(backend=current_platform.simple_compile_backend)
def sigmoid_logit_add_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k on `logits + bias`, weight by the unbiased
    `sigmoid(logits)`. vLLM's built-in sigmoid scoring instead selects on
    `sigmoid(logits) + bias`."""
    logits = gating_output.float()
    topk_ids = torch.topk(logits + e_score_correction_bias, k=topk, dim=-1)[1]
    topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_weights, topk_ids.to(torch.int32)


class Kolibri1SparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Differences from `Qwen3MoeSparseMoeBlock`:
    - fp32 router logits and `sigmoid_logit_add_routing`.
    - Shared expert without a shared-expert gate, named `shared_experts`.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        if get_tensor_model_parallel_world_size() > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {get_tensor_model_parallel_world_size()} "
                f"is greater than the number of experts {config.num_experts}."
            )
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.n_routed_experts = config.num_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = (
            self.n_physical_experts // get_ep_group().device_group.size()
        )

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=torch.float32)
        )
        self.shared_experts = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            custom_routing_function=partial(
                sigmoid_logit_add_routing,
                e_score_correction_bias=self.gate.e_score_correction_bias,
            ),
            router_logits_dtype=self.gate.out_dtype,
        )


class Kolibri1DecoderLayer(Qwen3MoeDecoderLayer):
    """Differences from `Qwen3MoeDecoderLayer`:
    - `Kolibri1Attention` with the pattern from `config.layer_types`.
    - Every layer is MoE.
    - Sandwich norms after attention and MoE.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config

        layer_type = config.layer_types[extract_layer_index(prefix)]
        self.self_attn = Kolibri1Attention(
            vllm_config,
            prefix=f"{prefix}.self_attn",
            is_full_attention=layer_type == "full_attention",
        )
        self.mlp = Kolibri1SparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.post_attn_norm(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Kolibri1Model(Qwen3MoeModel):
    hf_to_vllm_mapper = Qwen3MoeModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_substr={
            ".moe.router.expert_bias": ".mlp.gate.e_score_correction_bias",
        },
        orig_to_new_stacked={
            ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
            ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Kolibri1DecoderLayer,
        )


class Kolibri1ForCausalLM(Qwen3MoeForCausalLM):
    hf_to_vllm_mapper = Kolibri1Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        # DeepGEMM on Blackwell only takes UE8M0 scales, so it cannot run the
        # fp32 block scales at all.
        is_blackwell = any(
            current_platform.is_device_capability_family(family)
            for family in (100, 120)
        )
        if (
            isinstance(quant_config, Fp8Config)
            and quant_config.weight_block_size is not None
            and (
                is_deep_gemm_e8m0_used() or (is_blackwell and is_deep_gemm_supported())
            )
        ):
            raise RuntimeError(
                "Kolibri 1 FP8 weights carry fp32 block scales, which DeepGEMM "
                "would round to powers of two (UE8M0). Unset VLLM_USE_DEEP_GEMM "
                "and VLLM_USE_DEEP_GEMM_E8M0 or set them to 0."
            )
        self.config = config
        self.quant_config = quant_config
        self.model = Kolibri1Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        moe_blocks = [
            layer.mlp
            for layer in self.model.layers
            if not isinstance(layer, PPMissingLayer)
        ]
        example_layer = moe_blocks[0]
        self.moe_layers = [moe.experts for moe in moe_blocks]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 1
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
