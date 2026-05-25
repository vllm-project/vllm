# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only FlexOlmo model compatible with HuggingFace weights."""

import torch
from torch import nn
import torch.nn.functional as F

from collections.abc import Iterable
from itertools import islice

from transformers.integrations import use_kernel_forward_from_hub

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.attention import Attention
from vllm.transformers_utils.configs import FlexOlmoConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


logger = init_logger(__name__)


def _first_5_rows(x):
    """Return first 5 rows for tensors, otherwise return as-is."""
    if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.size(0) > 5:
        return x[:5]
    return x


def _log_info_5(msg):
    logger.info(msg)


@use_kernel_forward_from_hub("RMSNorm")
class FlexOlmoRMSNorm(nn.Module):
    def __init__(self, hidden_size: torch.Tensor, eps=1e-6):
        """
        FlexOlmoRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = (self.weight * hidden_states).to(input_dtype)
        return out

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class FlexOlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as
    `Attention(LN(x))` in `MLP(LN(x + Attention(LN(x))))`
    (plus another skip connection).
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.config = hf_config
        self.hidden_size = hf_config.hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = hf_config.num_attention_heads

        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tensor_model_parallel_world_size == 0

        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = getattr(
            hf_config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.max_position_embeddings = hf_config.max_position_embeddings
        self.total_num_key_value_heads = hf_config.num_key_value_heads
        if self.total_num_key_value_heads >= tensor_model_parallel_world_size:
            assert self.total_num_key_value_heads % tensor_model_parallel_world_size == 0
        else:
            assert tensor_model_parallel_world_size % self.total_num_key_value_heads == 0
        self.num_key_value_heads = max(
            1, self.total_num_key_value_heads // tensor_model_parallel_world_size
        )

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_key_value_heads,
            bias=hf_config.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=hf_config.rope_parameters,
        )

        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            scale=self.scaling,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )

        # Attention output projection: (num_heads * head_dim) -> hidden_size
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=hf_config.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Match HF: always create q_norm/k_norm. When use_head_qk_norm=False, norm over full q/k dim;
        # when use_head_qk_norm=True, overwrite with per-head norm (head_dim only).
        self.use_head_qk_norm = getattr(hf_config, "use_head_qk_norm", False)
        self.q_norm = FlexOlmoRMSNorm(
            self.head_dim if self.use_head_qk_norm else self.q_size,
            eps=hf_config.rms_norm_eps,
        )
        self.k_norm = FlexOlmoRMSNorm(
            self.head_dim if self.use_head_qk_norm else self.kv_size,
            eps=hf_config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if not self.use_head_qk_norm:
            # Full q/k norm over last dim (num_heads*head_dim, num_kv_heads*head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            # Per-head norm: reshape to (..., num_heads, head_dim), norm over head_dim
            q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            q = self.q_norm(q)
            q = q.view(*q.shape[:-2], -1)
            k = k.view(*k.shape[:-1], self.num_key_value_heads, self.head_dim)
            k = self.k_norm(k)
            k = k.view(*k.shape[:-2], -1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
    

class GroupedGemmExperts(nn.Module):
    """
    A dropless expert MLP module with SwiGLU activation that uses grouped GEMM for efficient computation.
    This is adapted from olmo-core's DroplessMoEMLP implementation.
    """

    _gmm_fn = None
    _gmm_checked = False

    @classmethod
    def _get_gmm(cls):
        """Lazily import grouped_gemm and cache the result."""
        if not cls._gmm_checked:
            try:
                import grouped_gemm  # type: ignore

                cls._gmm_fn = grouped_gemm.ops.gmm
            except ImportError:
                cls._gmm_fn = None
            cls._gmm_checked = True
        return cls._gmm_fn

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        hf_config = vllm_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        tp_size = get_tensor_model_parallel_world_size()

        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.num_experts = hf_config.num_experts

        # Weight layout: (num_experts * intermediate_size, hidden_size)
        # This layout is optimized for grouped GEMM operations
        self.w1 = nn.Parameter(torch.empty(self.num_experts * self.intermediate_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts * self.intermediate_size, self.hidden_size))
        self.w3 = nn.Parameter(torch.empty(self.num_experts * self.intermediate_size, self.hidden_size))

        # Check if grouped_gemm is available
        if self._get_gmm() is None:
            logger.warning(
                "Grouped GEMM not available, so the MoE will be substantially slower. "
                "Please install the 'grouped_gemm' package if possible.\n"
                "https://github.com/tgale96/grouped_gemm"
            )

    @torch._dynamo.disable()
    def gmm(self, x: torch.Tensor, w: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool = False) -> torch.Tensor:
        """
        Grouped matrix multiplication.

        Args:
            x: Input tensor of shape (total_tokens, hidden_size)
            w: Weight tensor of shape (num_experts, intermediate_size, hidden_size)
            batch_sizes: Number of tokens for each expert, shape (num_experts,)
            trans_b: Whether to transpose the weight matrix
        """
        gmm_fn = self._get_gmm()
        if gmm_fn is not None:
            # grouped_gemm only accepts BF16
            result: torch.Tensor = gmm_fn(
                x.to(torch.bfloat16),
                w.to(torch.bfloat16),
                batch_sizes,
                trans_b=trans_b,
            )
            return result
        else:
            # Fallback to sequential computation
            out: list[torch.Tensor] = []
            start = 0
            for i, size in enumerate(batch_sizes.cpu().numpy()):
                if size > 0:
                    rhs = w[i, :, :].t() if trans_b else w[i, :, :]
                    out.append(x[start : start + size, :] @ rhs)
                start += size
            if out:
                return torch.cat(out)
            else:
                # Return empty tensor with correct shape
                out_dim = w.shape[1] if trans_b else w.shape[2]
                return torch.empty(0, out_dim, device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, batch_size_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Compute the expert outputs using grouped GEMM.

        Args:
            x: Input tensor of shape (total_tokens, hidden_size), tokens are sorted by expert assignment
            batch_size_per_expert: Number of tokens for each expert, shape (num_experts,)

        Returns:
            Output tensor of shape (total_tokens, hidden_size)
        """
        # Reshape weights: (num_experts, intermediate_size, hidden_size)
        w1 = self.w1.view(self.num_experts, self.intermediate_size, self.hidden_size)
        w2 = self.w2.view(self.num_experts, self.intermediate_size, self.hidden_size)
        w3 = self.w3.view(self.num_experts, self.intermediate_size, self.hidden_size)

        # SwiGLU: silu(x @ w1^T) * (x @ w3^T) @ w2
        x1 = self.gmm(x, w1, batch_size_per_expert, trans_b=True)  # (total_tokens, intermediate_size)
        x3 = self.gmm(x, w3, batch_size_per_expert, trans_b=True)  # (total_tokens, intermediate_size)
        x1 = F.silu(x1) * x3
        out = self.gmm(x1, w2, batch_size_per_expert)  # (total_tokens, hidden_size)
        return out


class GroupGemmMoe(nn.Module):
    """
    MoE block that uses grouped GEMM for efficient computation.
    This is adapted from olmo-core's DroplessMoE implementation.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.num_experts_per_tok
        self.norm_topk_prob = hf_config.norm_topk_prob
        self.hidden_size = hf_config.hidden_size

        self.gate = ReplicatedLinear(
            hf_config.hidden_size,
            hf_config.num_experts,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = GroupedGemmExperts(vllm_config=vllm_config, prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Support both 2D (num_tokens, hidden_dim) and 3D (batch, seq, hidden_dim)
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        # Router: (num_tokens, num_experts)
        # IMPORTANT: Match olmo-core's router which computes logits in float32
        # OLMo Core does: F.linear(x.float(), weight.float())
        # This is critical for numerical equivalence
        router_logits = F.linear(hidden_states.float(), self.gate.weight.float())

        # Compute routing weights and select top-k experts
        # Use float32 for softmax precision, matching olmo-core's router
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Flatten expert indices: (num_tokens * top_k,)
        flat_expert_indices = selected_experts.view(-1)
        flat_routing_weights = routing_weights.view(-1)

        # Count tokens per expert
        # NOTE: grouped_gemm requires batch_sizes to be on CPU
        batch_size_per_expert = torch.bincount(flat_expert_indices, minlength=self.num_experts).to(device="cpu", dtype=torch.int64)

        # Sort tokens by expert assignment
        sorted_indices = torch.argsort(flat_expert_indices, stable=True)
        sorted_routing_weights = flat_routing_weights[sorted_indices]

        # Map back to original token indices (handle top_k expansion)
        token_indices = sorted_indices // self.top_k
        sorted_hidden_states = hidden_states[token_indices]

        # Compute expert outputs using grouped GEMM
        # Output dtype is bfloat16 when using grouped_gemm (matching olmo-core DroplessMoEMLP)
        expert_outputs = self.experts(sorted_hidden_states, batch_size_per_expert)

        # IMPORTANT: Match olmo-core's padded_scatter behavior EXACTLY
        # olmo-core does:
        # 1. out = torch.empty((tokens, top_k, hidden), dtype=x.dtype)  # bfloat16
        # 2. For each sorted_idx: out[original_idx] = mlp[sorted_idx] * weight[original_idx]
        #    where multiplication is in float32, then stored as bfloat16
        # 3. return out.sum(dim=1)  # sum in bfloat16
        output_dtype = expert_outputs.dtype
        
        # Create 3D output tensor matching olmo-core's padded_scatter
        out = torch.empty(
            num_tokens, self.top_k, hidden_dim, dtype=output_dtype, device=hidden_states.device
        )
        
        # Compute (token, k) indices from original flat indices
        # sorted_indices[i] = original position for sorted position i
        original_indices = sorted_indices
        original_token_indices = original_indices // self.top_k
        original_k_indices = original_indices % self.top_k
        
        # Apply routing weights: float32 multiplication, then store as bfloat16
        # Use flat_routing_weights[original_indices] to match olmo-core's weights[index_a]
        weighted_outputs = (
            expert_outputs.float() * flat_routing_weights[original_indices].unsqueeze(-1).float()
        ).to(output_dtype)
        
        # Write to 3D tensor at (token, k) positions
        out[original_token_indices, original_k_indices] = weighted_outputs
        
        # Sum along top_k dimension (in bfloat16), matching olmo-core's out.sum(dim=1)
        final_hidden_states = out.sum(dim=1)

        final_hidden_states = final_hidden_states.view(orig_shape)
        return final_hidden_states


class SonicMLP(nn.Module):
    """
    A Sonic expert MLP module with SwiGLU activation.
    This is adapted from olmo-core's SonicMoEMLP implementation.
    """

    _sonic_checked = False
    _sonic_available = False

    @classmethod
    def _check_sonic(cls):
        """Lazily check if sonicmoe is available."""
        if not cls._sonic_checked:
            try:
                import sonicmoe.functional  # type: ignore
                cls._sonic_available = True
            except ImportError:
                cls._sonic_available = False
            cls._sonic_checked = True
        return cls._sonic_available

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.num_experts = hf_config.num_experts
        self.d_model = hf_config.hidden_size

        # Weight layout matches olmo-core's SonicMoEMLP:
        # fc: (num_experts, 2 * intermediate_size, hidden_size) - gate + up projection combined
        # proj: (num_experts, hidden_size, intermediate_size) - down projection
        self.fc = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )

        # Check if sonicmoe is available
        if not self._check_sonic():
            logger.warning(
                "SonicMoE not available, please install the 'sonicmoe' package. "
                "https://github.com/NVIDIA/sonicmoe"
            )

    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Compute the expert outputs using SonicMoE.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) or (total_tokens, hidden_size)
            expert_weights: Routing weights for each token-expert pair
            expert_indices: Expert indices for each token
            top_k: Number of experts per token

        Returns:
            Output tensor of same shape as input
        """
        from sonicmoe.functional import _DownProjection, _UpProjection, ActivationType
        from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton

        fc = self.fc.view(self.num_experts, 2 * self.intermediate_size, self.d_model)
        proj = self.proj.view(self.num_experts, self.d_model, self.intermediate_size)

        # fc shape is (2 * intermediate_size, d_model, num_experts)
        # proj shape is (d_model, intermediate_size, num_experts)
        fc = fc.permute(1, 2, 0)
        proj = proj.permute(1, 2, 0)

        T, K = expert_indices.size()
        TK = T * K
        device = expert_indices.device
        expert_frequency = torch.empty(self.num_experts, dtype=torch.int32, device=device)
        expert_frequency_offset = torch.empty(self.num_experts + 1, dtype=torch.int32, device=device)
        x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
        TC_topk_router_metadata_triton(
            expert_indices,
            self.num_experts,
            expert_frequency,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        original_shape = x.shape
        x = x.view(-1, self.d_model)
        seq_len = x.size(0)

        a, h = _UpProjection.apply(
            x,
            fc,
            None,
            expert_frequency_offset,
            seq_len * top_k,
            top_k,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,  # is_each_token_has_variable_activated_experts
            ActivationType.SWIGLU,
            False,  # is_inference_mode_enabled
        )

        o = _DownProjection.apply(
            a,
            h,
            proj,
            None,
            expert_weights,
            expert_frequency_offset,
            seq_len,
            top_k,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            None,
            False,  # is_each_token_has_variable_activated_experts
            ActivationType.SWIGLU,
        )

        assert o is not None
        out = o.view(original_shape)
        return out


class SonicMoe(nn.Module):
    """
    MoE block that uses SonicMoE for efficient computation.
    This is adapted from olmo-core's SonicMoE implementation.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.num_experts_per_tok
        self.norm_topk_prob = hf_config.norm_topk_prob
        self.hidden_size = hf_config.hidden_size

        self.gate = ReplicatedLinear(
            hf_config.hidden_size,
            hf_config.num_experts,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        
        self.experts = SonicMLP(vllm_config=vllm_config, prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Support both 2D (num_tokens, hidden_dim) and 3D (batch, seq, hidden_dim)
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]

        # Router: (num_tokens, num_experts)
        # Match olmo-core's router which computes logits in float32
        router_logits = F.linear(hidden_states_flat.float(), self.gate.weight.float())

        # Compute routing weights and select top-k experts
        # Use float32 for softmax precision, matching olmo-core's router
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)

        # expert_weights shape: (num_tokens, top_k)
        # expert_indices shape: (num_tokens, top_k)
        final_hidden_states = self.experts(
            hidden_states_flat,
            routing_weights_topk,
            selected_experts,
            self.top_k,
        )

        final_hidden_states = final_hidden_states.view(orig_shape)
        return final_hidden_states


class FlexOlmoDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        assert isinstance(hf_config, FlexOlmoConfig)

        self.self_attn = FlexOlmoAttention(
            vllm_config=vllm_config, prefix=f"{prefix}.self_attn"
        )
        self.input_layernorm = FlexOlmoRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.post_attention_layernorm = FlexOlmoRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.config = hf_config

        if hf_config.pre_norm:
            self.pre_attention_layernorm = FlexOlmoRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            self.pre_feedforward_layernorm = FlexOlmoRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        else:
            self.post_attention_layernorm = FlexOlmoRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            self.post_feedforward_layernorm = FlexOlmoRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            del self.input_layernorm

        # HF config uses "grouped_gemm", vLLM uses "group_mm" for the same backend
        moe_impl = getattr(hf_config, "moe_implementation", "group_mm")
        if moe_impl == "sonic":
            self.mlp = SonicMoe(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        elif moe_impl == "group_mm":
            self.mlp = GroupGemmMoe(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        else:
            raise ValueError(f"Invalid MoE implementation: {hf_config.moe_implementation}")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Attention block.
        residual = hidden_states
        if self.config.pre_norm:
            hidden_states = self.pre_attention_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        if not self.config.pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        # MLP block.
        residual = hidden_states
        if self.config.pre_norm:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if not self.config.pre_norm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
        

class FlexOlmoModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: FlexOlmoDecoderLayer(
                vllm_config=vllm_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = FlexOlmoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
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
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # Apply blocks one-by-one.
        for layer_idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            # shape: (batch_size, seq_len, d_model)
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    raise KeyError(
                        f"Checkpoint weight maps to parameter '{name}' which is not in the model. "
                        f"Available model params (prefix): {sorted(set(p.rsplit('.', 1)[0] for p in params_dict))}"
                    )
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
                if name not in params_dict:
                    raise KeyError(
                        f"Checkpoint weight maps to parameter '{name}' which is not in the model. "
                        f"Available model params (prefix): {sorted(set(p.rsplit('.', 1)[0] for p in params_dict))}"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                try:
                    weight_loader(param, loaded_weight)
                except AssertionError as e:
                    raise AssertionError(
                        f"Parameter '{name}': {e}"
                    ) from e
            loaded_params.add(name)
        return loaded_params


class FlexOlmoForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = FlexOlmoModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
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
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.weight"] if self.config.tie_word_embeddings else None
            ),
        )
        return loader.load_weights(weights)