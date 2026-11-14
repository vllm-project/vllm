# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Inference-only Zaqeris AXIOM model (ZAQERIS-M3 quantized sparse MoE).
# AXIOM is a 79.43B-total / 12.39B-active sparse Mixture-of-Experts model with
# a custom 3-bit "ZAQERIS-M3" block-quantized weight format.

from collections.abc import Iterable
from itertools import islice
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP, SupportsQuant

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.zaqeris_m3 import (
        ZaqerisM3QuantConfig,
    )


# ---------------------------------------------------------------------------
# ZAQERIS-M3 weight format
# ---------------------------------------------------------------------------
# Each quantized weight tensor is stored as a bundle of compact buffers:
#   m3_cb1/2/3 : 3-stage codebooks (float16), one row per codebook vector.
#   m3_idx1/2/3 : uint8 indices selecting one codebook vector per block.
#   m3_meta     : [out_features, in_features, block_size] float32 metadata.
#   m3_outliers : float16 salient outlier values.
#   m3_outlier_idx : (N, 2) int32 [row, col] positions of the outliers.
#
# Reconstruction (mirrors the reference transformers implementation):
#   w = (cb1[idx1] + cb2[idx2] + cb3[idx3]).view(out_features, in_features)
#   w[outlier_rows, outlier_cols] = outliers
# ---------------------------------------------------------------------------
M3_BUFFER_SUFFIXES = (
    ".m3_cb1",
    ".m3_cb2",
    ".m3_cb3",
    ".m3_idx1",
    ".m3_idx2",
    ".m3_idx3",
    ".m3_meta",
    ".m3_outliers",
    ".m3_outlier_idx",
)


def _strip_m3_suffix(name: str) -> str:
    """Return the underlying weight name without its ZAQERIS-M3 buffer suffix."""
    for suffix in M3_BUFFER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class ZaqerisAxiomM3Weight(nn.Module):
    """Container for a ZAQERIS-M3 quantized weight tensor.

    The buffers are populated directly from the safetensors checkpoint by
    :class:`ZaqerisAxiomM3QuantMethod`; no ``weight`` parameter is created.
    """

    def __init__(self, out_features: int, in_features: int, block_size: int = 8):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.block_size = block_size

        self.register_buffer(
            "m3_cb1", torch.empty(0, dtype=torch.float16), persistent=True
        )
        self.register_buffer(
            "m3_cb2", torch.empty(0, dtype=torch.float16), persistent=True
        )
        self.register_buffer(
            "m3_cb3", torch.empty(0, dtype=torch.float16), persistent=True
        )
        self.register_buffer(
            "m3_idx1", torch.empty(0, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "m3_idx2", torch.empty(0, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "m3_idx3", torch.empty(0, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "m3_meta",
            torch.tensor(
                [float(out_features), float(in_features), float(block_size)],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "m3_outliers", torch.empty(0, dtype=torch.float16), persistent=True
        )
        self.register_buffer(
            "m3_outlier_idx", torch.empty((0, 2), dtype=torch.int32), persistent=True
        )

    def reconstruct(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Materialize the full dequantized weight matrix."""
        cb1 = self.m3_cb1.to(dtype)
        cb2 = self.m3_cb2.to(dtype)
        cb3 = self.m3_cb3.to(dtype)
        w = (
            cb1[self.m3_idx1.long()]
            + cb2[self.m3_idx2.long()]
            + cb3[self.m3_idx3.long()]
        ).view(self.out_features, self.in_features)
        if self.m3_outliers.numel() > 0:
            rows = self.m3_outlier_idx[:, 0].long()
            cols = self.m3_outlier_idx[:, 1].long()
            w[rows, cols] = self.m3_outliers.to(dtype)
        return w


# ---------------------------------------------------------------------------
# ZAQERIS-M3 quantization method
# ---------------------------------------------------------------------------
class ZaqerisAxiomM3QuantMethod(QuantizeMethodBase):
    """Linear quantization method for ZAQERIS-M3.

    Instead of creating a ``weight`` parameter, this method registers the M3
    compact buffers directly on the layer. The layer's ``weight`` property
    dequantizes on demand, so ``apply`` is a plain matmul.
    """

    requires_device_loading = True
    uses_meta_device = False
    supports_pre_processed_weights = True

    def __init__(self, quant_config: "ZaqerisM3QuantConfig"):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # The M3 bundle lives in a ZaqerisAxiomM3Weight submodule so that the
        # buffers are discovered by the weight loader via named_modules().
        block_size = self.quant_config.block_size
        layer.m3_weight = ZaqerisAxiomM3Weight(
            output_size, input_size, block_size=block_size
        )
        # Expose the dequantized weight as the canonical "weight" attribute so
        # vLLM's generic machinery (weight tying, refit, repr) keeps working.
        layer.weight = layer.m3_weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m3 = layer.m3_weight
        out_features = m3.out_features
        in_features = m3.in_features
        compute_dtype = x.dtype if x.is_floating_point() else torch.float16
        x_dev = x.device

        # Small weights: dequantize once and run a single matmul.
        if out_features <= 16384:
            weight = m3.reconstruct(compute_dtype).to(x_dev)
            if bias is not None:
                bias = bias.to(compute_dtype)
            return F.linear(x, weight, bias)

        # Large weights (e.g. the 148k-row lm_head): reconstruct in row chunks
        # so peak memory stays bounded regardless of batch size.
        vecs_per_row = in_features // m3.block_size
        cb1 = m3.m3_cb1.to(compute_dtype)
        cb2 = m3.m3_cb2.to(compute_dtype)
        cb3 = m3.m3_cb3.to(compute_dtype)
        has_outliers = m3.m3_outliers.numel() > 0
        out_rows = m3.m3_outlier_idx[:, 0].contiguous() if has_outliers else None
        idx1 = m3.m3_idx1
        idx2 = m3.m3_idx2
        idx3 = m3.m3_idx3

        outputs = []
        chunk_rows = 16384
        for r_start in range(0, out_features, chunk_rows):
            r_end = min(r_start + chunk_rows, out_features)
            n_rows = r_end - r_start
            v_start = r_start * vecs_per_row
            v_end = r_end * vecs_per_row
            w_chunk = (
                cb1[idx1[v_start:v_end].long()]
                + cb2[idx2[v_start:v_end].long()]
                + cb3[idx3[v_start:v_end].long()]
            ).view(n_rows, in_features)
            if has_outliers:
                s = int(torch.searchsorted(out_rows, r_start, right=False).item())
                e = int(torch.searchsorted(out_rows, r_end, right=False).item())
                if e > s:
                    sub_r = (m3.m3_outlier_idx[s:e, 0] - r_start).long()
                    sub_c = m3.m3_outlier_idx[s:e, 1].long()
                    w_chunk[sub_r, sub_c] = m3.m3_outliers[s:e].to(compute_dtype)
            b_chunk = (
                bias[r_start:r_end].to(compute_dtype) if bias is not None else None
            )
            outputs.append(F.linear(x, w_chunk.to(x_dev), b_chunk))
        return torch.cat(outputs, dim=-1)

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Row-selective lookup without materializing the full vocabulary matrix.
        return layer.m3_weight.embed(x)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        layer.m3_weight = embed_tokens.m3_weight
        layer.weight = layer.m3_weight
        return layer

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

    def dequantize_weight(self, layer: torch.nn.Module):
        return layer.m3_weight.reconstruct()


class ZaqerisAxiomLinear(LinearBase):
    """A linear layer backed by a ZAQERIS-M3 quantized weight bundle.

    TP is intentionally disabled: M3 codebooks are not sharded across ranks in
    this first implementation. Use ``tp_size == 1`` (the default for single-GPU
    or CPU-offloaded serving).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        block_size = getattr(quant_config, "block_size", None) or 8
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=True,
            return_bias=False,
        )
        # ZaqerisAxiomLinear is always M3-backed. ``LinearBase.__init__`` only
        # selects a quant_method; it deliberately does NOT call create_weights
        # (the concrete LinearBase subclass is responsible, see
        # ``ReplicatedLinear.__init__``). When no quant_config is supplied (e.g.
        # unit tests) we still install the M3 method so that create_weights and
        # apply() dequantize instead of falling back to
        # ``UnquantizedLinearMethod``, which expects a dense ``weight``
        # parameter this layer never creates.
        if quant_config is None:
            # No quant_config supplied (e.g. unit tests): still install the M3
            # method so create_weights/apply() dequantize instead of falling
            # back to ``UnquantizedLinearMethod``, which expects a dense
            # ``weight`` parameter this layer never creates.
            self.quant_config = SimpleNamespace(block_size=block_size)
            self.quant_method = ZaqerisAxiomM3QuantMethod(self.quant_config)
        self.quant_method.create_weights(
            self,
            input_size,
            [output_size],
            input_size,
            output_size,
            self.params_dtype,
        )
        if bias:
            # AXIOM's checkpoint ships no bias tensors, so the bias parameter is
            # never loaded from disk; register it as a plain parameter.
            self.bias = torch.nn.Parameter(
                torch.empty(output_size, dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, self.bias if self.has_bias else None)


class ZaqerisAxiomEmbedding(nn.Module):
    """Token embedding backed by a ZAQERIS-M3 weight bundle.

    Performs row-selective lookup so the full 148480 x 4224 matrix is never
    materialized in memory.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        block_size: int = 8,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.m3_weight = ZaqerisAxiomM3Weight(num_embeddings, embedding_dim, block_size)
        self.weight = self.m3_weight

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        target_dev = self.m3_weight.m3_cb1.device
        flat_ids = input_ids.reshape(-1).to(target_dev, dtype=torch.long)
        vecs_per_row = self.embedding_dim // self.m3_weight.block_size
        offsets = flat_ids.unsqueeze(1) * vecs_per_row + torch.arange(
            vecs_per_row, device=target_dev, dtype=torch.long
        ).unsqueeze(0)
        cb1 = self.m3_weight.m3_cb1
        cb2 = self.m3_weight.m3_cb2
        cb3 = self.m3_weight.m3_cb3
        emb = (
            cb1[self.m3_weight.m3_idx1[offsets].long()]
            + cb2[self.m3_weight.m3_idx2[offsets].long()]
            + cb3[self.m3_weight.m3_idx3[offsets].long()]
        ).reshape(flat_ids.shape[0], self.embedding_dim)
        if self.m3_weight.m3_outliers.numel() > 0:
            out_rows = self.m3_weight.m3_outlier_idx[:, 0].contiguous()
            left = torch.searchsorted(out_rows, flat_ids, right=False)
            right = torch.searchsorted(out_rows, flat_ids, right=True)
            has_match = right > left
            if has_match.any():
                matched_pos = torch.nonzero(has_match, as_tuple=False).squeeze(-1)
                for pos in matched_pos.tolist():
                    lo = int(left[pos].item())
                    hi = int(right[pos].item())
                    cols = self.m3_weight.m3_outlier_idx[lo:hi, 1].long()
                    emb[pos, cols] = self.m3_weight.m3_outliers[lo:hi].to(emb.dtype)
        return emb.view(*input_ids.shape, self.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class ZaqerisAxiomAttention(nn.Module):
    """Decoder self-attention with GQA and YaRN RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int,
        rms_norm_eps: float,
        cache_config: Any = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.q_proj = ZaqerisAxiomLinear(
            hidden_size,
            self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ZaqerisAxiomLinear(
            hidden_size,
            self.kv_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ZaqerisAxiomLinear(
            hidden_size,
            self.kv_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = ZaqerisAxiomLinear(
            self.q_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
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
        qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


class ZaqerisAxiomSparseMoeBlock(nn.Module):
    """Sparse Mixture-of-Experts block.

    AXIOM uses 16 routed experts with top-2 routing and no shared expert. This
    first implementation uses a plain per-expert loop (no expert parallelism,
    no EPLB) so that the M3 compact buffers can be loaded directly; the fused
    ``FusedMoEFactory`` path can be adopted once an M3 fused-MoE method exists.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )

        self.gate = ZaqerisAxiomLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        self.experts = nn.ModuleList(
            [
                ZaqerisAxiomExpert(
                    hidden_size,
                    intermediate_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.experts.{idx}",
                )
                for idx in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden_size)
        router_logits = self.gate(flat)
        router_logits = router_logits.float()
        routing_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1), self.num_experts_per_tok, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        final = torch.zeros_like(flat)
        for expert_idx in range(self.num_experts):
            expert_mask = expert_indices == expert_idx
            if not expert_mask.any():
                continue
            token_idx, k_pos = torch.nonzero(expert_mask, as_tuple=True)
            tokens = flat[token_idx]
            out = self.experts[expert_idx](tokens)
            final.index_add_(
                0, token_idx, out * routing_weights[token_idx, k_pos].unsqueeze(-1)
            )
        return final.view(orig_shape)


class ZaqerisAxiomExpert(nn.Module):
    """A single AXIOM MoE expert: gate_proj / up_proj / down_proj (all M3)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_proj = ZaqerisAxiomLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ZaqerisAxiomLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = ZaqerisAxiomLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ZaqerisAxiomDecoderLayer(nn.Module):
    """Pre-norm decoder layer: attention block then MoE block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_act: str,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int,
        rms_norm_eps: float,
        cache_config: Any = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = ZaqerisAxiomAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_parameters=rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.block_sparse_moe = ZaqerisAxiomSparseMoeBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe",
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
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


@support_torch_compile
class ZaqerisAxiomModel(nn.Module):
    """AXIOM backbone: embedding + decoder layers + final RMSNorm."""

    hf_to_vllm_mapper = WeightsMapper()

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = ZaqerisAxiomEmbedding(
            config.vocab_size,
            config.hidden_size,
            block_size=getattr(config, "block_size", 8),
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ZaqerisAxiomDecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                hidden_act=config.hidden_act,
                rope_parameters=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                cache_config=vllm_config.cache_config,
                quant_config=quant_config,
                prefix=prefix,
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
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=[
                ".bias",
                "_bias",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class ZaqerisAxiomForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsLoRA,
    MixtureOfExperts,
    SupportsQuant,
):
    """Inference-only Zaqeris AXIOM for causal language modeling."""

    fall_back_to_pt_during_load = False

    packed_modules_mapping: dict[str, list[str]] = {}

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = ZaqerisAxiomModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # The lm_head is a plain M3 linear (not a ParallelLMHead): AXIOM's
        # 148480-row output projection is M3-quantized and cannot use vocab
        # parallelism, which requires a dense ``weight`` parameter.
        self.lm_head = ZaqerisAxiomLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.quant_method.tie_weights(self.lm_head, self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, ZaqerisAxiomDecoderLayer):
                example_moe = layer.block_sparse_moe
                self.moe_layers.append(layer.block_sparse_moe)
        if example_moe is None:
            raise RuntimeError("No ZaqerisAxiomSparseMoeBlock found in model.layers.")
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.num_experts
        self.num_physical_experts = example_moe.num_experts
        self.num_local_physical_experts = example_moe.num_experts
        self.num_routed_experts = example_moe.num_experts
        self.num_redundant_experts = 0

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, ZaqerisAxiomDecoderLayer):
                moe = layer.block_sparse_moe
                moe.num_physical_experts = num_physical_experts
                moe.num_local_physical_experts = num_local_physical_experts
                moe.num_redundant_experts = self.num_redundant_experts

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
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint stores each M3 bundle under the parent weight name, e.g.
        #   model.layers.0.self_attn.q_proj.weight.m3_cb1
        # vLLM's module traversal expects the buffer under the module that owns
        # it, which is the ``m3_weight`` submodule:
        #   model.layers.0.self_attn.q_proj.m3_weight.m3_cb1
        # Rewrite the checkpoint names before delegating to AutoWeightsLoader.
        rewritten: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if ".weight.m3_" in name:
                name = name.replace(".weight.m3_", ".m3_weight.m3_", 1)
            rewritten.append((name, tensor))
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=[
                ".bias",
                "_bias",
            ],
        )
        return loader.load_weights(weights=rewritten)
