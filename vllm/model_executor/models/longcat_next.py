# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LongCat-Next unified multimodal model."""

import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank
from vllm.inputs import MultiModalDataDict, PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import (
    get_processor_kwargs_type,
)

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    isin_list,
    maybe_prefix,
)

logger = init_logger(__name__)


class Int4PerChannelEmbeddingMethod(QuantizeMethodBase):
    """Online int4 per-channel quantization for huge vocabulary embeddings.

    Used for LongCat-Next's n-gram embeddings, which dominate the parameter
    count. Weights are packed two 4-bit values per byte and dequantized on-the-fly
    during lookup, cutting storage to ~0.5 bytes per parameter.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.weight_block_size = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # The Int4VocabParallelEmbedding class replaces this weight with a
        # packed uint8 buffer immediately after init, so this path is only used
        # to satisfy the base-class contract.
        weight = torch.nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        from vllm.model_executor.parameter import set_weight_attrs

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_int4_quantized", False):
            return
        weight = layer.weight
        hidden_size = weight.shape[1]
        if hidden_size % 2 != 0:
            raise ValueError(
                f"Int4 embedding requires even embedding_dim, got {hidden_size}"
            )

        logger.warning(
            "Quantizing %s weight %s to int4 (shape %s, device %s, "
            "current mem %.2f GB)",
            layer.__class__.__name__,
            getattr(layer, "prefix", "?"),
            tuple(weight.shape),
            weight.device,
            torch.accelerator.memory_allocated(weight.device) / 1e9,
        )

        # Per-channel symmetric int4 scale: range [-7, 7].
        max_abs = weight.abs().max(dim=0, keepdim=True).values
        scale = (max_abs / 7.0).to(weight.dtype)
        q = (weight / scale).round().clamp(-7, 7).to(torch.int8)
        # Offset to unsigned 4-bit values [0, 14] and pack two per byte.
        q_uint4 = (q + 8).to(torch.uint8)
        packed = q_uint4[:, ::2] | (q_uint4[:, 1::2] << 4)

        replace_parameter(
            layer, "weight", torch.nn.Parameter(packed, requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(scale, requires_grad=False),
        )
        layer._already_int4_quantized = True
        torch.accelerator.empty_cache()
        logger.warning(
            "Finished int4 quant for %s (new mem %.2f GB)",
            getattr(layer, "prefix", "?"),
            torch.accelerator.memory_allocated(weight.device) / 1e9,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Int4PerChannelEmbeddingMethod only supports embedding"
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        # Unpack and dequantize only the selected rows.
        packed = layer.weight[input_]
        low = (packed & 0xF).to(torch.int16)
        high = (packed >> 4).to(torch.int16)
        q_uint4 = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
        q = q_uint4.to(layer.params_dtype) - 8.0
        return q * layer.weight_scale


class Int4VocabParallelEmbedding(VocabParallelEmbedding):
    """VocabParallelEmbedding that stores its weight in int4 per-channel format.

    Unlike the base class, the int4 weight is allocated directly at init time
    so the full bf16/fp16 weight is never kept on the GPU.  The checkpoint weight
    is quantized on-the-fly inside the weight loader.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = 64,
        prefix: str = "",
    ) -> None:
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"Int4VocabParallelEmbedding requires even embedding_dim, "
                f"got {embedding_dim}"
            )

        # Let the base class compute TP shard metadata and allocate a temporary
        # full-precision weight.  We replace it with a packed int4 buffer below.
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            params_dtype=params_dtype,
            org_num_embeddings=org_num_embeddings,
            padding_size=padding_size,
            quant_config=None,
            prefix=prefix,
        )

        device = self.weight.device
        packed_shape = (
            self.num_embeddings_per_partition,
            self.embedding_dim // 2,
        )
        packed = torch.nn.Parameter(
            torch.empty(packed_shape, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        scale = torch.nn.Parameter(
            torch.empty(1, self.embedding_dim, dtype=self.params_dtype, device=device),
            requires_grad=False,
        )

        replace_parameter(self, "weight", packed)
        self.register_parameter("weight_scale", scale)

        # replace_parameter copies the old parameter's weight_loader onto a new
        # Parameter object, so we must install the int4 loader on self.weight.
        if hasattr(self.weight, "weight_loader"):
            delattr(self.weight, "weight_loader")
        set_weight_attrs(
            self.weight,
            {
                "weight_loader": self._int4_weight_loader,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        # Install the int4 embedding method so forward() uses the unpack path.
        self.quant_method = Int4PerChannelEmbeddingMethod(self)
        # The weight loader already quantizes the checkpoint; skip a second pass.
        self._already_int4_quantized = True

    def _pick_staging_device(self, target_device: torch.device) -> torch.device:
        """Pick an idle GPU for staging quantization chunks.

        The target GPU is usually full of model weights, so we stream chunks
        to another GPU that has free memory. Falls back to CPU if no GPU is
        available.

        NOTE: For multi-worker startup the "most free" GPU can be contended
        by several concurrent loaders, so we conservatively use CPU to avoid
        sporadic OOMs during N-gram embedding quantization.
        """
        return torch.device("cpu")

    def _int4_weight_loader(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        """Shard the loaded bf16/fp16 weight and quantize to int4 in-place."""
        start_idx = self.shard_indices.org_vocab_start_index
        shard_size = self.shard_indices.org_vocab_end_index - start_idx

        if shard_size <= 0:
            param.data.fill_(0)
            self.weight_scale.data.fill_(0)
            return

        # Keep the full bf16 shard on CPU so it does not compete with the
        # model weights on the target GPU. We then stream modest chunks to an
        # idle staging GPU for fast quantization.
        loaded_shard_cpu = loaded_weight.narrow(0, start_idx, shard_size).detach().cpu()

        target_device = param.device
        staging_device = self._pick_staging_device(target_device)

        # ~512 MiB bf16 per chunk keeps staging memory modest.
        chunk_rows = 1_048_576

        # First pass: compute per-channel max abs.
        max_abs = torch.zeros(
            1, self.embedding_dim, dtype=torch.float32, device=staging_device
        )
        for row_start in range(0, shard_size, chunk_rows):
            row_end = min(row_start + chunk_rows, shard_size)
            chunk = loaded_shard_cpu[row_start:row_end].to(staging_device)
            chunk_max = chunk.abs().amax(dim=0, keepdim=True)
            max_abs = torch.maximum(max_abs, chunk_max)
        scale = (max_abs / 7.0).to(self.params_dtype)

        # Second pass: quantize each chunk and copy the packed result back.
        for row_start in range(0, shard_size, chunk_rows):
            row_end = min(row_start + chunk_rows, shard_size)
            chunk = loaded_shard_cpu[row_start:row_end].to(staging_device)
            q = (
                (chunk.to(torch.float32) / scale.to(torch.float32))
                .round()
                .clamp(-7, 7)
                .to(torch.int8)
            )
            q_uint4 = (q + 8).to(torch.uint8)
            packed = q_uint4[:, ::2] | (q_uint4[:, 1::2] << 4)
            param.data[row_start:row_end].copy_(packed)

        param.data[shard_size:].fill_(0)
        self.weight_scale.data.copy_(scale)


class NgramEmbedding(nn.Module):
    """N-gram embedding layer with polynomial rolling hash."""

    def __init__(self, config: PretrainedConfig, embed_tokens: VocabParallelEmbedding):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens

        # N-gram parameters
        self.emb_neighbor_num = config.emb_neighbor_num  # n = 4
        self.emb_split_num = config.emb_split_num  # k = 4
        self.ngram_vocab_size_ratio = config.ngram_vocab_size_ratio  # 78

        # Base vocab size for N-gram hash: m = ratio * text_vocab_size
        # Each embedder vocab_size = m + index*2 + 1
        self.m = (
            self.ngram_vocab_size_ratio * config.text_vocab_size
        )  # 78 * 131072 = 10223616

        # Only create N-gram embedders on the first PP rank
        # On non-first ranks, embed_tokens is PPMissingLayer
        is_first = get_pp_group().is_first_rank
        with open("/tmp/longcat_ngram_debug.log", "a") as _f:
            _f.write(
                f"NgramEmbedding init is_first_rank={is_first} "
                f"tp_rank={get_tensor_model_parallel_rank()}\n"
            )
        if is_first:
            self.is_first_pp_rank = True
            # Create embedders for each N-gram combination
            # Total embedders = k * (n - 1) = 4 * 3 = 12
            num_embedders = self.emb_split_num * (self.emb_neighbor_num - 1)
            emb_dim_per_embedder = (
                config.hidden_size // num_embedders
            )  # 3072 // 12 = 256
            self.embedders = nn.ModuleList()
            self.post_projs = nn.ModuleList()
            for index in range(num_embedders):
                emb_vocab_dim = int(self.m + index * 2 + 1)
                # Use VocabParallelEmbedding for TP sharding per embedder
                self.embedders.append(
                    Int4VocabParallelEmbedding(
                        emb_vocab_dim,
                        emb_dim_per_embedder,
                    )
                )
                # NOTE: Use nn.Linear (replicated), NOT ColumnParallelLinear.
                # VocabParallelEmbedding already does an internal AllReduce,
                # returning a full 256-dim tensor on each rank.
                self.post_projs.append(
                    nn.Linear(
                        emb_dim_per_embedder,
                        config.hidden_size,
                        bias=False,
                    )
                )
        else:
            self.is_first_pp_rank = False
            self.embedders = nn.ModuleList()
            self.post_projs = nn.ModuleList()

    def _shift_right_ignore_eos(
        self, tensor: torch.Tensor, n: int, eos_token_id: int = 2
    ) -> torch.Tensor:
        """Shift tensor right by n positions, respecting EOS boundaries.

        Args:
            tensor: 1D tensor of token IDs
            n: Number of positions to shift
            eos_token_id: EOS token ID (default 2)

        Returns:
            Shifted tensor with zeros in invalid positions
        """
        q = tensor.numel()
        if n == 0:
            return tensor.clone()
        if n >= q:
            return torch.zeros_like(tensor)

        result = torch.zeros_like(tensor)

        # Find EOS and special token locations
        special_tokens = 0
        total_mask = (tensor == eos_token_id) | (tensor == special_tokens)

        # Calculate segment IDs based on EOS positions
        eos_cumsum = total_mask.long().cumsum(dim=0)
        segment_ids = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=tensor.device), eos_cumsum[:-1]],
            dim=0,
        )

        col_indices = torch.arange(q, device=tensor.device)
        max_segments = segment_ids.max().item() + 1
        segment_starts = torch.full(
            (max_segments,), q, dtype=torch.long, device=tensor.device
        )
        segment_starts.scatter_reduce_(
            0, segment_ids, col_indices, reduce="amin", include_self=False
        )

        segment_start_per_pos = segment_starts[segment_ids]
        offset_in_segment = col_indices - segment_start_per_pos
        source_offset = offset_in_segment - n
        valid_mask = source_offset >= 0
        source_indices = segment_start_per_pos + torch.clamp(source_offset, min=0)

        result = torch.gather(tensor, 0, source_indices)
        result = result * valid_mask * (~total_mask)

        return result

    def _precompute_vocab_mods(self) -> dict:
        """Precompute modular arithmetic values for N-gram hash."""
        if hasattr(self, "_vocab_mods_cache") and self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        vocab_size = self.config.text_vocab_size

        for i in range(2, self.emb_neighbor_num + 1):
            for j in range(self.emb_split_num):
                index = (i - 2) * self.emb_split_num + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self, context: torch.Tensor, shifted_ids: dict, vocab_mods: list, ngram: int
    ) -> torch.Tensor:
        """Compute N-gram hash IDs using polynomial rolling hash."""
        ngram_ids = context.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_first_pp_rank:
            raise RuntimeError(
                "NgramEmbedding should not be called on non-first PP rank. "
                "inputs_embeds should be passed via intermediate_tensors."
            )

        seq_len = input_ids.numel()

        # Determine complete context
        if ngram_context is not None:
            context = torch.cat(
                [ngram_context[..., -(self.emb_neighbor_num - 1) :], input_ids], dim=-1
            )
        else:
            context = input_ids.clone()

        # Skip N-gram look-up for oe_ignored_token_ids.
        # Use isin_list (which uses async_tensor_h2d) instead of
        # torch.tensor(..., device=device) to avoid CPU→GPU transfers that
        # are unsafe inside CUDA graph capture regions.
        oe_ignored_mask = isin_list(input_ids,
                                    self.config.oe_ignored_token_ids)
        context_ignored_mask = isin_list(context,
                                         self.config.oe_ignored_token_ids)
        context = context.clone()
        context[context_ignored_mask] = 0

        # Base word embeddings
        base_embeds = self.embed_tokens(input_ids)
        x = base_embeds.clone()

        # If context is too short for N-gram computation, just apply scaling
        if context.numel() < 2:
            x[~oe_ignored_mask] /= 1 + self.emb_split_num * (self.emb_neighbor_num - 1)
            return x

        # Precompute modular values
        vocab_mods = self._precompute_vocab_mods()

        # Compute shifted IDs for each N-gram size
        shifted_ids = {}
        for i in range(2, self.emb_neighbor_num + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(
                context, i - 1, eos_token_id=self.config.eos_token_id
            )

        # Add N-gram embeddings
        for i in range(2, self.emb_neighbor_num + 1):
            for j in range(self.emb_split_num):
                index = (i - 2) * self.emb_split_num + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[-seq_len:]
                text_mask = new_ids > 0

                # Look up in embedder
                embedder = self.embedders[index]
                x_ngram = embedder(new_ids)

                # Zero out masked positions
                x_ngram = x_ngram * text_mask.unsqueeze(-1).to(x_ngram.dtype)

                # Project to hidden_size
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj.to(x.device)

        # Normalize
        x[~oe_ignored_mask] /= 1 + self.emb_split_num * (self.emb_neighbor_num - 1)

        return x


# =============================================================================
# Output Heads (text / visual / audio)
# =============================================================================


class LongcatNextHeadAttention(nn.Module):
    """Causal self-attention used by the depth-wise visual/audio heads.

    Operates on a fixed ``depth`` sequence (the number of residual-quantization
    codebooks). The checkpoint stores q/v/out projections with bias and k
    projection without bias, matching the HF reference.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = embed_dim // 128
        self.head_dim = 128

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch_size, depth, embed_dim]"""
        bsz, depth, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, depth, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, depth, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, depth, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, depth, self.embed_dim)
        )
        return self.out_proj(attn_output)


class LongcatNextHeadTransformerLayer(nn.Module):
    """Single depth-wise transformer layer for visual/audio heads.

    Matches the HF ``CasualDepthTransformerLayer`` layout and weight shapes:
    - RMSNorm -> causal self-attention -> residual
    - RMSNorm -> depth-wise FFN (weights reshaped and contracted over depth)
      -> residual
    """

    def __init__(
        self, transformer_dim: int, transformer_ffn_scale: int, depth: int
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer_ffn_scale = transformer_ffn_scale
        self.depth = depth

        assert transformer_dim % depth == 0
        self.self_attention = LongcatNextHeadAttention(transformer_dim)
        self.layernorm1 = RMSNorm(transformer_dim, eps=1e-6)
        self.layernorm2 = RMSNorm(transformer_dim, eps=1e-6)
        self.linear1 = nn.Linear(
            transformer_dim, transformer_ffn_scale * transformer_dim
        )
        self.linear2 = nn.Linear(
            transformer_ffn_scale * transformer_dim, transformer_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch_size, depth, transformer_dim]"""
        res = x
        x = self.layernorm1(x)
        x = self.self_attention(x)
        _res = x + res

        res = self.layernorm2(_res)
        # Depth-wise FFN: linear1 weight is reshaped to
        # (ffn_scale * dim / depth, depth, dim)
        w1 = self.linear1.weight.view(
            self.transformer_ffn_scale * self.transformer_dim // self.depth,
            self.depth,
            self.transformer_dim,
        )
        x = torch.einsum("bld,tld->blt", res, w1)
        x = nn.functional.gelu(x)
        w2 = self.linear2.weight.view(
            self.transformer_dim,
            self.depth,
            self.transformer_ffn_scale * self.transformer_dim // self.depth,
        )
        x = torch.einsum("blt,dlt->bld", x, w2)
        return _res + x


class LongcatNextTransformerHead(nn.Module):
    """Bifurcated visual/audio head for LongCat-Next.

    Replaces the simple ``ParallelLMHead`` used for text tokens. The HF
    checkpoint stores a small causal depth-transformer followed by one
    classification head per residual-quantization codebook.
    """

    def __init__(
        self,
        hidden_size: int,
        codebook_sizes: list[int],
        transformer_dim: int,
        transformer_layers: int,
        transformer_ffn_scale: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.codebook_sizes = codebook_sizes
        self.transformer_dim = transformer_dim
        self.transformer_ffn_scale = transformer_ffn_scale
        self.depth = len(codebook_sizes)

        if transformer_ffn_scale > 0:
            self.hidden_norm = RMSNorm(hidden_size, eps=1e-6)
            self.hidden_proj = nn.Linear(hidden_size, transformer_dim, bias=False)

        self.transformer_layers = nn.ModuleList(
            [
                LongcatNextHeadTransformerLayer(
                    transformer_dim=transformer_dim,
                    transformer_ffn_scale=transformer_ffn_scale,
                    depth=self.depth,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.headnorm = RMSNorm(transformer_dim, eps=1e-6)
        self.heads = nn.ModuleList(
            [nn.Linear(transformer_dim, size + 1) for size in codebook_sizes]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        """Simple forward used only for generation-mode logits.

        For text generation this head is not invoked by default; it is kept
        here so the checkpoint loads and can be wired later for image/audio
        generation.
        """
        hidden_states = hidden_states.view(-1, 1, self.hidden_size)
        hidden_states = hidden_states.expand(-1, self.depth, -1)

        if self.transformer_ffn_scale > 0:
            hidden_states = self.hidden_norm(hidden_states)
            hidden_states = self.hidden_proj(hidden_states)

        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.headnorm(hidden_states)
        return self.heads[level](hidden_states[:, level])


# =============================================================================
# Visual Tower
# =============================================================================


class LongcatNextVisualEncoder(nn.Module):
    """Visual encoder for LongCat-Next.

    Wraps vLLM's Qwen2_5_VisionTransformer but removes the merger,
    since LongCat-Next uses OmniVisualBridge instead.
    Returns hidden_states in window-indexed order.
    """

    def __init__(
        self,
        vision_config: Any,
        norm_eps: float = 1e-6,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer

        # Create the full vLLM vision transformer (includes merger)
        self._vision_transformer = Qwen2_5_VisionTransformer(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )

        # Remove the merger since LongCat-Next doesn't use it
        if hasattr(self._vision_transformer, "merger"):
            del self._vision_transformer.merger

        # Expose attributes for convenience
        self.patch_embed = self._vision_transformer.patch_embed
        self.blocks = self._vision_transformer.blocks
        self.hidden_size = self._vision_transformer.hidden_size
        self.num_heads = self._vision_transformer.num_heads
        self.spatial_merge_size = self._vision_transformer.spatial_merge_size
        self.spatial_merge_unit = self._vision_transformer.spatial_merge_unit
        self.fullatt_block_indexes = self._vision_transformer.fullatt_block_indexes
        self.window_size = self._vision_transformer.window_size
        self.patch_size = self._vision_transformer.patch_size
        self.dtype = self._vision_transformer.dtype
        self.device = self._vision_transformer.device

    def prepare_encoder_metadata(
        self, grid_thw: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        return self._vision_transformer.prepare_encoder_metadata(grid_thw)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]] | None = None,
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
        require_window_index: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        seq_len = hidden_states.shape[0]
        if encoder_metadata is None:
            if grid_thw is None:
                raise ValueError("Either grid_thw or encoder_metadata must be provided")
            encoder_metadata = self.prepare_encoder_metadata(grid_thw)

        rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
        rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]
        window_index = encoder_metadata["window_index"]
        cu_seqlens = encoder_metadata["cu_seqlens"]
        cu_window_seqlens = encoder_metadata["cu_window_seqlens"]
        max_seqlen_full = encoder_metadata["max_seqlen_full"]
        max_seqlen_window = encoder_metadata["max_seqlen_window"]
        sequence_lengths_full = encoder_metadata.get("sequence_lengths_full")
        sequence_lengths_window = encoder_metadata.get("sequence_lengths_window")

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
                sequence_lengths_now = sequence_lengths_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
                sequence_lengths_now = sequence_lengths_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen_now,
                sequence_lengths=sequence_lengths_now,
            )

        # Skip merger — squeeze the middle dim instead
        hidden_states = hidden_states.squeeze(1)

        if require_window_index:
            return hidden_states, window_index
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, bridging the `_vision_transformer` wrapper."""
        stripped_weights = [
            (name[len("_vision_transformer.") :], loaded_weight)
            if name.startswith("_vision_transformer.")
            else (name, loaded_weight)
            for name, loaded_weight in weights
        ]
        loaded = self._vision_transformer.load_weights(stripped_weights)
        return {f"_vision_transformer.{n}" for n in loaded}


class MLP(nn.Module):
    """Simple MLP with separate gate/up/down projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Single decoder layer used by the visual embedding bridge."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.visual_embedding_layer_intermediate_size,
            hidden_act=config.visual_embedding_layer_hidden_act,
        )
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class VQEmbedding(nn.Module):
    """VQ embedding module with EMA-style parameter layout.

    Adapted from HF modular_longcat_next_visual.py. The checkpoint stores an
    extra padding vector, so ``embed`` has shape ``(n_embed + 1, embed_dim)``;
    inference uses ``embed[:-1]`` as the actual codebook.
    """

    def __init__(self, n_embed: int, embed_dim: int) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        embed = torch.randn(n_embed + 1, embed_dim, dtype=torch.float32)
        nn.init.kaiming_uniform_(embed)
        self.embed = nn.Parameter(embed)
        # EMA buffers stored in the checkpoint for training; not used at inference.
        self.embed_ema = nn.Parameter(
            torch.randn(n_embed, embed_dim, dtype=torch.float32)
        )
        self.cluster_size_ema = nn.Parameter(torch.randn(n_embed, dtype=torch.float32))

    def compute_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        codebook = self.embed[:-1]
        codebook_t = codebook.t()
        inputs_shape = inputs.shape
        inputs_flat = inputs.reshape(-1, self.embed_dim)
        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_norm_sq = codebook.pow(2.0).sum(dim=1, keepdim=True).t()
        distances = torch.addmm(
            inputs_norm_sq + codebook_norm_sq, inputs_flat, codebook_t, alpha=-2.0
        )
        return distances.reshape(*inputs_shape[:-1], self.n_embed)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns quantized features and selected indices."""
        ori_dtype = inputs.dtype
        inputs = inputs.to(torch.float32)
        distances = self.compute_distances(inputs)
        indices = distances.argmin(dim=-1)
        quantize = self.embed[:-1][indices]
        return quantize.to(ori_dtype), indices


class RQBottleneck(nn.Module):
    """Residual quantization bottleneck."""

    def __init__(
        self,
        latent_shape: tuple[int, int, int],
        code_shape: tuple[int, int, int],
        n_embed: int,
        shared_codebook: bool = False,
    ) -> None:
        super().__init__()
        self.latent_shape = torch.Size(latent_shape)
        self.code_shape = torch.Size(code_shape)
        self.shape_divisor = torch.Size(
            [latent_shape[i] // code_shape[i] for i in range(3)]
        )
        self.shared_codebook = False

        n_embed_list = (
            n_embed
            if isinstance(n_embed, Iterable)
            else [n_embed] * self.code_shape[-1]
        )
        embed_dim = (
            np.prod(latent_shape[:2]) // np.prod(code_shape[:2]) * latent_shape[2]
        )

        if self.shared_codebook:
            codebook0 = VQEmbedding(n_embed_list[0], embed_dim)
            self.codebooks = nn.ModuleList(
                [codebook0 for _ in range(self.code_shape[-1])]
            )
        else:
            self.codebooks = nn.ModuleList(
                [VQEmbedding(size, embed_dim) for size in n_embed_list]
            )

    def to_code_shape(self, x: torch.Tensor) -> torch.Tensor:
        (B, H, W, D) = x.shape
        (rH, rW, _) = self.shape_divisor
        x = x.reshape(B, H // rH, rH, W // rW, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H // rH, W // rW, -1)
        return x

    def to_latent_shape(self, x: torch.Tensor) -> torch.Tensor:
        (B, h, w, _) = x.shape
        (_, _, D) = self.latent_shape
        (rH, rW, _) = self.shape_divisor
        x = x.reshape(B, h, w, rH, rW, D)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h * rH, w * rW, D)
        return x

    def quantize(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        residual_feature = x.detach().clone()
        quant_list: list[torch.Tensor] = []
        code_list: list[torch.Tensor] = []
        aggregated_quants = torch.zeros_like(x)
        for codebook in self.codebooks:
            quant, code = codebook(residual_feature)
            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)
            quant_list.append(aggregated_quants.clone().to(dtype=ori_dtype))
            code_list.append(code.unsqueeze(-1))
        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_reshaped = self.to_code_shape(x)
        quant_list, codes = self.quantize(x_reshaped)
        quants_trunc = self.to_latent_shape(quant_list[-1])
        quants_trunc = x + (quants_trunc - x).detach()
        return quants_trunc, codes


class OmniVisualBridge(nn.Module):
    """OmniVisualBridge for LongCat-Next.

    From HF modular_longcat_next_visual.py L391.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.merge_size = getattr(config, "merge_size", 2)
        self.hidden_size = config.hidden_size * (self.merge_size**2)

        from vllm.model_executor.layers.layernorm import RMSNorm

        self.ln_q = RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, window_index: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]
        return x


class VisualQuantizer(nn.Module):
    """Residual Quantization (RQ) quantizer for visual tokens.

    From HF modular_longcat_next_visual.py L413.
    """

    def __init__(self, quantizer_config: Any) -> None:
        super().__init__()
        self.config = quantizer_config
        self.depth = self.config.depth
        self.codebook_size = self.config.codebook_size
        self.codebook_dim = self.config.codebook_dim
        self.in_channels = getattr(self.config, "in_channels", self.codebook_dim)
        self.shared_codebook = getattr(self.config, "shared_codebook", False)
        self.quantizer_type = getattr(self.config, "quantizer_type", "rq")

        code_h_w = int(448 / 14)
        latent_shape = [code_h_w, code_h_w, self.codebook_dim]
        code_shape = [code_h_w, code_h_w, self.depth]

        self.quantize = RQBottleneck(
            latent_shape=latent_shape,
            code_shape=code_shape,
            n_embed=self.codebook_size,
            shared_codebook=self.shared_codebook,
        )

        if getattr(self.config, "quant_conv", False):
            self.quant_conv = nn.Sequential(
                nn.LayerNorm(self.in_channels),
                nn.Linear(self.in_channels, self.in_channels, bias=True),
                nn.GELU(),
                nn.Linear(self.in_channels, self.codebook_dim, bias=True),
            )
        else:
            self.quant_conv = None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L, D = x.shape
        to_qnt_feat = x.unsqueeze(0)  # [1, L, D]

        if self.quant_conv is not None:
            to_qnt_feat = self.quant_conv(to_qnt_feat)

        # NCHW format expected by RQBottleneck
        to_qnt_feat = to_qnt_feat.reshape(1, 1, L, self.codebook_dim).permute(
            0, 3, 1, 2
        )
        if self.quantizer_type == "rq":
            to_qnt_feat = to_qnt_feat.permute(0, 2, 3, 1).contiguous()
            quant, codes = self.quantize(to_qnt_feat)
            codes = codes.reshape(-1, codes.shape[-1])
        else:
            quant, codes = self.quantize(to_qnt_feat)
        return quant, codes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, codes = self.encode(x)
        return codes


class VisualEmbeddingBridge(nn.Module):
    """VisualEmbeddingBridge for LongCat-Next.

    From HF modular_longcat_next_visual.py L519.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.pre_buffer = DecoderLayer(config)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.pre_buffer(embedding)


class VisualVQBridge(nn.Module):
    """Combines OmniVisualBridge and VisualQuantizer.

    From HF modular_longcat_next_visual.py L528.
    """

    def __init__(self, visual_config: Any) -> None:
        super().__init__()
        self.bridge = OmniVisualBridge(visual_config)
        self.quantizer = VisualQuantizer(visual_config.vq_config)

    def forward(
        self, visual_embed: torch.Tensor, window_index: torch.Tensor
    ) -> torch.Tensor:
        visual_embed = self.bridge(visual_embed, window_index)
        indices = self.quantizer(visual_embed)
        return indices


class LongcatNextVisualTokenizer(nn.Module):
    """Complete visual tokenizer for LongCat-Next.

    Combines: VisualEncoder -> VisualVQBridge -> VisualEmbeddingBridge
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # Visual encoder (Qwen2.5-VL style ViT without merger)
        self.visual_model = LongcatNextVisualEncoder(
            vision_config=config.visual_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual_model"),
        )

        # Bridge + quantizer
        self.visual_bridge_model = VisualVQBridge(config.visual_config)

        # Embedding bridge: refines embeddings after lookup
        self.visual_embedding_layer = VisualEmbeddingBridge(config)

    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images to discrete visual token indices.

        Args:
            pixel_values: [num_patches, patch_dim] or [num_patches * C, H, W]
            grid_thw: [num_images, 3] (t, h, w)

        Returns:
            Visual token indices of shape [num_tokens, depth]
        """
        visual_embed, window_index = self.visual_model(
            pixel_values,
            grid_thw=grid_thw.tolist(),
            require_window_index=True,
        )
        indices = self.visual_bridge_model(visual_embed, window_index)
        return indices

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for visual tokenizer."""
        loaded_params: set[str] = set()

        model_weights: dict[str, list[tuple[str, torch.Tensor]]] = {
            "visual_model": [],
            "visual_bridge_model": [],
            "visual_embedding_layer": [],
        }

        for name, loaded_weight in weights:
            if name.startswith("visual_model."):
                model_weights["visual_model"].append(
                    (name[len("visual_model.") :], loaded_weight)
                )
            elif name.startswith("visual_bridge_model."):
                model_weights["visual_bridge_model"].append(
                    (name[len("visual_bridge_model.") :], loaded_weight)
                )
            elif name.startswith("visual_embedding_layer."):
                model_weights["visual_embedding_layer"].append(
                    (name[len("visual_embedding_layer.") :], loaded_weight)
                )

        if model_weights["visual_model"]:
            loaded = self.visual_model.load_weights(model_weights["visual_model"])
            loaded_params.update(f"visual_model.{n}" for n in loaded)

        for key in ["visual_bridge_model", "visual_embedding_layer"]:
            if model_weights[key]:
                submodule = getattr(self, key)
                loader = AutoWeightsLoader(submodule)
                loaded = loader.load_weights(model_weights[key])
                loaded_params.update(f"{key}.{n}" for n in loaded)

        return loaded_params


class LongcatNextAudioEncoderAttention(nn.Module):
    """Multi-head self-attention matching the Whisper checkpoint layout."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        return attn_output


class LongcatNextAudioEncoderLayer(nn.Module):
    """Single Whisper-style encoder layer."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = LongcatNextAudioEncoderAttention(
            config.d_model, config.encoder_attention_heads
        )
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(nn.functional.gelu(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


class LongcatNextAudioEncoder(nn.Module):
    """Audio encoder for LongCat-Next (Whisper-style with sinusoidal positions).

    Matches the HF modular_longcat_next_audio.py encoder layout:
    conv1 -> conv2 -> positional_embedding -> Transformer layers -> layer_norm.
    """

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_mel_bins = getattr(config, "num_mel_bins", 128)

        self.conv1 = nn.Conv1d(
            self.num_mel_bins, self.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=3, stride=2, padding=1
        )

        max_source_positions = getattr(config, "max_source_positions", 1500)
        self.register_buffer(
            "positional_embedding",
            self._sinusoids(max_source_positions, self.d_model),
        )

        self.layers = nn.ModuleList(
            [
                LongcatNextAudioEncoderLayer(config)
                for _ in range(getattr(config, "encoder_layers", 32))
            ]
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    @staticmethod
    def _sinusoids(
        length: int, channels: int, max_timescale: float = 10000.0
    ) -> torch.Tensor:
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2)
        )
        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(
        self,
        input_features: torch.Tensor,
        output_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode audio features.

        Args:
            input_features: [bs, num_mel_bins, frames]
            output_length: [bs] or None

        Returns:
            Hidden states of shape [bs, frames', d_model]
        """
        # Align input dtype with the convolution weights (fp16/bf16).
        input_features = input_features.to(dtype=self.conv1.weight.dtype)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # [bs, frames', d_model]

        bsz, tgt_len, _ = inputs_embeds.size()
        if tgt_len < self.positional_embedding.shape[0]:
            pos_emb = self.positional_embedding[:tgt_len]
        else:
            pos_emb = self.positional_embedding
        hidden_states = (inputs_embeds.float() + pos_emb).to(inputs_embeds.dtype)

        attention_mask = None
        if output_length is not None:
            mask = (
                torch.arange(tgt_len, device=hidden_states.device)[None, :]
                < output_length[:, None]
            )
            attention_mask = (~mask).to(hidden_states.dtype) * -1e9
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.layer_norm(hidden_states)

        if output_length is not None:
            mask = (
                torch.arange(tgt_len, device=hidden_states.device)[None, :, None]
                < output_length[:, None, None]
            )
            hidden_states = torch.where(mask, hidden_states, 0)

        return hidden_states


class AudioEuclideanCodebook(nn.Module):
    """Euclidean codebook for audio VQ with EMA parameters."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        embed = torch.randn(codebook_size, dim)
        nn.init.kaiming_uniform_(embed)
        self.embed = nn.Parameter(embed)
        # EMA parameters are stored in the checkpoint but unused at inference.
        self.embed_avg = nn.Parameter(embed.clone())
        self.cluster_size = nn.Parameter(torch.zeros(codebook_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize x using nearest neighbor in codebook.

        Args:
            x: [N, dim] float32

        Returns:
            quantize: [N, dim]
            indices: [N]
        """
        # Match the embedding dtype when the model runs in fp16/bf16.
        x = x.to(self.embed.dtype)
        x_norm = (x**2).sum(dim=1, keepdim=True)  # [N, 1]
        e_norm = (self.embed**2).sum(dim=1, keepdim=True).t()  # [1, codebook_size]
        distances = x_norm + e_norm - 2.0 * torch.mm(x, self.embed.t())
        indices = distances.argmin(dim=-1)  # [N]
        quantize = self.embed[indices]  # [N, dim]
        return quantize, indices


class AudioVectorQuantize(nn.Module):
    """Vector quantize layer for audio."""

    def __init__(self, dim: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook = AudioEuclideanCodebook(dim, codebook_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize x.

        Args:
            x: [N, dim]

        Returns:
            quantize: [N, dim]
            indices: [N]
        """
        orig_dtype = x.dtype
        x = x.to(self.codebook.embed.dtype)
        quantize, indices = self.codebook(x)
        return quantize.to(orig_dtype), indices

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook.embed[indices]


class LongcatNextAudioVQBridger(nn.Module):
    """Audio VQ bridger for LongCat-Next.

    From HF modular_longcat_next_audio.py L1671.
    """

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.avg_pooler = getattr(config, "avg_pooler", 4)
        self.d_model = config.d_model
        self.intermediate_dim = self.d_model * self.avg_pooler

        self.gate_proj = nn.Conv1d(
            self.d_model,
            self.intermediate_dim,
            self.avg_pooler,
            stride=self.avg_pooler,
            bias=False,
        )
        self.up_proj = nn.Conv1d(
            self.d_model,
            self.intermediate_dim,
            self.avg_pooler,
            stride=self.avg_pooler,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_dim, self.intermediate_dim, bias=False
        )
        self.act_fn = nn.SiLU()
        self.layer_norm = nn.LayerNorm(self.intermediate_dim)
        self.proj_decoder = nn.Linear(self.intermediate_dim, self.d_model, bias=True)

        # RVQ layers
        self.vq_list = nn.ModuleList()
        codebook_sizes = config.vq_config.codebook_sizes
        for codebook_size in codebook_sizes:
            self.vq_list.append(
                AudioVectorQuantize(self.intermediate_dim, codebook_size)
            )

    def forward(
        self, x: torch.Tensor, output_length: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode audio embeddings to discrete tokens.

        Args:
            x: [bs, seq_len, d_model]
            output_length: [bs] or None

        Returns:
            Audio token indices of shape [num_valid_tokens, depth]
        """
        batch_size, seq_len, _ = x.shape

        # Pad to multiple of avg_pooler
        pad_len = (self.avg_pooler - seq_len % self.avg_pooler) % self.avg_pooler
        if pad_len > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_len), "constant", 0)

        xt = x.permute(0, 2, 1)  # [bs, d_model, seq_len]
        g = self.gate_proj(xt).permute(0, 2, 1)  # [bs, seq_len//pooler, d*pooler]
        u = self.up_proj(xt).permute(0, 2, 1)

        # Reshape x to match pooled length
        x_pooled = x.reshape(batch_size, -1, self.intermediate_dim)

        c = self.down_proj(self.act_fn(g) * u)
        res = self.layer_norm(c + x_pooled)

        # Compute valid mask after pooling. `output_length` is the number of
        # valid tokens after the avg_pooler downsampling (the HF processor calls
        # this `bridge_length`). Use it directly, do not divide by avg_pooler.
        if output_length is not None:
            valid_mask = (
                torch.arange(res.shape[1], device=res.device)[None, :]
                < output_length[:, None]
            )
        else:
            valid_mask = torch.ones(res.shape[:2], dtype=torch.bool, device=res.device)

        # RVQ
        residual = res
        code_list = []
        for vq in self.vq_list:
            residual_flat = residual.reshape(-1, self.intermediate_dim)
            valid_flat = valid_mask.reshape(-1)
            residual_valid = residual_flat[valid_flat]

            _, indices = vq(residual_valid)
            quantized = vq.get_output_from_indices(indices)

            residual_flat_valid = residual_flat[valid_flat]
            residual_flat_valid = residual_flat_valid.float() - quantized.float()
            residual_flat[valid_flat] = residual_flat_valid.to(residual.dtype)
            residual = residual_flat.reshape_as(residual)

            full_indices = torch.full(
                (residual_flat.shape[0],), -1, dtype=torch.long, device=res.device
            )
            full_indices[valid_flat] = indices
            code_list.append(full_indices)

        # Stack codes and filter valid
        codes = torch.stack(code_list, dim=-1)  # [bs*seq_pooled, depth]
        codes = codes[valid_flat]  # [num_valid, depth]
        return codes

    def decode(self, code_ids: torch.Tensor) -> torch.Tensor:
        """Decode audio token IDs to embeddings.

        Args:
            code_ids: [N, depth]

        Returns:
            Embeddings of shape [N, d_model]
        """
        vq_num = code_ids.shape[-1]
        res = sum(
            self.vq_list[i].get_output_from_indices(code_ids[:, i]).float()
            for i in range(vq_num - 1, -1, -1)
        )
        decoder_emb = self.proj_decoder(res.to(self.proj_decoder.weight.dtype))
        return decoder_emb


class LongcatNextAudioTokenizer(nn.Module):
    """Complete audio tokenizer for LongCat-Next.

    Combines: AudioEncoder -> AudioVQBridger
    """

    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.audio_model = LongcatNextAudioEncoder(
            config.audio_config,
            prefix=maybe_prefix(prefix, "audio_model"),
        )
        self.audio_bridge_model = LongcatNextAudioVQBridger(
            config.audio_config,
            prefix=maybe_prefix(prefix, "audio_bridge_model"),
        )

    def encode(
        self,
        input_features: torch.Tensor,
        encoder_length: torch.Tensor | None = None,
        bridge_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode audio to discrete token indices.

        Args:
            input_features: [bs, num_mel_bins, frames]
            encoder_length: [bs] or None
            bridge_length: [bs] or None

        Returns:
            Audio token indices of shape [num_tokens, depth]
        """
        audio_emb = self.audio_model(input_features, output_length=encoder_length)
        audio_tokens = self.audio_bridge_model(audio_emb, output_length=bridge_length)
        return audio_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for audio tokenizer."""
        loaded_params: set[str] = set()
        model_weights: dict[str, list[tuple[str, torch.Tensor]]] = {
            "audio_model": [],
            "audio_bridge_model": [],
        }
        for name, loaded_weight in weights:
            if name.startswith("audio_model."):
                model_weights["audio_model"].append(
                    (name[len("audio_model.") :], loaded_weight)
                )
            elif name.startswith("audio_bridge_model."):
                model_weights["audio_bridge_model"].append(
                    (name[len("audio_bridge_model.") :], loaded_weight)
                )
            # audio_decoder / audio_flow_matching_decoder weights are loaded by
            # the HF generation pipeline, not by vLLM's prefill tower.

        for key in ["audio_model", "audio_bridge_model"]:
            if model_weights[key]:
                submodule = getattr(self, key)
                loader = AutoWeightsLoader(submodule)
                loaded = loader.load_weights(model_weights[key])
                loaded_params.update(f"{key}.{n}" for n in loaded)

        return loaded_params


class LongcatNextProcessingInfo(BaseProcessingInfo):
    """Processing info for LongCat-Next."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Video uses the same visual tower as image; both are supported.
        return {"audio": None, "image": None, "video": None}

    def get_data_parser(self) -> MultiModalDataParser:
        """Return a data parser that resamples audio to the model's rate."""
        hf_config = self.get_hf_config()
        sampling_rate = getattr(
            getattr(hf_config, "audio_config", None), "sampling_rate", 16000
        )
        return MultiModalDataParser(
            target_sr=float(sampling_rate),
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class LongcatNextDummyInputsBuilder(BaseDummyInputsBuilder[LongcatNextProcessingInfo]):
    """Dummy inputs builder for LongCat-Next."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        parts = []
        for _ in range(mm_counts.get("image", 0)):
            parts.append("<longcat_img_start><longcat_img_pad><longcat_img_end>")
        for _ in range(mm_counts.get("video", 0)):
            # LongCat-Next does not have separate video tokens; reuse image tokens
            parts.append("<longcat_img_start><longcat_img_pad><longcat_img_end>")
        for _ in range(mm_counts.get("audio", 0)):
            parts.append("<longcat_audio_start><longcat_audio_pad><longcat_audio_end>")
        return "\n".join(parts)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        """Return dummy multimodal data for profiling."""
        hf_config = self.info.get_hf_config()

        # Image dummy data
        image_size = getattr(hf_config.visual_config, "image_size", 448)
        num_images = mm_counts.get("image", 0)
        dummy_images = self._get_dummy_images(
            width=image_size,
            height=image_size,
            num_images=num_images,
            overrides=mm_options.get("image"),
        )

        # Video dummy data
        num_videos = mm_counts.get("video", 0)
        dummy_videos = self._get_dummy_videos(
            width=image_size,
            height=image_size,
            num_frames=4,
            num_videos=num_videos,
            overrides=mm_options.get("video"),
        )

        # Audio dummy data
        # LongCat-Next audio: max_audio_seconds * sampling_rate samples
        max_audio_seconds = getattr(hf_config.audio_config, "max_audio_seconds", 30)
        sampling_rate = getattr(hf_config.audio_config, "sampling_rate", 16000)
        max_audio_length = max_audio_seconds * sampling_rate
        num_audios = mm_counts.get("audio", 0)
        dummy_audios = self._get_dummy_audios(
            length=max_audio_length,
            num_audios=num_audios,
            overrides=mm_options.get("audio"),
        )

        mm_data: MultiModalDataDict = {}
        if dummy_images:
            mm_data["image"] = dummy_images
        if dummy_videos:
            mm_data["video"] = dummy_videos
        if dummy_audios:
            mm_data["audio"] = dummy_audios

        return mm_data


class LongcatNextMultiModalProcessor(
    BaseMultiModalProcessor[LongcatNextProcessingInfo]
):
    """Multimodal processor for LongCat-Next.

    Handles:
    - Tuple return from HF processor (text, image, audio BatchFeatures)
    - Audio mel-spectrogram padding and stacking
    - Prompt replacement for image/video/audio placeholders
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Call HF processor and merge tuple output into single BatchFeature.

        LongCat-Next's HF processor returns a tuple of 3 BatchFeatures:
        (text_inputs, image_inputs, audio_inputs).
        We merge them into one BatchFeature for vLLM's pipeline.

        Additionally, we preserve per-file audio segment counts so that
        prompt replacements can correctly compute the number of audio
        tokens per file.
        """
        import regex as re

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        # The HF processor returns a tuple of 3 BatchFeatures when
        # the underlying processor is LongcatNextProcessor.
        # Handle both tuple and single BatchFeature cases.
        if isinstance(hf_inputs, tuple):
            text_batch, image_batch, audio_batch = hf_inputs
            merged = BatchFeature(data=dict(text_batch))
            if image_batch is not None:
                merged.update(image_batch)
            if audio_batch is not None:
                merged.update(audio_batch)
            hf_inputs = merged

        # Rename visual_grid_thw -> image_grid_thw for vLLM compatibility
        if "visual_grid_thw" in hf_inputs:
            hf_inputs["image_grid_thw"] = hf_inputs.pop("visual_grid_thw")
        if "visual_pixel_values" in hf_inputs:
            hf_inputs["pixel_values"] = hf_inputs.pop("visual_pixel_values")

        # Pad and stack audio mel-spectrograms if needed
        audio_data = hf_inputs.get("audio")
        if audio_data is not None and isinstance(audio_data, list):
            # audio_data is a list of numpy arrays with different lengths
            max_len = max(a.shape[-1] for a in audio_data)
            padded = []
            for arr in audio_data:
                pad_width = max_len - arr.shape[-1]
                if pad_width > 0:
                    arr = np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
                padded.append(arr)
            hf_inputs["audio"] = np.stack(padded)

        # Compute and store per-file audio segment counts for prompt replacement.
        # The HF processor flattens audio segments, so we need to reconstruct
        # the per-file grouping to know how many tokens each audio file contributes.
        audios = mm_data.get("audios", [])
        if audios:
            if not isinstance(audios, list):
                audios = [audios]
            processor = self.info.get_hf_processor(**mm_kwargs)
            # Only count segments for files that were actually in the prompt
            audio_paths = re.findall(
                rf"{processor.audio_start_token}(.*?){processor.audio_end_token}",
                prompt,
            )
            num_segments = []
            for path in audio_paths:
                _, _, bridge_lengths = processor.audio_processor.process(path)
                num_segments.append(len(bridge_lengths))
            hf_inputs["audio_num_segments"] = torch.tensor(num_segments)

        return hf_inputs

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        """Apply the HF processor on the prompt text and multi-modal data.

        LongCat-Next's HF processor extracts image/audio file paths directly
        from between the start/end special tokens in the prompt text. This
        conflicts with vLLM's placeholder-token flow (where the prompt contains
        pad placeholders and the actual multi-modal data is passed separately).
        We therefore bypass the HF processor's prompt-update path and rely on
        vLLM's prompt replacements instead.
        """
        return super()._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=False,
        )

    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        """Tokenize prompt text without invoking the HF processor.

        LongCat-Next's HF processor eagerly searches for image/audio paths in
        the text and fails when no multi-modal data is provided (e.g., during
        dummy-input profiling). Bypass it for text-only tokenization.
        """
        tokenizer = self.info.get_tokenizer()
        encoded = tokenizer.encode(prompt_text, **tokenization_kwargs)
        if not isinstance(encoded, list):
            encoded = encoded.tolist()
        return encoded

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process multi-modal data without generating dummy text.

        The base implementation falls back to `_apply_hf_processor_text_mm`
        when `_call_hf_processor` is overridden, which causes LongCat-Next's
        HF processor to misinterpret placeholder tokens as file paths. Instead,
        we call the HF processor's image/audio/video processors directly on
        the multi-modal data.
        """
        valid_mm_items = mm_items.select(
            {k for k, c in mm_items.get_all_counts().items() if c > 0}
        )

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        output_kwargs = processor._merge_kwargs(
            get_processor_kwargs_type(processor),
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            **hf_processor_mm_kwargs,
            **tokenization_kwargs,
        )

        merged_data: dict[str, Any] = {}

        if "image" in valid_mm_items:
            images = valid_mm_items["image"].get_all()
            images_inputs = processor.image_processor(
                images, **output_kwargs["images_kwargs"]
            )
            merged_data.update(images_inputs)

        if "video" in valid_mm_items:
            videos = valid_mm_items["video"].get_all()
            # LongCat-Next's HF processor treats video same as image
            videos_inputs = processor.image_processor(
                videos, **output_kwargs["images_kwargs"]
            )
            # Rename video-specific keys to match what the model expects
            if "pixel_values" in videos_inputs:
                videos_inputs["pixel_values_videos"] = videos_inputs.pop("pixel_values")
            if "image_grid_thw" in videos_inputs:
                videos_inputs["video_grid_thw"] = videos_inputs.pop("image_grid_thw")
            merged_data.update(videos_inputs)

        if "audio" in valid_mm_items:
            audio_items = valid_mm_items["audio"].get_all()
            # LongCat-Next's audio processor only accepts file paths, but vLLM
            # passes audio as numpy arrays. Write each array to a temporary WAV
            # file so the HF processor can load it.
            audio_paths: list[str] = []
            sampling_rate = getattr(
                self.info.get_hf_config().audio_config, "sampling_rate", 16000
            )
            for audio in audio_items:
                if audio is None:
                    continue
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                audio = np.asarray(audio, dtype=np.float32)
                fd, path = tempfile.mkstemp(suffix=".wav")
                try:
                    os.close(fd)
                    wavfile.write(path, sampling_rate, audio)
                    audio_paths.append(path)
                except Exception:
                    os.close(fd)
                    raise

            if audio_paths:
                audio_inputs = processor.audio_processor(
                    audio_paths, **output_kwargs["audio_kwargs"]
                )
                # HF processor flattens audio data across files
                for key in list(audio_inputs.keys()):
                    val = audio_inputs[key]
                    if isinstance(val, list):
                        audio_inputs[key] = [v for b_val in val for v in b_val]
                merged_data.update(audio_inputs)

        return BatchFeature(
            data=merged_data,
            tensor_type=tokenization_kwargs.get("return_tensors", "pt"),
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Map HF output fields to MultiModalFieldConfig."""
        # Image fields
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_pixel_grid_sizes = image_grid_thw.prod(-1)

        # Audio fields
        audio = hf_inputs.get("audio")

        fields: dict[str, MultiModalFieldConfig] = {}

        if "pixel_values" in hf_inputs:
            fields["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image", image_pixel_grid_sizes
            )
        if "image_grid_thw" in hf_inputs:
            fields["image_grid_thw"] = MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            )

        # Video fields
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_pixel_grid_sizes = video_grid_thw.prod(-1)

        if "pixel_values_videos" in hf_inputs:
            fields["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video", video_pixel_grid_sizes
            )
        if "video_grid_thw" in hf_inputs:
            fields["video_grid_thw"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )

        if audio is not None:
            fields["audio"] = MultiModalFieldConfig.batched("audio")
        if "encoder_length" in hf_inputs:
            fields["encoder_length"] = MultiModalFieldConfig.batched("audio")
        if "bridge_length" in hf_inputs:
            fields["bridge_length"] = MultiModalFieldConfig.batched("audio")
        if "audio_num_segments" in hf_inputs:
            fields["audio_num_segments"] = MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=True
            )

        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Return prompt replacements for image/video/audio placeholders."""
        hf_config = self.info.get_hf_config()

        # Get token IDs from config
        image_start_id = getattr(
            hf_config.visual_config, "image_start_token_id", 131106
        )
        image_end_id = getattr(hf_config.visual_config, "image_end_token_id", 131107)
        image_pad_id = getattr(hf_config.visual_config, "image_pad_token_id", 131108)
        audio_start_id = getattr(hf_config.audio_config, "audio_start_token_id", 131103)
        audio_end_id = getattr(hf_config.audio_config, "audio_end_token_id", 131104)
        audio_pad_id = getattr(hf_config.audio_config, "audio_pad_token_id", 131105)

        spatial_merge_size = getattr(hf_config.visual_config, "spatial_merge_size", 2)
        merge_length = spatial_merge_size * spatial_merge_size

        out_mm_data = out_mm_kwargs.get_data()

        def get_image_replacement(item_idx: int):
            grid_thw = out_mm_data["image_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            n_tokens = int(grid_thw.prod()) // merge_length
            full = [image_start_id] + [image_pad_id] * n_tokens + [image_end_id]
            return PromptUpdateDetails.select_token_ids(full, [image_pad_id])

        def get_video_replacement(item_idx: int):
            # Video now has its own grid_thw field after renaming
            # in _apply_hf_processor_mm_only
            grid_thw = out_mm_data["video_grid_thw"][item_idx]
            assert isinstance(grid_thw, torch.Tensor)
            n_tokens = int(grid_thw.prod()) // merge_length
            full = [image_start_id] + [image_pad_id] * n_tokens + [image_end_id]
            return PromptUpdateDetails.select_token_ids(full, [image_pad_id])

        def get_audio_replacement(item_idx: int):
            bridge_lengths = out_mm_data["bridge_length"]
            audio_num_segments = out_mm_data.get("audio_num_segments")
            assert isinstance(bridge_lengths, torch.Tensor)

            if audio_num_segments is not None:
                # Reconstruct per-file bridge lengths from flattened data
                assert isinstance(audio_num_segments, torch.Tensor)
                segments = audio_num_segments.tolist()
                start_idx = sum(segments[:item_idx])
                end_idx = start_idx + segments[item_idx]
                file_bridge_lengths = bridge_lengths[start_idx:end_idx]
                n_tokens = int(file_bridge_lengths.sum().item())
            else:
                # Fallback: assume one segment per file
                n_tokens = int(bridge_lengths[item_idx].sum().item())

            full = [audio_start_id] + [audio_pad_id] * n_tokens + [audio_end_id]
            return PromptUpdateDetails.select_token_ids(full, [audio_pad_id])

        updates: list[PromptUpdate] = []

        if mm_items.get_count("image", strict=False) > 0:
            updates.append(
                PromptReplacement(
                    modality="image",
                    target=[image_pad_id],
                    replacement=get_image_replacement,
                )
            )

        if mm_items.get_count("video", strict=False) > 0:
            updates.append(
                PromptReplacement(
                    modality="video",
                    target=[image_pad_id],
                    replacement=get_video_replacement,
                )
            )

        if mm_items.get_count("audio", strict=False) > 0:
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[audio_pad_id],
                    replacement=get_audio_replacement,
                )
            )

        return updates


@MULTIMODAL_REGISTRY.register_processor(
    LongcatNextMultiModalProcessor,
    info=LongcatNextProcessingInfo,
    dummy_inputs=LongcatNextDummyInputsBuilder,
)
class LongcatNextForCausalLM(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    SupportsLoRA,
    SupportsTranscription,
):
    """LongCat-Next unified multimodal model.

    Extends longcat_flash.py's backbone with:
    - N-gram embedding
    - Visual/audio towers
    - Bifurcated generation heads (text/visual/audio)
    """

    # Enable batch-level data parallelism for multimodal encoders.
    # This allows the vision/audio towers to replicate weights across TP ranks
    # and shard input data instead, bypassing dimension divisibility constraints
    # (LongCat-Next's vision tower has dims like 2730, 21, 7 not divisible by 4/8).
    supports_encoder_tp_data = True

    # Supported languages for transcription (ISO 639-1 codes)
    # LongCat-Next is primarily trained on Chinese and English audio
    supported_languages: Mapping[str, str] = {
        "zh": "Chinese",
        "en": "English",
    }

    # Packed modules for fused QKV / gate-up projections.
    # CRITICAL: Explicitly define ONLY gate_up_proj here. Do NOT inherit
    # "qkv_proj" from LongcatFlashForCausalLM — that class has:
    #   "qkv_proj": ["q_proj", "k_proj", "v_proj"]
    # which is WRONG for MLA (MLA uses q_a_proj/q_b_proj/kv_a_proj, not
    # standard q/k/v projections). If this class ever inherits from
    # LongcatFlashForCausalLM, the "qkv_proj" entry MUST be removed.
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA embedding modules (required by SupportsLoRA)
    embedding_modules = {
        "language_model.embed_tokens": "input_embeddings",
        "text_head": "output_embeddings",
    }
    embedding_padding_modules = ["text_head"]

    # HF to vLLM weight name mapping
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # N-gram embeddings are at the top level, NOT inside FlashModel.
            "model.ngram_embeddings.": "ngram_embeddings.",
            # embed_tokens lives inside FlashModel.
            "model.embed_tokens.": "language_model.embed_tokens.",
            "model.layers.": "language_model.layers.",
            "model.norm.": "language_model.norm.",
            # Visual/audio tokenizers are not yet implemented; map them so they
            # can be loaded when towers are ported. For now they are ignored.
            "model.visual_tokenizer.": "visual.",
            "model.audio_tokenizer.": "audio_tower.",
            "lm_head.": "text_head.",
            "visual_head.": "visual_head.",
            "audio_head.": "audio_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<longcat_img_start><longcat_img_pad><longcat_img_end>"
        if modality.startswith("video"):
            # LongCat-Next does not have separate video tokens; reuse image tokens
            return "<longcat_img_start><longcat_img_pad><longcat_img_end>"
        if modality.startswith("audio"):
            return (
                f"Audio {i}: <longcat_audio_start>"
                "<longcat_audio_pad><longcat_audio_end>"
            )
        raise ValueError(f"Unsupported modality: {modality}")

    @classmethod
    def get_generation_prompt(
        cls,
        stt_params: SpeechToTextParams,
    ) -> PromptType:
        """Construct a transcription/translation prompt for LongCat-Next.

        Unlike Kimi-Audio's minimal prompt (just audio placeholder), this
        implementation follows the Qwen3-Omni pattern with explicit task
        instructions to steer the model toward transcription mode rather
        than audio description mode.

        The prompt includes:
        - Explicit "transcribe" or "translate" instruction
        - Language specification (e.g., "into Chinese")
        - Optional custom request prompt
        - Audio placeholder in LongCat-Next format
        """
        from vllm.tokenizers import cached_get_tokenizer

        audio = stt_params.audio
        model_config = stt_params.model_config
        task_type = stt_params.task_type
        language = stt_params.language
        to_language = stt_params.to_language
        request_prompt = stt_params.request_prompt

        # Validate task type
        if task_type not in ("transcribe", "translate"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Supported task types are 'transcribe' and 'translate'."
            )

        # Build instruction with explicit task description
        # Following Qwen3-Omni pattern: "Transcribe this audio into <language>"
        if task_type == "transcribe":
            instruction = "Transcribe this audio"
            if language and language in cls.supported_languages:
                instruction += f" into {cls.supported_languages[language]}"
        else:  # translate
            instruction = "Translate this audio"
            if language and language in cls.supported_languages:
                instruction += f" from {cls.supported_languages[language]}"
            if to_language and to_language in cls.supported_languages:
                instruction += f" into {cls.supported_languages[to_language]}"
            elif to_language is None:
                # Default to English for translation
                instruction += " into English"

        instruction += ". Provide only the transcribed/translated text."

        # Append custom request prompt if provided
        if request_prompt:
            instruction += f" {request_prompt}"

        # Construct prompt with LongCat-Next audio placeholder
        # Format: <longcat_audio_start><longcat_audio_pad><longcat_audio_end>
        audio_placeholder = (
            "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"
        )
        user_content = f"{audio_placeholder} {instruction}"

        # Use the model's chat template format with thinking DISABLED.
        # LongCat-Next (Qwen3-based) defaults to thinking mode.
        # Disable it with /think_off + empty think block to prevent
        # repetitive thinking loops on transcription requests.
        THINK_END = "</longcat_think>"
        prompt = (
            f" /think_off<longcat_think>\n\n</longcat_think>\n"
            f"<longcat_user>\n{user_content}</longcat_user>\n"
            f"<longcat_assistant>\n{THINK_END}\n"
        )

        # Return as TokensPrompt with audio data attached
        # CRITICAL: multi_modal_data must include the audio for the processor
        # to replace the placeholder tokens with actual audio embeddings.
        from vllm.inputs import TokensPrompt

        tokenizer = cached_get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        prompt_token_ids = tokenizer.encode(prompt)

        return TokensPrompt(
            prompt_token_ids=prompt_token_ids,
            multi_modal_data={"audio": audio},
        )

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: Literal["transcribe", "translate"],
    ) -> SpeechToTextConfig:
        """Get speech-to-text configuration for LongCat-Next.

        Returns a config with appropriate defaults for audio transcription.
        LongCat-Next can handle long audio clips without chunking.
        """
        return SpeechToTextConfig(
            # LongCat-Next can handle up to 30 seconds of audio
            # Disable chunking for now since the model handles long sequences
            max_audio_clip_s=None,
            sample_rate=16000,  # Standard 16kHz for speech
            min_energy_split_window_size=None,  # No chunking
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        """Estimate number of audio tokens for a given duration.

        LongCat-Next's audio pipeline:
        - Sample rate: 16kHz
        - Conv1: no downsampling (kernel=3, padding=1)
        - Conv2: 2x downsampling (stride=2)
        - VQ bridger: 4x downsampling (avg_pooler=4)
        - Total: ~8x downsampling → 2000 tokens/sec before VQ
        - VQ compression: further reduces to ~50-200 tokens/sec

        We use a conservative estimate of 100 tokens per second.
        """
        # Conservative estimate: 100 tokens per second of audio
        # This accounts for the VQ compression in the audio bridge
        tokens_per_second = 100
        return int(audio_duration_s * tokens_per_second)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        # Configure multimodal token handling
        # CRITICAL: Use text_vocab_size (131072), NOT vocab_size (282624).
        # All multimodal tokens (131103-131109, 131116, 131120-131124) are >= text_vocab_size, so
        # _has_oov_mm_tokens becomes True. This causes _embed_text_input_ids
        # to mask them to 0 BEFORE ngram embedding, matching the HF model
        # which explicitly zeroes them at line 153:
        #   input_ids[:, special_audio_mask | special_visual_mask] = 0
        # Without this, the multimodal token IDs pollute the n-gram hash
        # context for surrounding text tokens, degrading output quality.
        self.configure_mm_token_handling(
            vocab_size=config.text_vocab_size,
            mm_token_ids=[
                # Image tokens
                config.visual_config.image_start_token_id,  # 131106
                config.visual_config.image_end_token_id,  # 131107
                config.visual_config.image_pad_token_id,  # 131108
                config.visual_config.image_newline_token_id,  # 131109
                # Audio tokens (basic)
                config.audio_config.audio_start_token_id,  # 131103
                config.audio_config.audio_end_token_id,  # 131104
                config.audio_config.audio_pad_token_id,  # 131105
                config.audio_config.audio_delim_token_id,  # 131116
                # Audio-text tokens
                config.audio_config.audiotext_start_token_id,  # 131120
                config.audio_config.audiotext_end_token_id,  # 131121
                config.audio_config.audiotext_pad_token_id,  # 131122
                # Audio-generation tokens
                config.audio_config.audiogen_start_token_id,  # 131123
                config.audio_config.audiogen_end_token_id,  # 131124
            ],
        )

        # Language Model FIRST (so we can reuse its embed_tokens for NgramEmbedding)
        # Reuse FlashModel from longcat_flash.py directly.
        # NOTE: _mark_language_model and _mark_tower_model are defined in
        # vllm/model_executor/models/interfaces.py (lines 216 and 251) as part
        # of SupportsMultiModal. They are context managers that track which
        # child modules belong to the language model vs tower models.
        with self._mark_language_model(vllm_config):
            self.language_model = self._build_language_model(
                vllm_config, config, prefix
            )

        # Pipeline parallelism: delegate intermediate tensor factory to
        # backbone (FlashModel implements this via
        # make_empty_intermediate_tensors_factory).
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # N-gram Embedding — reuses FlashModel's embed_tokens, NO separate
        # embedding table. FlashModel already creates embed_tokens internally
        # (TP-sharded). Creating a separate self.embed_tokens would create a
        # SECOND, uninitialized table. On non-first PP ranks, embed_tokens is a
        # PPMissingLayer. NgramEmbedding only computes N-gram embeddings on the
        # first PP rank (where embed_tokens is real). On later ranks,
        # inputs_embeds is passed via intermediate_tensors, so NgramEmbedding is
        # never called, but the module must not crash during init.
        self.ngram_embeddings = NgramEmbedding(config, self.language_model.embed_tokens)

        # Visual / Audio towers are only needed on the first PP rank.
        # On other ranks the inputs are already embedded and forwarded as
        # hidden_states through the pipeline, so we skip the large encoders
        # to avoid duplicating them on every GPU.
        if get_pp_group().is_first_rank:
            # Visual Tower (image + video)
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = self._build_visual_tower(config, prefix)

            # Audio Tower
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = self._build_audio_tower(config, prefix)
        else:
            self.visual = None
            self.audio_tower = None

        # Output Heads — only instantiated on the last PP rank.
        # On intermediate PP ranks the heads are replaced by PPMissingLayer
        # to avoid duplicating large parameter matrices on every GPU.
        visual_codebook_sizes = list(config.visual_config.vq_config.codebook_sizes)
        audio_codebook_sizes = list(config.audio_config.vq_config.codebook_sizes)
        if get_pp_group().is_last_rank:
            # Text head: predicts text tokens + multimodal special tokens
            self.text_head = ParallelLMHead(
                config.text_vocab_plus_multimodal_special_token_size,  # 131125
                config.hidden_size,
            )
            # Visual head: small causal depth-transformer + per-codebook classifiers
            self.visual_head = LongcatNextTransformerHead(
                hidden_size=config.hidden_size,
                codebook_sizes=visual_codebook_sizes,
                transformer_dim=getattr(
                    config.visual_config, "image_head_transformer_dims", 2048
                ),
                transformer_layers=getattr(
                    config.visual_config, "image_head_transformer_layers", 4
                ),
                transformer_ffn_scale=getattr(
                    config.visual_config, "image_head_transformer_ffn_scale", 16
                ),
            )
            # Audio head: small causal depth-transformer + per-codebook classifiers
            self.audio_head = LongcatNextTransformerHead(
                hidden_size=config.hidden_size,
                codebook_sizes=audio_codebook_sizes,
                transformer_dim=getattr(
                    config.audio_config, "audio_head_transformer_dims", 3072
                ),
                transformer_layers=getattr(
                    config.audio_config, "audio_head_transformer_layers", 4
                ),
                transformer_ffn_scale=getattr(
                    config.audio_config, "audio_head_transformer_ffn_scale", 16
                ),
            )
        else:
            self.text_head = PPMissingLayer()
            self.visual_head = PPMissingLayer()
            self.audio_head = PPMissingLayer()

        # Visual/audio logits processors are kept for future generation-mode work.
        # Text logits use the standard ParallelLMHead + LogitsProcessor pattern.
        visual_vocab_size = config.visual_offset + sum(visual_codebook_sizes)
        audio_vocab_size = config.audio_offset + sum(audio_codebook_sizes)
        self.logits_processor = LogitsProcessor(
            config.text_vocab_plus_multimodal_special_token_size
        )
        self.visual_logits_processor = LogitsProcessor(
            config.visual_offset + visual_vocab_size
        )
        self.audio_logits_processor = LogitsProcessor(
            config.audio_offset + audio_vocab_size
        )

        # Special token IDs (from config)
        self.image_start_token_id = getattr(
            config.visual_config, "image_start_token_id", 131106
        )
        self.image_end_token_id = getattr(
            config.visual_config, "image_end_token_id", 131107
        )
        self.image_pad_token_id = getattr(
            config.visual_config, "image_pad_token_id", 131108
        )
        self.image_newline_token_id = getattr(
            config.visual_config, "image_newline_token_id", 131109
        )
        self.audio_start_token_id = getattr(
            config.audio_config, "audio_start_token_id", 131103
        )
        self.audio_end_token_id = getattr(
            config.audio_config, "audio_end_token_id", 131104
        )
        self.audio_pad_token_id = getattr(
            config.audio_config, "audio_pad_token_id", 131105
        )
        self.visual_offset = config.visual_offset  # 150581
        self.audio_offset = config.audio_offset  # 131125

        # CRITICAL: Audio and visual offsets are cumulative sums across
        # codebook levels, matching the HF model's _init_multimodal_constants:
        #   audio_offset_list = [audio_offset] + codebook_sizes[:-1]
        #   audio_offset_vals = cumsum(audio_offset_list)
        # Each codebook level occupies a distinct region of the embedding
        # table. Using a single scalar offset for all levels (as the original
        # vLLM implementation did) causes levels 1..N-1 to look up embeddings
        # at completely wrong positions, producing garbage multimodal features.
        audio_codebook_sizes = config.audio_config.vq_config.codebook_sizes
        audio_offset_list = [config.audio_offset] + audio_codebook_sizes[:-1]
        self.register_buffer(
            "audio_offset_vals",
            torch.tensor(audio_offset_list, dtype=torch.long).cumsum(dim=0),
            persistent=False,
        )

        visual_codebook_sizes = config.visual_config.vq_config.codebook_sizes
        visual_offset_list = [config.visual_offset] + visual_codebook_sizes[:-1]
        self.register_buffer(
            "visual_offset_vals",
            torch.tensor(visual_offset_list, dtype=torch.long).cumsum(dim=0),
            persistent=False,
        )

    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int, int, float]]:
        """Iterate over multimodal features and yield grid information.

        CRITICAL: This method is defined directly on Qwen2VLForConditionalGeneration
        (qwen2_vl.py lines 1202-1238) — it is NOT inherited from any base class.
        LongcatNextForCausalLM must implement its own version.

        LongCat-Next uses Qwen2.5-VL's vision config, so spatial_merge_size
        and tokens_per_second come from self.config.vision_config.
        """
        spatial_merge_size = getattr(self.config.visual_config, "spatial_merge_size", 2)
        tokens_per_second = getattr(self.config.visual_config, "tokens_per_second", 1.0)
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                yield offset, 1, h // spatial_merge_size, w // spatial_merge_size, 1.0
            elif mm_feature.modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                second_per_grid_ts = 1.0
                if mm_feature.data.get("second_per_grid_ts", None):
                    second_per_grid_ts = mm_feature.data[
                        "second_per_grid_ts"
                    ].data.item()
                t_factor = second_per_grid_ts * tokens_per_second
                yield (
                    offset,
                    t,
                    h // spatial_merge_size,
                    w // spatial_merge_size,
                    t_factor,
                )
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Compute M-RoPE input positions for Qwen2.5-VL style vision encoder.

        Reference: qwen2_vl.py lines 1240-1277, qwen2_5_omni_thinker.py lines 1320-1400+
        Returns 3D positions [3, num_tokens] with T/H/W axes.

        CRITICAL: This method is REQUIRED for SupportsMRoPE. Without it, position
        IDs for visual tokens will be computed incorrectly, breaking attention.
        """
        llm_pos_ids_list: list = []
        st = 0

        for (
            offset,
            llm_grid_t,
            llm_grid_h,
            llm_grid_w,
            t_factor,
        ) in self.iter_mm_grid_thw(mm_features):
            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

            grid_indices = np.indices((llm_grid_t, llm_grid_h, llm_grid_w))
            if t_factor != 1.0:
                grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)
            llm_pos_ids_list.append(grid_indices.reshape(3, -1) + text_len + st_idx)
            st = offset + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

        return torch.from_numpy(llm_positions), mrope_position_delta

    def get_mm_mapping(self) -> MultiModelKeys:
        """Return multimodal module prefix mapping.

        Required for multimodal models. Describes which module prefixes
        correspond to the language model, connector, and tower.
        Reference: qwen2_5_omni_thinker.py lines 1509-1517, gemma4_mm.py lines 1686-1696
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            # LongCat-Next's bridges are submodules INSIDE the towers, not
            # standalone top-level modules. Verified against HF source:
            #   - modular_longcat_next_visual.py L550
            #   - modular_longcat_next_audio.py L1979
            # In vLLM, towers are at self.visual and self.audio_tower, so
            # bridge paths are "visual.visual_bridge_model" and
            # "audio_tower.audio_bridge_model".
            connector=["visual.visual_bridge_model", "audio_tower.audio_bridge_model"],
            tower_model=["visual.", "audio_tower."],
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Delegate MoE expert mapping to the backbone.

        AutoWeightsLoader calls this to stack MoE expert weights.
        Without this, expert weight loading fails silently.
        Reference: qwen3_moe.py lines 774-775, longcat_flash.py lines 556-567
        """
        return self.language_model.get_expert_mapping()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply N-gram token embeddings to `input_ids`.

        Overrides the default SupportsMultiModal.embed_input_ids to use
        NgramEmbedding instead of the base FlashModel.embed_input_ids.
        """
        from .interfaces import _require_is_multimodal

        # Get text embeddings via N-gram embedding layer
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.ngram_embeddings,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        result = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

        return result

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Run visual/audio towers and return per-item embeddings.

        Called by vLLM's multimodal pipeline during prefill.
        Returns one 2-D tensor per multimodal item so that the sanity checks in
        `vllm.v1.worker.utils.sanity_check_mm_encoder_outputs` pass and the
        embeddings can be merged back into the token stream.
        Reference: qwen2_5_omni_thinker.py embed_multimodal pattern
        """
        # On non-first pipeline-parallel ranks the multimodal towers are not
        # instantiated (inputs are already embedded by the first rank). vLLM's
        # memory profiler still calls embed_multimodal on every rank, so we
        # return a dummy list of the expected length to keep the sanity checks
        # happy without duplicating the heavy encoders on every GPU.
        if not get_pp_group().is_first_rank:
            num_items = 0
            for key in ("audio", "pixel_values_videos"):
                value = kwargs.get(key)
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    num_items = value.shape[0]
                    break
            if num_items == 0:
                for key in ("image_grid_thw", "video_grid_thw"):
                    value = kwargs.get(key)
                    if isinstance(value, torch.Tensor) and value.ndim > 0:
                        num_items = value.shape[0]
                        break

            if num_items == 0:
                return []

            param = next(self.parameters())
            return [
                torch.zeros(
                    1,
                    self.config.hidden_size,
                    device=param.device,
                    dtype=param.dtype,
                )
                for _ in range(num_items)
            ]

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result must be a tuple/list of 2-D tensors, one per input item.
        # vLLM batches consecutive items of the same modality and then asserts
        # that the number of returned embeddings equals the number of input items.
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in mm_input_by_modality:
            mm_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(mm_input)
                multimodal_embeddings += tuple(image_embeddings)
            elif modality == "video":
                video_embeddings = self._process_video_input(mm_input)
                multimodal_embeddings += tuple(video_embeddings)
            elif modality == "audio":
                audio_embeddings = self._process_audio_input(mm_input)
                multimodal_embeddings += tuple(audio_embeddings)

        return multimodal_embeddings

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, dict[str, object]]:
        """Parse and validate multimodal inputs from the processor.

        Returns a dict mapping modality -> dict of input tensors.
        Reference: qwen2_5_omni_thinker.py lines 1104-1131
        """
        mm_input_by_modality: dict[str, dict[str, object]] = {}

        # Preserve the order of modalities from the kwargs order.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds", "image_grid_thw"):
                mm_input_by_modality.setdefault("image", {})[input_key] = kwargs[
                    input_key
                ]
            elif input_key in (
                "pixel_values_videos",
                "video_embeds",
                "video_grid_thw",
            ):
                mm_input_by_modality.setdefault("video", {})[input_key] = kwargs[
                    input_key
                ]
            elif input_key in ("audio", "encoder_length", "bridge_length"):
                mm_input_by_modality.setdefault("audio", {})[input_key] = kwargs[
                    input_key
                ]

        return mm_input_by_modality

    def _process_image_input(
        self, mm_input: dict[str, object]
    ) -> tuple[torch.Tensor, ...]:
        """Run visual tower on image input and split into per-image embeddings.

        Flow:
        1. Encode images via visual tokenizer -> discrete visual_ids
        2. Add visual_offset to map to vocab space
        3. Look up embeddings via embed_tokens
        4. Apply visual_embedding_layer
        5. Split concatenated embeddings for each image item
        """
        pixel_values = mm_input["pixel_values"]
        image_grid_thw = mm_input.get("image_grid_thw")

        if self.visual is None:
            raise RuntimeError("Visual tower is not initialized")
        assert isinstance(pixel_values, torch.Tensor)

        # Run visual encoder + bridge + quantizer
        visual_ids = self.visual.encode(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,  # type: ignore[arg-type]
        )

        # Map to vocab space using per-level cumulative offsets.
        # visual_ids shape: [L, num_levels]. visual_offset_vals: [num_levels].
        visual_ids = visual_ids + self.visual_offset_vals.to(visual_ids.device)

        # Look up embeddings: [L, depth] -> [L, depth, hidden] -> [L, hidden]
        visual_embeds = self.language_model.embed_tokens(visual_ids).sum(dim=1)

        # Apply embedding bridge
        visual_embeds = self.visual.visual_embedding_layer(visual_embeds)

        # Split concatenated embeddings for each image item.
        assert isinstance(image_grid_thw, torch.Tensor)
        merge_size = getattr(self.config.visual_config, "spatial_merge_size", 2)
        merge_length = merge_size * merge_size
        sizes = (image_grid_thw.prod(-1) // merge_length).tolist()
        return visual_embeds.split(sizes)

    def _process_video_input(
        self, mm_input: dict[str, object]
    ) -> tuple[torch.Tensor, ...]:
        """Run visual tower on video input and split into per-video embeddings.

        Video processing is similar to image processing but with temporal dim.
        """
        pixel_values = mm_input["pixel_values_videos"]
        video_grid_thw = mm_input.get("video_grid_thw")

        if self.visual is None:
            raise RuntimeError("Visual tower is not initialized")
        assert isinstance(pixel_values, torch.Tensor)

        # Run visual encoder + bridge + quantizer
        visual_ids = self.visual.encode(
            pixel_values=pixel_values,
            grid_thw=video_grid_thw,  # type: ignore[arg-type]
        )

        # Map to vocab space using per-level cumulative offsets.
        visual_ids = visual_ids + self.visual_offset_vals.to(visual_ids.device)

        # CRITICAL: Protect against OOV visual IDs (same as audio).
        vocab_size = self.language_model.embed_tokens.weight.shape[0]
        max_visual_id = int(visual_ids.max())
        if max_visual_id >= vocab_size:
            logger.warning(
                "[VISUAL_FIX] visual_ids contain OOV tokens: max=%d >= vocab_size=%d. "
                "Clamping to 0. This indicates a vocab_size mismatch in the model config.",
                max_visual_id,
                vocab_size,
            )
            visual_ids = visual_ids.clamp(max=vocab_size - 1)

        # Look up embeddings
        visual_embeds = self.language_model.embed_tokens(visual_ids).sum(dim=1)

        # Apply embedding bridge
        visual_embeds = self.visual.visual_embedding_layer(visual_embeds)

        # Split concatenated embeddings for each video item.
        assert isinstance(video_grid_thw, torch.Tensor)
        merge_size = getattr(self.config.visual_config, "spatial_merge_size", 2)
        merge_length = merge_size * merge_size
        sizes = (video_grid_thw.prod(-1) // merge_length).tolist()
        return visual_embeds.split(sizes)

    def _process_audio_input(
        self, mm_input: dict[str, object]
    ) -> tuple[torch.Tensor, ...]:
        """Run audio tower on audio input and split into per-audio embeddings.

        Flow:
        1. Encode audio via audio tokenizer -> discrete audio_ids
        2. Add audio_offset to map to vocab space
        3. Look up embeddings via embed_tokens
        4. Split concatenated embeddings for each audio item
        """
        input_audio_features = mm_input["audio"]
        audio_length = mm_input.get("encoder_length")
        bridge_length = mm_input.get("bridge_length")

        if self.audio_tower is None:
            raise RuntimeError("Audio tower is not initialized")
        assert isinstance(input_audio_features, torch.Tensor)

        logger.warning(
            "[AUDIO_DIAG] _process_audio_input: "
            "audio_features shape=%s dtype=%s min=%.4f max=%.4f "
            "encoder_length=%s bridge_length=%s",
            tuple(input_audio_features.shape),
            input_audio_features.dtype,
            float(input_audio_features.min())
            if input_audio_features.numel() > 0
            else float("nan"),
            float(input_audio_features.max())
            if input_audio_features.numel() > 0
            else float("nan"),
            tuple(audio_length.shape) if audio_length is not None else None,
            tuple(bridge_length.shape) if bridge_length is not None else None,
        )
        if audio_length is not None:
            logger.warning("[AUDIO_DIAG] encoder_length values: %s", audio_length)
        if bridge_length is not None:
            logger.warning("[AUDIO_DIAG] bridge_length values: %s", bridge_length)

        # Run audio encoder + bridge
        audio_ids = self.audio_tower.encode(
            input_features=input_audio_features,
            encoder_length=audio_length,  # type: ignore[arg-type]
            bridge_length=bridge_length,  # type: ignore[arg-type]
        )

        logger.warning(
            "[AUDIO_DIAG] after encode: audio_ids shape=%s dtype=%s "
            "unique_count=%d min=%d max=%d",
            tuple(audio_ids.shape),
            audio_ids.dtype,
            int(audio_ids.unique().numel()),
            int(audio_ids.min()) if audio_ids.numel() > 0 else -1,
            int(audio_ids.max()) if audio_ids.numel() > 0 else -1,
        )

        # Map to vocab space using per-level cumulative offsets.
        # audio_ids shape: [L, num_levels]. audio_offset_vals shape: [num_levels].
        # Each codebook level occupies a distinct embedding-table region, so
        # level k must be offset by cumsum(audio_offset_list)[:k+1].
        logger.warning(
            "[AUDIO_DIAG] audio_offset_vals: %s",
            self.audio_offset_vals.tolist()
            if self.audio_offset_vals is not None
            else None,
        )
        audio_ids = audio_ids + self.audio_offset_vals.to(audio_ids.device)

        vocab_size = self.language_model.embed_tokens.weight.shape[0]
        logger.warning(
            "[AUDIO_DIAG] after offset: audio_ids min=%d max=%d vocab_size=%d",
            int(audio_ids.min()),
            int(audio_ids.max()),
            vocab_size,
        )

        # CRITICAL: Protect against OOV audio IDs.
        # The audio encoder produces discrete token IDs per codebook level,
        # which are then offset to different regions of the embedding table.
        # If the config's vocab_size is smaller than the max offset ID,
        # the embedding lookup will fail (with TP=1) or silently produce
        # wrong embeddings (with TP=2 due to OOV masking in
        # VocabParallelEmbedding). Clamp OOV IDs to 0 to prevent crashes
        # and ensure deterministic behavior regardless of TP size.
        max_audio_id = int(audio_ids.max())
        if max_audio_id >= vocab_size:
            logger.warning(
                "[AUDIO_FIX] audio_ids contain OOV tokens: max=%d >= vocab_size=%d. "
                "Clamping to 0. This indicates a vocab_size mismatch in the model config.",
                max_audio_id,
                vocab_size,
            )
            audio_ids = audio_ids.clamp(max=vocab_size - 1)

        # Look up embeddings: [L, depth] -> [L, depth, hidden] -> [L, hidden]
        audio_embeds = self.language_model.embed_tokens(audio_ids).sum(dim=1)

        logger.warning(
            "[AUDIO_DIAG] audio_embeds shape=%s dtype=%s min=%.6f max=%.6f",
            tuple(audio_embeds.shape),
            audio_embeds.dtype,
            float(audio_embeds.min()),
            float(audio_embeds.max()),
        )

        # Split concatenated embeddings for each audio item.
        # `bridge_length` returned by the HF processor is already the number of
        # valid tokens after the bridge's convolutional pooler, so use it
        # directly without dividing by `avg_pooler` again.
        if bridge_length is not None:
            if bridge_length.ndim > 1:
                per_item_lengths = bridge_length.sum(dim=-1)
            else:
                per_item_lengths = bridge_length
            sizes = per_item_lengths.tolist()
        else:
            # Fallback: assume one segment per audio file.
            sizes = [audio_embeds.shape[0] // input_audio_features.shape[0]]
        return audio_embeds.split(sizes)

    def _build_language_model(self, vllm_config, config, prefix):
        """Build the language model backbone.

        CRITICAL: Use `FlashModel` (not `nn.ModuleList` of decoder layers).
        `FlashModel` is a complete backbone with:
        - `embed_tokens`: TP-sharded embedding (handled by PP rank)
        - `layers`: 14x `FlashDecoderLayer` (dual MLA + dual MLP + MoE)
        - `norm`: RMSNorm (handled by PP rank)
        - `make_empty_intermediate_tensors`: Required for `SupportsPP`
        - `embed_input_ids`: Required for `_mark_language_model` fallback
        - `forward`: PP-aware forward with `intermediate_tensors` support

        `FlashModel` internally handles TP/PP via `get_tp_group()` / `get_pp_group()`.
        """
        from vllm.model_executor.models.longcat_flash import FlashModel

        return FlashModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

    def _build_visual_tower(self, config, prefix):
        """Build the visual tower (image + video encoder).

        LongCat-Next's visual pipeline (from HF modular_longcat_next_visual.py):
        1. Qwen2_5_VisionTransformer: pixel_values (num_patches, 1176) -> (L, 1280)
           32 layers, hidden=1280, patch_size=14, spatial_merge_size=2
        2. OmniVisualBridge: (L, 1280) -> (L_out, 3584)
           MLP: [5120,5120] -> [3584,5120], SiLU activation
        3. VisualQuantizer (RQ): 8 codebooks x 16384, dim=3584
           -> codes (h, w, 8) -> + visual_offset(150581) -> visual_ids

        In vLLM, the tower must return embeddings that replace placeholder
        tokens in the input stream. The RQ quantizer produces discrete codes,
        but during prefill we need the continuous embeddings before quantization.
        Reference: modular_longcat_next_visual.py L550 (visual_bridge_model)
        """
        return LongcatNextVisualTokenizer(
            config=config,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

    def _build_audio_tower(self, config, prefix):
        """Build the audio tower (audio encoder).

        LongCat-Next's audio pipeline (from HF modular_longcat_next_audio.py):
        1. Whisper AudioEncoder: audio (bs, 128 mel, frames) -> (bs, frames', 1280)
           32 encoder layers, 20 heads, ffn_dim=5120
        2. avg_pooler=4: downsamples by 4x
        3. AudioVQBridger: (bs, frames', 1280) -> (bs x seq, 8)
           conv1[1280,128,3] -> conv2[1280,1280,3] -> N x OmniWhisperTransformerLayer
           -> gate_proj[5120,1280,4] + up_proj -> SiLU MLP -> down_proj[5120,5120]
           -> RQ 8 codebooks dim=5120
        4. + audio_offset(131125) -> audio_ids

        In vLLM, the tower must return embeddings that replace placeholder
        tokens in the input stream.
        Reference: modular_longcat_next_audio.py L1979 (audio_bridge_model)
        """
        return LongcatNextAudioTokenizer(
            config=config,
            prefix=maybe_prefix(prefix, "audio_tower"),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.ngram_embeddings(input_ids, ngram_context=None)
            else:
                hidden_states = inputs_embeds
        else:
            assert intermediate_tensors is not None, (
                "intermediate_tensors are required on non-first PP rank"
            )
            hidden_states = intermediate_tensors["hidden_states"]

        # Run through language model backbone
        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=hidden_states,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits using the appropriate head for the generation mode.

        CRITICAL: vLLM's model runner ONLY passes hidden_states to compute_logits.
        It does NOT pass **kwargs or generation_status.
        See: vllm/v1/worker/gpu_model_runner.py:
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)

        The generation mode must be tracked per-request and made available here.
        This is the hardest architectural challenge — see Section 6.1.

        NOTE: LogitsProcessor is REQUIRED for TP correctness. It performs the
        all-gather across TP ranks. Calling self.text_head(hidden_states) directly
        returns a (output, bias) tuple and skips the all-gather, breaking TP > 1.
        Reference: qwen2_5_omni_thinker.py lines 1499-1502
        """
        # TODO: Implement generation mode switching for bifurcated heads.
        # vLLM's inference pipeline does not pass generation mode metadata
        # (text/audio/visual) to compute_logits. To support multimodal
        # generation (audio/visual token prediction), the generation mode
        # must be tracked per-request via forward context or sampling metadata.
        #
        # For now, default to text_head since text generation is the primary
        # use case. All three heads (text_head, visual_head, audio_head) are
        # registered and load weights correctly for future extension.
        logits = self.logits_processor(self.text_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with HF to vLLM name mapping.

        CRITICAL: FlashModel.load_weights performs post-load MLA processing
        (w_kc/w_vc split, layernorm scaling) that AutoWeightsLoader does NOT do.
        We must delegate backbone weights to self.language_model.load_weights().
        """
        # Apply HF to vLLM name mapping
        mapped_weights = list(self.hf_to_vllm_mapper.apply(weights))

        # Split weights into categories
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        visual_weights: list[tuple[str, torch.Tensor]] = []
        audio_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []

        for name, weight in mapped_weights:
            if name.startswith("language_model."):
                backbone_weights.append((name[len("language_model.") :], weight))
            elif name.startswith("visual."):
                visual_weights.append((name[len("visual.") :], weight))
            elif name.startswith("audio_tower."):
                audio_weights.append((name[len("audio_tower.") :], weight))
            else:
                other_weights.append((name, weight))

        loaded_params: set[str] = set()

        # Delegate backbone weights to FlashModel.load_weights for MLA post-processing
        if backbone_weights:
            loaded_backbone = self.language_model.load_weights(backbone_weights)
            loaded_params.update(f"language_model.{n}" for n in loaded_backbone)

        # Load visual tower weights
        if self.visual is not None and visual_weights:
            loaded_visual = self.visual.load_weights(visual_weights)
            loaded_params.update(f"visual.{n}" for n in loaded_visual)

        # Load audio tower weights
        if self.audio_tower is not None and audio_weights:
            loaded_audio = self.audio_tower.load_weights(audio_weights)
            loaded_params.update(f"audio_tower.{n}" for n in loaded_audio)

        # Load remaining weights (heads) via AutoWeightsLoader
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "language_model.",
                "visual.",
                "audio_tower.",
            ],
            ignore_unexpected_prefixes=[
                "ngram_embeddings.",  # Empty on non-first PP ranks
            ],
        )
        loaded_other = loader.load_weights(other_weights)
        loaded_params.update(loaded_other)

        return loaded_params
