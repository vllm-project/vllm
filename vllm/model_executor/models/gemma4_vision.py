# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 vision tower.

Images (and video frames) are packed into one sequence with ``cu_seqlens``:
no padding patches and no attention mask, so attention runs on any
``MMEncoderAttention`` backend and the tower can be captured in encoder CUDA
graphs. Parameter names match transformers' ``Gemma4VisionModel``.
"""

from collections.abc import Mapping, Sequence
from functools import partial

import numpy as np
import torch
from torch import nn
from transformers.models.gemma4 import Gemma4VisionConfig

from vllm.model_executor.layers.activation import GELUTanh, get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.vision import (
    get_fp8_padded_hidden_size,
    get_vit_attn_backend,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .moonvit import _build_merge_gather_idx, patch_merger_packed
from .utils import maybe_prefix


class Gemma4VisionClippableLinear(nn.Module):
    """Linear layer whose input and output are clamped to checkpoint bounds."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        input_size: int,
        output_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "linear"),
            return_bias=False,
        )
        if self.use_clipped_linears:
            self.input_min: torch.Tensor
            self.input_max: torch.Tensor
            self.output_min: torch.Tensor
            self.output_max: torch.Tensor
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_proj = ReplicatedLinear(
            3 * config.patch_size**2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "input_proj"),
            return_bias=False,
        )
        self.position_embedding_table = nn.Parameter(
            torch.empty(2, config.position_embedding_size, config.hidden_size)
        )

    def forward(
        self, pixel_values: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        table = self.position_embedding_table
        pixel_values = (2 * (pixel_values - 0.5)).to(table.dtype)
        position_embeddings = table[0][positions[:, 0]] + table[1][positions[:, 1]]
        return self.input_proj(pixel_values) + position_embeddings


class Gemma4VisionAttention(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden_size = config.hidden_size

        linear = partial(Gemma4VisionClippableLinear, config, quant_config=quant_config)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = linear(hidden_size, q_size, prefix=maybe_prefix(prefix, "q_proj"))
        self.k_proj = linear(
            hidden_size, kv_size, prefix=maybe_prefix(prefix, "k_proj")
        )
        self.v_proj = linear(
            hidden_size, kv_size, prefix=maybe_prefix(prefix, "v_proj")
        )
        self.o_proj = linear(q_size, hidden_size, prefix=maybe_prefix(prefix, "o_proj"))
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)
        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            prefix=maybe_prefix(prefix, "attn"),
        )

    def _rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        half = self.head_dim // 2
        return torch.cat(
            (
                self.apply_rotary_emb(x[..., :half], cos[0], sin[0]),
                self.apply_rotary_emb(x[..., half:], cos[1], sin[1]),
            ),
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(-1, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(-1, self.head_dim))
        v = self.v_norm(self.v_proj(x).view(-1, self.head_dim))
        q = self._rope(q.view(num_tokens, self.num_heads, self.head_dim), cos, sin)
        k = self._rope(k.view(num_tokens, self.num_kv_heads, self.head_dim), cos, sin)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        out = self.attn(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        return self.o_proj(out.reshape(num_tokens, -1))


class Gemma4VisionMLP(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        linear = partial(Gemma4VisionClippableLinear, config, quant_config=quant_config)
        hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
        self.gate_proj = linear(
            hidden_size, intermediate_size, prefix=maybe_prefix(prefix, "gate_proj")
        )
        self.up_proj = linear(
            hidden_size, intermediate_size, prefix=maybe_prefix(prefix, "up_proj")
        )
        self.down_proj = linear(
            intermediate_size, hidden_size, prefix=maybe_prefix(prefix, "down_proj")
        )
        # get_act_fn swaps gelu_pytorch_tanh for the exact GELU on ROCm, for
        # torch.compile; this tower is not compiled, so keep the model's tanh.
        self.act_fn = (
            GELUTanh()
            if config.hidden_activation == "gelu_pytorch_tanh"
            else get_act_fn(config.hidden_activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4VisionAttention(
            config, quant_config=quant_config, prefix=maybe_prefix(prefix, "self_attn")
        )
        self.mlp = Gemma4VisionMLP(
            config, quant_config=quant_config, prefix=maybe_prefix(prefix, "mlp")
        )
        hidden_size, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        x = self.self_attn(self.input_layernorm(hidden_states), **attn_kwargs)
        hidden_states = hidden_states + self.post_attention_layernorm(x)
        x = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return hidden_states + self.post_feedforward_layernorm(x)


class Gemma4VisionEncoder(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert config.rope_parameters is not None
        # Axial 2D RoPE: the first half of each head rotates by the patch's
        # x, the second half by its y, each with head_dim / 2 rotary dims.
        self.rotary_emb = get_rope(
            head_size=config.head_dim,
            max_position=config.position_embedding_size,
            is_neox_style=True,
            rope_parameters={
                "rope_theta": config.rope_parameters["rope_theta"],
                "partial_rotary_factor": 0.5,
            },
        )
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i}"),
                )
                for i in range(config.num_hidden_layers)
            ]
        )


class Gemma4VisionModel(nn.Module):
    """Encodes packed patches and pools each image's ``k x k`` blocks into
    soft tokens."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "patch_embedder"),
        )
        self.encoder = Gemma4VisionEncoder(
            config, quant_config=quant_config, prefix=maybe_prefix(prefix, "encoder")
        )
        if config.standardize:
            self.std_bias: torch.Tensor
            self.std_scale: torch.Tensor
            self.register_buffer("std_bias", torch.empty(config.hidden_size))
            self.register_buffer("std_scale", torch.empty(config.hidden_size))

        self.attn_backend = get_vit_attn_backend(
            head_size=config.head_dim, dtype=torch.get_default_dtype()
        )
        # FlashInfer reads q, k and v with one per-token stride: no GQA.
        if (
            self.attn_backend == AttentionBackendEnum.FLASHINFER
            and config.num_key_value_heads != config.num_attention_heads
        ):
            raise NotImplementedError(
                "FLASHINFER for Gemma 4's vision tower needs as many key/value "
                "heads as query heads."
            )
        self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(
            config.num_attention_heads, config.head_dim
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedder.position_embedding_table.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embedder.position_embedding_table.device

    def prepare_encoder_metadata(
        self,
        positions: Sequence[np.ndarray],
        *,
        max_num_seqs: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Host-side metadata for packed images, shared by the eager path and
        CUDA graph capture and replay.

        Args:
            positions: ``(x, y)`` patch positions of each image or video
                frame, padding excluded.
            max_num_seqs: Pad ``cu_seqlens`` with empty sequences to this
                many sequences (CUDA graph buffers).
            max_seqlen_override: ``max_seqlen`` to use instead of the longest
                sequence (CUDA graph capture, where it is baked in).
            device: Device for the tensors. Defaults to the tower's.

        """
        if device is None:
            device = self.device
        k = self.config.pooling_kernel_size

        seqlens = np.array([len(p) for p in positions], dtype=np.int32)
        cu_seqlens = np.concatenate([[0], seqlens.cumsum()]).astype(np.int32)
        # The processor lays out each image's patches in row-major order, in a
        # grid whose sides are multiples of the pooling kernel.
        grids = []
        for pos in positions:
            columns = int(pos[:, 0].max()) + 1
            rows = len(pos) // columns
            patches = np.arange(len(pos))
            if (
                rows % k
                or columns % k
                or not np.array_equal(pos[:, 0], patches % columns)
                or not np.array_equal(pos[:, 1], patches // columns)
            ):
                raise ValueError(
                    f"Expected a row-major grid of patches with sides that are "
                    f"multiples of {k}, got {len(pos)} patches in "
                    f"{columns} columns."
                )
            grids.append((rows, columns))
        pool_gather_idx = _build_merge_gather_idx(grids, (k, k))
        if max_num_seqs is not None and len(seqlens) < max_num_seqs:
            cu_seqlens = np.pad(
                cu_seqlens, (0, max_num_seqs - len(seqlens)), mode="edge"
            )

        if max_seqlen_override is not None:
            max_seqlen = max_seqlen_override
        else:
            max_seqlen = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens
            )
        metadata = {
            "pixel_position_ids": async_tensor_h2d(
                np.concatenate(positions).astype(np.int64), device=device
            ),
            "pool_gather_idx": async_tensor_h2d(pool_gather_idx, device=device),
            "cu_seqlens": MMEncoderAttention.maybe_recompute_cu_seqlens(
                self.attn_backend,
                cu_seqlens,
                self.config.num_attention_heads * self.config.head_dim,
                1,
                device,
                fp8_padded_hidden_size=self.fp8_padded_hidden_size,
                separate_qkv=True,
            ),
            # On the host: backends read it there, and CUDA graphs bake it in.
            "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32),
        }
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, device
        )
        if sequence_lengths is not None:  # FlashInfer
            metadata["sequence_lengths"] = sequence_lengths
        return metadata

    def forward(
        self,
        pixel_values: torch.Tensor,
        encoder_metadata: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode packed patches ``(num_patches, 3 * patch_size**2)`` into
        soft tokens ``(num_patches / k**2, hidden_size)``."""
        positions = encoder_metadata["pixel_position_ids"]
        hidden_states = self.patch_embedder(pixel_values, positions)
        cos, sin = self.encoder.rotary_emb.get_cos_sin(
            self.config.position_embedding_size
        )
        # (2, num_patches, head_dim / 4): x's tables, then y's, each contiguous.
        cos, sin = cos[positions.T], sin[positions.T]
        for layer in self.encoder.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                max_seqlen=encoder_metadata["max_seqlen"],
                sequence_lengths=encoder_metadata.get("sequence_lengths"),
            )
        return self._pool(hidden_states, encoder_metadata["pool_gather_idx"])

    def _pool(
        self, hidden_states: torch.Tensor, pool_gather_idx: torch.Tensor
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        k = self.config.pooling_kernel_size
        blocks = patch_merger_packed(hidden_states, pool_gather_idx, (k, k))
        # Average and scale in float32 like transformers: the sqrt(hidden_size)
        # scaling can exceed the float16 range before standardization.
        pooled = blocks.float().mean(1).to(hidden_states.dtype).float()
        pooled = pooled * hidden_size**0.5
        if self.config.standardize:
            pooled = (pooled - self.std_bias.float()) * self.std_scale.float()
        return pooled.to(hidden_states.dtype)
