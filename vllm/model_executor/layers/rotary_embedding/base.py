# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings Base Class."""

import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform

from .common import ApplyRotaryEmb
from .functional import native_rope


# --8<-- [start:rotary_embedding]
@CustomOp.register("rotary_embedding")
class RotaryEmbeddingBase(CustomOp):
    """Original rotary positional embedding."""

    # --8<-- [end:rotary_embedding]

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.spec = current_platform.spec

        # Specialized RoPE variants may require an fp32 cache for FlashInfer.
        if not hasattr(self, "use_flashinfer"):
            self.use_flashinfer = False

        if init_cache:
            cache = self._compute_cos_sin_cache()
            if not self.use_flashinfer:
                cache = cache.to(dtype)
            self.cos_sin_cache: torch.Tensor
            self.register_buffer("cos_sin_cache", cache, persistent=False)

            alternate_dtype = self.spec.rope_cache_dtype
            self.cos_sin_cache_alternate: torch.Tensor | None
            if (
                self.enabled()
                and alternate_dtype is not None
                and cache.dtype != alternate_dtype
            ):
                self.register_buffer(
                    "cos_sin_cache_alternate",
                    cache.to(alternate_dtype),
                    persistent=False,
                )
            else:
                self.cos_sin_cache_alternate = None

        self.apply_rotary_emb = ApplyRotaryEmb(
            is_neox_style=self.is_neox_style,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> torch.Tensor:
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        cos_sin_cache = self.cos_sin_cache
        if (
            cos_sin_cache.device == query.device
            and self.cos_sin_cache.dtype == query.dtype
        ):
            return cos_sin_cache

        alternate = getattr(self, "cos_sin_cache_alternate", None)
        if (
            torch.compiler.is_compiling()
            and alternate is not None
            and alternate.dtype == query.dtype
            and alternate.device == query.device
        ):
            return alternate

        cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)
        # Avoid mutating buffers during torch.compile (cudagraph) tracing.
        if torch.compiler.is_compiling():
            return cos_sin_cache

        self.cos_sin_cache = cos_sin_cache
        return cos_sin_cache

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[:seqlen]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin


class RotaryEmbedding(RotaryEmbeddingBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
        )

    forward_static = staticmethod(native_rope)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )

    def forward_platform(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rope = self.spec.rope
        if rope is None:
            return self.forward_native(positions, query, key)
        return rope(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            self._match_cos_sin_cache_dtype(query),
            self.is_neox_style,
        )

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
