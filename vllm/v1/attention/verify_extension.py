# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension contract for non-chain verification on full-attention layers.

The contract is intentionally separate from any attention kernel. An
out-of-tree backend must explicitly opt in, attach the extension to its
metadata, and call :func:`dispatch_full_attention_verify`. The in-tree
FlashAttention and ROCmAttention backends do not do that yet.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.spec_decode.registry import get_speculative_method

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import (
        AttentionBackend,
        CommonAttentionMetadata,
    )


@dataclass(frozen=True)
class FullAttentionVerifyBackend:
    """Stable identity of an attention backend that implements this ABI."""

    name: str
    module: str
    qualname: str

    @classmethod
    def from_backend(
        cls, backend: type["AttentionBackend"]
    ) -> "FullAttentionVerifyBackend":
        module, qualname = backend.full_cls_name()
        return cls(name=backend.get_name(), module=module, qualname=qualname)

    @property
    def class_path(self) -> str:
        return f"{self.module}.{self.qualname}"


@dataclass(frozen=True)
class FullAttentionVerifyMetadata:
    """Backend-neutral batch metadata exposed to a verify extension.

    Tensor fields are zero-copy views of ``CommonAttentionMetadata``. The
    extension must not retain metadata beyond the forward that consumes it.
    """

    backend: FullAttentionVerifyBackend
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool | torch.Tensor
    positions: torch.Tensor | None
    is_prefilling: torch.Tensor | None


@dataclass(frozen=True)
class FullAttentionVerifyRows:
    """Data-only view of one full-attention layer's verify forward.

    The extension writes every handled row to ``output``. There is no
    "not handled" result: a capable backend calls this only for a non-chain
    verify and must use its normal path for all other forwards.

    ``kv_cache`` keeps the identified backend's native layout and strides.
    This ABI does not pretend FlashAttention and ROCmAttention caches are
    interchangeable; each backend wiring must document and test its layout.
    """

    layer_name: str
    query: torch.Tensor
    key: torch.Tensor | None
    value: torch.Tensor | None
    kv_cache: torch.Tensor
    output: torch.Tensor
    metadata: FullAttentionVerifyMetadata
    num_heads: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    scale: float
    kv_cache_dtype: str
    attn_type: str
    sliding_window: tuple[int, int] | None
    logits_soft_cap: float | None
    alibi_slopes: torch.Tensor | None
    sinks: torch.Tensor | None
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    output_scale: torch.Tensor | None = None
    output_block_scale: torch.Tensor | None = None


class FullAttentionVerifyExtension:
    """Runs a non-chain verify on full-attention rows.

    Both sides opt in. ``supports_backend`` must recognize the exact backend
    identity, and that backend must implement
    ``AttentionBackend.supports_full_attention_verify_extension`` plus the
    metadata and forward dispatch. The default supports nothing.
    """

    def supports_backend(self, backend: FullAttentionVerifyBackend) -> bool:
        return False

    def forward(self, rows: FullAttentionVerifyRows) -> None:
        """Write the attention result for all rows to ``rows.output``."""
        raise NotImplementedError


def build_full_attention_verify_metadata(
    backend: FullAttentionVerifyBackend,
    common_attn_metadata: "CommonAttentionMetadata",
) -> FullAttentionVerifyMetadata:
    """Build the frozen, zero-copy batch contract for a capable backend."""
    return FullAttentionVerifyMetadata(
        backend=backend,
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=common_attn_metadata.max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        query_start_loc=common_attn_metadata.query_start_loc,
        seq_lens=common_attn_metadata.seq_lens,
        block_table=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=common_attn_metadata.causal,
        positions=common_attn_metadata.positions,
        is_prefilling=common_attn_metadata.is_prefilling,
    )


def create_full_attn_verify_extension(
    vllm_config: "VllmConfig",
    backend: type["AttentionBackend"],
) -> FullAttentionVerifyExtension | None:
    """Create and validate the registered method's extension for ``backend``."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return None
    registered = get_speculative_method(speculative_config.method)
    if registered is None or registered.full_attn_verify_extension is None:
        return None

    factory = resolve_obj_by_qualname(
        registered.full_attn_verify_extension.replace(":", ".")
    )
    extension = factory(vllm_config)
    if extension is None:
        return None
    if not isinstance(extension, FullAttentionVerifyExtension):
        raise TypeError(
            "A full-attention verify-extension factory must return "
            "FullAttentionVerifyExtension or None."
        )

    capability = FullAttentionVerifyBackend.from_backend(backend)
    if not backend.supports_full_attention_verify_extension():
        raise ValueError(
            f"Attention backend {capability.name} ({capability.class_path}) "
            "does not wire the full-attention verify-extension ABI."
        )
    if not extension.supports_backend(capability):
        raise ValueError(
            f"{type(extension).__name__} does not support attention backend "
            f"{capability.name} ({capability.class_path})."
        )
    return extension


def dispatch_full_attention_verify(
    extension: FullAttentionVerifyExtension,
    rows: FullAttentionVerifyRows,
) -> None:
    """Dispatch without a causal fallback.

    A metadata/backend mismatch raises instead of silently verifying a tree as
    a chain.
    """
    if not extension.supports_backend(rows.metadata.backend):
        backend = rows.metadata.backend
        raise RuntimeError(
            f"{type(extension).__name__} cannot dispatch attention backend "
            f"{backend.name} ({backend.class_path})."
        )
    extension.forward(rows)
