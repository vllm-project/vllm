# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SplitQ attention backend for AMD GPUs (native kernels: HIP or FlyDSL).

KV cache format and reference: ``vllm/v1/attention/ops/rocm_splitq.py``.
Kernels: ``csrc/attention/splitq_attn.cu`` (portable) and the per-architecture
tiers in ``_ARCH_TIERS``; see ``docs/design/rocm_splitq.md``.

Decode, MTP verification and short continuation chunks (every request with
at most ``_DECODE_MAX_QUERY_LEN`` query tokens) run the split-KV decode
kernel over the packed cache; consecutive query tokens of one request share
each K/V read. Longer prefills run the prefill kernel, which reads the cached
prefix from the packed cache and the current chunk's K/V unquantized.
"""

import contextlib
from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops import rocm_splitq as sq
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

# Per-architecture kernel tiers (gcnArchName prefix -> (decode, prefill)).
# Decode always has the portable dot4 kernel; prefill needs a native kernel
# (HIP or FlyDSL) per architecture and there is no Triton fallback.
_ARCH_TIERS: dict[str, tuple[str, str | None]] = {
    "gfx11": ("wmma", "wmma"),
}
_PORTABLE_TIERS: tuple[str, str | None] = ("dot4", None)


def _arch_tiers() -> tuple[str, tuple[str, str | None]]:
    arch = torch.cuda.get_device_properties().gcnArchName.split(":")[0]
    for prefix, tiers in _ARCH_TIERS.items():
        if arch.startswith(prefix):
            return arch, tiers
    return arch, _PORTABLE_TIERS


_DECODE_MAX_QUERY_LEN = 128
_KERNEL_HEAD_SIZE = 256
_KERNEL_ROPE_DIM = 64


class RocmSplitQAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "splitq_k3v4",
        "splitq_k3v3",
        "splitq_k3v3_compact",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_SPLITQ"

    @staticmethod
    def get_impl_cls() -> type["RocmSplitQAttentionImpl"]:
        return RocmSplitQAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["RocmSplitQMetadataBuilder"]:
        return RocmSplitQMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_KERNEL_HEAD_SIZE]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The kernels take block/head/token strides; the slot is contiguous.
        return (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """One packed K+V slot per (token, KV head)."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_splitq:
            return spec
        fmt = sq.SplitQFormat.from_cache_dtype(
            spec.kv_quant_mode.name.lower(),
            spec.head_size,
            sq.registered_rope_dim(spec.head_size),
        )
        return replace(spec, state_content_bytes=fmt.slot_bytes)


class RocmSplitQMetadataBuilder(TritonAttentionMetadataBuilder):
    """Reuses the Triton builder: it fills the per-query request and causal
    length maps (``q_to_req`` / ``q_to_klen``) the kernels index by, including
    for spec-decode verification steps."""


class RocmSplitQAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None or sinks is not None:
            raise NotImplementedError("ROCM_SPLITQ does not support ALiBi or sinks")
        if sliding_window is not None or logits_soft_cap:
            raise NotImplementedError(
                "ROCM_SPLITQ does not support sliding window or soft cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("ROCM_SPLITQ supports decoder attention only")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        vllm_config = get_current_vllm_config()
        rope_dim = sq.rope_dim_from_hf_config(
            vllm_config.model_config.hf_text_config, head_size
        )
        sq.register_rope_dim(head_size, rope_dim)
        self.fmt = sq.SplitQFormat.from_cache_dtype(kv_cache_dtype, head_size, rope_dim)
        if not (
            current_platform.is_rocm()
            and head_size == _KERNEL_HEAD_SIZE
            and rope_dim == _KERNEL_ROPE_DIM
            and hasattr(torch.ops._C, "splitq_decode")
        ):
            raise NotImplementedError(
                "ROCM_SPLITQ kernels need ROCm, head_size=256 and 64 rotary "
                f"dims; got head_size={head_size}, rotary dims={rope_dim}"
            )
        arch, (decode_tier, prefill_tier) = _arch_tiers()
        if prefill_tier is None:
            raise NotImplementedError(
                f"ROCM_SPLITQ has no native prefill kernel for {arch} yet; see the "
                "architecture contract in docs/design/rocm_splitq.md"
            )
        self._use_wmma = decode_tier == "wmma"

        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self._max_capture_size = (
            vllm_config.compilation_config.max_cudagraph_capture_size or 4
        )
        self._k_signs: torch.Tensor | None = None
        self._v_signs: torch.Tensor | None = None
        self._mid_o: torch.Tensor | None = None

    def _signs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._k_signs is None or self._k_signs.device != device:
            self._k_signs = sq.sign_bits(self.head_size).to(device)
            self._v_signs = self._k_signs
        assert self._v_signs is not None
        return self._k_signs, self._v_signs

    def _mid_o_buffer(
        self, num_q: int, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """Split-KV partials. The persistent buffer is allocated once at the
        cudagraph capture size and never replaced (captured graphs hold its
        address); larger eager batches get a bounded transient one."""
        if self._mid_o is None:
            self._mid_o = torch.zeros(
                self._max_capture_size,
                self.num_heads,
                self.max_num_kv_splits,
                self.head_size + 2,
                dtype=torch.float32,
                device=device,
            )
        if num_q <= self._mid_o.shape[0]:
            return self._mid_o, self.max_num_kv_splits
        budget = self._max_capture_size * self.max_num_kv_splits
        splits = max(1, min(self.max_num_kv_splits, budget // num_q))
        buf = torch.empty(
            num_q,
            self.num_heads,
            splits,
            self.head_size + 2,
            dtype=torch.float32,
            device=device,
        )
        return buf, splits

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if slot_mapping.numel() == 0:
            return
        k_signs, v_signs = self._signs(key.device)
        n = slot_mapping.shape[0]
        torch.ops._C.splitq_cache_store(
            key[:n],
            value[:n],
            kv_cache,
            slot_mapping,
            k_signs,
            v_signs,
            self.fmt.kernel_code,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("ROCM_SPLITQ does not fuse output quant")
        if attn_metadata is None:
            return output.fill_(0)

        n = attn_metadata.num_actual_tokens
        out = output[:n].view(n, self.num_heads, self.head_size)
        q = query[:n].view(n, self.num_heads, self.head_size)

        if attn_metadata.max_query_len <= _DECODE_MAX_QUERY_LEN:
            assert attn_metadata.q_to_req is not None
            self._decode(
                out,
                q,
                kv_cache,
                attn_metadata,
                attn_metadata.q_to_req,
                attn_metadata.q_to_klen,
            )
            return output

        num_dt = attn_metadata.num_decode_tokens
        if num_dt > 0:
            q_to_req, q_to_klen = attn_metadata.q_to_req, attn_metadata.q_to_klen
            assert q_to_req is not None and q_to_klen is not None
            self._decode(
                out[:num_dt],
                q[:num_dt],
                kv_cache,
                attn_metadata,
                q_to_req[:num_dt],
                q_to_klen[:num_dt],
            )
        self._prefill(out, q, key, value, kv_cache, attn_metadata)
        return output

    def _decode(
        self,
        out: torch.Tensor,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        md: TritonAttentionMetadata,
        q_to_req: torch.Tensor,
        q_to_klen: torch.Tensor,
    ) -> None:
        num_q = q.shape[0]
        if num_q == 0:
            return
        k_signs, v_signs = self._signs(q.device)
        mid_o, splits = self._mid_o_buffer(num_q, q.device)
        # Async scheduling can free and reuse an input's memory before this
        # kernel reads it; tell the allocator these are in use on the stream.
        stream = torch.cuda.current_stream()
        for t in (q, out, md.block_table, q_to_req, q_to_klen, mid_o):
            with contextlib.suppress(Exception):
                t.record_stream(stream)
        torch.ops._C.splitq_decode(
            out,
            q,
            kv_cache,
            md.block_table,
            q_to_req,
            q_to_klen,
            mid_o,
            k_signs,
            v_signs,
            self.scale,
            splits,
            self.fmt.kernel_code,
            4 if md.max_query_len > 1 else 1,
            self._use_wmma,
        )

    def _prefill(
        self,
        out: torch.Tensor,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        md: TritonAttentionMetadata,
    ) -> None:
        """Requests after the decodes, in the rotated space, without host
        syncs: one call covers them all, reading the cached prefixes straight
        from the packed cache and the chunks' own K/V unquantized."""
        nd, ndt, n = md.num_decodes, md.num_decode_tokens, md.num_actual_tokens
        num_reqs = md.query_start_loc.shape[0] - 1
        if num_reqs - nd <= 0 or ndt >= n:
            return
        k_signs, v_signs = self._signs(q.device)
        # The kernel runs in fp16; the rotated copies are made in fp16.
        q_rot = q[ndt:n].to(torch.float16, copy=True)
        k_rot = key[ndt:n].to(torch.float16, copy=True)
        v_rot = value[ndt:n].to(torch.float16, copy=True)
        k_blocks = not self.fmt.compact
        torch.ops._C.splitq_rotate(q_rot, k_signs, k_blocks, False)
        torch.ops._C.splitq_rotate(k_rot, k_signs, k_blocks, False)
        torch.ops._C.splitq_rotate(v_rot, v_signs, False, False)
        o_rot = torch.empty_like(q_rot)
        torch.ops._C.splitq_prefill(
            o_rot,
            q_rot,
            k_rot,
            v_rot,
            kv_cache,
            md.block_table[nd:num_reqs],
            (md.query_start_loc[nd:] - ndt).to(torch.int32),
            md.seq_lens[nd:num_reqs].to(torch.int32),
            md.max_query_len,
            self.scale,
            self.fmt.kernel_code,
        )
        torch.ops._C.splitq_rotate(o_rot, v_signs, False, True)
        out[ndt:n] = o_rot
