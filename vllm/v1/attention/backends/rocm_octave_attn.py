# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Octave attention backend for AMD GPUs (native kernels: HIP or FlyDSL).

KV cache format and reference: ``vllm/v1/attention/ops/rocm_octave.py``.
Kernels: ``csrc/attention/octave_attn.cu`` (portable) and the per-architecture
tiers in ``_ARCH_TIERS``; see ``docs/design/rocm_octave.md``.

Decode, MTP verification and short continuation chunks (every request with
at most ``_DECODE_MAX_QUERY_LEN`` query tokens) run the split-KV decode
kernel over the packed cache; consecutive query tokens of one request share
each K/V read. Longer prefills run the prefill kernel, which reads the cached
prefix from the packed cache and the current chunk's K/V unquantized.
"""

import contextlib
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops import rocm_octave as sq
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

# Per-architecture kernel tiers (gcnArchName prefix -> (decode, prefill)).
# Decode always has the portable dot4 kernel; prefill needs a native kernel
# (HIP or FlyDSL) per architecture and there is no Triton fallback.
_ARCH_TIERS: dict[str, tuple[str, str | None]] = {
    # Only the targets the WMMA kernels are compiled for; on others they are
    # empty stubs.
    "gfx110": ("wmma", "wmma"),
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


def _mid_o_shape(
    num_q: int, num_heads: int, max_splits: int, capture_size: int
) -> tuple[tuple[int, int, int, int], int]:
    """Split-KV partials for num_q queries: every split up to the cudagraph
    capture size, fewer splits past it so the buffer never outgrows that."""
    splits = max_splits
    if num_q > capture_size:
        splits = max(1, min(max_splits, capture_size * max_splits // num_q))
    return (num_q, num_heads, splits, _KERNEL_HEAD_SIZE + 2), splits


def _max_capture_size(vllm_config: VllmConfig) -> int:
    return vllm_config.compilation_config.max_cudagraph_capture_size or 4


class RocmOctaveAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "octave_k3v4",
        "octave_k3v3",
        "octave_k3v3_compact",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_OCTAVE"

    @staticmethod
    def get_impl_cls() -> type["RocmOctaveAttentionImpl"]:
        return RocmOctaveAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["RocmOctaveMetadataBuilder"]:
        return RocmOctaveMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: "KVCacheSpec | None" = None,
    ) -> list[int | MultipleOf]:
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
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_octave:
            return spec
        fmt = sq.OctaveFormat.from_cache_dtype(
            spec.kv_quant_mode.name.lower(),
            spec.head_size,
            sq.registered_rope_dim(spec.head_size),
        )
        return replace(spec, state_content_bytes=fmt.slot_bytes)


@dataclass
class RocmOctaveMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # Decode requests (one query token) come first in the batch.
    num_decodes: int
    num_decode_tokens: int
    # Per query token: its request and its causal K length (0 for padding).
    q_to_req: torch.Tensor
    q_to_klen: torch.Tensor


class RocmOctaveMetadataBuilder(AttentionMetadataBuilder[RocmOctaveMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        # Persistent so captured graphs keep reading the current contents.
        # Sized for the larger of the scheduler's token budget and one KV
        # block: block-aligned chunks can exceed the budget on hybrid models,
        # where a block spans thousands of tokens.
        max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            kv_cache_spec.block_size,
        )
        self._q_to_req = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self._q_to_klen = torch.empty(max_tokens, dtype=torch.int32, device=device)
        # Layers run one after another, so they share one set of split-KV
        # partials; reserve it before the workspace is locked.
        if is_workspace_manager_initialized():
            num_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            capture_size = _max_capture_size(vllm_config)
            shape, _ = _mid_o_shape(
                capture_size,
                num_heads,
                vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph,
                capture_size,
            )
            current_workspace_manager().get_simultaneous((shape, torch.float32))

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmOctaveMetadata:
        cm = common_attn_metadata
        n = cm.num_actual_tokens
        num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(cm)
        md = RocmOctaveMetadata(
            num_actual_tokens=n,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            q_to_req=self._q_to_req[:n],
            q_to_klen=self._q_to_klen[:n],
        )
        self._fill_query_maps(md)
        return md

    def update_draft_decode_metadata(self, metadata: RocmOctaveMetadata) -> None:
        # Draft steps advance seq_lens in place; the K lengths follow it.
        self._fill_query_maps(metadata)

    @staticmethod
    def _fill_query_maps(md: RocmOctaveMetadata) -> None:
        """On the device, without host syncs (also inside graph capture).
        Queries past the last request (cudagraph padding) get K length 0."""
        n = md.num_actual_tokens
        num_reqs = md.query_start_loc.shape[0] - 1
        if n == 0 or num_reqs == 0:
            return
        idx = torch.arange(n, dtype=torch.int32, device=md.q_to_req.device)
        qsl = md.query_start_loc.to(torch.int32)
        req = torch.searchsorted(qsl[1:], idx, right=True).to(torch.int32)
        r = req.clamp(max=num_reqs - 1).long()
        q_lens = qsl[1:] - qsl[:-1]
        klen = md.seq_lens.to(torch.int32)[r] - q_lens[r] + (idx - qsl[r]) + 1
        md.q_to_req.copy_(req)
        md.q_to_klen.copy_(torch.where(req < num_reqs, klen, 0))


class RocmOctaveAttentionImpl(AttentionImpl):
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
            raise NotImplementedError("ROCM_OCTAVE does not support ALiBi or sinks")
        if sliding_window is not None or logits_soft_cap:
            raise NotImplementedError(
                "ROCM_OCTAVE does not support sliding window or soft cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("ROCM_OCTAVE supports decoder attention only")

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
        self.fmt = sq.OctaveFormat.from_cache_dtype(kv_cache_dtype, head_size, rope_dim)
        if not (
            current_platform.is_rocm()
            and head_size == _KERNEL_HEAD_SIZE
            and rope_dim == _KERNEL_ROPE_DIM
            and hasattr(torch.ops._C, "octave_decode")
        ):
            raise NotImplementedError(
                "ROCM_OCTAVE kernels need ROCm, head_size=256 and 64 rotary "
                f"dims; got head_size={head_size}, rotary dims={rope_dim}"
            )
        arch, (decode_tier, prefill_tier) = _arch_tiers()
        if prefill_tier is None:
            raise NotImplementedError(
                f"ROCM_OCTAVE has no native prefill kernel for {arch} yet; see the "
                "architecture contract in docs/design/rocm_octave.md"
            )
        self._use_wmma = decode_tier == "wmma"

        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self._max_capture_size = _max_capture_size(vllm_config)
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
        """Split-KV partials, from the workspace every layer shares. Without a
        workspace, a buffer of this layer's own, allocated once at the
        cudagraph capture size since captured graphs hold its address."""
        shape, splits = _mid_o_shape(
            num_q, self.num_heads, self.max_num_kv_splits, self._max_capture_size
        )
        if is_workspace_manager_initialized():
            (buf,) = current_workspace_manager().get_simultaneous(
                (shape, torch.float32)
            )
            return buf, splits
        if self._mid_o is None:
            full, _ = _mid_o_shape(
                self._max_capture_size,
                self.num_heads,
                self.max_num_kv_splits,
                self._max_capture_size,
            )
            self._mid_o = torch.empty(full, dtype=torch.float32, device=device)
        if num_q <= self._mid_o.shape[0]:
            return self._mid_o, splits
        return torch.empty(shape, dtype=torch.float32, device=device), splits

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
        torch.ops._C.octave_cache_store(
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
        attn_metadata: RocmOctaveMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("ROCM_OCTAVE does not fuse output quant")
        if attn_metadata is None:
            return output.fill_(0)

        n = attn_metadata.num_actual_tokens
        out = output[:n].view(n, self.num_heads, self.head_size)
        q = query[:n].view(n, self.num_heads, self.head_size)

        if attn_metadata.max_query_len <= _DECODE_MAX_QUERY_LEN:
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
        md: RocmOctaveMetadata,
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
        torch.ops._C.octave_decode(
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
        md: RocmOctaveMetadata,
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
        torch.ops._C.octave_rotate(q_rot, k_signs, k_blocks, False)
        torch.ops._C.octave_rotate(k_rot, k_signs, k_blocks, False)
        torch.ops._C.octave_rotate(v_rot, v_signs, False, False)
        o_rot = torch.empty_like(q_rot)
        torch.ops._C.octave_prefill(
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
        torch.ops._C.octave_rotate(o_rot, v_signs, False, True)
        out[ndt:n] = o_rot
