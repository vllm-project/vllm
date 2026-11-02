# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UltraQuant 4-bit KV-cache attention on CPU.

The slot layout is the one in ``v1/attention/ops/ultraquant/format.py``, so a
cache written here is byte-identical to the GPU backend's. Only the seams
differ: the store and decode kernels are the AVX-512 ops in
``csrc/cpu/ultraquant.cpp``, a first prefill chunk attends the unquantized
K/V it was handed, and a continuation chunk stages the dequantized prefix
into a bf16 page so vLLM's own CPU attention kernel can run over it.
"""

from dataclasses import replace
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionBackendImpl,
    CPUAttentionMetadata,
    CPUAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.ultraquant.format import (
    DEFAULT_CONSTANT_C,
    GROUP_SIZE,
    slot_size,
)
from vllm.v1.attention.ops.ultraquant.reference import (
    UltraQuantEncoded,
    hadamard_matrix,
    ultraquant_dequant,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

logger = init_logger(__name__)

# Reading the 4-bit cache in place re-reads it once per query row, so it beats
# dequantizing the prefix into a bf16 page only while the chunk is a small
# fraction of the context. An eighth is where the two cross on Turin, which is
# why this is a ratio and not the flat token count the GPU backend uses.
_CONTINUATION_DECODE_RATIO = 8


class CPUUltraQuantAttentionBackend(CPUAttentionBackend):
    """CPU backend for the ``ultraquant_4bit`` KV-cache dtype."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["ultraquant_4bit"]

    @staticmethod
    def get_name() -> str:
        return "CPU_ULTRAQUANT"

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """UltraQuant packs K and V into one 4-bit slot per head."""
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_ultraquant:
            return spec
        return replace(spec, state_content_bytes=slot_size(spec.head_size))

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Head-major block interiors, as the kernels index them, and as
        # CPU_ATTN wants: the boundary layers it keeps at the native dtype
        # share one cache layout with the quantized ones.
        return (KVCacheLayout.LBHNC,)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Powers of two for the Hadamard rotation, and a multiple of the
        # group size of 32 so a head is a whole number of scale groups.
        return [32, 64, 128, 256]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "ultraquant_4bit"

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["CPUUltraQuantAttentionImpl"]:
        return CPUUltraQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[CPUAttentionMetadataBuilder]:
        return CPUAttentionMetadataBuilder


class CPUUltraQuantAttentionImpl(CPUAttentionBackendImpl):
    """FP4 E2M1 codes with UE8M0 group-of-32 scales and Hadamard-rotated K."""

    @property
    def rotation(self) -> torch.Tensor:
        return hadamard_matrix(self.head_size, torch.device("cpu"), torch.float32)

    @property
    def rotation_t(self) -> torch.Tensor:
        """Transposed rotation in the model dtype, for the decode query."""
        t = getattr(self, "_rot_t", None)
        if t is None:
            t = self.rotation.T.to(torch.bfloat16).contiguous()
            self._rot_t = t
        return t

    @property
    def fp8_widen(self) -> torch.Tensor:
        """Every FP8 E4M3 bit pattern as bf16.

        Torch's own fp8->bf16 conversion is scalar and costs more than the
        rotation it follows; a 256-entry gather gives the identical bytes.
        """
        t = getattr(self, "_fp8_tbl", None)
        if t is None:
            t = (
                torch.arange(256, dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .to(torch.bfloat16)
            )
            self._fp8_tbl = t
        return t

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        ops.ultraquant_store(
            key, value, kv_cache, slot_mapping, self.rotation, DEFAULT_CONSTANT_C
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "CPUUltraQuantAttentionImpl"
            )
        if attn_metadata is None:
            return output

        num_tokens = attn_metadata.num_actual_tokens
        if attn_metadata.max_query_len == 1:
            # Pure-decode batch: every row is its own request, so the metadata
            # tensors already line up with the kernel's arguments.
            self._decode(
                query[:num_tokens],
                kv_cache,
                attn_metadata.block_table[:num_tokens],
                attn_metadata.seq_lens[:num_tokens],
                output[:num_tokens],
            )
            return output

        qsl = attn_metadata.query_start_loc.tolist()
        ctx_lens = attn_metadata.seq_lens.tolist()
        decode_rows: list[int] = []
        prefill_rows: list[int] = []
        cont_rows: list[int] = []
        for req, ctx in enumerate(ctx_lens):
            start, end = qsl[req], qsl[req + 1]
            if start >= num_tokens:
                break
            q_len = end - start
            if q_len == ctx:
                prefill_rows.append(req)
            elif q_len == 1:
                decode_rows.append(req)
            else:
                cont_rows.append(req)

        if prefill_rows:
            lo, hi = qsl[prefill_rows[0]], qsl[prefill_rows[-1] + 1]
            lens = [qsl[r + 1] - qsl[r] for r in prefill_rows]
            # The scratch page is addressed by a dense token range, so a
            # scattered set of prefill rows has to go one request at a time.
            if hi - lo == sum(lens):
                self._prefill(
                    layer,
                    query[lo:hi],
                    key[lo:hi],
                    value[lo:hi],
                    output[lo:hi],
                    lens,
                    kv_cache.size(2),
                )
            else:
                for r in prefill_rows:
                    s, e = qsl[r], qsl[r + 1]
                    self._prefill(
                        layer,
                        query[s:e],
                        key[s:e],
                        value[s:e],
                        output[s:e],
                        [e - s],
                        kv_cache.size(2),
                    )

        for r in cont_rows:
            s, e = qsl[r], qsl[r + 1]
            self._continuation(
                layer,
                query[s:e],
                key[s:e],
                value[s:e],
                output[s:e],
                kv_cache,
                attn_metadata.block_table[r],
                ctx_lens[r],
            )

        if decode_rows:
            idx = torch.tensor(decode_rows, dtype=torch.int64)
            tok = torch.tensor([qsl[r] for r in decode_rows], dtype=torch.int64)
            out = torch.empty(
                (len(decode_rows), self.num_heads, self.head_size),
                dtype=torch.float32,
            )
            self._decode(
                query.index_select(0, tok),
                kv_cache,
                attn_metadata.block_table.index_select(0, idx),
                attn_metadata.seq_lens.index_select(0, idx),
                out,
            )
            output.index_copy_(0, tok, out.to(output.dtype))
        return output

    def _scratch_cache(self, num_blocks: int, block_size: int) -> torch.Tensor:
        """A bf16 paged cache for staged K/V, grown as needed and reused.

        The kernel still reads the unwritten tail of each prompt's last block
        before masking it, so the buffer has to start finite: uninitialized
        memory there turns the whole request's output into NaN. Stale finite
        values left by an earlier request are masked out and so are harmless.
        """
        tail = (self.num_kv_heads, block_size, 2 * self.head_size)
        buf = getattr(self, "_scratch", None)
        if buf is None or buf.shape[1:] != tail or buf.size(0) < num_blocks:
            buf = torch.zeros((num_blocks, *tail), dtype=torch.bfloat16)
            self._scratch = buf
        return buf[:num_blocks]

    @staticmethod
    def _split_scratch(scratch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Carve the scratch page into the two caches the CPU kernel wants."""
        nb, hk, bs, _ = scratch.size()
        kc, vc = scratch.view(nb, hk, bs * 2, -1).chunk(2, dim=2)
        return kc, vc

    def _rotation_like(self, t: torch.Tensor) -> torch.Tensor:
        """The store's rotation in ``t``'s dtype; hadamard_matrix caches these."""
        return hadamard_matrix(self.head_size, t.device, t.dtype)

    def _prefill(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        lens: list[int],
        block_size: int,
    ) -> None:
        """Attend the unquantized K/V of prompts that have no prior context.

        The UltraQuant cache holds nothing these requests can read yet, so the
        raw tensors are the exact inputs. Staging them into a bf16 scratch page
        buys vLLM's own CPU attention kernel, which beats per-request SDPA.
        """
        lens_t = torch.tensor(lens, dtype=torch.int32)
        blocks = (lens_t + block_size - 1) // block_size
        start_blk = torch.zeros_like(blocks)
        torch.cumsum(blocks[:-1], 0, out=start_blk[1:])
        block_table = start_blk[:, None] + torch.arange(
            int(blocks.max()), dtype=blocks.dtype
        )

        qsl = torch.zeros(len(lens) + 1, dtype=torch.int32)
        torch.cumsum(lens_t, 0, out=qsl[1:])
        offs = torch.arange(int(qsl[-1]), dtype=torch.int64) - qsl[:-1].to(
            torch.int64
        ).repeat_interleave(lens_t)
        slot_mapping = (start_blk.to(torch.int64) * block_size).repeat_interleave(
            lens_t
        ) + offs

        scratch = self._scratch_cache(int(blocks.sum()), block_size)
        kc, vc = self._split_scratch(scratch)
        ops.cpu_attn_reshape_and_cache(key, value, kc, vc, slot_mapping, layer.isa)  # type: ignore
        self._attend(layer, query, kc, vc, output, qsl, lens_t, block_table)

    def _attend(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kc: torch.Tensor,
        vc: torch.Tensor,
        output: torch.Tensor,
        qsl: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        ops.cpu_attention_with_kv_cache(
            query=query,
            key_cache=kc,
            value_cache=vc,
            output=output,
            query_start_loc=qsl,
            seq_lens=seq_lens,
            scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=ops.cpu_attn_get_scheduler_metadata(
                num_reqs=seq_lens.numel(),
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                seq_lens=seq_lens,
                dtype=query.dtype,
                query_start_loc=qsl,
                causal=True,
                sliding_window_size=self.sliding_window,
                isa=layer.isa,  # type: ignore
                enable_kv_split=envs.VLLM_CPU_ATTN_SPLIT_KV,
            ),
            s_aux=self.sinks,
            dynamic_causal=None,
        )

    def _dequant_prefix(
        self, kv_cache: torch.Tensor, block_ids: torch.Tensor, prefix_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One request's cached K/V as bf16; K stays in the rotated basis."""
        d = self.head_size
        g = d // GROUP_SIZE
        slots = (
            kv_cache.index_select(0, block_ids)
            .permute(0, 2, 1, 3)
            .reshape(-1, self.num_kv_heads, kv_cache.size(3))[:prefix_len]
        )
        k = ultraquant_dequant(
            UltraQuantEncoded(slots[..., : d // 2], slots[..., d // 2 : d // 2 + g]), d
        )
        v = ultraquant_dequant(
            UltraQuantEncoded(slots[..., d // 2 + g : d + g], slots[..., d + g :]), d
        )
        return k.to(torch.bfloat16), v.to(torch.bfloat16)

    def _continuation(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        ctx_len: int,
    ) -> None:
        """Attend a chunk whose prefix is already in the 4-bit cache.

        The store ran before this call, so a short chunk is just its own rows
        of a decode batch, reading the cache in place; only a long one earns
        back the cost of dequantizing the prefix into a bf16 page.
        """
        q_len = query.size(0)
        if q_len * _CONTINUATION_DECODE_RATIO <= ctx_len:
            self._decode(
                query,
                kv_cache,
                block_table.expand(q_len, -1),
                torch.arange(ctx_len - q_len + 1, ctx_len + 1, dtype=torch.int32),
                output,
            )
            return

        # Rotating the query and this chunk's K costs less than un-rotating the
        # cached prefix and leaves the scores unchanged, the Hadamard being
        # orthonormal.
        block_size = kv_cache.size(2)
        prefix_len = ctx_len - q_len
        pre_blocks = (prefix_len + block_size - 1) // block_size
        k_pre, v_pre = self._dequant_prefix(
            kv_cache, block_table[:pre_blocks].to(torch.int64), prefix_len
        )

        rot = self._rotation_like(key)
        k_all = torch.cat([k_pre.to(key.dtype), key @ rot])
        v_all = torch.cat([v_pre.to(value.dtype), value])

        scratch = self._scratch_cache(
            (ctx_len + block_size - 1) // block_size, block_size
        )
        kc, vc = self._split_scratch(scratch)
        ops.cpu_attn_reshape_and_cache(
            k_all,
            v_all,
            kc,
            vc,
            torch.arange(ctx_len, dtype=torch.int64),
            layer.isa,  # type: ignore
        )
        self._attend(
            layer,
            query @ self._rotation_like(query),
            kc,
            vc,
            output,
            torch.tensor([0, q_len], dtype=torch.int32),
            torch.tensor([ctx_len], dtype=torch.int32),
            torch.arange(scratch.size(0), dtype=torch.int32)[None, :],
        )

    def _decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # The format feeds Q through the rotation and FP8 E4M3 before the dot,
        # so the kernel sees exactly what the reference decoder does. Three
        # bf16 ops rather than four fp32 ones: the kernel dots in bf16 anyway,
        # and FP8 E4M3 keeps only 3 mantissa bits regardless.
        q = (query @ self.rotation_t).to(torch.float8_e4m3fn)
        q = self.fp8_widen[q.view(torch.uint8).int()]

        out = torch.empty(q.shape, dtype=torch.float32)
        ops.ultraquant_decode_attention(
            q, kv_cache, block_table, seq_lens, out, self.scale, self.sinks
        )
        output.copy_(out)
