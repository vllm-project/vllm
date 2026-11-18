# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent gfx1200/gfx1201 Octave kernels and vLLM serving adapter."""

from functools import lru_cache
from math import prod

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionImpl,
    AttentionLayer,
)
from vllm.v1.attention.backends.rocm_octave_attn import (
    _DECODE_MAX_QUERY_LEN,
    RocmOctaveMetadata,
    RocmOctaveMetadataBuilder,
)
from vllm.v1.attention.ops import rocm_octave as sq
from vllm.v1.attention.ops.rocm_octave_rdna4 import OctaveGeometry, TunedHipOctave
from vllm.v1.attention.ops.rocm_octave_rdna4.splits import DEFAULT_WORKSPACE_BYTES
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)


@lru_cache
def _engine(head_size: int, rope_dim: int, arch: str) -> TunedHipOctave:
    return TunedHipOctave(head_size, rope_dim, arch)


def _reserve_workspace(config: VllmConfig, head_size: int) -> None:
    if not is_workspace_manager_initialized():
        return
    tokens = config.scheduler_config.max_num_batched_tokens
    heads = config.model_config.get_num_attention_heads(config.parallel_config)
    scratch_bytes = min(
        DEFAULT_WORKSPACE_BYTES, tokens * heads * 256 * (head_size + 2) * 4
    )
    current_workspace_manager().get_simultaneous(
        ((tokens, heads, head_size), torch.float32),
        ((scratch_bytes // 4,), torch.float32),
    )


class RocmOctaveRdna4MetadataBuilder(RocmOctaveMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        _reserve_workspace(vllm_config, kv_cache_spec.head_size)


class RocmOctaveRdna4AttentionImpl(AttentionImpl):
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
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        if (
            alibi_slopes is not None
            or sinks is not None
            or sliding_window is not None
            or logits_soft_cap
            or attn_type != "decoder"
        ):
            raise NotImplementedError(
                "ROCM_OCTAVE (RDNA4) supports dense causal decoder attention "
                "without sliding windows, ALiBi, sinks or soft caps"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads <= 0 or self.num_kv_heads <= 0 or num_heads % self.num_kv_heads:
            raise ValueError("ROCM_OCTAVE (RDNA4) requires an integer GQA ratio")
        self.head_size, self.scale = head_size, float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.config = get_current_vllm_config()
        rope_dim = sq.rope_dim_from_hf_config(
            self.config.model_config.hf_text_config, head_size
        )
        self.layout = OctaveGeometry(head_size, rope_dim).layout(kv_cache_dtype)
        sq.register_rope_dim(head_size, rope_dim)
        arch = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).gcnArchName.split(":")[0]
        if arch not in ("gfx1200", "gfx1201"):
            raise NotImplementedError(f"ROCM_OCTAVE (RDNA4) requires gfx1200/1: {arch}")
        self.engine = _engine(head_size, rope_dim, arch)
        self._sign_words: torch.Tensor | None = None
        logger.info_once(
            "ROCM_OCTAVE (RDNA4): native HIP %s, D=%d RoPE=%d, Q=%d KV=%d, %s",
            arch,
            head_size,
            rope_dim,
            num_heads,
            self.num_kv_heads,
            kv_cache_dtype,
        )

    def _signs(self, device: torch.device) -> torch.Tensor:
        if self._sign_words is None or self._sign_words.device != device:
            self._sign_words = sq.sign_bits(self.layout.head_size).to(device)
        return self._sign_words

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        n = slot_mapping.numel()
        if n == 0:
            return
        key = key[:n].view(n, self.num_kv_heads, self.head_size)
        value = value[:n].view(n, self.num_kv_heads, self.head_size)
        self.engine.store(
            key,
            value,
            kv_cache,
            slot_mapping,
            self._signs(key.device),
            self.kv_cache_dtype,
        )

    def _run(
        self,
        out,
        query,
        cache,
        table,
        mapping,
        lengths,
        *,
        key=None,
        value=None,
        context_capacity=None,
        max_query_len=None,
    ) -> None:
        if (
            key is not None
            and self.num_heads // self.num_kv_heads == 6
            and query.shape[0] >= 256
            and context_capacity is not None
            and context_capacity > 0
            and max_query_len is not None
            and max_query_len > 32
        ):
            capacity = min(context_capacity, table.shape[1] * cache.shape[2])
            blocks = (capacity + 127) // 128
            dense_shape = (
                table.shape[0] * blocks,
                128,
                self.num_kv_heads,
                self.head_size,
            )
            temporary = (
                (dense_shape, torch.float16),
                (dense_shape, torch.float16),
                (tuple(query.shape), torch.float16),
                ((table.shape[0], blocks), torch.int32),
            )
            temporary_bytes = sum(
                ((prod(shape) * dtype.itemsize + 255) // 256) * 256
                for shape, dtype in temporary
            )
            if temporary_bytes <= DEFAULT_WORKSPACE_BYTES:
                logger.info_once(
                    "ROCM_OCTAVE (RDNA4) prefill uses temporary FP16 KV and "
                    "Triton attention within the existing bounded workspace"
                )
                shapes = ((tuple(query.shape), torch.float32), *temporary)
                if is_workspace_manager_initialized():
                    buffers = current_workspace_manager().get_simultaneous(*shapes)
                else:
                    buffers = [
                        torch.empty(shape, dtype=dtype, device=query.device)
                        for shape, dtype in shapes
                    ]
                fp32, key_tmp, value_tmp, query_tmp, table_tmp = buffers
                self.engine.prefill_expanded(
                    query,
                    key,
                    value,
                    cache,
                    table,
                    mapping,
                    lengths,
                    self._signs(query.device),
                    self.kv_cache_dtype,
                    key_tmp=key_tmp,
                    value_tmp=value_tmp,
                    query_tmp=query_tmp,
                    table_tmp=table_tmp,
                    out=fp32,
                    max_query_len=max_query_len,
                    scale=self.scale,
                )
                out.copy_(fp32)
                out.record_stream(torch.cuda.current_stream(query.device))
                return
        splits = self.engine.suggest_splits(
            query,
            cache,
            table,
            self.kv_cache_dtype,
            prefill=key is not None,
            context_capacity=context_capacity if key is not None else None,
        )
        shape = (*query.shape[:2], splits, self.head_size + 2)
        scratch_shape = shape if splits > 1 else (0,)
        if is_workspace_manager_initialized():
            fp32, scratch = current_workspace_manager().get_simultaneous(
                (tuple(query.shape), torch.float32), (scratch_shape, torch.float32)
            )
        else:
            fp32 = torch.empty(query.shape, device=query.device, dtype=torch.float32)
            scratch = torch.empty(
                scratch_shape, device=query.device, dtype=torch.float32
            )
        signs = self._signs(query.device)
        args = (cache, table, mapping, lengths, signs, self.kv_cache_dtype)
        options = dict(
            scale=self.scale,
            splits=splits,
            out=fp32,
            scratch=scratch if splits > 1 else None,
        )
        if key is None:
            self.engine.decode(query, *args, **options)
        else:
            self.engine.prefill(query, key, value, *args, **options)
        out.copy_(fp32)
        stream = torch.cuda.current_stream(query.device)
        for tensor in (
            query,
            key,
            value,
            cache,
            table,
            mapping,
            lengths,
            fp32,
            scratch,
        ):
            if tensor is not None:
                tensor.record_stream(stream)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmOctaveMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("ROCM_OCTAVE (RDNA4) does not fuse output quant")
        if attn_metadata is None:
            _reserve_workspace(self.config, self.head_size)
            self._signs(query.device)
            return output.zero_()
        md = attn_metadata
        n = md.num_actual_tokens
        if n == 0:
            return output
        q = query[:n].view(n, self.num_heads, self.head_size)
        out = output[:n].view_as(q)
        ndt = n if md.max_query_len <= _DECODE_MAX_QUERY_LEN else md.num_decode_tokens
        if ndt:
            self._run(
                out[:ndt],
                q[:ndt],
                kv_cache,
                md.block_table,
                md.q_to_req[:ndt],
                md.q_to_klen[:ndt],
            )
        if ndt < n:
            nd = md.num_decodes
            reqs = md.query_start_loc.numel() - 1
            self._run(
                out[ndt:],
                q[ndt:],
                kv_cache,
                md.block_table[nd:reqs],
                md.query_start_loc[nd:] - ndt,
                md.seq_lens[nd:reqs],
                key=key[ndt:n].view(n - ndt, self.num_kv_heads, self.head_size),
                value=value[ndt:n].view(n - ndt, self.num_kv_heads, self.head_size),
                context_capacity=md.max_seq_len,
                max_query_len=md.max_query_len,
            )
        return output
