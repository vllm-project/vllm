# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference CPU sparse MLA backend for GLM5Next."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType


class Glm5NextCPUSparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_CPU_SPARSE_MLA"

    @staticmethod
    def get_metadata_cls() -> type["Glm5NextCPUSparseMetadata"]:
        return Glm5NextCPUSparseMetadata

    @staticmethod
    def get_builder_cls() -> type["Glm5NextCPUSparseMetadataBuilder"]:
        return Glm5NextCPUSparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["Glm5NextCPUSparseImpl"]:
        return Glm5NextCPUSparseImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class Glm5NextCPUIndexerBackend(DeepseekV32IndexerBackend):
    """Metadata-only backend for the GLM indexer cache."""

    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_CPU_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @staticmethod
    def get_builder_cls() -> type[  # type: ignore[override]
        "Glm5NextCPUIndexerMetadataBuilder"
    ]:
        return Glm5NextCPUIndexerMetadataBuilder


class Glm5NextCPUIndexerMetadataBuilder(AttentionMetadataBuilder):
    """Eager CPU metadata for the Python KeyPool indexer."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        query_lens = torch.diff(common_attn_metadata.query_start_loc_cpu).to(
            self.device
        )
        req_ids = torch.repeat_interleave(
            torch.arange(common_attn_metadata.num_reqs, device=self.device),
            query_lens,
        )
        seq_lens = common_attn_metadata.seq_lens.to(self.device)
        start_positions = seq_lens - query_lens
        positions = torch.repeat_interleave(start_positions, query_lens)
        request_starts = torch.repeat_interleave(
            common_attn_metadata.query_start_loc[:-1].to(self.device), query_lens
        )
        offsets = torch.arange(num_tokens, device=self.device) - request_starts
        token_seq_lens = positions + offsets + 1
        block_table = common_attn_metadata.block_table_tensor.to(self.device)
        ratio = self.kv_cache_spec.tokens_per_state
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
        if ratio > 1:
            kernel_block_size = self.kernel_block_size
            if (
                kernel_block_size is not None
                and self.kv_cache_spec.block_size != kernel_block_size
                and self.kv_cache_spec.block_size % kernel_block_size == 0
            ):
                factor = self.kv_cache_spec.block_size // kernel_block_size
                block_table = (block_table[:, ::factor] // factor).contiguous()
            pool_positions = (token_seq_lens - 1) // ratio
            valid = (token_seq_lens % ratio == 0) & (slot_mapping >= 0)
            compressed_slots = torch.full_like(slot_mapping, -1)
            block_size = self.kv_cache_spec.num_states
            physical = block_table[req_ids[valid], pool_positions[valid] // block_size]
            compressed_slots[valid] = (
                physical * block_size + pool_positions[valid] % block_size
            )
            slot_mapping = compressed_slots
            seq_lens = seq_lens // ratio
            token_seq_lens = token_seq_lens // ratio
        decode_block_table = block_table.index_select(0, req_ids)
        decode = DeepSeekV32IndexerDecodeMetadata(
            block_table=decode_block_table,
            seq_lens=token_seq_lens.to(torch.int32),
            decode_lens=torch.ones(num_tokens, dtype=torch.int32, device=self.device),
            requires_padding=False,
            schedule_metadata=torch.empty(
                (0, 2), dtype=torch.int32, device=self.device
            ),
            per_req_decode_lens=query_lens,
            decode_is_uniform=False,
            write_max_decode_len=int(query_lens.max().item())
            if query_lens.numel()
            else 0,
        )
        return DeepseekV32IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len // ratio,
            slot_mapping=slot_mapping,
            num_decodes=common_attn_metadata.num_reqs,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=decode,
            prefill=None,
        )


@dataclass(kw_only=True)
class Glm5NextCPUSparseMetadata(SparseMLACommonMetadata):
    pass


class Glm5NextCPUSparseMetadataBuilder(SparseMLACommonMetadataBuilder):
    """Metadata builder that routes every row through CPU sparse MQA."""

    metadata_cls = Glm5NextCPUSparseMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.req_id_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> Glm5NextCPUSparseMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        starts = common_attn_metadata.query_start_loc_cpu
        req_ids = torch.repeat_interleave(
            torch.arange(common_attn_metadata.num_reqs, device=self.device),
            torch.diff(starts.to(self.device)),
        )
        self.req_id_buffer[:num_tokens].copy_(req_ids.to(torch.int32))
        return Glm5NextCPUSparseMetadata(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=self.req_id_buffer[:num_tokens],
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            num_decodes=common_attn_metadata.num_reqs,
            num_prefills=0,
            num_decode_tokens=num_tokens,
            decode_max_query_len=common_attn_metadata.max_query_len,
            prefill_max_seq_len=0,
            prefill=None,
            cp_kv_cache_interleave_size=1,
        )


class Glm5NextCPUSparseImpl(SparseMLACommonImpl[Glm5NextCPUSparseMetadata]):
    can_return_lse_for_decode = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype: str,
        logits_soft_cap,
        attn_type: str,
        kv_sharing_target_layer_name,
        **mla_args,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("GLM5Next CPU sparse MLA is decoder-only")
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            q_lora_rank=mla_args["q_lora_rank"],
            kv_lora_rank=mla_args["kv_lora_rank"],
            qk_nope_head_dim=mla_args["qk_nope_head_dim"],
            qk_rope_head_dim=mla_args["qk_rope_head_dim"],
            qk_head_dim=mla_args["qk_head_dim"],
            v_head_dim=mla_args["v_head_dim"],
            kv_b_proj=mla_args["kv_b_proj"],
            indexer=mla_args.get("indexer"),
            topk_indices_buffer=mla_args.get("topk_indices_buffer"),
            index_group_builder=None,
        )
        self.q_pad_num_heads = None
        self.dcp_world_size = 1

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        width = kv_c_normed.shape[-1] + k_pe.shape[-1]
        if kv_cache.shape[-1] != width:
            raise ValueError(
                f"GLM5Next CPU sparse MLA cache width must be {width}, "
                f"got {kv_cache.shape[-1]}"
            )
        flat = kv_cache.view(-1, width)
        slots = slot_mapping.flatten().to(torch.long)
        valid = slots >= 0
        values = torch.cat((kv_c_normed, k_pe.squeeze(1)), dim=-1)
        flat[slots[valid]] = values[valid].to(flat.dtype)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: Glm5NextCPUSparseMetadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if q.shape[-1] != self.kv_lora_rank:
            raise NotImplementedError(
                "GLM5Next CPU sparse MLA currently requires qk_rope_head_dim=0"
            )
        assert self.topk_indices_buffer is not None
        topk = self.topk_indices_buffer[: q.shape[0]]
        req_ids = attn_metadata.req_id_per_token[: q.shape[0]]
        cache = kv_cache.view(-1, kv_cache.shape[-1])
        latent_cache = cache[:, : self.kv_lora_rank]
        block_size = attn_metadata.block_size
        output = q.new_zeros((q.shape[0], self.num_heads, self.kv_lora_rank))
        for row in range(q.shape[0]):
            req = int(req_ids[row])
            local = topk[row]
            valid = local >= 0
            local = local[valid].to(torch.long)
            if local.numel() == 0:
                continue
            blocks = torch.div(local, block_size, rounding_mode="floor")
            offsets = local.remainder(block_size)
            physical = attn_metadata.block_table[req, blocks]
            slots = physical * block_size + offsets
            latent = latent_cache[slots]
            query = q[row]
            logits = torch.einsum("nd,sd->ns", query, latent)
            probs = torch.softmax(logits * self.scale, dim=-1)
            output[row] = torch.einsum("hs,sd->hd", probs, latent)
        return output, None

    def forward_mha(self, *args, **kwargs) -> None:
        raise NotImplementedError("GLM5Next CPU sparse MLA routes prefill through MQA")
