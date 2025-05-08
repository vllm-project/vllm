# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm._ipex_ops import ipex_ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.utils.fa_utils import get_flash_attn_version_xpu
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata, FlashAttentionMetadataBuilder,
    make_local_attention_virtual_batches)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.xpu_model_runner import XPUModelRunner


@dataclass
class IPEXAttentionMetadata(FlashAttentionMetadata):
    seq_start_loc: torch.Tensor = torch.tensor([0], dtype=torch.int64)


class IPEXAttentionMetadataBuilder(FlashAttentionMetadataBuilder):

    def __init__(self, runner: "XPUModelRunner"):
        model_config = runner.model_config

        self.runner: XPUModelRunner = runner
        self.aot_schedule = (get_flash_attn_version_xpu() == 3)
        self.num_heads = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.headdim = model_config.get_head_size()
        self.page_size = self.runner.block_size

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        # XPU flush_attn V2 require this.
        seq_start_loc_cpu = self.runner.seq_start_loc_cpu[:num_reqs + 1]
        seq_start_loc = seq_start_loc_cpu.to(self.runner.device,
                                             non_blocking=True)
        query_start_loc_cpu = self.runner.query_start_loc_cpu[:num_reqs + 1]
        query_start_loc = query_start_loc_cpu.to(self.runner.device,
                                                 non_blocking=True)
        seq_lens_cpu = self.runner.seq_lens_cpu[:num_reqs]
        seq_lens = seq_lens_cpu.to(self.runner.device, non_blocking=True)
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            return None

        # for local attention
        local_attn_metadata = None
        if self.runner.attention_chunk_size is not None:
            seqlens_q_local_np, virt_q_cu_seqlens_np, virt_k_seqlens_np, \
                virt_block_table = make_local_attention_virtual_batches(
                    self.runner.attention_chunk_size,
                    self.runner.query_start_loc_np[:num_reqs + 1],
                    self.runner.seq_lens_np[:num_reqs],
                    block_table,
                    self.runner.block_size,
                )
            local_query_start_loc = torch.from_numpy(virt_q_cu_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_seqused_k = torch.from_numpy(virt_k_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_max_query_len = seqlens_q_local_np.max()
            local_max_seq_len = virt_k_seqlens_np.max()
            local_scheduler_metadata = schedule(
                batch_size=local_query_start_loc.shape[0] - 1,
                cu_query_lens=local_query_start_loc,
                max_query_len=local_max_query_len,
                seqlens=local_seqused_k,
                max_seq_len=local_max_seq_len,
                causal=True)

            local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
                local_query_start_loc=local_query_start_loc,
                local_seqused_k=local_seqused_k,
                local_block_table=virt_block_table,
                local_max_query_len=local_max_query_len,
                local_max_seq_len=local_max_seq_len,
                local_scheduler_metadata=local_scheduler_metadata,
            )

        # FIXME(kunshang): support cascade attn
        # use_cascade = common_prefix_len > 0
        use_cascade = False

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.runner.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.runner.device)
            suffix_kv_lens = (self.runner.seq_lens_np[:num_reqs] -
                              common_prefix_len)
            suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(
                self.runner.device)
            prefix_scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False)
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=suffix_kv_lens,
                                          max_seq_len=max_seq_len -
                                          common_prefix_len,
                                          causal=True)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=seq_lens,
                                          max_seq_len=max_seq_len,
                                          causal=True)

        attn_metadata = IPEXAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            local_attn_metadata=local_attn_metadata,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_start_loc=seq_start_loc,
        )
        return attn_metadata


class IPEXAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "IPEX_V1"

    @staticmethod
    def get_impl_cls() -> type["IPEXAttentionImpl"]:
        return IPEXAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return IPEXAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["IPEXAttentionMetadataBuilder"]:
        return IPEXAttentionMetadataBuilder


class IPEXAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = IPEXAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "IpexAttnBackendImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: IPEXAttentionBackend,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with IPEXAttention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output

        # NOTE(woosuk): IPEXAttention does not support FP8 KV cache.
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0, (
            "key/v_scale is not supported in IPEXAttention.")

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_heads = self.num_heads
        head_size = self.head_size
        num_kv_heads = self.num_kv_heads
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)

        # Reshape the input keys and values and store them in the cache.
        key_cache, value_cache = kv_cache.unbind(0)

        ipex_ops.reshape_and_cache_flash(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        ipex_ops.chunked_prefill(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            None,
            attn_metadata.block_table,
            self.alibi_slopes,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
            0.0,
            self.scale,
            False,
            True,
            False,
            None,
        )
        return output
