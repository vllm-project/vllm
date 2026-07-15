# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm import _custom_ops as ops
from vllm.distributed.parallel_state import get_dcp_group
from vllm.model_executor.layers.attention.pcp import pcp_dcp_a2a_lse_reduce
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_ar
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


@dataclass
class MLAPCPChunkedContextMetadata:
    seq_lens: list[list[int]]
    cu_seq_lens: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]


def build_pcp_chunked_context_metadata(
    context_lens_cpu: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    dcp_local_block_size: int,
    local_chunk_starts: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> MLAPCPChunkedContextMetadata:
    local_context_lens = get_dcp_local_seq_lens(
        context_lens_cpu,
        dcp_world_size,
        dcp_rank,
        dcp_local_block_size,
    )
    local_chunk_seq_lens = (
        torch.min(
            local_context_lens.unsqueeze(0),
            local_chunk_starts + chunk_size,
        )
        - local_chunk_starts
    ).clamp(min=0)
    local_cu_seq_lens = torch.zeros(
        local_chunk_seq_lens.shape[0],
        local_chunk_seq_lens.shape[1] + 1,
        dtype=torch.int32,
        pin_memory=True,
    )
    torch.cumsum(
        local_chunk_seq_lens,
        dim=1,
        out=local_cu_seq_lens[:, 1:],
        dtype=torch.int32,
    )
    return MLAPCPChunkedContextMetadata(
        seq_lens=local_chunk_seq_lens.tolist(),
        cu_seq_lens=local_cu_seq_lens.to(device, non_blocking=True),
        seq_tot=local_cu_seq_lens[:, -1].tolist(),
        max_seq_lens=local_chunk_seq_lens.max(dim=1).values.tolist(),
    )


class MLAPCPImplMixin:
    num_heads: int
    v_head_dim: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    kv_b_proj: Any
    lse_base_on_e: bool
    dcp_tp_size: int
    dcp_world_size: int
    dcp_a2a: bool

    if TYPE_CHECKING:

        def _concat_k_nope_k_pe(
            self, k_nope: torch.Tensor, k_pe: torch.Tensor
        ) -> torch.Tensor: ...

        def _context_parallel_compute_prefill_context(
            self,
            q: torch.Tensor,
            kv_c_and_k_pe_cache: torch.Tensor,
            attn_metadata: Any,
            k_scale: torch.Tensor,
            dcp_world_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def _empty_context_chunk_output(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            q.new_zeros((q.shape[0], self.num_heads, self.v_head_dim)),
            torch.full(
                (self.num_heads, q.shape[0]),
                -float("inf"),
                dtype=torch.float32,
                device=q.device,
            ),
        )

    @staticmethod
    def _unpad_dcp_local_kvcache(
        local_kvcache: torch.Tensor,
        padded_seq_lens: list[int],
        local_seq_lens: list[int],
        local_toks: int,
    ) -> torch.Tensor:
        if local_toks == local_kvcache.shape[0]:
            return local_kvcache

        kv_segments: list[torch.Tensor] = []
        src_token_idx = 0
        for padded_len, local_len in zip(padded_seq_lens, local_seq_lens):
            if local_len:
                kv_segments.append(
                    local_kvcache[src_token_idx : src_token_idx + local_len]
                )
            src_token_idx += padded_len

        local_kvcache = torch.cat(kv_segments, dim=0)
        assert local_kvcache.shape[0] == local_toks
        return local_kvcache

    def _project_context_kv(
        self, local_kvcache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_c_normed = local_kvcache[..., : self.kv_lora_rank]
        k_pe = local_kvcache[..., self.kv_lora_rank :].unsqueeze(1)
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return self._concat_k_nope_k_pe(k_nope, k_pe), v

    def _compute_pcp_dcp_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: Any,
        use_dcp_a2a: bool = False,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        assert prefill_metadata.chunked_context is not None
        chunked_context = prefill_metadata.chunked_context
        assert chunked_context.padded_local_chunk_seq_lens is not None
        assert chunked_context.padded_local_cu_seq_lens is not None
        assert chunked_context.pcp_chunk_metadata is not None
        pcp_chunk_metadata = chunked_context.pcp_chunk_metadata

        local_chunked_context = replace(
            chunked_context,
            cu_seq_lens=pcp_chunk_metadata.cu_seq_lens,
            seq_tot=pcp_chunk_metadata.seq_tot,
            max_seq_lens=pcp_chunk_metadata.max_seq_lens,
        )
        local_prefill_metadata = replace(
            prefill_metadata, chunked_context=local_chunked_context
        )
        prefill_metadata.prefill_backend.prepare_metadata(local_prefill_metadata)

        output = None
        merge_output = None
        workspace = chunked_context.workspace
        for i, padded_toks in enumerate(chunked_context.seq_tot):
            local_toks = pcp_chunk_metadata.seq_tot[i]
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=chunked_context.padded_local_cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=chunked_context.starts[i],
            )

            if local_toks == 0:
                attn_output, attn_softmax_lse = self._empty_context_chunk_output(q)
            else:
                local_kvcache = self._unpad_dcp_local_kvcache(
                    workspace[:padded_toks],
                    chunked_context.padded_local_chunk_seq_lens[i],
                    pcp_chunk_metadata.seq_lens[i],
                    local_toks,
                )
                k, v = self._project_context_kv(local_kvcache)
                attn_output, attn_softmax_lse = (
                    prefill_metadata.prefill_backend.run_prefill_context_chunk(
                        chunk_idx=i,
                        q=q,
                        k=k,
                        v=v,
                    )
                )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                if merge_output is None:
                    merge_output = torch.empty_like(output)
                    merge_output_lse = torch.empty_like(output_lse)
                merge_attn_states(
                    output=merge_output,
                    output_lse=merge_output_lse,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output, merge_output = merge_output, output
                output_lse, merge_output_lse = merge_output_lse, output_lse

        assert output is not None
        if use_dcp_a2a:
            output, output_lse_t = pcp_dcp_a2a_lse_reduce(
                output,
                output_lse.transpose(0, 1),
                return_lse=True,
                is_lse_base_on_e=self.lse_base_on_e,
            )
        else:
            output, output_lse_t = cp_lse_ag_out_ar(
                output,
                output_lse.transpose(0, 1),
                get_dcp_group(),
                return_lse=True,
                is_lse_base_on_e=self.lse_base_on_e,
            )
        return output, output_lse_t.transpose(0, 1)

    def _compute_dcp_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: Any,
        k_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dcp_tp_size == 1:
            return self._compute_pcp_dcp_prefill_context(
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
                self.dcp_a2a,
            )
        return self._context_parallel_compute_prefill_context(
            q,
            kv_c_and_k_pe_cache,
            attn_metadata,
            k_scale=k_scale,
            dcp_world_size=self.dcp_world_size,
        )
