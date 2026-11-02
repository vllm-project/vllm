# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA region summaries as their own KV cache group, and the per-step
metadata the CSA op reads from it."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.models.step5.common.config import get_step5_sparse_config
from vllm.models.step5.common.ops.prefill_indexer_metadata import (
    PrefillIndexerMetadata,
    prepare_prefill_indexer,
)
from vllm.models.step5.common.ops.summary_update import build_prefill_groups
from vllm.models.step5.common.sparse_summary_cache import (
    Step5SparseSummaryCacheLayer,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class Step5CSAMetadata(AttentionMetadata):
    # Block table and token slots of the summary group.
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    token_req: torch.Tensor
    positions: torch.Tensor
    num_regions: int = 0
    # None for uniform decode and MTP verify batches.
    prefill_groups: torch.Tensor | None = None
    # None when every earlier region fits in top-k.
    prefill_indexer: list[PrefillIndexerMetadata] | None = None


class Step5CSAMetadataBuilder(AttentionMetadataBuilder[Step5CSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sparse = get_step5_sparse_config(vllm_config.model_config.hf_text_config)
        assert sparse is not None
        self.region_size = sparse.region_block_size
        self.topk = sparse.topk
        spec = vllm_config.speculative_config
        self.max_uniform_query_len = 1 + (
            spec.num_speculative_tokens if spec is not None else 0
        )
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.token_req_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Step5CSAMetadata:
        del common_prefix_len, fast_build
        m = common_attn_metadata
        n = m.num_actual_tokens
        token_req = m.token_to_req_indices(self.token_req_buffer)
        if m.positions is None:
            positions = (
                m.compute_num_computed_tokens().index_select(0, token_req)
                + torch.arange(n, device=token_req.device)
                - m.query_start_loc.index_select(0, token_req)
            )
        else:
            positions = m.positions[:n]
        md = Step5CSAMetadata(
            block_table=m.block_table_tensor,
            slot_mapping=m.slot_mapping[:n],
            query_start_loc=m.query_start_loc,
            seq_lens=m.seq_lens,
            num_actual_tokens=n,
            num_reqs=m.num_reqs,
            max_query_len=m.max_query_len,
            token_req=token_req,
            positions=positions.to(torch.int64),
        )
        uniform = (
            n == m.num_reqs * m.max_query_len
            and m.max_query_len <= self.max_uniform_query_len
        )
        if n and not uniform:
            md.prefill_groups = build_prefill_groups(
                md.slot_mapping,
                md.positions,
                token_req,
                m.query_start_loc[: m.num_reqs + 1],
            )
            # Upper bound: scores past each row's history are masked.
            md.num_regions = cdiv(m.max_seq_len, self.region_size)
            if md.num_regions > self.topk:
                md.prefill_indexer = prepare_prefill_indexer(
                    token_req, md.positions, md.num_regions, self.topk, self.num_sms
                )
        return md


class Step5CSABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "STEP5_CSA_SUMMARY"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("CSA summaries have no attention impl")

    @staticmethod
    def get_builder_cls() -> type[Step5CSAMetadataBuilder]:
        return Step5CSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class Step5CSASummaryCacheLayer(Step5SparseSummaryCacheLayer):
    """FP8 region summaries in their own KV cache group: under the
    layer-outer ROCm KV layout they use a 384-token block, so the CSA op
    reads them through that group's metadata."""

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Step5CSABackend
