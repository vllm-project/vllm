# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main-attention metadata shared by the QSA platform implementations."""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass(frozen=True)
class QSAPrefillLayout:
    """The trailing run of prefilling requests of a step, with the exact
    host lengths an op that plans on the host needs.

    Requests ``first_req`` .. the last, rows ``first_row`` .. the last; the
    requests before them (decodes, whose lengths the host only bounds) are
    not described. ``prefix_lens`` count the tokens already in the cache.
    """

    first_req: int
    first_row: int
    query_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    # Device tensors derived from this layout and a block table, shared by
    # the layers of one step.
    cache: dict[Any, Any] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @property
    def num_rows(self) -> int:
        return sum(self.query_lens)


def host_prefill_layout(
    common_attn_metadata: CommonAttentionMetadata,
) -> QSAPrefillLayout | None:
    """The step's trailing prefilling requests from host metadata only (no
    device read), or None when the step has none or the host cannot give
    their lengths exactly.

    ``seq_lens_cpu_upper_bound`` is exact for prefilling rows (not for decode
    rows under async spec decode), and the host query lengths of prefilling
    rows are exact as well (``supports_device_cpu_query_lens_mismatch``), so
    the run of prefilling requests at the end of the batch (decodes come
    first) is described exactly. A step whose rows run past its requests
    (padding) is not described.
    """
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    is_prefilling = common_attn_metadata.is_prefilling
    if seq_lens is None or is_prefilling is None:
        return None
    num_reqs = common_attn_metadata.num_reqs
    starts = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].tolist()
    if starts[-1] != common_attn_metadata.num_actual_tokens:
        return None
    prefilling = is_prefilling[:num_reqs].tolist()
    first = num_reqs
    while first and prefilling[first - 1]:
        first -= 1
    if first == num_reqs:
        return None
    query_lens = tuple(b - a for a, b in zip(starts[first:-1], starts[first + 1 :]))
    prefix_lens = tuple(
        seq - query for seq, query in zip(seq_lens[first:num_reqs].tolist(), query_lens)
    )
    if min(query_lens) <= 0 or min(prefix_lens) < 0:
        return None
    return QSAPrefillLayout(first, starts[first], query_lens, prefix_lens)


@dataclass
class Qwen4ExpQSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # Host lengths of the step's prefilling requests; set only by a platform
    # builder whose attention op plans on them.
    prefill_layout: QSAPrefillLayout | None = None


class Qwen4ExpQSAMetadataBuilder(AttentionMetadataBuilder[Qwen4ExpQSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table = True
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Qwen4ExpQSAMetadata:
        """Keep cache addresses shared with the current runner batch."""
        return Qwen4ExpQSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def update_block_table(
        self,
        metadata: Qwen4ExpQSAMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Qwen4ExpQSAMetadata:
        """Reuse batch bounds while replacing cache-group-specific addresses."""
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)

    def update_draft_decode_metadata(self, metadata: Qwen4ExpQSAMetadata) -> None:
        """Decode batch bounds stay fixed and addresses reference runner buffers."""
        pass
