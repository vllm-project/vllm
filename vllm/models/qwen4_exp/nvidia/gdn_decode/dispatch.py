# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata predicates; never inspect device tensor values during dispatch."""

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def is_pure_decode(attn_metadata: object, batch_size: int, max_batch_size: int) -> bool:
    """Require one non-speculative token per request, allowing graph padding.

    GDN counts distinguish a one-token prefill from a decode. Its
    num_actual_tokens includes FULL-graph token padding, while num_decodes and
    num_decode_tokens count real requests/tokens. Unknown/profiling metadata
    must use the ordinary GDN output projection.
    """
    if not 1 <= batch_size <= max_batch_size or not isinstance(attn_metadata, dict):
        return False
    found = False
    real_tokens = None
    for metadata in attn_metadata.values():
        if not isinstance(metadata, GDNAttentionMetadata):
            continue
        found = True
        if not (
            metadata.num_prefills == 0
            and metadata.num_prefill_tokens == 0
            and metadata.num_spec_decodes == 0
            and metadata.num_spec_decode_tokens == 0
            and metadata.spec_sequence_masks is None
            and 0 < metadata.num_decodes == metadata.num_decode_tokens <= batch_size
            and metadata.num_actual_tokens == batch_size
        ):
            return False
        if real_tokens is not None and real_tokens != metadata.num_decode_tokens:
            return False
        real_tokens = metadata.num_decode_tokens
    return found


def is_gdn_decode_context(metadata: object, batch: int, runtime_mode: object) -> bool:
    return runtime_mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL) and is_pure_decode(
        metadata, batch, 2
    )
