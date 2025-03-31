# SPDX-License-Identifier: Apache-2.0
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.request import Request, RequestStatus


def check_stop(request: Request, max_model_len: int) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False


def force_recompute_last_token_for_full_hit(
    kv_cache_config: KVCacheConfig, num_computed_tokens: int,
    num_new_tokens: int, computed_blocks: list[KVCacheBlock]
) -> tuple[int, int, list[KVCacheBlock]]:
    """
    Adjust the number of computed tokens and new tokens to force recompute the
    last token for full prefix cache hit.

    Args:
        kv_cache_config: The kv cache config.
        num_computed_tokens: The number of computed tokens.
        num_new_tokens: The number of new tokens.
        computed_blocks: The computed blocks.
    
    Returns:
        A tuple containing the updated num_computed_tokens, num_new_tokens, and
        computed_blocks.
    """
    kv_cache_groups = kv_cache_config.kv_cache_groups
    if len(kv_cache_groups) == 1 and isinstance(
            kv_cache_groups[0].kv_cache_spec, FullAttentionSpec):
        # Force to recompute the last block instead of the last token.
        # We have to re-compute an entire block because allocate_slots()
        # assumes num_computed_tokens is always a multiple of the block size.
        # This limitation can potentially be removed in the future to slightly
        # improve the performance.
        block_size = kv_cache_groups[0].kv_cache_spec.block_size
        num_computed_tokens -= block_size
        num_new_tokens += block_size
        computed_blocks.pop()
    else:
        # For layers other than full attention, the recomputation of the last
        # token may require a block that not cached. For example, request [ABCD]
        # with block_size 1 and sliding_window_size 3, the cache may be [AxCD] (
        # x means cache miss). The recomputation of D requires B, which is not
        # cached, and the optimal choice is to recompute [BCD].
        # The logic is too complicated to implement here, so we just recompute
        # all tokens in the request as a temporary solution.
        num_new_tokens += num_computed_tokens
        num_computed_tokens = 0
        computed_blocks = []

    return num_computed_tokens, num_new_tokens, computed_blocks
