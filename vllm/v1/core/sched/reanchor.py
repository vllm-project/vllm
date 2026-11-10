# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position shifts of streaming sessions for --enable-unbounded-streaming-input."""

from dataclasses import replace

from vllm.config import SchedulerConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import round_down
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.request import Request

logger = init_logger(__name__)


class SessionReanchor:
    """Keeps streaming sessions under max_model_len by shifting their positions
    down. The worker re-rotates the cached keys."""

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
        encoder_cache_manager: EncoderCacheManager,
        max_model_len: int,
        block_size: int,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.encoder_cache_manager = encoder_cache_manager
        self.max_model_len = max_model_len
        self.block_size = block_size

        windows = []
        for mgr in kv_cache_manager.coordinator.single_type_managers:
            # The worker has already checked every KV cache group.
            assert isinstance(mgr, SlidingWindowManager)
            windows.append(mgr.sliding_window + mgr.extra_retained_tokens)
        self.window = max(windows)

        margin = scheduler_config.unbounded_streaming_input_margin_tokens
        if margin is None:
            # Half for the tokens generated per input chunk, half between shifts.
            margin = max((max_model_len - self.window) // 2, 0)
            minimum = self.window + 2 * block_size - 1
        else:
            minimum = self.window + margin + block_size
        # A shift drops whole blocks, so at least one block must fit between the
        # window and the trigger.
        if max_model_len < minimum:
            raise ValueError(
                f"--enable-unbounded-streaming-input needs max_model_len >= {minimum} "
                f"(sliding window {self.window}, KV cache block {block_size}), got "
                f"{max_model_len}."
            )
        self.trigger = max_model_len - margin

    def shift(self, request: Request) -> bool:
        """Shift a session's positions down by d, dropping its d oldest tokens,
        which are out of the sliding window. Returns False if nothing is dropped."""
        processed = request.num_computed_tokens
        has_keys = processed > 0
        if not has_keys:
            # Preempted: only the re-prefill from position 0 has to fit.
            if request.num_tokens < self.max_model_len:
                return False
            processed = request.num_tokens - 1
        # A multiple of the scheduler block size (the LCM of the group block
        # sizes), so every group drops whole blocks.
        d = round_down(processed - self.window, self.block_size)
        if d <= 0:
            return False

        if has_keys:
            # A step still in flight after a stop runs at the frontier or past it,
            # so it only reads what the next step reads.
            self.kv_cache_manager.remove_skipped_blocks(request.request_id, processed)
            self.kv_cache_manager.drop_leading_blocks(request.request_id, d)
            request.num_computed_tokens -= d
            request.pending_reanchor_d += d

        del request._all_token_ids[:d]
        # Resuming folded the outputs into the prompt.
        assert request.prompt_token_ids is not None and not request._output_token_ids
        del request.prompt_token_ids[:d]
        request.num_prompt_tokens -= d

        # Drop the mm features that end before d and shift the others.
        n_drop = 0
        for mm_feature in request.mm_features:
            pos = mm_feature.mm_position
            if pos.offset + pos.length > d:
                break
            n_drop += 1
        if n_drop:
            self.encoder_cache_manager.drop_leading_inputs(request, n_drop)
        request.mm_features = [
            replace(
                f, mm_position=replace(f.mm_position, offset=f.mm_position.offset - d)
            )
            for f in request.mm_features[n_drop:]
        ]

        logger.info(
            "Re-anchored %s: positions shifted down by %d.", request.request_id, d
        )
        return True
