# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched GPU exact no-repeat n-gram logits processor."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


MAX_WHITELIST_TOKEN_IDS = 1024


class NoRepeatNGramState(LogitsProcessor):
    """Per-request configuration for output-only no-repeat n-gram masking.

    Token history already lives in :class:`RequestState`; retaining a second
    Python or GPU index costs more than scanning it in parallel for the long
    n-grams used by generative OCR models. The only additional persistent
    state is therefore the constraint configuration for each request slot.
    """

    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "no-repeat n-gram masking does not support speculative decoding"
            )
        self.req_states = req_states
        self.max_model_len = req_states.all_token_ids.gpu.shape[1]
        self.ngram_sizes = UvaBackedTensor(req_states.max_num_reqs, dtype=torch.int32)
        self.window_sizes = UvaBackedTensor(req_states.max_num_reqs, dtype=torch.int32)
        self.whitelist_lens = UvaBackedTensor(
            req_states.max_num_reqs, dtype=torch.int32
        )
        self.whitelist_token_ids = StagedWriteTensor(
            (req_states.max_num_reqs, MAX_WHITELIST_TOKEN_IDS),
            dtype=torch.int32,
            device=req_states.device,
        )
        self.ngram_sizes.np.fill(0)
        self.window_sizes.np.fill(0)
        self.whitelist_lens.np.fill(0)

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra_args = sampling_params.extra_args or {}
        native_size = extra_args.get("no_repeat_ngram_size")
        compatible_size = extra_args.get("ngram_size")
        if native_size is not None and compatible_size is not None:
            raise ValueError("Specify only one of no_repeat_ngram_size and ngram_size")
        raw_size = native_size if native_size is not None else compatible_size
        if raw_size is None:
            return
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ValueError("ngram size must be an integer")
        if raw_size <= 0:
            raise ValueError("ngram size must be positive")
        window_size = extra_args.get("window_size", 100)
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise ValueError("window_size must be an integer")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        whitelist = extra_args.get("whitelist_token_ids")
        if whitelist is not None:
            if isinstance(whitelist, (str, bytes)) or not isinstance(
                whitelist, Iterable
            ):
                raise ValueError("whitelist_token_ids must be an iterable of integers")
            whitelist = list(whitelist)
            if any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in whitelist
            ):
                raise ValueError("whitelist_token_ids must contain only integers")
            if len(set(whitelist)) > MAX_WHITELIST_TOKEN_IDS:
                raise ValueError(
                    f"whitelist_token_ids supports at most "
                    f"{MAX_WHITELIST_TOKEN_IDS} unique IDs"
                )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        extra_args = sampling_params.extra_args or {}
        native_size = extra_args.get("no_repeat_ngram_size")
        compatible_size = extra_args.get("ngram_size")
        raw_size = native_size if native_size is not None else compatible_size
        if raw_size is None:
            self.ngram_sizes.np[req_idx] = 0
            self.window_sizes.np[req_idx] = 0
            self.whitelist_lens.np[req_idx] = 0
            return False
        # The legacy OCR processor treats ngram_size=1 as disabled because its
        # prefix slice covers the whole output. Preserve that behavior for the
        # compatibility alias while canonical no_repeat_ngram_size=1 keeps the
        # standard unigram constraint.
        if native_size is None and compatible_size == 1:
            self.ngram_sizes.np[req_idx] = 0
            self.window_sizes.np[req_idx] = 0
            self.whitelist_lens.np[req_idx] = 0
            return False
        if raw_size > self.max_model_len:
            raise ValueError("ngram size cannot exceed the configured max model length")
        window_size = extra_args.get("window_size", 100)
        whitelist = sorted(set(extra_args.get("whitelist_token_ids") or []))
        if whitelist and (
            whitelist[0] < 0 or whitelist[-1] >= self.req_states.vocab_size
        ):
            raise ValueError("whitelist_token_ids contains an out-of-vocabulary ID")
        self.ngram_sizes.np[req_idx] = raw_size
        self.window_sizes.np[req_idx] = window_size
        self.whitelist_lens.np[req_idx] = len(whitelist)
        if whitelist:
            self.whitelist_token_ids.stage_write(req_idx, 0, whitelist)
        return True

    def apply_staged_writes(self) -> None:
        self.ngram_sizes.copy_to_uva()
        self.window_sizes.copy_to_uva()
        self.whitelist_lens.copy_to_uva()
        self.whitelist_token_ids.apply_write()

    def apply(
        self,
        logits: torch.Tensor,
        ctx: LogitsContext,
    ) -> torch.Tensor:
        if not (self.ngram_sizes.np[ctx.idx_mapping_np] >= 1).any():
            return logits
        apply_no_repeat_ngram(
            logits,
            ctx.expanded_idx_mapping,
            self.req_states.all_token_ids.gpu,
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            self.ngram_sizes.gpu,
            self.window_sizes.gpu,
            self.whitelist_token_ids.gpu,
            self.whitelist_lens.gpu,
        )
        return logits


@triton.jit
def _no_repeat_ngram_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    ngram_sizes_ptr,
    window_sizes_ptr,
    whitelist_ids_ptr,
    whitelist_ids_stride,
    whitelist_lens_ptr,
    vocab_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    req_idx = tl.load(expanded_idx_mapping_ptr + row)
    ngram_size = tl.load(ngram_sizes_ptr + req_idx)
    prompt_len = tl.load(prompt_len_ptr + req_idx)
    output_len = tl.load(total_len_ptr + req_idx) - prompt_len
    window_size = tl.load(window_sizes_ptr + req_idx)
    if ngram_size < 1 or output_len < ngram_size:
        return

    starts = tl.program_id(1) * block_size + tl.arange(0, block_size)
    valid = (starts <= output_len - ngram_size) & (
        starts >= tl.maximum(0, output_len - window_size)
    )
    output = all_token_ids_ptr + req_idx * all_token_ids_stride + prompt_len
    matches = valid
    if ngram_size > 1:
        current_prefix = output_len - ngram_size + 1
        matches &= tl.load(output + starts, mask=valid, other=-1) == tl.load(
            output + current_prefix
        )
        for offset in range(1, ngram_size - 1):
            matches &= tl.load(
                output + starts + offset, mask=matches, other=-1
            ) == tl.load(output + current_prefix + offset)

    banned = tl.load(output + starts + ngram_size - 1, mask=matches, other=0)
    whitelist_len = tl.load(whitelist_lens_ptr + req_idx)
    whitelisted = tl.zeros((block_size,), dtype=tl.int1)
    whitelist_idx = 0
    while whitelist_idx < whitelist_len:
        whitelist_token = tl.load(
            whitelist_ids_ptr + req_idx * whitelist_ids_stride + whitelist_idx
        )
        whitelisted |= banned == whitelist_token
        whitelist_idx += 1
    tl.atomic_xchg(
        logits_ptr + row * logits_stride + banned,
        -float("inf"),
        mask=matches & ~whitelisted & (banned >= 0) & (banned < vocab_size),
        sem="relaxed",
    )


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    ngram_sizes: torch.Tensor,
    window_sizes: torch.Tensor,
    whitelist_token_ids: torch.Tensor,
    whitelist_lens: torch.Tensor,
) -> None:
    _no_repeat_ngram_kernel[
        (logits.shape[0], triton.cdiv(all_token_ids.shape[1], 256))
    ](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_lens,
        total_lens,
        ngram_sizes,
        window_sizes,
        whitelist_token_ids,
        whitelist_token_ids.stride(0),
        whitelist_lens,
        logits.shape[1],
        256,
    )
