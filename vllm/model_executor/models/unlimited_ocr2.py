# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional page-count stopping guard for Unlimited-OCR-2."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

DEFAULT_MAX_PAGE_TOKENS = 20000


def _force_token(logits: torch.Tensor, token_id: int) -> None:
    keep = logits[token_id].clone()
    logits[:] = -float("inf")
    logits[token_id] = keep


class PageCountGuard:
    """Per-request page-count state machine over the generated tokens.

    CPU-friendly core of the processor: token bookkeeping is plain Python
    ints/lists, only the one-row logits edit touches torch.
    """

    def __init__(
        self,
        page_token_id: int,
        eos_token_id: int,
        max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
    ):
        self.page_token_id = page_token_id
        self.eos_token_id = eos_token_id
        self.max_page_tokens = max_page_tokens
        self.expected_pages = -1
        self.marker_cum = 0
        self.page_len = 0
        self.scanned = 0

    def seed_expected_pages(self, prompt_ids: list[int]) -> None:
        """Count the prompt's page markers; called once the prompt is known."""
        self.expected_pages = prompt_ids.count(self.page_token_id)

    def step(self, output_ids: list[int], logits: torch.Tensor) -> None:
        """Apply the guard to one row of logits, in place."""
        if self.expected_pages <= 0:
            return
        n = len(output_ids)
        if n > self.scanned:
            tail = output_ids[self.scanned :]
            num_markers = tail.count(self.page_token_id)
            if num_markers:
                self.marker_cum += num_markers
                self.page_len = tail[::-1].index(self.page_token_id)
            else:
                self.page_len += len(tail)
            self.scanned = n

        if self.marker_cum > self.expected_pages:
            _force_token(logits, self.eos_token_id)
        elif 0 < self.max_page_tokens <= self.page_len:
            force = (
                self.page_token_id
                if self.marker_cum < self.expected_pages
                else self.eos_token_id
            )
            _force_token(logits, force)
        elif self.marker_cum < self.expected_pages:
            logits[self.eos_token_id] = -float("inf")


@triton.jit
def _page_count_guard_kernel(
    Logits,
    Tokens,
    PromptLen,
    Positions,
    Slots,
    Params,
    State,
    LOGIT_STRIDE: tl.constexpr,
    LOGIT_COL_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(Slots + row).to(tl.int64)
    page = tl.load(Params + slot * 3)
    if page < 0:
        return
    eos = tl.load(Params + slot * 3 + 1)
    cap = tl.load(Params + slot * 3 + 2)
    prompt_len = tl.load(PromptLen + slot)
    length = tl.maximum(tl.load(Positions + row) + 1 - prompt_len, 0)
    expected = tl.load(State + slot * 4)
    seen = tl.load(State + slot * 4 + 1)
    markers = tl.load(State + slot * 4 + 2)
    last = tl.load(State + slot * 4 + 3)
    offsets = tl.arange(0, BLOCK)
    if expected < 0:
        expected = tl.full((), 0, tl.int64)
        for prompt_start in range(0, prompt_len, BLOCK):
            prompt_pos = prompt_start + offsets
            token = tl.load(
                Tokens + slot * TOKEN_STRIDE + prompt_pos,
                prompt_pos < prompt_len,
                other=-1,
            )
            expected += tl.sum(
                ((prompt_pos < prompt_len) & (token == page)).to(tl.int64), 0
            )
        tl.store(State + slot * 4, expected)
    if expected == 0:
        return
    if length < seen:
        seen = tl.full((), 0, tl.int64)
        markers = tl.full((), 0, tl.int64)
        last = tl.full((), -1, tl.int64)
    for output_start in range(seen, length, BLOCK):
        output_pos = output_start + offsets
        token = tl.load(
            Tokens + slot * TOKEN_STRIDE + prompt_len + output_pos,
            output_pos < length,
            other=-1,
        )
        found = (output_pos < length) & (token == page)
        markers += tl.sum(found.to(tl.int64), 0)
        last = tl.maximum(last, tl.max(tl.where(found, output_pos, -1), 0))
    tl.store(State + slot * 4 + 1, length)
    tl.store(State + slot * 4 + 2, markers)
    tl.store(State + slot * 4 + 3, last)
    force = tl.full((), -1, tl.int64)
    if markers > expected:
        force = eos
    elif (cap > 0) & (length - last - 1 >= cap):
        force = tl.where(markers < expected, page, eos)
    if force >= 0:
        for fill_start in range(0, VOCAB_SIZE, BLOCK):
            logit_pos = fill_start + offsets
            # Preserve the forced token's existing logit, including other masks.
            tl.store(
                Logits + row * LOGIT_STRIDE + logit_pos * LOGIT_COL_STRIDE,
                -float("inf"),
                (logit_pos < VOCAB_SIZE) & (logit_pos != force),
            )
    elif markers < expected:
        tl.store(Logits + row * LOGIT_STRIDE + eos * LOGIT_COL_STRIDE, -float("inf"))


class PageCountGuardLogitsProcessor(LogitsProcessor):
    """Page-count guard for Unlimited-OCR-2 multi-page parsing.

    In the multi-page layout (``"<PAGE><image>" * n_pages + "Multi page
    parsing."``) the model must emit exactly one ``<PAGE>`` marker per prompt
    page marker. This request-level guard enforces that contract, mirroring
    the reference ``_uocr_page_count_guard``:

    * while fewer ``<PAGE>`` markers than prompt pages have been generated,
      EOS is masked so a page cannot be terminated early;
    * once the page count is reached EOS is allowed, and generating further
      ``<PAGE>`` markers forces EOS;
    * each page is capped at ``max_page_tokens`` generated tokens -- hitting
      the cap forces ``<PAGE>`` (next page) or EOS (last page);
    * prompts without a ``<PAGE>`` marker (single-page requests) pass
      through.

    Register it via the CLI::

        --logits-processors \\
          vllm.model_executor.models.unlimited_ocr2:PageCountGuardLogitsProcessor

    Per-request parameters are read from ``SamplingParams.extra_args``
    (``page_token_id``, ``eos_token_id``, ``max_page_tokens``); the token ids
    fall back to ``hf_config.sparse_attention_config.page_token_id`` and
    ``hf_config.eos_token_id`` when not given. The processor is inactive for
    requests (or v1 configs) where no page token id can be resolved.

    GPU state is keyed by the persistent request slot. The prompt is scanned
    once; subsequent steps scan only newly committed output tokens.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

        if vllm_config.speculative_config is not None:
            raise ValueError("Page count guard does not support speculative decoding")
        hf_config = vllm_config.model_config.hf_config
        sparse_config = getattr(hf_config, "sparse_attention_config", None)
        self._default_page_token_id = self._read(sparse_config, "page_token_id")
        eos_token_id = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        self._default_eos_token_id = eos_token_id

        self.all_token_ids = req_states.all_token_ids
        self.prompt_len_np = req_states.prompt_len.np
        self._prompt_len = req_states.prompt_len
        self._params = StagedWriteTensor(
            (req_states.max_num_reqs, 3), torch.int64, req_states.device
        )
        self._state = StagedWriteTensor(
            (req_states.max_num_reqs, 4), torch.int64, req_states.device
        )
        self._guards: dict[int, PageCountGuard] = {}

    @staticmethod
    def _read(config: Any, key: str) -> Any:
        if config is None:
            return None
        if isinstance(config, Mapping):
            return config.get(key)
        return getattr(config, key, None)

    @staticmethod
    def _validate_token_id(name: str, value: Any):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"`{name}` has to be a non-negative integer, got {value}.")

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra_args = sampling_params.extra_args or {}
        for name in ("page_token_id", "eos_token_id"):
            if (value := extra_args.get(name)) is not None:
                cls._validate_token_id(name, value)
        value = extra_args.get("max_page_tokens")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"`max_page_tokens` has to be an integer, got {value}.")

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self._guards.pop(req_idx, None)
        self._state.stage_write(req_idx, 0, [-1, 0, 0, -1])
        extra_args = sampling_params.extra_args or {}
        page_token_id = extra_args.get("page_token_id", self._default_page_token_id)
        if page_token_id is None:
            self._params.stage_write(req_idx, 0, [-1, 0, 0])
            return False
        self._validate_token_id("page_token_id", page_token_id)

        eos_token_id = extra_args.get("eos_token_id", self._default_eos_token_id)
        if eos_token_id is None:
            raise ValueError(
                "`eos_token_id` is required for the page count guard but could "
                "not be resolved from `extra_args` or the model config."
            )
        self._validate_token_id("eos_token_id", eos_token_id)

        max_page_tokens = extra_args.get("max_page_tokens", DEFAULT_MAX_PAGE_TOKENS)
        if not isinstance(max_page_tokens, int) or isinstance(max_page_tokens, bool):
            raise ValueError(
                f"`max_page_tokens` has to be an integer, got {max_page_tokens}."
            )

        self._guards[req_idx] = PageCountGuard(
            page_token_id, eos_token_id, max_page_tokens
        )
        self._params.stage_write(
            req_idx, 0, [page_token_id, eos_token_id, max_page_tokens]
        )
        return True

    def apply_staged_writes(self) -> None:
        self._params.apply_write()
        self._state.apply_write()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self._guards:
            return logits
        _page_count_guard_kernel[(logits.shape[0],)](
            logits,
            self.all_token_ids.gpu,
            self._prompt_len.gpu,
            ctx.pos,
            ctx.expanded_idx_mapping,
            self._params.gpu,
            self._state.gpu,
            logits.stride(0),
            logits.stride(1),
            self.all_token_ids.gpu.stride(0),
            logits.shape[1],
            1024,
        )
        return logits
