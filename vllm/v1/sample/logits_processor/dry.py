# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) repetition penalty.

Penalizes tokens that would extend a token sequence already present in the
context, with a penalty growing exponentially in the length of the repeated
sequence: ``multiplier * base ** (repeat_length - allowed_length)`` is
subtracted from the logit. ``repetition_penalty`` scales individual
tokens regardless of context; DRY targets verbatim loops.

The matching semantics follow llama.cpp's ``llama_sampler_dry_apply``
(itself ported from the koboldcpp implementation by pi6am; the DRY scheme
was designed by p-e-w for text-generation-webui), so identical settings
produce identical behavior in both runtimes. Parameter names and defaults
also follow llama.cpp.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.dry_utils import max_exponent
from vllm.v1.sample.logits_processor import dry_batched
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _dry_penalties(
    window: list[int],
    breakers: frozenset[int],
    multiplier: float,
    base: float,
    allowed_length: int,
    max_exp: int,
) -> dict[int, float]:
    """Compute DRY penalties for one request.

    Port of the scan in ``llama_sampler_dry_apply``: a reverse-direction
    Z-algorithm finds, for each position, the length of the match between
    the window suffix and the sequence ending at that position; each
    match's follower token is charged the penalty for the longest repeat
    it would extend.

    Returns a dict mapping token id -> penalty to subtract from its logit.
    """
    m = len(window)

    def rat(i: int) -> int:  # reverse access: rat(0) == last token
        return window[m - 1 - i]

    # Step 1: the nearest breaker from the end caps the match length.
    rep_limit = m
    for i in range(m):
        if rat(i) in breakers:
            rep_limit = i
            break
    if rep_limit < allowed_length:
        return {}

    # Step 2: reverse-direction Z-algorithm -> per-position repeat counts
    # (forward-indexed into ``window`` via cnt[last - k]).
    cnt = [0] * m
    last = m - 1
    lt = rt = 0
    for k in range(1, m):
        if k > rt:
            # Outside the current Z-box: extend naively.
            z = 0
            while z + k < m and rat(z) == rat(z + k):
                z += 1
            cnt[last - k] = min(z, rep_limit)
            if z > 0:
                lt, rt = k, k + z - 1
        else:
            p = k - lt
            right_part_len = rt - k + 1
            if cnt[last - p] < right_part_len:
                # Fully inside the Z-box: copy.
                cnt[last - k] = min(cnt[last - p], rep_limit)
            else:
                # Touches the right edge: extend past it.
                j = rt + 1
                while j < m and rat(j) == rat(j - k):
                    j += 1
                cnt[last - k] = min(j - k, rep_limit)
                lt, rt = k, j - 1

    # Step 3: map each repeat's follower token to the longest repeat that
    # it would extend.
    max_token_repeat: dict[int, int] = {}
    for i in range(m - 1):
        repeat_len = cnt[i]
        if repeat_len >= allowed_length:
            tok = window[i + 1]
            if max_token_repeat.get(tok, -1) < repeat_len:
                max_token_repeat[tok] = repeat_len

    # Step 4: exponential penalty, exponent clamped for float32 safety.
    # Breaker tokens are never penalized. A value overflowing float32
    # saturates the logit to -inf downstream, as in llama.cpp.
    penalties: dict[int, float] = {}
    for tok, repeat_len in max_token_repeat.items():
        if tok in breakers:
            continue
        exponent = repeat_len - allowed_length
        if max_exp and exponent > max_exp:
            exponent = max_exp
        penalties[tok] = multiplier * (base**exponent)
    return penalties


@dataclass
class _DryState:
    multiplier: float
    base: float
    allowed_length: int
    penalty_last_n: int
    breakers: frozenset[int]
    max_exponent: int
    prompt_tok_ids: list[int]
    # Live reference to the request's running output list (see the
    # BatchUpdate contract in interface.py); grows every step.
    output_tok_ids: list[int] = field(default_factory=list)
    # Lazily built by dry_batched._breaker_vocab_mask.
    _breaker_mask: torch.Tensor | None = None

    def __post_init__(self):
        # llama.cpp stores dry_multiplier and dry_base as float32 and the
        # penalty is computed from those rounded values (promoted to
        # double inside std::pow). Round once here so the sequential and
        # batched paths both match llama.cpp bit-for-bit (e.g. base=1.1
        # is 1.10000002384... in float32).
        self.multiplier = float(np.float32(self.multiplier))
        self.base = float(np.float32(self.base))

    def window(self) -> list[int] | None:
        """Trailing ``penalty_last_n`` tokens of prompt + output.

        Returns None when the effective window cannot produce a penalty.
        llama.cpp additionally caps the window at the model context size;
        vLLM's history cannot exceed the model context, so the cap is
        implicit here.
        """
        n_out = len(self.output_tok_ids)
        hist_len = len(self.prompt_tok_ids) + n_out
        if self.penalty_last_n == -1:
            last_n = hist_len
        else:
            last_n = min(hist_len, self.penalty_last_n)
        if last_n <= self.allowed_length:
            return None
        if last_n <= n_out:
            return self.output_tok_ids[-last_n:]
        return self.prompt_tok_ids[n_out - last_n :] + self.output_tok_ids


class DryLogitsProcessor(LogitsProcessor):
    """Builtin DRY penalty processor.

    Sparse per-request state (only requests with ``dry_multiplier > 0``);
    no-op when no request in the batch enables DRY.
    """

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        self.device = device
        self.reqs: dict[int, _DryState] = {}
        # Batched-vs-sequential dispatch: None = auto (batched on CUDA),
        # True/False force one path (used by tests and benchmarks).
        self.use_batched: bool | None = None
        self._warned_unresolved = False

    def is_argmax_invariant(self) -> bool:
        """DRY moves logits and can change the greedy argmax."""
        return False

    def needs_output_token_ids(self) -> bool:
        """DRY scans generation history, so the output token id lists must
        be kept fresh (see the base class: under async scheduling they are
        otherwise one token stale)."""
        return bool(self.reqs)

    def _new_state(
        self,
        params: SamplingParams,
        prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> _DryState | None:
        multiplier = params.dry_multiplier
        base = params.dry_base
        penalty_last_n = params.dry_penalty_last_n
        # Same gate as llama_sampler_dry_apply: disabled configurations
        # are not tracked.
        if not multiplier or base < 1.0 or penalty_last_n == 0:
            return None
        # Breaker strings are resolved to token ids by the engine frontend
        # (SamplingParams.update_from_tokenizer). If that step was skipped
        # (e.g. skip_tokenizer_init or direct library use), proceed without
        # breakers rather than loading a tokenizer in the worker.
        resolved = params._dry_breaker_ids
        if (
            resolved is None
            and params.dry_sequence_breakers
            and not self._warned_unresolved
        ):
            logger.warning(
                "DRY sequence breakers were not resolved to token ids; "
                "proceeding without breakers."
            )
            self._warned_unresolved = True
        return _DryState(
            multiplier=multiplier,
            base=base,
            allowed_length=params.dry_allowed_length,
            penalty_last_n=penalty_last_n,
            breakers=frozenset(resolved or ()),
            max_exponent=max_exponent(base),
            prompt_tok_ids=prompt_tok_ids or [],
            output_tok_ids=output_tok_ids,
        )

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self.reqs, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.reqs:
            return logits
        candidates: list[tuple[int, _DryState, list[int] | None]] = [
            (req_index, state, state.window()) for req_index, state in self.reqs.items()
        ]
        if self.use_batched is not None:
            use_batched = self.use_batched
        else:
            # The vectorized path has ~0.5 ms of fixed launch/H2D cost; the
            # sequential scan wins below ~2k total window tokens (measured
            # on an RTX PRO 6000; the crossover is flat across batch shapes).
            total_window = sum(len(w) for _, _, w in candidates if w is not None)
            use_batched = logits.device.type == "cuda" and total_window >= 2048
        batched_entries: list[tuple[int, _DryState, list[int]]] = []
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for req_index, state, window in candidates:
            if window is None:
                continue
            if use_batched and dry_batched.eligible(state):
                batched_entries.append((req_index, state, window))
                continue
            penalties = _dry_penalties(
                window,
                state.breakers,
                state.multiplier,
                state.base,
                state.allowed_length,
                state.max_exponent,
            )
            for tok, val in penalties.items():
                rows.append(req_index)
                cols.append(tok)
                vals.append(val)
        if batched_entries:
            logits = dry_batched.apply_dry_batched(logits, batched_entries)
        if rows:
            # float32 conversion saturates oversized penalties to inf, so
            # the logit lands at -inf, as in llama.cpp.
            logits[
                async_tensor_h2d(rows, self.device, torch.int64),
                async_tensor_h2d(cols, self.device, torch.int64),
            ] -= async_tensor_h2d(vals, self.device, torch.float32)
        return logits
