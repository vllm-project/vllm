# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) penalty as a Model Runner V2 custom logits
processor.

Penalizes tokens that would extend a token sequence already present in the
context, with a penalty growing exponentially in the repetition length
(``multiplier * base ** min(repeat_len - allowed_length, max_exp)``),
following llama.cpp's semantics and defaults. Unlike ``repetition_penalty``,
which scales tokens regardless of context, DRY targets verbatim loops.

Enable at engine init and configure per request via ``extra_args``::

    LLM(..., logits_processors=["vllm.v1.worker.gpu.sample.dry:DryState"])
    SamplingParams(extra_args={"dry_multiplier": 0.8})

Recognized extra_args keys (llama.cpp names and defaults):
``dry_multiplier`` (0.0 = off), ``dry_base`` (1.75), ``dry_allowed_length``
(2), ``dry_penalty_last_n`` (-1 = whole context), ``dry_sequence_breakers``
(llama.cpp's default set). Breaker strings are resolved to token ids with
llama.cpp's containment rule; resolution needs a tokenizer, which this
processor loads once at init from the model config.

Speculative decoding is not supported: with draft-expanded logits the
penalty is skipped with a one-time warning. The match computation lives in
``vllm.v1.sample.dry_core``.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.dry_core import _J_BUDGET, _dry_penalties, dry_core
from vllm.v1.sample.dry_utils import max_exponent, resolve_dry_breakers
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

DEFAULT_DRY_SEQUENCE_BREAKERS = ["\n", ":", '"', "*"]
"""llama.cpp's default DRY sequence breakers."""

_DRY_INT_MAX = 2**31 - 1
"""Upper bound on the integral DRY parameters, matching llama-server's
INT32_MAX cap. The state below stores them in an int64 numpy array, where
anything at or above 2**63 raises OverflowError inside execute_model
and takes the engine down."""

MAX_DRY_SEQUENCE_BREAKERS = 64
"""Upper bound on the dry_sequence_breakers list length. Each
previously-unseen breaker string costs a containment scan over the
vocabulary (cached afterwards)."""


class DryState(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        self.req_states = req_states
        max_num_reqs = req_states.max_num_reqs
        self.vocab_size = req_states.vocab_size
        self.device = req_states.device

        # float32 storage rounds multiplier/base exactly as llama.cpp's
        # float members do; the penalty math promotes to double from the
        # rounded values (see dry_core).
        self.multiplier = np.zeros(max_num_reqs, dtype=np.float32)
        self.base = np.zeros(max_num_reqs, dtype=np.float32)
        self.allowed_length = np.zeros(max_num_reqs, dtype=np.int64)
        self.penalty_last_n = np.zeros(max_num_reqs, dtype=np.int64)
        self.max_exponent = np.zeros(max_num_reqs, dtype=np.int64)
        self.use_dry = np.zeros(max_num_reqs, dtype=bool)

        # req_idx -> breaker token ids / cached [vocab] bool GPU mask.
        self.breaker_ids: dict[int, list[int]] = {}
        self._breaker_masks: dict[int, torch.Tensor] = {}

        self._warned_spec_decode = False
        self._warned_unresolved = False

        self._tokenizer = _load_tokenizer(vllm_config)

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """Validate the dry_* keys of extra_args at request admission."""
        ea = sampling_params.extra_args or {}
        # bool is an int subclass, so isinstance-based numeric checks alone
        # would accept JSON true/false for the dry_* numeric parameters.
        for name in (
            "dry_multiplier",
            "dry_base",
            "dry_allowed_length",
            "dry_penalty_last_n",
        ):
            value = ea.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a number, got {type(value).__name__}."
                )
        multiplier = ea.get("dry_multiplier", 0.0)
        base = ea.get("dry_base", 1.75)
        allowed_length = ea.get("dry_allowed_length", 2)
        penalty_last_n = ea.get("dry_penalty_last_n", -1)
        breakers = ea.get("dry_sequence_breakers", None)

        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(
                f"dry_multiplier must be non-negative and finite, got {multiplier}."
            )
        if multiplier and 0.0 <= base < 1.0:
            # libllama's own gate (llama-sampler.cpp), so this is not an error.
            # llama-server does NOT reach that gate: it coerces any base below
            # 1.0 back to its default (server-schema.cpp), so a config that
            # silently works there produces no penalty here.
            logger.warning(
                "dry_base=%s is below 1.0, which disables DRY entirely "
                "(llama.cpp semantics), even though dry_multiplier=%s was "
                "set. No repetition penalty will be applied.",
                base,
                multiplier,
            )
        if not np.isfinite(base) or base < 0.0:
            raise ValueError(f"dry_base must be non-negative and finite, got {base}.")
        # UPPER BOUNDS, not only signs. Both fields are stored into an int64
        # numpy array in the worker, and a value at or above 2**63 raises
        # OverflowError there - inside execute_model, which the engine core
        # turns into a fatal error. One request would end the server.
        # llama-server bounds both fields to INT32_MAX
        # (tools/server/server-schema.cpp), which is also far beyond any
        # useful context length, so that is the bound used here.
        if (
            not isinstance(allowed_length, int)
            or allowed_length < 0
            or allowed_length > _DRY_INT_MAX
        ):
            raise ValueError(
                f"dry_allowed_length must be an integer in [0, {_DRY_INT_MAX}], "
                f"got {allowed_length}."
            )
        if (
            not isinstance(penalty_last_n, int)
            or penalty_last_n < -1
            or penalty_last_n > _DRY_INT_MAX
        ):
            raise ValueError(
                "dry_penalty_last_n must be an integer: -1 (whole context), "
                f"0 (disable), or in [1, {_DRY_INT_MAX}], got {penalty_last_n}."
            )
        if breakers is not None and (
            not isinstance(breakers, (list, tuple))
            or any(not isinstance(s, str) for s in breakers)
        ):
            raise ValueError(
                f"dry_sequence_breakers must be a list of strings, got {breakers!r}."
            )
        if breakers is not None and len(breakers) > MAX_DRY_SEQUENCE_BREAKERS:
            raise ValueError(
                f"dry_sequence_breakers supports at most "
                f"{MAX_DRY_SEQUENCE_BREAKERS} entries, got {len(breakers)}."
            )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        ea = sampling_params.extra_args or {}
        multiplier = ea.get("dry_multiplier", 0.0)
        base = ea.get("dry_base", 1.75)
        allowed_length = ea.get("dry_allowed_length", 2)
        penalty_last_n = ea.get("dry_penalty_last_n", -1)
        # Same gate as llama_sampler_dry_apply.
        enabled = use_dry(multiplier, base, penalty_last_n)
        self.use_dry[req_idx] = enabled
        self.breaker_ids.pop(req_idx, None)
        self._breaker_masks.pop(req_idx, None)
        if not enabled:
            return False
        self.multiplier[req_idx] = multiplier
        self.base[req_idx] = base
        self.allowed_length[req_idx] = allowed_length
        self.penalty_last_n[req_idx] = penalty_last_n
        self.max_exponent[req_idx] = max_exponent(float(self.base[req_idx]))

        breakers = ea.get("dry_sequence_breakers", DEFAULT_DRY_SEQUENCE_BREAKERS)
        if breakers and self._tokenizer is not None:
            self.breaker_ids[req_idx] = resolve_dry_breakers(
                self._tokenizer, tuple(breakers)
            )
        elif breakers and not self._warned_unresolved:
            # No tokenizer was loaded at init (e.g. skip_tokenizer_init), so
            # breaker strings cannot be resolved to token ids.
            logger.warning(
                "DRY sequence breakers were not resolved to token ids; "
                "proceeding without breakers."
            )
            self._warned_unresolved = True
        return True

    def _breaker_mask(self, req_idx: int) -> torch.Tensor | None:
        ids = self.breaker_ids.get(req_idx)
        if not ids:
            return None
        mask = self._breaker_masks.get(req_idx)
        if mask is None:
            # BUILT ON THE HOST, then transferred once. Every device-side route
            # here synchronizes: boolean-mask indexing (``ids_t[ids_t < V]``)
            # reads the result size back, and an indexed write with a device
            # index tensor does the same. numpy does the selection for free and
            # one pinned copy of a [vocab] bool array (128 KB at 128k) leaves
            # nothing to read back. The result is cached per request, so this is
            # paid once, not per step.
            ids_np = np.asarray(ids, dtype=np.int64)
            ids_np = ids_np[ids_np < self.vocab_size]
            m_np = np.zeros(self.vocab_size, dtype=bool)
            m_np[ids_np] = True
            mask = async_tensor_h2d(m_np, self.device)
            self._breaker_masks[req_idx] = mask
        return mask

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        req_indices = ctx.idx_mapping_np
        active_rows = np.flatnonzero(self.use_dry[req_indices])
        if active_rows.size == 0:
            return logits
        if logits.shape[0] != ctx.idx_mapping_np.shape[0]:
            if not self._warned_spec_decode:
                logger.warning(
                    "DRY is not applied with speculative decoding yet; "
                    "requests with dry_multiplier set are unaffected."
                )
                self._warned_spec_decode = True
            return logits

        # NO DEVICE READ HERE, deliberately. The context visible to the token being
        # sampled is [0, seq_len), and the scheduler has already materialized those
        # lengths on the host (ctx.seq_lens_upper_bound_np, num_computed +
        # num_scheduled). Reading ``positions`` off the GPU and adding one gives the
        # same number and costs a host-device synchronization on every step a DRY
        # request is live, in the sampler, which is the hot path for every request
        # in the engine. The field is an upper bound because speculative decoding
        # can schedule more tokens than it accepts; DRY skips spec-decode batches
        # (above), so for these requests the bound is exact.
        cur_len = ctx.seq_lens_upper_bound_np[active_rows].astype(np.int64)

        reqs = req_indices[active_rows]
        last_n = self.penalty_last_n[reqs]
        window_len = np.where(last_n == -1, cur_len, np.minimum(cur_len, last_n))
        allowed = self.allowed_length[reqs]
        keep = window_len > allowed
        if not np.any(keep):
            return logits
        active_rows = active_rows[keep]
        reqs = reqs[keep]
        cur_len = cur_len[keep]
        window_len = window_len[keep]
        allowed = allowed[keep]
        max_exp = self.max_exponent[reqs]

        # Route degenerate-clamp requests (base <= 1.000001 or oversized
        # cap) through the sequential reference implementation.
        fast = (max_exp > 0) & (allowed + max_exp <= _J_BUDGET)
        # dry_core reduces its penalty accumulator with amax, which picks the longest
        # match only while the penalty is non-decreasing in the match length: with a
        # base below 1 it would pick the shortest, and with a negative multiplier
        # every candidate would lose to the zero initialiser and nothing would be
        # penalized. use_dry() and validate_params() already guarantee both. Checked
        # on the numpy copies, because the same check inside dry_core would have to
        # read the device tensors back, which is the sync this path exists to avoid.
        assert (self.base[reqs[fast]] >= 1.0).all(), "dry_base < 1 reached dry_core"
        assert (self.multiplier[reqs[fast]] >= 0.0).all(), "dry_multiplier < 0"
        all_tokens = self.req_states.all_token_ids.gpu

        if np.any(fast):
            f_rows = active_rows[fast]
            f_reqs = reqs[fast]
            f_len = window_len[fast]
            N = int(f_len.max())
            reqs_t = async_tensor_h2d(f_reqs, self.device)
            cur_t = async_tensor_h2d(cur_len[fast], self.device)
            j = torch.arange(N, device=self.device)
            # Right-aligned gather: column j holds token (cur_len - N + j);
            # out-of-window columns are masked inside dry_core via n_r.
            gather_idx = (cur_t[:, None] - N + j[None, :]).clamp(min=0)
            W = all_tokens[reqs_t[:, None], gather_idx].long()
            dry_core(
                logits,
                row_idx=async_tensor_h2d(f_rows, self.device),
                W=W,
                n_r=async_tensor_h2d(f_len, self.device),
                allowed=async_tensor_h2d(allowed[fast], self.device),
                max_exp=async_tensor_h2d(max_exp[fast], self.device),
                mult=async_tensor_h2d(self.multiplier[f_reqs], self.device),
                base=async_tensor_h2d(self.base[f_reqs], self.device),
                breaker_masks=[self._breaker_mask(r) for r in f_reqs],
                j_budget=int((allowed[fast] + max_exp[fast]).max()),
            )

        # THE SEQUENTIAL FALLBACK SYNCHRONIZES, and cannot not: it runs a Z-algorithm
        # in Python over the window, so the window has to come to the host. It is
        # reached two ways, both visible in `fast` above: max_exp == 0, which is a
        # dry_base at or below 1.000001, or allowed + max_exp over _J_BUDGET, which
        # at the default dry_allowed_length is a dry_base up to about 1.044 and at a
        # dry_allowed_length at or over _J_BUDGET is any base at all. Marked
        # explicitly rather than left to trip VLLM_GPU_SYNC_CHECK, so the exemption
        # is a decision in the source and not a surprise. The vectorized path is
        # sync-free.
        slow = ~fast
        if np.any(slow):
            with gpu_sync_allowed():
                rows_list = []
                cols_list = []
                vals_list = []
                for row, req, w_len, cur in zip(
                    active_rows[slow], reqs[slow], window_len[slow], cur_len[slow]
                ):
                    window = (
                        all_tokens[int(req), int(cur) - int(w_len) : int(cur)]
                        .cpu()
                        .tolist()
                    )
                    penalties = _dry_penalties(
                        window,
                        frozenset(self.breaker_ids.get(int(req), ())),
                        float(self.multiplier[req]),
                        float(self.base[req]),
                        int(self.allowed_length[req]),
                        int(self.max_exponent[req]),
                    )
                    for tok, val in penalties.items():
                        rows_list.append(int(row))
                        cols_list.append(tok)
                        vals_list.append(val)
                if rows_list:
                    logits[
                        torch.tensor(rows_list, dtype=torch.int64, device=self.device),
                        torch.tensor(cols_list, dtype=torch.int64, device=self.device),
                    ] -= torch.tensor(
                        vals_list, dtype=torch.float32, device=self.device
                    )
        return logits


def _load_tokenizer(vllm_config: "VllmConfig") -> "TokenizerLike | None":
    """Load the model's tokenizer for breaker resolution, once at init.

    Returns None when the engine runs without a tokenizer
    (skip_tokenizer_init); breaker strings are then dropped with a warning
    at add_request time.
    """
    if vllm_config is None:
        return None
    model_config = vllm_config.model_config
    if model_config.skip_tokenizer_init:
        return None
    # Deferred import: loading a tokenizer pulls in transformers.
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.tokenizer_revision,
    )


def use_dry(multiplier: float, base: float, penalty_last_n: int) -> bool:
    """Whether DRY modifies logits for a request with these parameters."""
    return bool(multiplier) and base >= 1.0 and penalty_last_n != 0
