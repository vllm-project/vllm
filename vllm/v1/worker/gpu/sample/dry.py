# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) penalty as a custom logits processor.

Penalizes tokens that would extend a token sequence already present in the
context, with a penalty growing exponentially in the length of the repeated
sequence. Parameter names, defaults and matching semantics follow
llama.cpp's ``llama_sampler_dry_apply``.

Load it by FQCN and configure it per request through ``extra_args``::

    LLM(..., logits_processors=["vllm.v1.worker.gpu.sample.dry:DryState"])
    SamplingParams(extra_args={"dry_multiplier": 0.8})

The recognized keys are ``dry_multiplier`` (0.0, off), ``dry_base`` (1.75),
``dry_allowed_length`` (2), ``dry_penalty_last_n`` (-1, the whole context)
and ``dry_sequence_breakers``. This processor claims the ``dry_*`` namespace:
any other ``dry_*`` key fails the request.

The match computation lives in ``vllm.v1.sample.dry_core``; windows are
gathered directly from the GPU-resident ``req_states.all_token_ids``, so no
per-step host-to-device copy of token history is needed. Speculative
decoding is not supported and is refused at construction.
"""

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.dry_core import _J_BUDGET, _dry_penalties, dry_core
from vllm.v1.sample.dry_utils import max_exponent, resolve_dry_breakers
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Phrased after STR_SPEC_DEC_REJECTS_LOGITSPROCS in
# vllm.v1.sample.logits_processor, which refuses the V1 equivalent.
STR_SPEC_DEC_REJECTS_DRY = (
    "The DRY logits processor is not supported when speculative decoding is enabled."
)

DEFAULT_DRY_SEQUENCE_BREAKERS = ("\n", ":", '"', "*")
"""llama.cpp's default DRY sequence breakers."""

MAX_DRY_SEQUENCE_BREAKERS = 64
"""Upper bound on the dry_sequence_breakers list length. A previously-unseen
multi-character breaker string costs a containment scan over the vocabulary
(cached afterwards); single-character ones resolve by lookup."""

_DRY_INT_MAX = 2**31 - 1
"""Upper bound on the integral DRY parameters, matching llama-server's
INT32_MAX cap. The state below stores them in an int64 numpy array, where
anything at or above 2**63 raises OverflowError inside execute_model and
takes the engine down."""

_MAX_CACHED_BREAKER_MASKS = 64
"""Upper bound on distinct breaker sets holding a device mask, mirroring
dry_utils._MAX_CACHED_BREAKER_SETS on the host side. Evicted sets rebuild
from one host-to-device copy."""

_DRY_DEFAULTS: dict[str, Any] = {
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "dry_sequence_breakers": DEFAULT_DRY_SEQUENCE_BREAKERS,
}
"""The recognized extra_args keys and llama.cpp's defaults for them. The one
place either is written down: validate_params and add_request both read it."""


def _dry_args(sampling_params: SamplingParams) -> dict[str, Any]:
    """Read the request's DRY arguments, filling in the defaults.

    A key present with value None counts as unset, matching how
    ``SamplingParams`` coerces its own None-valued arguments.
    """
    extra = sampling_params.extra_args or {}
    return {
        name: default if extra.get(name) is None else extra[name]
        for name, default in _DRY_DEFAULTS.items()
    }


class DryState(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", req_states: LogitsProcRequestState):
        if vllm_config.speculative_config is not None:
            raise ValueError(STR_SPEC_DEC_REJECTS_DRY)
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

        # req_idx -> its breaker set, and breaker set -> [vocab] bool device
        # mask shared by every request that asked for the same breakers.
        self.breaker_ids: dict[int, frozenset[int]] = {}
        self._breaker_masks: dict[frozenset[int], torch.Tensor] = {}

        self._warned_unresolved = False

        # Deferred import: the tokenizer registry pulls in transformers, and
        # the frontend imports this module only to validate params.
        from vllm.tokenizers import cached_tokenizer_from_config

        # None under skip_tokenizer_init.
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        # EAGERLY, because add_request runs inside execute_model: resolving a
        # breaker set decodes the whole vocabulary, which would block the batch.
        # A client's own set then resolves off that cached decode.
        self._default_breaker_ids = self._resolve_breakers(
            DEFAULT_DRY_SEQUENCE_BREAKERS
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """Check the ``dry_*`` keys of ``extra_args`` at request admission.

        Raises:
            ValueError: on an unknown or out-of-range DRY argument. The
                loader turns it into a ``VLLMValidationError``, which is why
                this cannot raise one itself: ``VLLMValidationError`` is not
                a ``ValueError``, so the loader would not catch it.

        """
        extra = sampling_params.extra_args or {}
        unknown = sorted(
            key for key in extra if key.startswith("dry_") and key not in _DRY_DEFAULTS
        )
        if unknown:
            # Rejected rather than ignored: a misspelled key would otherwise
            # disable DRY, or a parameter of it, without saying so.
            raise ValueError(
                f"Unknown dry_* extra_args: {', '.join(unknown)}. "
                f"Supported keys: {', '.join(_DRY_DEFAULTS)}."
            )

        args = _dry_args(sampling_params)
        for name in (
            "dry_multiplier",
            "dry_base",
            "dry_allowed_length",
            "dry_penalty_last_n",
        ):
            value = args[name]
            # bool is an int subclass, so an isinstance check alone would
            # accept JSON true/false for a numeric parameter.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a number, got {type(value).__name__}."
                )

        multiplier = args["dry_multiplier"]
        base = args["dry_base"]
        allowed_length = args["dry_allowed_length"]
        penalty_last_n = args["dry_penalty_last_n"]
        breakers = args["dry_sequence_breakers"]

        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError(
                f"dry_multiplier must be non-negative and finite, got {multiplier}."
            )
        if not math.isfinite(base) or base < 0.0:
            raise ValueError(f"dry_base must be non-negative and finite, got {base}.")
        # UPPER BOUNDS, not only signs. Both integer fields are stored into an
        # int64 numpy array in the worker, where a value at or above 2**63
        # raises OverflowError inside execute_model; the engine core escalates
        # that to a fatal error, so one request would end the server.
        # llama-server's own cap is INT32_MAX (tools/server/server-schema.cpp),
        # far past any useful context length, so that is the bound used here.
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
        if not isinstance(breakers, (list, tuple)) or any(
            not isinstance(s, str) for s in breakers
        ):
            raise ValueError(
                f"dry_sequence_breakers must be a list of strings, got {breakers!r}."
            )
        if len(breakers) > MAX_DRY_SEQUENCE_BREAKERS:
            raise ValueError(
                f"dry_sequence_breakers supports at most "
                f"{MAX_DRY_SEQUENCE_BREAKERS} entries, got {len(breakers)}."
            )
        if multiplier and 0.0 <= base < 1.0:
            # libllama's own gate (llama-sampler.cpp), so this is not an error.
            # llama-server does NOT reach that gate: it coerces any base below
            # 1.0 back to its default (server-schema.cpp), so a config that
            # silently works there produces no penalty here. That is the reason
            # to say something rather than nothing.
            logger.warning(
                "dry_base=%s is below 1.0, which disables DRY entirely "
                "(llama.cpp semantics), even though dry_multiplier=%s was "
                "set. No repetition penalty will be applied.",
                base,
                multiplier,
            )

    def _resolve_breakers(self, breakers: tuple[str, ...]) -> frozenset[int]:
        if self._tokenizer is None or not breakers:
            return frozenset()
        return frozenset(resolve_dry_breakers(self._tokenizer, breakers))

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        args = _dry_args(sampling_params)
        multiplier = args["dry_multiplier"]
        base = args["dry_base"]
        penalty_last_n = args["dry_penalty_last_n"]
        enabled = use_dry(multiplier, base, penalty_last_n)
        self.use_dry[req_idx] = enabled
        self.breaker_ids.pop(req_idx, None)
        if not enabled:
            return False
        self.multiplier[req_idx] = multiplier
        self.base[req_idx] = base
        self.allowed_length[req_idx] = args["dry_allowed_length"]
        self.penalty_last_n[req_idx] = penalty_last_n
        self.max_exponent[req_idx] = max_exponent(float(self.base[req_idx]))

        breakers = tuple(args["dry_sequence_breakers"])
        ids = (
            self._default_breaker_ids
            if breakers == DEFAULT_DRY_SEQUENCE_BREAKERS
            else self._resolve_breakers(breakers)
        )
        if ids:
            self.breaker_ids[req_idx] = ids
        elif breakers and self._tokenizer is None and not self._warned_unresolved:
            logger.warning(
                "DRY sequence breakers were not resolved to token ids: this "
                "engine has no tokenizer. Proceeding without breakers."
            )
            self._warned_unresolved = True
        return True

    def _breaker_mask(self, req_idx: int) -> torch.Tensor | None:
        ids = self.breaker_ids.get(req_idx)
        if not ids:
            return None
        mask = self._breaker_masks.get(ids)
        if mask is None:
            # Built on the host: every device-side route here reads a size
            # back and synchronizes. Paid once per distinct breaker set.
            ids_np = np.fromiter(ids, dtype=np.int64, count=len(ids))
            ids_np = ids_np[ids_np < self.vocab_size]
            m_np = np.zeros(self.vocab_size, dtype=bool)
            m_np[ids_np] = True
            mask = async_tensor_h2d(m_np, self.device)
            if len(self._breaker_masks) >= _MAX_CACHED_BREAKER_MASKS:
                # Oldest first. MAX_DRY_SEQUENCE_BREAKERS bounds the length of
                # one list, not how many distinct sets clients can ask for.
                self._breaker_masks.pop(next(iter(self._breaker_masks)))
            self._breaker_masks[ids] = mask
        return mask

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        req_indices = ctx.idx_mapping_np
        active_rows = np.flatnonzero(self.use_dry[req_indices])
        if active_rows.size == 0:
            return logits
        # Draft tokens would give a request more than one row; __init__ refuses
        # to run under speculative decoding, so this holds.
        assert logits.shape[0] == req_indices.shape[0], (
            "DRY received draft-expanded logits"
        )

        # host-side bound; reading positions off GPU would sync per step
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

        # The Python Z-algorithm fallback has to bring the window to the host.
        # Declared here rather than left to trip VLLM_GPU_SYNC_CHECK.
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
                        self.breaker_ids.get(int(req), frozenset()),
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


def use_dry(multiplier: float, base: float, penalty_last_n: int) -> bool:
    """Whether DRY changes logits for a request with these parameters.

    The same gate as ``llama_sampler_dry_apply``, and the only guarantee
    ``dry_core`` has that ``base >= 1.0``: it reduces its penalty accumulator
    with ``amax``, which picks the longest match only while the penalty does
    not decrease in the match length. A base below 1.0 would pick the
    shortest instead. ``validate_params`` only warns about such a base, for
    llama.cpp parity, so it is this predicate returning False - and
    ``add_request`` returning False with it - that keeps one out of the
    kernel. A negative multiplier, which would lose to the accumulator's zero
    initializer, ``validate_params`` does reject outright.
    """
    return bool(multiplier) and base >= 1.0 and penalty_last_n != 0
