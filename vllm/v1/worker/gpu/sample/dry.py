# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) penalty for the model runner.

State module following the ``PenaltiesState`` pattern. The match
computation lives in ``vllm.v1.sample.dry_core``; windows are gathered
directly from the GPU-resident ``req_states.all_token_ids``, so no
per-step host-to-device copy of token history is needed.

Speculative decoding is not supported. Requests enabling DRY are refused
up front by ``SamplingParams._validate_spec_decode``; the expanded-logits
branch below is a defensive backstop and should be unreachable on normal
paths. It cannot be the primary gate, because it keys on the logits being
draft-expanded, which is false on any step where no request happens to
carry draft tokens - so relying on it applied DRY intermittently.
"""

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.dry_core import _J_BUDGET, _dry_penalties, dry_core
from vllm.v1.sample.dry_utils import max_exponent
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)


class DryState:
    def __init__(self, req_states: RequestState):
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

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        # Same gate as llama_sampler_dry_apply.
        enabled = use_dry(sampling_params)
        self.use_dry[req_idx] = enabled
        self.breaker_ids.pop(req_idx, None)
        self._breaker_masks.pop(req_idx, None)
        if not enabled:
            return
        self.multiplier[req_idx] = sampling_params.dry_multiplier
        self.base[req_idx] = sampling_params.dry_base
        self.allowed_length[req_idx] = sampling_params.dry_allowed_length
        self.penalty_last_n[req_idx] = sampling_params.dry_penalty_last_n
        self.max_exponent[req_idx] = max_exponent(float(self.base[req_idx]))

        ids = sampling_params._dry_breaker_ids
        if ids:
            self.breaker_ids[req_idx] = list(ids)
        elif (
            ids is None
            and sampling_params.dry_sequence_breakers
            and not self._warned_unresolved
        ):
            # The engine frontend resolves breaker strings to ids
            # (SamplingParams.update_from_tokenizer). Reaching here means
            # that step was skipped (e.g. skip_tokenizer_init).
            logger.warning(
                "DRY sequence breakers were not resolved to token ids; "
                "proceeding without breakers."
            )
            self._warned_unresolved = True

    def apply_staged_writes(self) -> None:
        # All state is CPU-side numpy; nothing to stage.
        pass

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

    def apply_dry(
        self,
        logits: torch.Tensor,
        idx_mapping_np: np.ndarray,
        seq_lens_np: np.ndarray | None,
        expanded_logits: bool,
    ) -> None:
        req_indices = idx_mapping_np
        active_rows = np.flatnonzero(self.use_dry[req_indices])
        if active_rows.size == 0:
            return
        if expanded_logits:
            if not self._warned_spec_decode:
                logger.warning(
                    "DRY is not applied with speculative decoding yet; "
                    "requests with dry_multiplier set are unaffected."
                )
                self._warned_spec_decode = True
            return
        if seq_lens_np is None:
            # Reachable only from a caller that has a live DRY request and did not pass
            # the host-side lengths. In-tree there is none - the one caller that omits
            # them is the rejection sampler, and DRY is refused under speculative
            # decoding - but silently is the wrong way to find that out.
            raise ValueError(
                "apply_dry needs seq_lens_np (InputBatch.seq_lens_cpu_upper_bound) "
                "when any request in the batch enables DRY"
            )

        # NO DEVICE READ HERE, deliberately. The context visible to the token being
        # sampled is [0, seq_len), and the model runner has already materialized those
        # lengths on the host as ``InputBatch.seq_lens_cpu_upper_bound``
        # (num_computed_tokens_np + num_scheduled). Reading ``positions`` off the GPU
        # and adding one gives the same number and costs a host-device synchronization
        # on every step a DRY request is live, in the sampler, which is the hot path for
        # every request in the engine. The name says "upper bound" because speculative
        # decoding can schedule more tokens than it accepts; DRY is refused under
        # speculative decoding (SamplingParams._validate_spec_decode), so for these
        # requests the bound is exact.
        cur_len = seq_lens_np[active_rows].astype(np.int64)

        reqs = req_indices[active_rows]
        last_n = self.penalty_last_n[reqs]
        window_len = np.where(last_n == -1, cur_len, np.minimum(cur_len, last_n))
        allowed = self.allowed_length[reqs]
        keep = window_len > allowed
        if not np.any(keep):
            return
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
        # base below 1 it would pick the shortest, and with a negative multiplier every
        # candidate would lose to the zero initialiser and nothing would be penalized.
        # use_dry() and SamplingParams._verify_args() already guarantee both. Checked on
        # the numpy copies, because the same check inside dry_core would have to read
        # the device tensors back, which is the sync this path exists to avoid.
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
        # dry_base at or below 1.000001, or allowed + max_exp over _J_BUDGET, which at
        # the default dry_allowed_length is a dry_base up to about 1.044 and at a
        # dry_allowed_length at or over _J_BUDGET is any base at all. Marked explicitly
        # rather than left to trip VLLM_GPU_SYNC_CHECK, so the exemption is a decision
        # in the source and not a surprise. The vectorized path is sync-free.
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


def use_dry(sampling_params: SamplingParams) -> bool:
    return (
        bool(sampling_params.dry_multiplier)
        and sampling_params.dry_base >= 1.0
        and sampling_params.dry_penalty_last_n != 0
    )
