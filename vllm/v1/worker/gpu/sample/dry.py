# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DRY (Don't Repeat Yourself) penalty for the V2 model runner.

State module following the ``PenaltiesState`` pattern. The match
computation is shared with the V1-runner logits processor
(``vllm.v1.sample.logits_processor.dry_batched.dry_core``); windows are
gathered directly from the GPU-resident ``req_states.all_token_ids``, so
no per-step host-to-device copy of token history is needed.

Speculative decoding is not supported yet (matching min_p/logit_bias in
the V1 runner): with expanded draft logits DRY is skipped with a one-time
warning.
"""

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.dry_utils import max_exponent
from vllm.v1.sample.logits_processor.dry import _dry_penalties
from vllm.v1.sample.logits_processor.dry_batched import _J_BUDGET, dry_core
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
        # rounded values (see dry_batched).
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
            mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
            ids_t = torch.tensor(ids, dtype=torch.int64, device=self.device)
            mask[ids_t[ids_t < self.vocab_size]] = True
            self._breaker_masks[req_idx] = mask
        return mask

    def apply_dry(
        self,
        logits: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
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

        # ``pos`` holds the position of the last input token of each logits
        # row; the context visible to the token being sampled is
        # [0, pos + 1). Using ``pos`` alone drops the final context token
        # and shifts every match by one: the penalty lands on the
        # continuation of the previous suffix instead of the token being
        # sampled. Single small D2H sync, only when DRY is in use.
        active_t = torch.from_numpy(active_rows).to(self.device)
        cur_len = pos[active_t].cpu().numpy().astype(np.int64) + 1

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
        all_tokens = self.req_states.all_token_ids.gpu

        if np.any(fast):
            f_rows = active_rows[fast]
            f_reqs = reqs[fast]
            f_len = window_len[fast]
            N = int(f_len.max())
            reqs_t = torch.from_numpy(f_reqs).to(self.device)
            cur_t = torch.from_numpy(cur_len[fast]).to(self.device)
            j = torch.arange(N, device=self.device)
            # Right-aligned gather: column j holds token (cur_len - N + j);
            # out-of-window columns are masked inside dry_core via n_r.
            gather_idx = (cur_t[:, None] - N + j[None, :]).clamp(min=0)
            W = all_tokens[reqs_t[:, None], gather_idx].long()
            dry_core(
                logits,
                row_idx=torch.from_numpy(f_rows).to(self.device),
                W=W,
                n_r=torch.from_numpy(f_len).to(self.device),
                allowed=torch.from_numpy(allowed[fast]).to(self.device),
                max_exp=torch.from_numpy(max_exp[fast]).to(self.device),
                mult=torch.from_numpy(self.multiplier[f_reqs]).to(self.device),
                base=torch.from_numpy(self.base[f_reqs]).to(self.device),
                breaker_masks=[self._breaker_mask(r) for r in f_reqs],
            )

        slow = ~fast
        if np.any(slow):
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
                ] -= torch.tensor(vals_list, dtype=torch.float32, device=self.device)


def use_dry(sampling_params: SamplingParams) -> bool:
    return (
        bool(sampling_params.dry_multiplier)
        and sampling_params.dry_base >= 1.0
        and sampling_params.dry_penalty_last_n != 0
    )
