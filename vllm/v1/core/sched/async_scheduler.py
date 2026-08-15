# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size
            elif self._pp_decode_cadence:
                # P3/M3b (flag VLLM_V1_PP_DECODE_CADENCE): throttle this request's
                # decode to once per pp_size steps so each step runs one
                # independent cohort the worker handoff ring overlaps across PP
                # stages. Gap == pp_size == worker ring depth (INV-RING).
                #
                # P7 Phase B (flag VLLM_PPMTP_COHORT_BALANCE): the residue
                # (next_decode_eligible_step % pp_size) is FIXED at a request's
                # first decode. [PPB-DIAG 2026-07-25] confirmed the first decode ==
                # prefill completion (ncomp==nprompt, spec=0) and a synchronized
                # burst all completes prefill on the SAME step -> all one residue ->
                # one cadence cohort full, the other empty -> the overlap window is
                # half empty (why cadence is a net loss). Rotate the residue ONLY at
                # the first decode (or preempt-resume, both marked by next==0) via
                # an engine-level admit counter; keep the +pp_size base so no gap is
                # ever shorter than pp_size (the < pp_size "runs before the slot
                # lands" corruption). offset in [0, pp_size-1] lands the first decode
                # on residue r. Subsequent decodes fall through to the plain
                # +pp_size (gap == pp_size). P4's decode-layer shift (offset on
                # EVERY decode) is the wrong version and corrupts; this rotates only
                # the first.
                #
                # P7 Phase C low-concurrency guard: only rotate when there are at
                # least pp_size running requests (enough to fill >1 cadence cohort).
                # Below that, residues already alternate naturally (no imbalance),
                # so rotating only pays the offset>0 gap-shift's one-time draft loss
                # for zero benefit ([PPB-BAL] at conc1 showed EVERY request forced
                # offset=1). len(self.running) includes the just-transitioned req.
                if (
                    self._pp_cohort_balance
                    and request.next_decode_eligible_step == 0
                    and len(self.running) >= self.pp_size
                ):
                    r = self._admit_counter % self.pp_size
                    self._admit_counter += 1
                    offset = (r - self.current_step % self.pp_size) % self.pp_size
                    request.next_decode_eligible_step = (
                        self.current_step + self.pp_size + offset
                    )
                    _n = getattr(self, "_ppb_bal_n", 0)
                    if _n < 80:
                        self._ppb_bal_n = _n + 1
                        logger.info(
                            "[PPB-BAL] step=%d req=%s r=%d offset=%d next=%d "
                            "residue=%d admit=%d",
                            self.current_step, str(request.request_id)[-6:], r,
                            offset, request.next_decode_eligible_step,
                            request.next_decode_eligible_step % self.pp_size,
                            self._admit_counter,
                        )
                else:
                    request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
