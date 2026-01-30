# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SCHEDULER_CLS = {
    "ewsjf": "vllm.v1.core.sched.ewsjf_scheduler.scheduler.EWSJFScheduler",
    "chunked_prefill": (
        "vllm.v1.core.sched.ewsjf_scheduler.chunked_prefill_scheduler."
        "ChunkedPrefillScheduler"
    ),
    "chunked_prefill_decode": (
        "vllm.v1.core.sched.ewsjf_scheduler.chunked_prefill_decode_scheduler."
        "ChunkedPrefillSchedulerDecode"
    ),
    "fcfs": "vllm.v1.core.sched.scheduler.Scheduler",
    "sjf": "vllm.v1.core.sched.ewsjf_scheduler.sjf_scheduler.SJFScheduler",
}
